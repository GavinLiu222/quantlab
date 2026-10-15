"""Send a notification e-mail (in Chinese) for the daily data update and the paper trading.

The cron wrappers call it when a job ends: ``scripts/data_update/update_daily.sh`` after
each data update, quantlab-ibkr ``scripts/live_daily.sh`` after the morning and the
record step. Standard library only, so any interpreter runs it.

The body is read from stdin (``--body -``) or given with ``--body``; ``--log FILE`` appends
the file's last ``--tail`` lines (from line ``--from-line`` on); ``--update-status FILE``
writes the body from the data update's ``update_status.json`` and appends its state and t
to the subject.

``--html FILE`` adds an HTML body beside the plain one, and each ``--image CID=PATH`` (repeatable)
an inline image the HTML shows with ``<img src="cid:CID">``. The message is then
``multipart/related``: a ``multipart/alternative`` (the plain body, then the HTML) followed by the
images, each with its content id; a client without HTML shows the plain body. Without ``--html``
the message is a single plain-text part. ``--image`` without ``--html``, a malformed ``CID=PATH``,
a repeated CID, a missing or unreadable file or one that is not a PNG, JPEG or GIF is refused
(exit 2) before anything is sent.

Environment (``~/.config/quantlab/mail.env``, readable by the owner only)::

    QUANTLAB_SMTP_HOST      default smtp.163.com
    QUANTLAB_SMTP_PORT      default 465 (SSL); 587 uses STARTTLS
    QUANTLAB_SMTP_USER      the sending account, also the From address
    QUANTLAB_SMTP_PASSWORD  its SMTP authorisation code (not the login password)
    QUANTLAB_MAIL_TO        the recipients, comma separated

Without a user, password or recipient nothing is sent and the exit status is 0: a
missing mail setup never fails a job. A send that fails exits 1; the wrappers ignore it.

Usage::

    echo "正文" | python scripts/notify/send_mail.py --subject "主题" --body -
    python scripts/notify/send_mail.py --subject "数据更新" \\
        --update-status /data/quantlab/update_status.json --log update.log
    python scripts/notify/send_mail.py --subject "收盘简报" --body "纯文本正文" \\
        --html briefing.html --image nav=nav.png --image drawdown=drawdown.png
"""

from __future__ import annotations  # the system python3 (3.9 on macOS) runs it too

import argparse
import json
import mimetypes
import os
import smtplib
import socket
import ssl
import sys
from collections.abc import Sequence
from email.header import Header
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr, formatdate, make_msgid
from pathlib import Path
from typing import NamedTuple

#: The Chinese name of each state of ``update_status.json``.
UPDATE_STATES = {
    "done": "完成",
    "no_new_bar": "没有新数据（节假日或供应商未发布）",
    "failed": "失败",
    "running": "仍在运行（进程可能异常退出）",
}


def tail(path: Path, lines: int, from_line: int = 1) -> str:
    """The last ``lines`` lines of ``path`` from line ``from_line`` (1-based) on."""
    try:
        text = path.read_text(errors="replace").splitlines()[max(from_line, 1) - 1:]
    except OSError as error:
        return f"（无法读取日志 {path}：{error}）"
    return "\n".join(text[-lines:])


def update_status_body(path: Path) -> str:
    """A Chinese summary of the data update's ``update_status.json``."""
    try:
        status = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        return f"无法读取状态文件 {path}：{error}"
    state = status.get("state")
    lines = [
        f"日期：{status.get('date')}",
        f"状态：{UPDATE_STATES.get(state, state)}",
        f"最新数据日 t：{status.get('t')}",
        f"上次完成的数据日：{status.get('last_done_t')}",
        f"开始：{status.get('started')}",
        f"结束：{status.get('finished') or status.get('updated')}",
    ]
    if status.get("reason"):
        lines.append(f"原因：{status['reason']}")
    steps = status.get("steps") or []
    if steps:
        lines += ["", f"步骤（{len(steps)}）："]
        for step in steps:
            name = step.get("store") or step.get("script") or ",".join(step.get("writes", []))
            seconds = step.get("seconds")
            took = f"，{seconds:.0f} 秒" if isinstance(seconds, (int, float)) else ""
            lines.append(f"- [{step.get('stage')}] {step.get('action')} {name}：{step.get('result')}{took}")
    return "\n".join(lines)


class InlineImage(NamedTuple):
    """An image shown in the HTML body as ``cid:<cid>``."""

    cid: str
    name: str
    subtype: str
    data: bytes


#: The image types mail clients show inline (SVG and WebP are not shown by many).
INLINE_TYPES = {"image/png", "image/jpeg", "image/gif"}


def parse_image(spec: str) -> InlineImage:
    """Read ``CID=PATH``; ValueError when malformed, unreadable or not a PNG, JPEG or GIF."""
    cid, sep, path = spec.partition("=")
    cid, path = cid.strip(), path.strip()
    if not sep or not cid or not path or any(c in cid for c in "<>@ \t"):
        raise ValueError(f"--image {spec!r}: expected CID=PATH (CID without spaces, <, > or @)")
    path = Path(path)
    kind = mimetypes.guess_type(path.name)[0]
    if kind not in INLINE_TYPES:
        raise ValueError(f"--image {spec!r}: {path.name} is not a .png, .jpg or .gif")
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ValueError(f"--image {spec!r}: {error}") from None
    return InlineImage(cid, path.name, kind.split("/", 1)[1], data)


def build_message(subject: str, body: str, user: str, to: list[str], html: str | None = None,
                  images: Sequence[InlineImage] = ()) -> MIMEText | MIMEMultipart:
    """The message: plain text alone, or with ``html`` a related part of plain/HTML and images."""
    if html is None:
        message = MIMEText(body, "plain", "utf-8")
    else:
        message = MIMEMultipart("related", type="multipart/alternative")
        alternative = MIMEMultipart("alternative")
        alternative.attach(MIMEText(body, "plain", "utf-8"))
        alternative.attach(MIMEText(html, "html", "utf-8"))
        message.attach(alternative)
        for inline in images:
            image = MIMEImage(inline.data, _subtype=inline.subtype)
            image["Content-ID"] = f"<{inline.cid}>"
            image.add_header("Content-Disposition", "inline", filename=inline.name)
            message.attach(image)
    message["Subject"] = Header(subject, "utf-8")
    message["From"] = formataddr((str(Header(f"quantlab@{socket.gethostname()}", "utf-8")), user))
    message["To"] = ", ".join(to)
    message["Date"] = formatdate(localtime=True)
    message["Message-ID"] = make_msgid()
    return message


def send(subject: str, body: str, html: str | None = None,
         images: Sequence[InlineImage] = ()) -> bool:
    """Send ``body`` (and ``html`` with ``images``) to ``QUANTLAB_MAIL_TO``.

    False (and a note on stderr) when not configured.
    """
    user = os.environ.get("QUANTLAB_SMTP_USER", "")
    password = os.environ.get("QUANTLAB_SMTP_PASSWORD", "")
    to = [a.strip() for a in os.environ.get("QUANTLAB_MAIL_TO", "").split(",") if a.strip()]
    if not (user and password and to):
        print("mail not configured (QUANTLAB_SMTP_USER/PASSWORD, QUANTLAB_MAIL_TO), skipped",
              file=sys.stderr)
        return False
    host = os.environ.get("QUANTLAB_SMTP_HOST", "smtp.163.com")
    port = int(os.environ.get("QUANTLAB_SMTP_PORT", "465"))
    message = build_message(subject, body, user, to, html, images)
    context = ssl.create_default_context()
    if port == 465:
        with smtplib.SMTP_SSL(host, port, context=context, timeout=60) as smtp:
            smtp.login(user, password)
            smtp.sendmail(user, to, message.as_string())
    else:
        with smtplib.SMTP(host, port, timeout=60) as smtp:
            smtp.starttls(context=context)
            smtp.login(user, password)
            smtp.sendmail(user, to, message.as_string())
    return True


def main() -> None:
    """Parse the arguments, write the body and send it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subject", required=True)
    parser.add_argument("--body", default="", help="The body; '-' reads stdin.")
    parser.add_argument("--update-status", type=Path, help="Write the body from this update_status.json.")
    parser.add_argument("--log", type=Path, help="Append this log's last lines.")
    parser.add_argument("--tail", type=int, default=60)
    parser.add_argument("--from-line", type=int, default=1, help="Read the log from this line on.")
    parser.add_argument("--html", type=Path, help="Add this HTML file as the HTML body.")
    parser.add_argument("--image", action="append", default=[], metavar="CID=PATH",
                        help="Attach PATH inline as content id CID (repeatable; needs --html).")
    args = parser.parse_args()
    if args.image and not args.html:
        parser.error("--image needs --html")
    html = None
    if args.html:
        try:
            html = args.html.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            parser.error(f"--html {args.html}: {error}")
    try:
        images = [parse_image(spec) for spec in args.image]
    except ValueError as error:
        parser.error(str(error))
    cids = [image.cid for image in images]
    repeated = sorted({cid for cid in cids if cids.count(cid) > 1})
    if repeated:
        parser.error(f"--image: content id given more than once: {', '.join(repeated)}")
    parts, subject = [], args.subject
    if args.update_status:
        parts.append(update_status_body(args.update_status))
        try:
            status = json.loads(args.update_status.read_text())
            state = status.get("state")
            subject += f"：{str(UPDATE_STATES.get(state, state)).split('（')[0]}，t = {status.get('t')}"
        except (OSError, ValueError):
            subject += "：状态文件不可读"
    if args.body:
        parts.append(sys.stdin.read() if args.body == "-" else args.body)
    if args.log:
        parts.append(f"—— 日志 {args.log} 最后 {args.tail} 行 ——\n"
                     + tail(args.log, args.tail, args.from_line))
    parts.append(f"（发送自 {socket.gethostname()}）")
    try:
        sent = send(subject, "\n\n".join(p.strip("\n") for p in parts if p), html, images)
    except (OSError, smtplib.SMTPException) as error:
        print(f"mail failed: {error!r}", file=sys.stderr)
        sys.exit(1)
    if sent:
        print(f"mail sent: {subject}")


if __name__ == "__main__":
    main()

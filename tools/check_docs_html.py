"""Check generated documentation for broken local links and leaked RST markup."""

import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class Page(HTMLParser):
    def __init__(self, html: str):
        super().__init__()
        self.ids: set[str] = set()
        self.links: list[str] = []
        self.markup: list[str] = []
        self.in_article = False
        self.skip = 0
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        if tag == "a" and "href" in attrs:
            self.links.append(attrs["href"])
        if tag == "article":
            self.in_article = True
        if tag in {"pre", "code", "script", "style"}:
            self.skip += 1

    def handle_endtag(self, tag):
        if tag == "article":
            self.in_article = False
        if tag in {"pre", "code", "script", "style"}:
            self.skip = max(0, self.skip - 1)

    def handle_data(self, text):
        if self.in_article and not self.skip and "``" in text:
            self.markup.append(text.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Sphinx HTML output directory")
    root = parser.parse_args().directory.resolve()
    if not (root / "index.html").is_file():
        raise SystemExit(f"No built documentation index at {root}")
    pages = {p.resolve(): Page(p.read_text()) for p in root.rglob("*.html")}
    failures: set[str] = set()
    for path, page in pages.items():
        label = path.relative_to(root)
        for text in page.markup:
            if "_modules" not in label.parts:
                failures.add(f"{label}: unparsed inline markup: {text!r}")
        for href in page.links:
            url = urlsplit(href)
            if url.scheme or url.netloc:
                continue
            target = (path.parent / unquote(url.path)).resolve() if url.path else path
            if target.is_dir():
                target /= "index.html"
            if not target.exists():
                failures.add(f"{label}: missing target: {href}")
            elif (
                url.fragment
                and target in pages
                and unquote(url.fragment) not in pages[target].ids
            ):
                failures.add(f"{label}: missing anchor: {href}")
    for failure in sorted(failures):
        print(failure)
    if failures:
        raise SystemExit(f"Documentation checks failed: {len(failures)} issue(s).")
    print(f"Checked {len(pages)} HTML pages: no broken local links or unparsed inline markup.")


if __name__ == "__main__":
    main()

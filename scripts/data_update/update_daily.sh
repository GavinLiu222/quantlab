#!/usr/bin/env bash
# The daily data update (scripts/data_update/update.py, config/data_update.yaml) from cron.
#
# cron calls this every hour (Debian cron has no CRON_TZ); it checks the time in
# New York itself, so daylight-saving changes need no edit:
#
#   0 * * * * $HOME/projects/quantlab2/scripts/data_update/update_daily.sh
#
# At 04 ET on a weekday it runs the update, which retries the vendor download every
# 15 minutes until 08:30 ET while the day's bar is not published, and writes
# <data-dir>/update_status.json; the paper trading (quantlab-ibkr live_daily.sh) waits
# for "done" on its day. `update_daily.sh now` runs it at once.
#
# Each run ends with an e-mail (in Chinese) of the status file and the log's tail,
# sent by scripts/notify/send_mail.py; a missing or failing mail setup never fails the run.
#
# Environment (files readable by the owner only):
#   ~/.config/quantlab/sharadar.env  SHARADAR_API_KEY
#   $QUANTLAB_DIR/.env               MASSIVE_API_KEY, MASSIVE_S3_ACCESS_KEY_ID (the intraday
#                                    stage; cron does not read ~/.zshenv, which loads it)
#   ~/.config/quantlab/mail.env      QUANTLAB_SMTP_USER, QUANTLAB_SMTP_PASSWORD,
#                                    QUANTLAB_MAIL_TO (see scripts/notify/send_mail.py)
# Overrides: DATA_DIR, QUANTLAB_DIR, CPUS, UPDATE_FLAGS (e.g. "--dry-run").
set -u

DATA_DIR=${DATA_DIR:-/data/quantlab}
QUANTLAB_DIR=${QUANTLAB_DIR:-$HOME/projects/quantlab2}
CPUS=${CPUS:-64-114}
PY=$QUANTLAB_DIR/.venv/bin/python

ny() { TZ=America/New_York date "$@"; }
log_dir=$DATA_DIR/logs/data_update
mkdir -p "$log_dir"
log_file=$log_dir/$(ny +%F).log

run_update() {
    exec 9> "$log_dir/.lock"
    flock -n 9 || { echo "$(ny '+%F %T %Z') already running" >> "$log_file"; return 0; }
    load_env "$HOME/.config/quantlab/sharadar.env"
    load_env "$QUANTLAB_DIR/.env"
    echo "$(ny '+%F %T %Z') data update: start ${UPDATE_FLAGS:-}" >> "$log_file"
    # shellcheck disable=SC2086
    (cd "$QUANTLAB_DIR" && QUANTLAB_DATA_DIR=$DATA_DIR OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 taskset -c "$CPUS" "$PY" scripts/data_update/update.py \
        --data-dir "$DATA_DIR" --download-dir "$DATA_DIR/downloads" ${UPDATE_FLAGS:-}) >> "$log_file" 2>&1
    local status=$?
    echo "$(ny '+%F %T %Z') data update: exit $status" >> "$log_file"
    notify "$status"
    return "$status"
}

# E-mail the outcome: the status file (state, t, steps) and the log's last lines.
notify() {
    local flags=""
    [ "$1" -eq 0 ] || flags="--log $log_file --tail 80"
    case "${UPDATE_FLAGS:-}" in *--dry-run*|*--stage*|*--check*) return 0 ;; esac
    (load_env "$HOME/.config/quantlab/mail.env"
     # shellcheck disable=SC2086
     "$PY" "$QUANTLAB_DIR/scripts/notify/send_mail.py" --subject "【quantlab 数据更新】$(ny +%F) exit $1" \
         --update-status "$DATA_DIR/update_status.json" $flags) >> "$log_file" 2>&1 || true
}

load_env() {
    # shellcheck disable=SC1090
    [ -r "$1" ] && { set -a; . "$1"; set +a; }
}

case "${1:-cron}" in
    now) run_update ;;
    cron)
        [ "$(ny +%u)" -le 5 ] || exit 0
        [ "$(ny +%H)" = 04 ] && run_update
        ;;
    *) echo "usage: $0 [cron|now]" >&2; exit 2 ;;
esac

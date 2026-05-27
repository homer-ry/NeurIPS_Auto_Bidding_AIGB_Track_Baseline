# Project Memory

## KML/Jupyter Remote Command Access

This project can execute commands on the KML development machine through an already logged-in Google Chrome Jupyter page. The flow is:

```text
Codex local shell
  -> osascript controls Google Chrome
  -> logged-in KML/Jupyter /tree page
  -> Jupyter terminal API
  -> remote shell
```

Use the local helper:

```bash
python3 .codex_tmp/kml_jupyter_run.py '<remote command>' \
  --browser-app '/Applications/Google Chrome.app' \
  --window-id 121257493 \
  --tab-index 31 \
  --close-after 8 \
  --wait 10
```

Known working smoke test:

```bash
python3 .codex_tmp/kml_jupyter_run.py 'pwd && hostname' \
  --browser-app '/Applications/Google Chrome.app' \
  --window-id 121257493 \
  --tab-index 31 \
  --close-after 8 \
  --wait 10
```

Observed remote environment:

```text
/home/rongyu03
ad-bjx-gt10-ad203.idchb1az1.hb1.kwaidc.com
```

`nvidia-smi` works through this path. As of 2026-05-25, the machine showed 2 NVIDIA A10 GPUs with no active GPU processes.

If the Chrome window or tab changes, rediscover Jupyter tabs with:

```bash
osascript -e 'tell application "Google Chrome"
set out to ""
repeat with w in windows
repeat with i from 1 to count of tabs of w
set t to tab i of w
set u to URL of t
if u contains "kml-dtmachine" or u contains "kml-hb2az1" then
set out to out & (id of w) & "|" & i & "|" & (title of t as text) & "|" & u & linefeed
end if
end repeat
end repeat
return out
end tell'
```

Important notes:

- Google Chrome must have the KML/Jupyter page open and logged in.
- Chrome's AppleScript command syntax must use `execute active tab of window id ... javascript ...`; the Edge-style `execute javascript ... in tab ...` form fails in Chrome with `not allowed`.
- The helper creates a Jupyter terminal through `POST /api/terminals`, connects to `/terminals/websocket/<terminal-name>`, sends stdin, reads stdout, and then cleans up the terminal.
- For long-running remote jobs, start them with `nohup` and redirect logs, then poll with `ps`, `tail`, and `nvidia-smi`.
- Do not write KML tokens, cookies, passwords, or private keys into markdown files or the repository. This flow relies only on the existing browser login state.

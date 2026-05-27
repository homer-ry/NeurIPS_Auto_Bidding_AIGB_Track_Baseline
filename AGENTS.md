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

## uax_algo_1 Remote Machine

When the user says "在 uax_algo_1 执行", use the KML machine terminal below.

Machine terminal URL:

```text
https://kml.corp.kuaishou.com/v2/#/system/project/10098/machine-terminal/19951?fullScreen=1&originPid=10098&provider=undefined
```

Expected remote workspace root for this project family:

```bash
/share/rongyu03/rl/wentou
```

Preferred local entrypoint is the Universe Model webshell CLI. Resolve the current installed script path first because the plugin version can change:

```bash
WEBSHELL_CLI="$(find /Users/rongyu/.codex/plugins/cache -path '*/skills/webshell/scripts/webshell_cli.py' -type f | sort | tail -1)"
PATH=/usr/local/bin:$PATH "$WEBSHELL_CLI" exec \
  --url 'https://kml.corp.kuaishou.com/v2/#/system/project/10098/machine-terminal/19951?fullScreen=1&originPid=10098&provider=undefined' \
  --cmd 'pwd && hostname && whoami && date && nvidia-smi' \
  --timeout 30 \
  --total-timeout 120
```

If the `/v2/#/...` URL form times out, retry the normalized URL form:

```bash
WEBSHELL_CLI="$(find /Users/rongyu/.codex/plugins/cache -path '*/skills/webshell/scripts/webshell_cli.py' -type f | sort | tail -1)"
PATH=/usr/local/bin:$PATH "$WEBSHELL_CLI" exec \
  --url 'https://kml.corp.kuaishou.com/#/system/project/10098/machine-terminal/19951?fullScreen=1&originPid=10098' \
  --cmd 'pwd && hostname && whoami && date && nvidia-smi' \
  --timeout 30 \
  --total-timeout 120
```

For long-running jobs on `uax_algo_1`, start the command with `nohup` and write logs under the remote project directory, then poll with `ps`, `tail`, and `nvidia-smi`:

```bash
PATH=/usr/local/bin:$PATH "$WEBSHELL_CLI" exec \
  --url 'https://kml.corp.kuaishou.com/#/system/project/10098/machine-terminal/19951?fullScreen=1&originPid=10098' \
  --cmd 'cd /share/rongyu03/rl/wentou/<project> && nohup bash run.sh > run.log 2>&1 & echo PID=$!' \
  --timeout 10 \
  --total-timeout 120
```

Operational notes:

- `uv` is required locally for `webshell_cli.py`; if `env: uv: No such file or directory` appears, ensure `/usr/local/bin` is on `PATH` or install with `HOMEBREW_NO_AUTO_UPDATE=1 brew install uv`.
- This connection path relies on local Google Chrome / Kit / 0Pass login state. Do not store or print cookies, tokens, passwords, or private keys.
- If `webshell_cli` times out while acquiring token or cookies, open the machine terminal URL in Google Chrome and confirm KML/KaiWorks login manually, then retry. A useful reset is to remove stale webshell/K0Pass lock/cache files only for `kml.corp.kuaishou.com` / `kaiworks.corp.kuaishou.com`, then force-refresh cookies through the CLI.
- If the machine exposes a Jupyter `/tree` page instead of a machine-terminal page, use `.codex_tmp/kml_jupyter_run.py` as documented in the previous section.

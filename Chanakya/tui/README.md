# Chanakya TUI

Terminal UI for the Chanakya AI agent using `@mariozechner/pi-tui`.

## Setup

1. Start the Python server in one terminal:
   ```bash
   cd D:\Chanakya\agent
   python -m uvicorn server:app --host 127.0.0.1 --port 8000
   ```

2. Start the TUI in another terminal:
   ```bash
   cd D:\Chanakya\tui
   run-dev.cmd
   ```

   If you prefer `npm`, use `npm.cmd run dev` in PowerShell. Plain `npm run dev` can fail on Windows when `npm.ps1` is blocked by execution policy.

## Usage

- Enter a URL or article text in the input field
- Press Enter to submit
- Type `exit` to quit

## Project Structure

```text
Chanakya/
|-- agent/            # Python backend
|   |-- agent.py      # Original Python agent
|   `-- server.py     # FastAPI HTTP server used by the TUI
`-- tui/              # pi-tui frontend
    |-- package.json
    |-- tsconfig.json
    `-- src/index.ts  # TUI application
```

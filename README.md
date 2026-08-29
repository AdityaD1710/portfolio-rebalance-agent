# Portfolio Rebalance Agent
Setup for portfolio rebalance agent
Built for **The Agent Harness Hackathon** (WeMakeDevs x TrueFoundry) — "Best Use of TrueForge" track.

An agent that fetches your real Alpaca paper-trading portfolio, computes a
target-allocation rebalance in a sandboxed Python function, proposes trades
in plain language, and pauses for explicit human approval before executing
anything.

## What it does

1. Fetches current positions and cash balance via **Alpaca's official MCP
   server** (real paper-trading account — real tool reach, not mocked data).
2. Runs a verified rebalance calculation as actual Python, executed in a
   **Daytona sandbox** provisioned by TrueForge — not just an LLM guessing
   at arithmetic in text.
3. Proposes trades only for positions that have drifted more than 5
   percentage points from target, explaining the reasoning in plain
   language.
4. **Pauses for approval** before placing any trade — TrueForge/the Alpaca
   MCP server flags `place_stock_order` as requiring explicit human
   sign-off, with Allow/Deny controls, before anything irreversible
   happens.

## Architecture

```
TrueForge (Docker, localhost:8791)
  ├── Model: Mistral (mistral-small-latest, custom OpenAI-compatible provider)
  ├── Sandbox: Daytona
  └── Connector: alpaca-paper-trading-v2
        └── mcp-proxy (stdio→SSE bridge, runs on host, port 8000)
              └── alpaca-mcp-server (official, stdio-only)
                    └── Alpaca paper-trading account
```

## Setup

### Prerequisites
- Docker Desktop
- Python 3 + [uv](https://docs.astral.sh/uv/)
- An Alpaca account with paper-trading API keys
- A Mistral API key
- A Daytona account + API key

### 1. Start TrueForge
```bash
git clone <this repo>
cd portfolio-rebalance-agent
docker compose up -d
```
Open `http://localhost:8791`.

### 2. Bridge Alpaca's MCP server
Alpaca's official MCP server is stdio-only; TrueForge's custom connectors
need a URL, so it's bridged with `mcp-proxy`:
```bash
uv tool install mcp-proxy --with "mcp<2.0.0"
mcp-proxy --port=8000 \
  -e ALPACA_API_KEY <your_key> \
  -e ALPACA_SECRET_KEY <your_secret> \
  -e ALPACA_PAPER_TRADE true \
  -- uvx alpaca-mcp-server
```
Leave this running for as long as TrueForge needs to reach Alpaca.

### 3. Configure TrueForge
- **Settings → Models**: add Mistral as a custom OpenAI-compatible
  provider (`https://api.mistral.ai/v1`), model `mistral-small-latest`.
- **Settings → Sandbox providers**: connect Daytona with your API key.
- **Settings → Connectors → Add MCP Server**: point at
  `http://host.docker.internal:8000/sse`, Auth type: None.

### 4. Run it
Start a chat, enable the Alpaca connector, and ask it to rebalance your
portfolio toward a target allocation (e.g. "30% AAPL, 30% MSFT, 40% SPY").
See `rebalance.py` for the exact calculation logic used in the sandbox.

## Environment variables

| Variable | Where it's used |
|---|---|
| `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` | Passed to `mcp-proxy`, never stored in TrueForge itself |
| `ALPACA_PAPER_TRADE` | Set to `true` — this project only ever trades on Alpaca's paper account |
| Mistral API key | Configured directly in TrueForge's Models settings |
| Daytona API key | Configured directly in TrueForge's Sandbox providers settings |

No secrets are committed to this repository.

## Qodo Code Review Evidence

<!-- TODO: link the PR once Qodo's review is visible on it -->
PR #1: `Add target-allocation rebalance calculation` —
[link](https://github.com/AdityaD1710/portfolio-rebalance-agent/pull/1)

## Demo

<!-- TODO: link the 3-minute demo video here -->

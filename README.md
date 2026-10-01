# Jzbedin

A voice AI butler built on [LiveKit Agents](https://docs.livekit.io/agents/) and Gemini's
realtime model. Jzbedin talks to you, looks things up when it needs to, and drives a real
Chromium browser on your behalf.

The project is a fork of
[`livekit-examples/agent-starter-python`](https://github.com/livekit-examples/agent-starter-python),
extended with a browser, web search, a rewritten instruction set, and two clients.

## What makes it different

Most demo agents greet you, answer one question, and stop. Jzbedin was built against three
specific complaints:

- **Replies that stopped half way through.** The instructions now require a complete answer,
  and a single turn's output is capped so the agent cannot monologue past the point of use.
- **Invented facts.** The instructions tell the model to distinguish what a tool returned from
  what it is guessing, and to say when a result is too thin to support an answer.
- **A greeting that forgot its own name.** The opening line is a fixed constant spoken with
  `session.say()` rather than generated, so the agent always introduces itself. If you offer
  your name in the first reply, Jzbedin remembers it and uses it for the rest of the session.

Interruption handling is tuned so Jzbedin stops talking when you start talking, instead of
finishing a sentence over you.

## Tools

Jzbedin has ten tools. Nine drive a Chromium instance through Playwright, and one searches
the web.

| Tool | Purpose |
| --- | --- |
| `browse` | Open a page and return its visible structure. |
| `browse_click` | Click an element by its inspected target. |
| `browse_interact` | Type into a field, then optionally submit. |
| `browse_nav` | Go back, forward, or reload. |
| `browse_tabs` | List, open, or close browser tabs. |
| `browse_key` | Press a key such as `Enter` or `Tab`. |
| `scroll` | Scroll the page. |
| `screenshot` | Save a PNG of the current viewport. |
| `confirm_browser_action` | Record that the user approved a consequential action. |
| `search_web` | Search the web via DuckDuckGo. |

The browser tools share a single inspection path, so the model gets the same shape of
response whichever tool it calls. Submissions and `Enter` presses are consequential: they
require a `confirm_browser_action` call first, and the approval is consumed by the very next
action. A confirmation cannot be reused for a second click.

## Getting started

You need [uv](https://docs.astral.sh/uv/), a LiveKit Cloud project, and a Google AI Studio
key for the realtime model.

```bash
uv sync
uv run playwright install chromium
```

Create a `.env.local` and fill it in:

```
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=
GOOGLE_API_KEY=
```

Get the LiveKit keys from
[cloud.livekit.io/settings/keys](https://cloud.livekit.io/settings/keys) and the Google key
from [aistudio.google.com/apikey](https://aistudio.google.com/apikey). `.env.local` is
gitignored; never commit real keys.

Run the agent locally:

```bash
uv run src/agent.py dev
```

The same bootstrap is available as `task dev` if you use [Task](https://taskfile.dev).

### Configuration

| Variable | Meaning |
| --- | --- |
| `JZBEDIN_BROWSER_HEADLESS` | Set to `1` to hide the Chromium window. |
| `JZBEDIN_BROWSER_SCREENSHOT_DIR` | Where `screenshot` writes PNGs. Defaults to `./screenshots`. |

## Clients

Two clients talk to the agent. Both are part of this repository and were flattened out of
their own git histories when they were brought in.

- `frontend/` — Next.js web client with LiveKit's agent UI components.
- `livekit_flutter_starter/` — Flutter client for iOS, Android, macOS, and web.

The web client needs a token endpoint to mint session tokens:

```bash
cd frontend
pnpm install
pnpm dev
```

The client expects a token endpoint at `app/api/token/route.ts` to mint session tokens. The
version included is a development token server; see `frontend/README.md` before deploying it.

## Testing

```bash
uv run pytest          # unit and tool tests
uv run ruff check src tests
uv run ruff format --check src tests
```

The tests run in CI on every push. `tests/test_agent.py` covers agent behaviour, and
`tests/test_browser_tools.py` covers the browser tools and the confirmation gate.

Behaviour that needs a live conversation is covered by simulations instead. Each scenario in
`scenarios.yaml` is a full conversation with a simulated user, judged against expectations:

```bash
lk agent simulate --scenarios scenarios.yaml
```

These run in CI on every merge to `main`. When you change the instructions, the tool
descriptions, or a workflow, add a scenario for what you changed and write the test first.

## Deployment

The included `Dockerfile` runs the agent as a worker:

```bash
docker build -t jzbedin .
docker run -p 8081:8081 --env-file .env.local jzbedin
```

The image already sets `JZBEDIN_BROWSER_HEADLESS=1` and installs Chromium, so the browser
tools work in the container. The agent in `start` mode registers with LiveKit and serves a
health check on port 8081; set `PORT` to change it.

For a hosted deployment, see the
[LiveKit Agents documentation](https://docs.livekit.io/agents/build/deployment/).

## Repository layout

```
src/agent.py           the agent: session setup, greeting, name memory, turn limits
src/prompts.py         system instructions and the fixed greeting
src/browser_tools.py   the nine Playwright tools and the confirmation gate
src/tools.py           web search
frontend/              Next.js client
livekit_flutter_starter/  Flutter client
scenarios.yaml         simulation scenarios
```

`AGENTS.md` has notes for working on this codebase, including the LiveKit documentation
tooling and the testing conventions.

## Credits

Built on the
[LiveKit Agents Python starter](https://github.com/livekit-examples/agent-starter-python).
LiveKit provides the realtime transport, Gemini provides the model, and Playwright provides
the browser.


# Care for the Carers

Care for the Carers is a Telegram chatbot for family caregivers who are under
stress, built first for carers of people with dementia in Hong Kong. It listens,
replies like a kind friend would by text, notices when stress is building and
offers local support services. If a message suggests someone may be at risk, it
always shows crisis hotlines.

The project was built by a team of students from the Chinese University of Hong
Kong (CUHK). It won the CUHK round of the World's Challenge Challenge and came
runner-up at the world final hosted by Western University, earning CAD$15,000
in kick-start funding.

**Version 2** rebuilds the bot around a testable agent core, with an
evaluation harness that measures whether the bot is kind, brief and, above all,
safe. It also makes the bot easy to install and host.

---

## Quick start

You need Python 3.10+ (or Docker) and, for real replies, an
[Anthropic API key](https://console.anthropic.com/). Without a key, you can use
the offline `scripted` model or a local model through Ollama.

```bash
pip install "git+https://github.com/csauyong/Care-for-the-Carers"
# or: uv tool install "git+https://github.com/csauyong/Care-for-the-Carers"

# 1. Try it in your terminal, no keys needed (canned replies):
carebot --provider scripted chat

# 2. With Claude:
export ANTHROPIC_API_KEY=sk-ant-...
carebot chat

# 3. On Telegram: get a token from @BotFather, then
export TELEGRAM_BOT_TOKEN=123456:ABC...
carebot run
```

Settings can also live in a `.env` file. Copy [`.env.example`](.env.example) and
fill it in. `carebot check` validates your settings and makes one test call.

## Hosting

The bot keeps its sessions in a SQLite file and its traces in a JSONL file, both
under `CAREBOT_DATA_DIR` (default `./data`). It needs no database server.

| Where | How |
| --- | --- |
| **Docker** (any server, NAS, Raspberry Pi) | `cp .env.example .env`, fill it in, then `docker compose up -d`. Data persists in the `carebot-data` volume. |
| **Fully local, no API key** | Run a model with [Ollama](https://ollama.com). Set `CAREBOT_PROVIDER=openai`, `CAREBOT_MODEL=llama3.1` and `OPENAI_BASE_URL=http://ollama:11434/v1`, then `docker compose --profile local up -d` and `docker compose exec ollama ollama pull llama3.1`. |
| **Platforms that run containers** (Render, Railway, Fly.io, Cloud Run, ...) | Deploy the Dockerfile and set `CAREBOT_WEBHOOK_URL` to the service's public `https://` URL. The bot then uses a webhook instead of polling and listens on `$PORT`, with a health check at `/healthz`. Attach a persistent disk at `/data` or sessions reset on every deploy. |
| **Plain Python** | `pip install .` then `carebot run` under systemd, tmux or similar. |

**Polling or webhook?** Polling (the default) only needs outbound internet, so
it works behind home routers and firewalls. Webhook mode suits hosts that
expect an HTTP server or that sleep idle containers. Telegram sends a secret
header with each update, and the bot rejects requests without it.

## Configuration

| Variable | Default | |
| --- | --- | --- |
| `TELEGRAM_BOT_TOKEN` | – | From @BotFather. `TELEGRAM_API_KEY` from v1 also works. |
| `CAREBOT_PROVIDER` | `anthropic` | `anthropic`, `openai` (any OpenAI-compatible server) or `scripted` (offline). |
| `CAREBOT_MODEL` | `claude-opus-5-5` | Required for `openai`. |
| `CAREBOT_EFFORT` | `low` | Claude's reasoning effort. `low` suits short chat replies. |
| `CAREBOT_DETECTOR` | `keyword` | `keyword` (free, instant) or `llm` (Claude rates each message; better recall). |
| `CAREBOT_RESOURCES_FILE` | Hong Kong | YAML with your own hotlines and services. Copy [`hong_kong.yaml`](src/carebot/data/hong_kong.yaml). |
| `CAREBOT_TRACE_CONTENT` | `true` | `false` keeps message text out of traces. |
| `CAREBOT_WEBHOOK_URL` | – | Set to switch from polling to webhook mode. |

The full list, with comments, is in [`.env.example`](.env.example).

## How it works

```
Telegram ─┐                        ┌──────────── CareAgent (the harness) ────────────┐
terminal ─┼─▶ respond(user, text) ─▶│ 1. detector     stress score + crisis screen    │
eval run ─┘                         │ 2. session      SQLite, trimmed history window  │
                                    │ 3. model        writes the conversational reply │──▶ Claude / Ollama / scripted
                                    │ 4. policy       crisis lines? offer services?   │
                                    │ 5. trace        JSONL: signals, actions, tokens │
                                    └──────────────────────────────────────────────────┘
```

The model only writes the conversation. Everything safety-relevant is ordinary
code that can be tested:

- **Crisis screen.** Pattern rules in English and Cantonese run on every message,
  whatever detector or model is in use. If a message matches, the model is told
  to respond to the risk, and the crisis hotlines are appended verbatim. They are
  shown even if the model call fails or the model refuses.
- **Referral policy.** Stress accumulates with decay. Crossing the threshold
  triggers one offer with Yes/No buttons, followed by a cooldown. (Version 1
  re-offered on every message once the threshold was crossed.)
- **Grounding.** The model may only mention contacts from the resources file.
  The eval harness fails any reply that contains a phone number or link not in
  that file.
- **Privacy.** Traces hash user IDs and can drop all message text. `/forget`
  deletes a user's stored history.

Telegram commands: `/start`, `/resources`, `/forget`.

## Evaluation

The bot talks to people at a vulnerable moment, so a prompt tweak or a model
change has to be shown to be safe before it ships. The harness tests the bot at
five levels, from cheap and exact to expensive and nuanced:

| Layer | What it catches | Command | Cost |
| --- | --- | --- | --- |
| Unit tests | Harness logic: referral policy, crisis fallback, history window, webhook auth | `pytest` | free |
| Detector benchmark | Precision and recall of stress and crisis detection on [labelled messages](evals/detector_cases.yaml). **CI fails if crisis recall drops below 100%.** | `carebot eval detector` | free (keyword) |
| Scripted scenarios | Multi-turn conversations with expectations on each turn: crisis shown, service offered once, no invented phone numbers, no dosage advice | `carebot eval scenarios` | free offline, or live with any model |
| LLM judge | Empathy, style, safety and relevance, scored 1–5 against a rubric, plus criteria specific to each scenario | `carebot eval scenarios --judge` | judge calls |
| Simulated caregivers | A model plays a [persona](evals/personas/) (a stoic husband, a Cantonese-speaking son, a hidden crisis) for several turns, then the judge grades the conversation | `carebot eval personas` | most |

Every run writes `eval-results/<timestamp>/report.json` and `report.md`. Each
report records the model, the detector, a hash of the system prompt and the git
commit. Models are not deterministic, so use `--trials N`. The report gives
each case's pass rate, and whether all trials passed, which is the bar for
safety cases.

**Changing the prompt or the model:**

```bash
carebot eval scenarios --judge --trials 5                      # baseline
# ...edit src/carebot/prompts.py, or set CAREBOT_MODEL...
carebot eval scenarios --judge --trials 5                      # candidate
carebot eval compare eval-results/<baseline> eval-results/<candidate>
```

**CI.** Every push runs the tests, the detector benchmark and the scenarios with
the offline scripted model, and builds the Docker image. Run the
**Live evals** workflow manually (it needs an `ANTHROPIC_API_KEY` secret) for
judged and simulated runs against real models.

Adding a case is a YAML file. See [`evals/README.md`](evals/README.md).

## Contributing

Pull requests are welcome. Please add an eval case for any behaviour you change,
and add any real-world detector miss to `evals/detector_cases.yaml` (anonymised)
so it can't come back.

```bash
pip install -e ".[dev,openai]"
pytest
carebot --provider scripted eval scenarios
```

## Important

This bot offers emotional support. It is not a medical or crisis service. The
contacts in `hong_kong.yaml` were last reviewed in October 2026; check them
before deploying, and at least yearly.

## License

MIT. See [LICENSE](LICENSE).

# Care for the Carers

Care for the Carers is a Telegram chatbot for family caregivers who are under
stress, built first for carers of people with dementia in Hong Kong. It listens,
replies like a kind friend would by text, recommends local support services
from a verified resource bank and, if a message suggests someone may be at
risk, always shows crisis hotlines.

The project was built by a team of students from the Chinese University of Hong
Kong (CUHK). It won the CUHK round of the World's Challenge Challenge and came
runner-up at the world final hosted by Western University, earning CAD$15,000
in kick-start funding. **Version 2 (2026)** rebuilds it as an LLM agent with a
safety harness and an evaluation suite. It also makes the bot easy to install
and host.

---

## Quick start

You need Python 3.10+ (or Docker) and, for real replies, an
[Anthropic API key](https://console.anthropic.com/). Without a key, you can use
the offline `scripted` model or a local model through Ollama.

```bash
pip install "git+https://github.com/csauyong/Care-for-the-Carers"

carebot --provider scripted chat        # try it in the terminal, no keys
export ANTHROPIC_API_KEY=sk-ant-...
carebot --agent tools chat              # the tool-using agent, with Claude
export TELEGRAM_BOT_TOKEN=123456:ABC... # from @BotFather
carebot run                             # on Telegram
```

Settings can also live in a `.env` file (see [`.env.example`](.env.example)).
`carebot check` validates them and makes one test call.

## Hosting

The bot keeps its sessions in a SQLite file and its traces in a JSONL file under
`CAREBOT_DATA_DIR` (default `./data`). It needs no database server.

| Where | How |
| --- | --- |
| **Docker** (any server, NAS, Raspberry Pi) | `cp .env.example .env`, fill it in, then `docker compose up -d`. |
| **Fully local, no API key** | Set `CAREBOT_PROVIDER=openai`, `CAREBOT_MODEL=llama3.1` and `OPENAI_BASE_URL=http://ollama:11434/v1`, then `docker compose --profile local up -d` and `docker compose exec ollama ollama pull llama3.1`. |
| **Platforms that run containers** (Render, Railway, Fly.io, Cloud Run, ...) | Deploy the Dockerfile and set `CAREBOT_WEBHOOK_URL` to the public `https://` URL. The bot uses a webhook, listens on `$PORT`, and has a health check at `/healthz`. Attach a persistent disk at `/data`. |
| **Plain Python** | `pip install .` then `carebot run` under systemd or similar. |

Polling, the default, only needs outbound internet. Webhook mode suits hosts
that expect an HTTP server, and it rejects requests without Telegram's secret
header.

## How it works

```
Telegram ─┐                        ┌──────────────── CareAgent (the harness) ─────────────────┐
terminal ─┼─▶ respond(user, text) ─▶│ 1. detector   stress score + crisis screen (always runs)  │
eval run ─┘                         │ 2. session    SQLite, trimmed history window              │
                                    │ 3. model      policy mode: writes the reply               │──▶ Claude / Ollama / scripted
                                    │               tools mode:  agent loop with 3 tools        │
                                    │ 4. guardrails crisis lines guaranteed; offer cooldown     │──▶ resource bank (retrieval)
                                    │ 5. trace      JSONL: signals, tool trajectory, tokens     │
                                    └───────────────────────────────────────────────────────────┘
```

**Two agent modes**, so they can be compared on the same evals:

- `policy`: the model writes the conversation, and code decides when to offer
  services (stress builds up and fades; one offer, then a cooldown).
- `tools`: the model is an agent. It calls `search_services` to retrieve from
  the resource bank, `offer_services` to show Yes/No buttons, and
  `escalate_crisis` to show hotlines, and decides when.

**Guarantees in both modes:**

- **Crisis floor.** A crisis screen (English, Cantonese and slang) runs on every
  message. If it fires, the crisis lines are shown even if the model doesn't
  escalate, fails, or refuses. The trace records whether the model or the
  harness escalated.
- **Grounding.** Every phone number and link the bot mentions must come from
  [`hong_kong.yaml`](src/carebot/data/hong_kong.yaml), a resource bank of 12
  services with sources. The evals fail any reply that invents one.
- **Privacy.** Traces hash user IDs and can drop all message text and tool
  arguments. `/forget` deletes a user's history.

**Components you can swap**, each with its own benchmark:

| Component | Options |
| --- | --- |
| Model | Claude (default `claude-opus-5-5`), any OpenAI-compatible server, scripted |
| Stress detector | `keyword`, `llm` (Claude with structured output), `finetuned` (LoRA classifier, see [`finetune/`](finetune/)) |
| Retriever | `bm25` (English words and Cantonese character pairs, no dependencies), `llm` (Claude picks from the bank) |

## Evaluation

The bot talks to people at a vulnerable moment, so every change to the prompt,
model or components has to be shown to be safe first.

| What | How it's measured | Command |
| --- | --- | --- |
| **Detection** | Precision and recall for stress and crisis on [39 labelled messages](evals/detector_cases.yaml), comparing keyword, LLM and fine-tuned detectors. Crisis recall must be 100%. | `carebot eval detector` |
| **Recommendation accuracy** | hit@1, recall@3 and MRR on [24 labelled conversations](evals/recommendation_cases.yaml), with a dev/test split so tuning doesn't inflate the score. Compares BM25 with the LLM recommender. | `carebot eval retrieval` |
| **Task success and trajectory review** | Multi-turn [scenarios](evals/scenarios/) check each turn: crisis shown, services offered once, the *right* services recommended, the expected tool calls made (`tools_called`). Reports include every tool call and say what share of crisis turns the model escalated by itself. | `carebot eval scenarios` |
| **Safety and red-teaming** | Crisis, boundary and red-team scenarios: dosage jailbreaks through role-play, prompt injection (with an automatic prompt-leak check), slang and Cantonese euphemisms for suicide, restraint requests, invented-hotline requests. Plus *adaptive* red-team personas that change tactics turn by turn. Any failing `safety` case fails the run. | `carebot eval scenarios --only redteam` |
| **LLM-as-a-judge** | Claude scores each conversation 1–5 on empathy, style, safety, relevance and (in tools mode) tool use, plus criteria specific to each scenario. | `--judge` |
| **Simulated users** | A model plays caregivers ([personas](evals/personas/)): a stoic husband, a Cantonese-speaking son, a hidden crisis, a carer who needs respite but doesn't know it exists. | `carebot eval personas` |
| **Human-in-the-loop** | Export a blind, shuffled rating sheet; people score it; the tool reports judge-vs-human and human-vs-human agreement (weighted κ, Spearman, pass/fail κ). | `carebot eval export-ratings` / `agreement` |
| **Benchmarks and regression testing** | Run a suite across models × agent modes × detectors on the same frozen test set. The set is hashed into every report. Pass rates come with bootstrap 95% CIs. `compare` flags a regression only when the interval excludes zero. | `carebot eval benchmark` / `compare` |

### Results so far

These were measured offline (no model API) at the commit that added them:

| | Result |
| --- | --- |
| Keyword detector | crisis precision 1.00 / recall **1.00**; stress precision 0.96 / recall 0.86 |
| BM25 retrieval | dev split hit@1 0.90, recall@3 0.93 → **held-out test hit@1 0.64, recall@3 0.68**. Keyword matching doesn't generalise to new wording, especially colloquial Cantonese. This is the gap the LLM recommender and tools mode are meant to close. |
| Scenarios (policy mode, scripted model) | 17 of 18 pass; every safety and red-team case passes. The one failure is the Cantonese money conversation above. |
| Fine-tuning pipeline | Verified end to end on a tiny local model (train → merge → load → benchmark). Accuracy comes from a real GPU run: see [`finetune/README.md`](finetune/README.md). |

Live numbers (LLM detector, LLM recommender, tools vs policy, the judge,
personas, model comparison) come from the **Live evals** workflow, which needs
an `ANTHROPIC_API_KEY` secret.

**Changing the prompt or model:**

```bash
carebot eval benchmark --judge --trials 5 --models claude-opus-5-5 claude-sonnet-5-5 --agents policy tools
carebot eval compare eval-results/<baseline> eval-results/<candidate>
```

**CI** runs the tests, the detector, retrieval and scenario benchmarks (safety
as a hard gate), a fine-tuning smoke test and a Docker build on every push.
Adding a case is a YAML file: see [`evals/README.md`](evals/README.md).

## Contributing

```bash
pip install -e ".[dev,openai]"
pytest
carebot --provider scripted eval scenarios --min-pass-rate 0.9
```

Please add an eval case for any behaviour you change, and add any real-world
detector miss to `evals/detector_cases.yaml` (anonymised).

## Important

This bot offers emotional support. It is not a medical or crisis service. The
resource bank was last reviewed in October 2026. Every entry lists its source.
Check them before deploying, and at least yearly.

## License

MIT. See [LICENSE](LICENSE).

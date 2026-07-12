# Props Scorer

[![CI](https://github.com/bene-art/props-scorer/actions/workflows/ci.yml/badge.svg)](https://github.com/bene-art/props-scorer/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Status:** Stable. Pulled from a larger private system to demonstrate this slice as a standalone tool. Treat as a snapshot, not current production. Security reports via [SECURITY.md](./SECURITY.md).

## Why does this exist?

I needed to make money for med school. Sports betting seemed like a good way to apply math to something with real stakes—literally. So instead of using AI to help me study flashcards, I started building models to find edges in player props.

That side project turned into a full system: agents, workflows, databases, the whole thing. This repo is just the inference layer—the part that takes player stats and returns a prediction.

I pulled it out and cleaned it up because the full system is messy and personal. This is the version I can show people.

---

## What does it actually do?

You send it player info. It tells you the probability they'll go over or under a betting line.

That's it. Three endpoints:

| Endpoint | What it does |
|----------|--------------|
| `/health` | "You alive?" → "Yeah, here's my version" |
| `/model` | "What stats do you need?" → List of features |
| `/predict` | "Here's the player data" → Probability + recommendation |

---

## Quick example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sport": "NBA",
    "stat_type": "points",
    "line": 25.5,
    "glicko_mu": 1650.0,
    "last_5_avg": 27.2,
    "last_10_avg": 26.8,
    "is_home": true
  }'
```

```json
{
  "probability": 0.62,
  "confidence": "medium",
  "recommendation": "OVER",
  "model_version": "xgb-v3"
}
```

Translation: 62% chance he goes over 25.5 points. Medium confidence. Model says take the over.

`glicko_mu` and `glicko_phi` are Glicko ratings — a player skill estimate and its uncertainty, similar to chess Elo but with an explicit confidence interval. Defaults (1500/200) apply if you don't have them.

When probability sits between 0.45 and 0.55, the model returns `NO_EDGE` instead of a direction. Abstaining is a first-class output, not a fallback.

---

## Why sports betting?

Honestly? The math.

Baseball especially—so many stats, so many ways to slice it. Once you get past a certain level, math stops being something you have to do and becomes something you *get* to do. It's a tool, not a crutch.

Sports betting is also a brutally honest feedback loop. You're either right or you're not. The market doesn't care about your feelings. I like that.

---

## What's actually in here?

For my friend who bets but doesn't code:

```
props-scorer/
├── src/inference_api/
│   ├── main.py            # The front door. Handles requests.
│   ├── scorer.py          # The brain. Loads model, runs inference.
│   ├── schemas.py         # The bouncer. Rejects bad input.
│   └── logging_config.py  # The security camera. Logs everything.
├── models/
│   ├── xgb_scorer_v3.joblib  # Trained XGBoost model (synthetic data)
│   └── xgb_scorer_v3.json    # Model metadata and metrics
├── scripts/
│   ├── train_model.py     # Reproducible training on synthetic data
│   └── healthcheck.py     # Docker/CI health check
├── tests/                 # 32 checks to prove it works
├── Dockerfile             # How to ship it anywhere
└── .github/workflows/     # Robot that tests everything automatically
```

For the engineer:

- **XGBoost** classifier trained on synthetic data (reproducible via `scripts/train_model.py`)
- **FastAPI** with async handlers
- **Pydantic** schemas for request validation
- **Structured JSON logging** with request correlation IDs
- **Docker** containerization with health checks
- **GitHub Actions** CI across Python 3.10-3.13

**Model metrics** (synthetic data, reproducible via `scripts/train_model.py`):

| Metric | Value |
|--------|-------|
| Training samples | 4,000 |
| Test samples | 1,000 |
| Accuracy | 0.559 |
| AUC-ROC | 0.589 |
| Log loss | 0.691 |

Near-chance accuracy on synthetic data is expected — the model architecture and serving pipeline are the artifact here, not the edge.

Architectural rationale: [docs/architectural_decisions.md](./docs/architectural_decisions.md)

---

## The boring-but-important stuff

Every request gets a unique ID. That ID shows up in the logs and in the response headers. If something breaks, you can trace it.

Every prediction includes the model version. If you're comparing results across time, you know exactly which version made each call.

Logs are JSON, not messy text. Any log aggregation tool (Datadog, ELK, whatever) can parse them directly.

These aren't fancy features. They're just what you need when something goes wrong at 2am and you're trying to figure out why.

---

## Running it yourself

```bash
git clone https://github.com/bene-art/props-scorer.git
cd props-scorer

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

pytest              # 32 tests, should all pass
uvicorn inference_api.main:app --reload
```

Server runs at `localhost:8000`. Hit `/health` to make sure it's alive.

---

## Docker

```bash
docker build -t props-scorer .
docker run -p 8000:8000 props-scorer
```

Same thing, but containerized. Runs the same on your laptop, a server, or Kubernetes.

---

## What this doesn't include

The included model is trained on **synthetic data** to demonstrate the full serving pipeline. Real model weights, player rating databases, and calibration curves stay private.

This shows how I build and serve a model: reproducible training, serialization, feature engineering, and structured inference. The actual edge stays private.

---

## Where this fits

I started with sports betting, and the system grew past that. Websites, workflows, agents, networking, CI/CD pipelines. This repo is a slice — cleaned up and made public.

Four repos, four concerns:

| Repo | Question it answers |
|------|---------------------|
| [**props-scorer**](https://github.com/bene-art/props-scorer) | What's going to happen? |
| [**betting-math-kit**](https://github.com/bene-art/betting-math-kit) | What should I do about it? |
| [**bet-tracker**](https://github.com/bene-art/bet-tracker) | Did it work? |
| [**backtester**](https://github.com/bene-art/backtester) | Would it have worked? |

The `/predict` response — a probability between 0 and 1 — is the natural input to `betting_math_kit.calculate_edge_calibrated()`. That's where inference hands off to decision math.

---

## License

MIT. Do whatever you want with it.

---

## Author

Benjamin Easington — [GitHub](https://github.com/bene-art)

I work in logistics, go to school, and build stuff on the side. If you're reading this, you probably care about AI or sports betting or both. Either way, feel free to reach out.

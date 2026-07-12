# Props Scorer

[![CI](https://github.com/bene-art/props-scorer/actions/workflows/ci.yml/badge.svg)](https://github.com/bene-art/props-scorer/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Status:** Stable. Real NBA game log data (2023–25), calibrated XGBoost, production serving patterns. Run `scripts/fetch_data.py` then `scripts/train_model.py` before serving. Security reports via [SECURITY.md](./SECURITY.md).

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
    "stat_type":      "points",
    "line":           27.5,
    "rolling_5_avg":  28.2,
    "rolling_10_avg": 26.8,
    "rolling_5_std":  4.1,
    "rest_days":      2,
    "is_home":        true,
    "opp_def_rtg":    112.3
  }'
```

```json
{
  "probability":    0.61,
  "confidence":     "medium",
  "recommendation": "OVER",
  "model_version":  "nba_points_v1",
  "source":         "model"
}
```

Translation: 61% chance he goes over 27.5 points. Medium confidence. `source` tells you whether the trained model or the heuristic fallback made the call.

When probability sits between 0.45 and 0.55, the model returns `NO_EDGE` instead of a direction. Abstaining is a first-class output, not a fallback.

---

## Pipeline

```bash
# 1. Fetch real NBA game logs — cached after first run (~2-3 min)
python scripts/fetch_data.py

# 2. Train calibrated model on 2023-25 season data (~2-3 min)
python scripts/train_model.py

# 3. Evaluate calibration on held-out test set
python scripts/evaluate_model.py

# 4. Serve
uvicorn inference_api.main:app --reload
```

Three scripts, end to end. Everything is reproducible from raw data.

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
│   ├── fetch_data.py      # Pull NBA game logs via nba_api (cached)
│   ├── train_model.py     # Calibrated XGBoost on real game log data
│   ├── evaluate_model.py  # Calibration curve + edge bucket analysis
│   └── healthcheck.py     # Docker/CI health check
├── tests/                 # 32 checks to prove it works
├── Dockerfile             # How to ship it anywhere
└── .github/workflows/     # Robot that tests everything automatically
```

For the engineer:

- **XGBoost + isotonic calibration** trained on real NBA game logs (2023–25 regular seasons)
- **FastAPI** with async handlers
- **Pydantic** schemas for request validation
- **Structured JSON logging** with request correlation IDs
- **Docker** containerization with health checks
- **GitHub Actions** CI across Python 3.10-3.13

**Model metrics** (real data — run `python scripts/evaluate_model.py` after training):

| | Where to find it |
|-|-----------------|
| Brier score (calibrated vs raw) | `models/nba_points_v1.json` |
| Calibration curve | `python scripts/evaluate_model.py` |
| Feature importances | `python scripts/train_model.py` output |

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

python scripts/fetch_data.py   # pull game logs -> data/
python scripts/train_model.py  # calibrate model -> models/

pytest
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

The model uses publicly available NBA game log data and commodity features — rolling averages, rest days, home/away, opponent defensive rating. None of this generates real edge on its own; it's already priced into the market.

What's not here: the proprietary signal. The pipeline — fetch, engineer, calibrate, serve, evaluate — is the artifact.

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

I'm an engineer building production AI systems. This repo is one slice of a larger private stack. More at [github.com/bene-art](https://github.com/bene-art).

# Book Recommendations on Goodreads

End-to-end book recommender on the [UCSD Goodreads](https://mengtingwan.github.io/data/goodreads.html)
public datasets (228M user-book interactions, 2M+ books). Implements three
approaches — collaborative KNN, content-based KNN, and a hybrid — with a
reproducible offline evaluation showing pure collaborative filtering
beats the textbook hybrid by ~50% on NDCG@10 on this dataset.

[![CI](https://github.com/RickArko/BookRecGoodReads/actions/workflows/ci.yml/badge.svg)](https://github.com/RickArko/BookRecGoodReads/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)

## Highlights

- **Polars-streamed ingest** turns 4GB of CSV plus 2GB of gzipped JSON into a
  50k × 770k sparse interaction matrix in roughly a minute.
- **Three recommenders share infrastructure**: an item-based KNN over the
  sparse interaction matrix, a TF-IDF + author + numeric content KNN, and a
  hybrid that linearly combines the two.
- **Offline evaluation framework** (precision / recall / NDCG / MAP / hit-rate
  at K, catalog coverage, intra-list diversity) with deterministic
  leave-one-out sampling. Generated report:
  [`docs/EVALUATION.md`](docs/EVALUATION.md).
- **Streamlit app** with fuzzy title search, score breakdowns, genre
  distributions, and a metrics dashboard. Deployable via Docker.
- **Tests + CI**: 68 unit tests, GitHub Actions across Python 3.10 and 3.12,
  ruff and black on every commit.

## Headline Result (NDCG@10, 500 users, seed 42)

| Recommender | NDCG@10 | Hit-Rate@10 | Coverage |
|---|---:|---:|---:|
| **Collaborative KNN** | **0.333** | 0.758 | 0.114 |
| Hybrid (0.7 collab / 0.3 content) | 0.271 | 0.634 | 0.129 |
| Hybrid (0.6 / 0.4) — production default | 0.228 | 0.564 | 0.133 |
| Content KNN | 0.010 | 0.070 | 0.068 |

The content signal *hurts* ranking quality at every weight setting; the
production hybrid weights are due for re-tuning. Full discussion in
[`MODEL.md`](MODEL.md#measured-performance).

## Quickstart

Requires [`uv`](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux
# Windows PowerShell:
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then:

```bash
# 1. Install dependencies (project + dev + app groups)
uv sync --all-groups

# 2. Download UCSD Goodreads data (~6GB; one-time)
python src/downloader.py

# 3. JSON / CSV → parquet (polars streaming)
python src/serialize_data.py --batch_size=500_000

# 4. Build the sparse interaction matrix + ID mappings
python prepare_data.py
# Smaller smoke-test sample:
# python prepare_data.py --min_reads=50 --top_books=5000 --top_users=100000

# 5. (Optional) Build content features for the hybrid recommender
python src/extract_metadata.py
python src/create_content_features.py

# 6. Run the Streamlit app
uv run --group app streamlit run app/app_enhanced.py
# → http://localhost:8501
```

## Architecture

```
                ┌─────────────────────────┐
                │  UCSD Goodreads dataset │
                │  (~6GB CSV / NDJSON)    │
                └────────────┬────────────┘
                             │ src/downloader.py
                             ▼
              ┌──────────────────────────────┐
              │  Polars streaming ingest     │ src/serialize_data.py
              │  CSV/NDJSON → parquet        │
              └────────────┬─────────────────┘
                           │
                           ▼
              ┌────────────────────────────┐
              │ Sparse books × users CSR   │ prepare_data.py
              │ + ID mappings (parquet)    │
              └─────┬───────────────┬──────┘
                    │               │
                    │               │ src/extract_metadata.py
                    │               │ src/create_content_features.py
                    ▼               ▼
   ┌──────────────────────┐  ┌────────────────────────┐
   │ Collaborative KNN    │  │ Content KNN            │
   │ (sparse cosine)      │  │ (TF-IDF + authors +    │
   │                      │  │  scaled numerics)      │
   └─────────┬────────────┘  └──────────┬─────────────┘
             └──────────────┬───────────┘
                            ▼
                ┌─────────────────────────┐
                │ Hybrid (linear combine) │ src/hybrid_recommender.py
                │ + Streamlit app         │ app/app_enhanced.py
                │ + offline evaluation    │ evaluate.py
                └─────────────────────────┘
```

## Repo Layout

```
.
├── prepare_data.py              # Sparse interaction matrix + ID mappings
├── evaluate.py                  # End-to-end recommender benchmark
├── src/
│   ├── downloader.py            # UCSD data download
│   ├── serialize_data.py        # CSV/NDJSON → parquet (polars streaming)
│   ├── extract_metadata.py      # Per-book metadata extraction
│   ├── create_content_features.py  # TF-IDF + authors + numeric features
│   ├── knn_recommender_sparse.py   # Collaborative KNN on sparse matrix
│   ├── hybrid_recommender.py    # Collab + content linear combination
│   ├── train_spotlight.py       # ALS matrix factorization (implicit lib)
│   └── evaluation.py            # Ranking metrics + holdout sampler
├── app/
│   ├── app.py                   # Basic KNN UI
│   └── app_enhanced.py          # Hybrid UI (recommended)
├── tests/                       # Unit tests (run with `uv run pytest`)
├── docs/EVALUATION.md           # Generated benchmark report
├── notebooks/                   # EDA + training viz
├── Dockerfile, docker-compose.yml
└── pyproject.toml, uv.lock
```

## Reproduce the Benchmark

After running the data preparation steps above:

```bash
python evaluate.py --n-users 500 --seed 42
```

Writes [`docs/EVALUATION.md`](docs/EVALUATION.md) and
`data/eval_results.json`. Adjust `--n-users`, `--k`, and `--hybrid-weights`
to explore the cost / precision frontier.

## Development

```bash
# Run the full test suite (currently 77 tests)
uv run pytest

# Lint, format, and type checks
uv run ruff check .
uv run black --check .
uv run mypy
```

`mypy` is configured in `pyproject.toml` to strict-check the modules
that ship with full type annotations (`src/evaluation.py`,
`src/matching.py`, `evaluate.py`). The rest of the tree is excluded for
now and is being brought up to standard incrementally.

CI runs the same matrix on every push and pull request — see
[`.github/workflows/ci.yml`](.github/workflows/ci.yml).

For a deeper development walkthrough, see [`CONTRIBUTING.md`](CONTRIBUTING.md).
For an at-a-glance pipeline reference, see [`QUICK_START.md`](QUICK_START.md).

## Docker

```bash
# Build and run the hybrid Streamlit app on http://localhost:8501.
docker-compose up --build

# docker-compose.yml mounts ./data:/app/data:ro so the prepared
# parquet/npz files are visible to the container without baking
# them into the image.
```

## Manual Data Download

If `src/downloader.py` fails (Google Drive throttling, etc.), download these
files manually and save them to `data/`:

- [`goodreads_books.json.gz`](https://drive.google.com/uc?id=1LXpK1UfqtP89H1tYy0pBGHjYk8IhigUK)
- [`goodreads_interactions.csv`](https://drive.google.com/uc?id=1zmylV7XW2dfQVCLeg1LbllfQtHD2KUon)
- [`book_id_map.csv`](https://drive.google.com/uc?id=1CHTAaNwyzvbi1TR08MJrJ03BxA266Yxr)
- [`user_id_map.csv`](https://drive.google.com/uc?id=15ax-h0Oi_Oyee8gY_aAQN6odoijmiz6Q)

## Acknowledgments

- [UCSD Book Graph](https://mengtingwan.github.io/data/goodreads.html) —
  Mengting Wan and Julian McAuley, *Item Recommendation on Monotonic
  Behavior Chains*, RecSys 2018.
- Reference reading:
  [Collaborative Filtering Deep Dive](https://www.kaggle.com/code/jhoward/collaborative-filtering-deep-dive)
  (Jeremy Howard).

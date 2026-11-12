# Assignment 5 — Gaussian HMM for Financial Market Regimes

Gaussian Hidden Markov Model applied to ~11 years of S&P 500 (`^GSPC`) daily
returns to infer hidden market regimes (low- vs high-volatility), analyse the
transition matrix, and forecast the near-term regime.

- **Name:** Amrit Singh
- **Roll No:** 20262701001@iiitvadodara.ac.in

## Contents

- `gaussian_hmm_market_regimes.ipynb` — full analysis (Parts 1–5 + bonus)
- `gaussian_hmm_market_regimes_amrit.pdf` — exported report
- `data/sp500.csv` — cached S&P 500 adjusted-close data
- `build_notebook.py` — script that generates the notebook
- `requirements.txt` — Python dependencies

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace gaussian_hmm_market_regimes.ipynb
```

The notebook reads the cached CSV if present; delete `data/sp500.csv` to
re-download fresh data from Yahoo Finance.

# AI Agent Reliability Framework

MSc Dissertation — Applying Classical Reliability Engineering Techniques to Evaluate AI Agent Pipeline Reliability

## Dashboard

Live at: https://share.streamlit.io (deploy via Streamlit Cloud)

## Pages

| Page | Description |
|---|---|
| Overview | Summary of all results across V1/V2/V3 |
| Live Pipeline Demo | Interactive question → component pass/fail demo |
| Fault Tree Analysis | FTA results with config selector (V1/V2/V3) |
| Event Tree Analysis | ETA results with barrier effectiveness |
| Bayesian Network | BN CPT table and evidence injection |
| Sensitivity Analysis | Tornado diagram + interactive sliders |
| Cross-Validation Results | Full 9-row dissertation CV table |
| V1 vs V2 vs V3 Comparison | Three-config comparison with CI table |
| Apply to Any Agent | Domain-agnostic reliability calculator |

## Experimental Configurations

| Config | n | Intent | Retriever | LLM | Agent Failure |
|---|---|---|---|---|---|
| V1 | 500 | BART-large-mnli | MiniLM-L6 | Flan-T5-Large | 73.20% |
| V2 | 2,000 | DeBERTa-v3 | mpnet-base-v2 | Flan-T5-XL | 70.80% |
| V3 | 4,000 | DeBERTa-v3 | mpnet-base-v2 | Flan-T5-XL | 69.95% |

## Key Results

- **Naive FTA** overestimates agent failure by 12.74pp (V1) due to violated independence assumption
- **Architecture-aware corrected FTA** reduces error to 6.71pp
- **Bayesian Network** achieves mean absolute CV error of 0.0157 at V3 — 88.9% reduction vs naive FTA
- **Most critical component:** Retrieval Failure (BN sensitivity swing = 0.154)
- **ETA finding:** Ranking Error produces highest P(Unsafe Output) = 0.136 despite not being most frequent

## Run Locally

```bash
pip install streamlit plotly pandas numpy
streamlit run app.py
```

App opens at http://localhost:8501

## Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to https://share.streamlit.io
3. Sign in with GitHub
4. Click New app → select your repo → set Main file: `app.py`
5. Click Deploy — get a permanent public URL

## Repository Structure

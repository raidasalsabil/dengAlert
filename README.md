# dengueAlert
 Early-warning system that predicts ward-level dengue outbreak risk 2–4 weeks ahead from weather lags.

 ## Run locally (under 3 minutes)

Requires Python 3.10 or newer.

```powershell
git clone https://github.com/raidasalsabil/dengAlert.git
cd dengAlert
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run app/app.py
```

On macOS or Linux, activate with `source .venv/bin/activate`.

Then open http://localhost:8501. The trained model and processed features are included in the repo, so the app starts immediately with no training step.

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` first.

## Rebuild features and retrain (optional)

Run from the project root. The raw spreadsheet must be at `data/raw/hackathon_preli_dataset.xlsx`.

```powershell
python -m src.features
python -m src.train
```

`src.train` prints validation and test metrics, then rewrites `models/model.joblib`, `models/meta.json` and `data/processed/eval_predictions.parquet`.

## Using the dashboard

1. Pick a **city**, **ward** and **forecast date** in the sidebar. Use a 2025 date for a forecast the model never saw.
2. **Ward forecast**: risk tier, predicted cases for the 2-4 week window, the main weather drivers and the lagged weather behind them.
3. **City overview**: all wards ranked by risk for the chosen date, plus a heatmap of the last 20 weeks.
4. **Model accuracy**: chronological validation, with MAE and RMSE against climatology and persistence baselines.

## Project structure

```
app/app.py            Streamlit dashboard
src/features.py       multi-week lag features per ward
src/risk.py           risk tier mapping
src/train.py          chronological split and XGBoost training
models/               serialized model and metadata
data/processed/       features and evaluation predictions
```

## AI assistants and external APIs

**AI assistants used during development:**

- Claude (Anthropic): used for code review and explanation, the Streamlit dashboard code, and drafting this README and ARCHITECTURE.md.

**External APIs:** none. The app runs fully locally from a saved model, with no network calls and no LLM at runtime.

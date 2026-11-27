# dengueAlert

Early-warning system that predicts ward-level dengue outbreak risk 2--4
weeks ahead from weather lags.

## Run locally (under 3 minutes)

Requires Python 3.10 or newer.

``` powershell
git clone https://github.com/raidasalsabil/dengAlert.git
cd dengAlert
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run app/app.py
```

On macOS or Linux, activate with `source .venv/bin/activate`.

Then open <http://localhost:8501>. The trained model and processed
features are included in the repo, so the app starts immediately with no
training step.

If PowerShell blocks activation, run
`Set-ExecutionPolicy -Scope Process Bypass` first.

## Rebuild features and retrain (optional)

Run from the project root. The raw spreadsheet must be at
`data/raw/hackathon_preli_dataset.xlsx`.

``` powershell
python -m src.features
python -m src.train
```

`src.train` prints validation and test metrics, then rewrites
`models/model.joblib`, `models/meta.json` and
`data/processed/eval_predictions.parquet`.

## Using the dashboard

1.  Pick a **city**, **ward** and **forecast date** in the sidebar. Use
    a 2025 date for a forecast the model never saw.
2.  **Ward forecast**: risk tier, predicted cases for the 2--4 week
    window, the main weather drivers and the lagged weather behind them.
3.  **City overview**: all wards ranked by risk for the chosen date,
    plus a heatmap of the last 20 weeks.
4.  **Model accuracy**: chronological validation, with MAE and RMSE
    against climatology and persistence baselines.

### Operations Command Centre

The **Operations** tab helps turn dengue forecasts into practical
resource-allocation and response plans. It contains four sections:

#### 1. Resource Allocation

-   Set the available regional stock percentage.
-   Set the total number of spraying crews available.
-   View estimated requirements and allocations for saline bags,
    diagnostic kits, hospital beds and spraying crews.
-   See ward-level allocations based on predicted cases, risk level and
    population density.
-   Identify shortages through critical supply deficit alerts.

#### 2. Hospital Bed Overflow Routing

-   Identify wards where projected dengue patients may exceed available
    bed capacity.
-   Find other wards within the same city that have spare beds.
-   View suggested patient transfers from wards with shortages to wards
    with available capacity.
-   Identify patients who cannot be accommodated and require further
    escalation.

#### 3. What-if Simulator

-   Explore how different spraying crew deployment plans may affect
    predicted dengue cases.
-   Choose between **Auto-optimise** and **Manual** deployment.
-   Adjust crew effectiveness and the number of crews deployed.
-   Compare estimated cases prevented, beds saved and saline saved.
-   View ward rankings and estimated return on investment (ROI).

Simulator results are scenario estimates based on assumptions. They are
not guaranteed real-world intervention outcomes.

#### 4. Dispatch

-   Generate ward-level SMS and WhatsApp message templates.
-   Include risk levels, predicted cases, resource allocations and
    shortage alerts in the messages.
-   Generate city-level HTML briefs for commissioners.
-   Print or save the generated briefs as PDF.

Messages are generated as templates; the application does not
automatically send SMS or WhatsApp messages.

## Project structure

``` text
app/
    app.py                Streamlit dashboard
    ops_ui.py             Operations Command Centre interface

src/
    features.py           Multi-week lag features per ward
    risk.py               Risk tier mapping
    train.py              Chronological split and XGBoost training
    capacity.py            Capacity calculations
    resources.py          Resource allocation
    overflow.py            Hospital bed overflow routing
    simulator.py          What-if intervention simulator
    dispatch.py           Dispatch message and brief generation
    forecast.py           Forecast-related functionality
    config.py             Application configuration

models/
    Serialized model and metadata

data/
    raw/                  Raw datasets
    processed/            Features and evaluation predictions

tests/                    Project tests
exports/                  Generated dispatch exports
```

## AI assistants and external APIs

**AI assistants used during development:**

-   Claude (Anthropic): used for code review and explanation, the
    Streamlit dashboard code, and drafting this README and
    ARCHITECTURE.md.

**External APIs:** none. The app runs fully locally from a saved model,
with no network calls and no LLM at runtime.

It is also deployed with streamlit.

Here's the link! denguealertv2.streamlit.app

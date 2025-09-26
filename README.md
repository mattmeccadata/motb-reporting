# Streamlit Viewer for Notebook DataFrames

This app executes `motb-reporting.ipynb` (your existing analysis) and renders six DataFrames:

- A) Region revenue breakdown
- B) Region commitments
- C) Region balances
- D1) Unfulfilled 2024
- D2) Unfulfilled 2025
- E) Potential misidentified gifts

## How it works

The app uses `nbconvert.PythonExporter` to convert the notebook into Python code on the fly, executes it, then pulls the variables created by the notebook. It expects the notebook to define the following variables:

- `region_rev_breakdown`
- `region_commitments`
- `region_balances`
- `unfulfilled_2024_display` (or `pledge_detail_bal` with `balance_2024`)
- `unfulfilled_2025_display` (or `pledge_detail_bal` with `balance_2025`)
- `potential_matches_df`

If your notebook uses API keys (e.g., Monday.com), set them as Streamlit Secrets. At minimum, set:

```toml
# .streamlit/secrets.toml (Streamlit Cloud: App → Settings → Secrets)
MONDAY_API_TOKEN = "YOUR_TOKEN"
```

## Deploy

1. Add these files and your `motb-reporting.ipynb` to a GitHub repo.
2. Go to https://share.streamlit.io → New App → select repo/branch → Deploy.
3. In the app settings, add `MONDAY_API_TOKEN` under **Secrets**.
4. Open the app and click **Run / Refresh**.

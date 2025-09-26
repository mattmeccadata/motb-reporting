
import os
import io
import traceback
from typing import Dict, Any, List
from pathlib import Path

import streamlit as st
import pandas as pd
from nbconvert import PythonExporter
import nbformat

APP_TITLE = "Regional Funding Report"
DEFAULT_NOTEBOOK_NAME = "motb-reporting.ipynb"
APP_DIR = Path(__file__).resolve().parent

# ---- display/HTML shims so notebook code using IPython.display won't crash ----
def _display_shim(obj=None, *args, **kwargs):
    """Best-effort renderer for common notebook display() usage inside Streamlit."""
    try:
        # Avoid duplicating rendering; this is only for misc HTML or prints
        if obj is None:
            return
        if isinstance(obj, pd.DataFrame):
            # Let the main app render the canonical DataFrames; still show if called
            st.dataframe(obj, use_container_width=True)
        elif hasattr(obj, "_repr_html_"):
            st.markdown(obj._repr_html_(), unsafe_allow_html=True)
        elif isinstance(obj, str) and obj.strip().startswith("<"):
            st.markdown(obj, unsafe_allow_html=True)
        else:
            st.write(obj)
    except Exception:
        # Never let display issues crash the run
        pass

class HTML(str):
    """Lightweight stand-in for IPython.display.HTML"""
    def _repr_html_(self):
        return str(self)

def _default_candidates() -> List[Path]:
    return [
        APP_DIR / DEFAULT_NOTEBOOK_NAME,
        Path.cwd() / DEFAULT_NOTEBOOK_NAME,
        APP_DIR / "notebooks" / DEFAULT_NOTEBOOK_NAME,
        APP_DIR.parent / DEFAULT_NOTEBOOK_NAME,
    ]

def _find_default_path() -> str:
    for p in _default_candidates():
        if p.exists():
            return str(p)
    return str(APP_DIR / DEFAULT_NOTEBOOK_NAME)

def execute_notebook_from_file(nb_path: str) -> Dict[str, Any]:
    exporter = PythonExporter()
    code, _ = exporter.from_filename(nb_path)
    ns: Dict[str, Any] = {"display": _display_shim, "HTML": HTML}
    exec(compile(code, nb_path, "exec"), ns)
    return ns

def execute_notebook_from_bytes(nb_bytes: bytes) -> Dict[str, Any]:
    exporter = PythonExporter()
    nb = nbformat.read(io.BytesIO(nb_bytes), as_version=4)
    code, _ = exporter.from_notebook_node(nb)
    ns: Dict[str, Any] = {"display": _display_shim, "HTML": HTML}
    exec(compile(code, "<uploaded-notebook>", "exec"), ns)
    return ns

def collect_output_frames(ns: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}

    def get_df(name: str) -> pd.DataFrame:
        df = ns.get(name)
        if isinstance(df, pd.DataFrame):
            return df
        return pd.DataFrame()

    frames["A) Region revenue breakdown"] = get_df("region_rev_breakdown")
    frames["B) Region commitments"] = get_df("region_commitments")
    frames["C) Region balances"] = get_df("region_balances")

    d1 = ns.get("unfulfilled_2024_display")
    d2 = ns.get("unfulfilled_2025_display")

    if not isinstance(d1, pd.DataFrame) or not isinstance(d2, pd.DataFrame):
        pledge_detail_bal = get_df("pledge_detail_bal")
        if not pledge_detail_bal.empty:
            cols_common = [c for c in ["pledge_group","name","region","email","phone","address"] if c in pledge_detail_bal.columns]
            if "balance_2024" in pledge_detail_bal.columns and cols_common:
                d1 = pledge_detail_bal[cols_common + ["balance_2024"]].copy()
            if "balance_2025" in pledge_detail_bal.columns and cols_common:
                d2 = pledge_detail_bal[cols_common + ["balance_2025"]].copy()

    frames["D1) Unfulfilled 2024"] = d1 if isinstance(d1, pd.DataFrame) else pd.DataFrame()
    frames["D2) Unfulfilled 2025"] = d2 if isinstance(d2, pd.DataFrame) else pd.DataFrame()
    frames["E) Potential misidentified gifts"] = get_df("potential_matches_df")

    return frames

def to_excel_bytes(frames: Dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        for name, df in frames.items():
            if df is None or df.empty:
                continue
            sheet = name.replace("/", "-")[:31]
            df.to_excel(writer, index=False, sheet_name=sheet)
    return buffer.getvalue()

def _env_secrets_to_env():
    if "MONDAY_API_TOKEN" in st.secrets:
        os.environ["MONDAY_API_TOKEN"] = st.secrets["MONDAY_API_TOKEN"]

def diagnostics():
    with st.expander("Diagnostics", expanded=False):
        st.write("**Working directory**:", os.getcwd())
        try:
            st.write("**Files in working dir**:", os.listdir("."))
        except Exception as e:
            st.write("Could not list working dir:", e)
        st.write("**App directory**:", str(APP_DIR))
        try:
            st.write("**Files in app dir**:", os.listdir(str(APP_DIR)))
        except Exception as e:
            st.write("Could not list app dir:", e)
        st.write("**Notebook search candidates**:", [str(p) for p in _default_candidates()])

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("First, click the run/refresh button on the left. The report will take a few seconds to run. Then, click on the tabs below to navigate between reports. Download to Excel using the button.")

    _env_secrets_to_env()

    st.sidebar.header("Run")
    default_path = _find_default_path()
    run = st.sidebar.button("Run / Refresh", type="primary")

    if "MONDAY_API_TOKEN" not in os.environ and "MONDAY_API_TOKEN" not in st.secrets:
        with st.sidebar.expander("Secrets", expanded=False):
            st.info("Set **MONDAY_API_TOKEN** in Streamlit secrets for the notebook's API calls.")
        st.warning("No MONDAY_API_TOKEN found. The notebook may fail if it requires API access.")

    diagnostics()

    if not run:
        st.info("Click **Run / Refresh** to execute the notebook and render the DataFrames.")
        return

    try:
        with st.status("Executing notebook…", expanded=False) as status:
            if uploaded is not None:
                ns = execute_notebook_from_bytes(uploaded.read())
            else:
                if not Path(nb_path).exists():
                    raise FileNotFoundError(f"Notebook not found at: {nb_path}")
                ns = execute_notebook_from_file(nb_path)
            status.update(state="complete", label="Notebook executed")

        frames = collect_output_frames(ns)

        tabs = st.tabs(list(frames.keys()))
        for tab, (name, df) in zip(tabs, frames.items()):
            with tab:
                st.subheader(name)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    st.caption(f"Rows: {len(df):,} | Columns: {df.shape[1]}")
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No data to display.")

        excel_bytes = to_excel_bytes(frames)
        st.download_button(
            "Download all tables (Excel)",
            data=excel_bytes,
            file_name=f"regional_report_download_{pd.Timestamp.today().strftime('%m%d%Y')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    except Exception as e:
        st.error("Error while executing the notebook. See details below:")
        st.exception(e)
        with st.expander("Full traceback"):
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()

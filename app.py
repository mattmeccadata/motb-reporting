
import os
import io
import json
import traceback
from typing import Dict, Any, List, Tuple

import streamlit as st
import pandas as pd

# Optional: used to turn the notebook into executable Python code
from nbconvert import PythonExporter


APP_TITLE = "DataFrame Viewer (Notebook-powered)"
NOTEBOOK_PATH = "motb-reporting.ipynb"  # keep this file in the same repo directory


def execute_notebook_to_namespace(nb_path: str) -> Dict[str, Any]:
    """
    Convert a .ipynb notebook to plain Python code with nbconvert, then exec() it.
    Returns the globals namespace so we can pull variables defined by the notebook.
    """
    exporter = PythonExporter()
    code, _ = exporter.from_filename(nb_path)

    # Streamlit Secrets -> ENV (for notebooks that read tokens with os.getenv)
    # e.g., set st.secrets["MONDAY_API_TOKEN"] in Streamlit Cloud
    if "MONDAY_API_TOKEN" in st.secrets:
        os.environ["MONDAY_API_TOKEN"] = st.secrets["MONDAY_API_TOKEN"]

    ns: Dict[str, Any] = {}
    exec(compile(code, nb_path, "exec"), ns)  # execute notebook-derived code
    return ns


def collect_output_frames(ns: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Pull the six target DataFrames from the executed notebook namespace.
    Falls back to deriving the D-frames if only base frames exist.
    """
    frames: Dict[str, pd.DataFrame] = {}

    def get_df(name: str) -> pd.DataFrame:
        df = ns.get(name)
        if isinstance(df, pd.DataFrame):
            return df
        return pd.DataFrame()

    # A, B, C
    frames["A) Region revenue breakdown"] = get_df("region_rev_breakdown")
    frames["B) Region commitments"] = get_df("region_commitments")
    frames["C) Region balances"] = get_df("region_balances")

    # D1, D2 (if notebook didn't create the *_display, derive from bases)
    d1 = ns.get("unfulfilled_2024_display")
    d2 = ns.get("unfulfilled_2025_display")

    if not isinstance(d1, pd.DataFrame) or not isinstance(d2, pd.DataFrame):
        pledge_detail_bal = get_df("pledge_detail_bal")
        if not pledge_detail_bal.empty:
            cols_common = ["pledge_group","name","region","email","phone","address"]
            if "balance_2024" in pledge_detail_bal.columns:
                d1 = pledge_detail_bal[cols_common + ["balance_2024"]].copy()
            if "balance_2025" in pledge_detail_bal.columns:
                d2 = pledge_detail_bal[cols_common + ["balance_2025"]].copy()

    frames["D1) Unfulfilled 2024"] = d1 if isinstance(d1, pd.DataFrame) else pd.DataFrame()
    frames["D2) Unfulfilled 2025"] = d2 if isinstance(d2, pd.DataFrame) else pd.DataFrame()

    # E
    frames["E) Potential misidentified gifts"] = get_df("potential_matches_df")

    return frames


def to_excel_bytes(frames: Dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        for name, df in frames.items():
            if df is None or df.empty:
                continue
            # Sheet names limited to 31 chars
            sheet = name.replace("/", "-")[:31]
            df.to_excel(writer, index=False, sheet_name=sheet)
    return buffer.getvalue()


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Runs the bundled Jupyter Notebook and displays its resulting tables.")

    st.sidebar.header("Run")
    nb_file = st.sidebar.text_input("Notebook file path", NOTEBOOK_PATH)
    run = st.sidebar.button("Run / Refresh", type="primary")

    if "MONDAY_API_TOKEN" not in os.environ and "MONDAY_API_TOKEN" not in st.secrets:
        with st.sidebar.expander("Secrets", expanded=False):
            st.info("Set **MONDAY_API_TOKEN** in Streamlit secrets for the notebook's API calls.")
        st.warning("No MONDAY_API_TOKEN found. The notebook may fail if it requires API access.")

    if not run:
        st.info("Click **Run / Refresh** to execute the notebook and render the DataFrames.")
        return

    try:
        with st.status("Executing notebook…", expanded=False) as status:
            ns = execute_notebook_to_namespace(nb_file)
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
            file_name="notebook_tables.xlsx",
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

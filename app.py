#!/usr/bin/env python3
from __future__ import annotations

import ast
from typing import Sequence, Optional, List, Dict, Any
import os
import numpy as np
import pandas as pd

import dash
from dash import Dash, html, dcc, Input, Output, State, ALL, no_update
import dash_ag_grid as dag
import plotly.graph_objs as go

# =========================
# Config
# =========================
APP_TITLE = "CN — CryptoNest by QuantNest"
APP_BRAND = "CN"
DEFAULT_COINS = ["BTC/USD", "ETH/USD", "SOL/USD", "DAI/USD"]
DEFAULT_WEIGHTS = [0.5, 0.3, 0.1, 0.1]
DEFAULT_START = "2023-01-01"
DEFAULT_END = "2024-12-31"

# ---- Paths that work on Windows/Mac/Linux from VS Code ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("CN_DATA_DIR", os.path.join(BASE_DIR, "data"))

def _env_or_default(envkey: str, default_filename: str) -> str:
    p = os.environ.get(envkey)
    if p:
        return p
    return os.path.join(DATA_DIR, default_filename)

DATA_PATHS = {
    "betas_panel": _env_or_default("CN_BETAS_PANEL", "betas_panel_5y.csv"),
    "factors_all": _env_or_default("CN_FACTORS_ALL", "factor_returns_5y.csv"),
    "resids":      _env_or_default("CN_RESIDS",      "residuals_5y.csv"),
}

def _assert_exists(path: str, label: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{label} not found at '{path}'.\n"
            f"- Expected in: {os.path.join(BASE_DIR, 'data')}\n"
            f"- Or set env vars CN_BETAS_PANEL / CN_FACTORS_ALL / CN_RESIDS (or CN_DATA_DIR)."
        )

# =========================
# Data loaders & analytics
# =========================
def load_betas(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels != 2:
        if df.columns.dtype == object:
            try:
                tuples = [
                    ast.literal_eval(c) if isinstance(c, str) and c.startswith("(") else (c, "")
                    for c in df.columns
                ]
                df.columns = pd.MultiIndex.from_tuples(tuples, names=["asset", "factor"])
            except Exception:
                pass
    return df

def load_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)

def portfolio_factor_loadings(
    betas_panel: pd.DataFrame,
    coins: Sequence[str],
    weights: Sequence[float],
    start: Optional[str] = None,
    end: Optional[str] = None,
    renormalize: bool = True,
    include_alpha: bool = False,
) -> pd.DataFrame:
    if not isinstance(betas_panel.columns, pd.MultiIndex) or betas_panel.columns.nlevels != 2:
        raise ValueError("betas_panel must have MultiIndex columns (asset, factor).")

    b = betas_panel.loc[start:end] if (start or end) else betas_panel
    lvl1 = b.columns.get_level_values(1)
    beta_only = b.loc[:, lvl1 != "alpha"]
    alpha_df = b.xs("alpha", level=1, axis=1) if (include_alpha and "alpha" in set(lvl1)) else None

    available_assets = pd.Index(sorted(set(beta_only.columns.get_level_values(0))))
    w = pd.Series(weights, index=pd.Index(coins, name="asset"), dtype=float)
    use_assets = available_assets.intersection(w.index)
    if use_assets.empty:
        raise ValueError("None of the requested coins are present in betas_panel.")

    w = w.loc[use_assets]
    if renormalize:
        s = w.sum()
        if s == 0 or pd.isna(s):
            raise ValueError("Weights sum to zero/NaN after aligning to assets.")
        w = w / s

    exposures = {}
    for f in sorted(set(beta_only.columns.get_level_values(1))):
        B_f = beta_only.xs(f, level=1, axis=1).reindex(columns=use_assets)
        exposures[f] = B_f.mul(w, axis=1).sum(axis=1)
    out = pd.DataFrame(exposures).sort_index()

    if include_alpha and alpha_df is not None and not alpha_df.empty:
        A = alpha_df.reindex(columns=use_assets)
        out["ALPHA"] = A.mul(w, axis=1).sum(axis=1)
    return out

def portfolio_return_decomposition(
    betas_panel: pd.DataFrame,
    factors_all: pd.DataFrame | pd.Series,
    resids: pd.DataFrame | pd.Series,
    coins: Sequence[str],
    weights: Sequence[float],
    start: Optional[str] = None,
    end: Optional[str] = None,
    include_alpha: bool = False,
    renormalize_each_day: bool = True,
    asset_returns: Optional[pd.DataFrame | pd.Series] = None,
    name_residual: str = "Residual",
    name_alpha: str = "Alpha",
) -> pd.DataFrame:
    if isinstance(factors_all, pd.Series):
        factors_all = factors_all.to_frame(name=factors_all.name or "FACTOR")
    if isinstance(resids, pd.Series):
        resids = resids.to_frame(name=resids.name or "ASSET")
    if asset_returns is not None and isinstance(asset_returns, pd.Series):
        asset_returns = asset_returns.to_frame(name=asset_returns.name or "ASSET")

    if not isinstance(betas_panel, pd.DataFrame):
        raise ValueError("betas_panel must be a DataFrame with MultiIndex columns (asset, factor).")
    if not isinstance(betas_panel.columns, pd.MultiIndex) or betas_panel.columns.nlevels != 2:
        raise ValueError("betas_panel.columns must be MultiIndex with levels (asset, factor).")

    b = betas_panel.loc[start:end] if (start or end) else betas_panel
    f = factors_all.loc[start:end] if (start or end) else factors_all
    r = resids.loc[start:end] if (start or end) else resids

    lvl1 = b.columns.get_level_values(1)
    has_alpha = "alpha" in set(lvl1)
    beta_only = b.loc[:, lvl1 != "alpha"]
    alpha_df = (b.xs("alpha", level=1, axis=1) if (include_alpha and has_alpha) else pd.DataFrame(index=b.index))

    beta_factors = sorted(set(beta_only.columns.get_level_values(1)))
    factors = sorted(set(beta_factors).intersection(f.columns))
    if not factors:
        raise ValueError("No overlapping factor names between betas_panel and factors_all.")
    f = f[factors]

    common_idx = beta_only.index.intersection(f.index).intersection(r.index)
    if common_idx.empty:
        raise ValueError("No overlapping dates among betas, factors, and residuals in the requested range.")
    beta_only = beta_only.reindex(common_idx)
    alpha_df = alpha_df.reindex(common_idx)
    f = f.reindex(common_idx)
    r = r.reindex(common_idx)

    w_full = pd.Series(weights, index=pd.Index(coins, name="asset"), dtype=float)

    out = pd.DataFrame(index=common_idx, columns=factors, dtype=float)
    for t in common_idx:
        B_t = beta_only.loc[[t]]
        assets_avail = B_t.columns.get_level_values(0).unique().intersection(w_full.index)
        if len(assets_avail) == 0:
            out.loc[t] = np.nan
            continue
        w_t = w_full.loc[assets_avail]
        if renormalize_each_day:
            s = w_t.sum()
            if s == 0 or pd.isna(s):
                out.loc[t] = np.nan
                continue
            w_t = w_t / s
        for fac in factors:
            betas_fac_t = B_t.xs(fac, level=1, axis=1).reindex(columns=assets_avail)
            expo_t = betas_fac_t.mul(w_t, axis=1).sum(axis=1).iloc[0]
            out.at[t, fac] = float(expo_t) * float(f.loc[t, fac])

    resid_contrib = []
    for t in common_idx:
        assets_avail = r.columns.intersection(w_full.index)
        res_t = r.loc[t, assets_avail].dropna()
        if res_t.empty:
            resid_contrib.append(np.nan)
            continue
        w_t = w_full.loc[res_t.index]
        if renormalize_each_day:
            s = w_t.sum()
            if s == 0 or pd.isna(s):
                resid_contrib.append(np.nan)
                continue
            w_t = w_t / s
        resid_contrib.append(float(res_t.mul(w_t).sum()))
    out[name_residual] = resid_contrib

    if include_alpha and not alpha_df.empty:
        alpha_contrib = []
        for t in common_idx:
            A_t = alpha_df.loc[t].dropna()
            use = A_t.index.intersection(w_full.index)
            if len(use) == 0:
                alpha_contrib.append(np.nan)
                continue
            w_t = w_full.loc[use]
            if renormalize_each_day:
                s = w_t.sum()
                if s == 0 or pd.isna(s):
                    alpha_contrib.append(np.nan)
                    continue
                w_t = w_t / s
            alpha_contrib.append(float(A_t.loc[use].mul(w_t).sum()))
        out[name_alpha] = alpha_contrib

    parts = factors + ([name_alpha] if (include_alpha and name_alpha in out.columns) else []) + [name_residual]
    out["TotalModel"] = out[parts].sum(axis=1)
    return out

# =========================
# Load data at startup
# =========================
_assert_exists(DATA_PATHS["betas_panel"], "betas_panel CSV")
_assert_exists(DATA_PATHS["factors_all"], "factor_returns CSV")
_assert_exists(DATA_PATHS["resids"],      "residuals CSV")

betas_panel = load_betas(DATA_PATHS["betas_panel"])
factors_all = load_df(DATA_PATHS["factors_all"])
resids      = load_df(DATA_PATHS["resids"])

AVAILABLE_COINS = sorted(set(betas_panel.columns.get_level_values(0)))
lvl1 = betas_panel.columns.get_level_values(1)
FACTOR_LIST = sorted(set([f for f in lvl1 if f != "alpha"]))

# =========================
# App UI (DARK THEME)
# =========================
external_stylesheets = [
    # Bootswatch Darkly
    "https://cdn.jsdelivr.net/npm/bootswatch@5.3.3/dist/darkly/bootstrap.min.css",
    # AG Grid (light + dark)
    "https://cdn.jsdelivr.net/npm/ag-grid-community/styles/ag-grid.css",
    "https://cdn.jsdelivr.net/npm/ag-grid-community/styles/ag-theme-quartz.css",
    "https://cdn.jsdelivr.net/npm/ag-grid-community/styles/ag-theme-quartz-dark.css",
]

app: Dash = dash.Dash(__name__, external_stylesheets=external_stylesheets, title=APP_TITLE)
server = app.server

def brand_header() -> html.Div:
    return html.Div(
        className="container-fluid py-3 border-bottom border-secondary",
        children=[
            html.Div(
                className="d-flex align-items-center justify-content-between",
                children=[
                    html.Div(
                        className="d-flex align-items-center gap-3",
                        children=[
                            html.Div(
                                APP_BRAND,
                                className="fw-bold text-white bg-primary rounded-circle d-flex align-items-center justify-content-center shadow",
                                style={"width": "44px", "height": "44px", "fontSize": "18px", "letterSpacing": "1px"},
                            ),
                            html.Div(
                                children=[
                                    html.Div("CryptoNest by QuantNest", className="fw-semibold", style={"fontSize": "1.25rem"}),
                                    html.Small("Portfolio factor analytics • exposures • contributions • correlations", className="text-muted"),
                                ]
                            ),
                        ],
                    ),
                    html.Small("v1", className="text-muted"),
                ],
            )
        ],
    )

def dark_card(title: str, body_children, card_id=None):
    return html.Div(
        id=card_id,
        className="card bg-dark border-secondary shadow-sm",
        children=[
            html.Div(className="card-header bg-dark border-secondary fw-semibold", children=title),
            html.Div(className="card-body", children=body_children),
        ],
        style={"display": "none"} if card_id else None,  # hidden by default for result cards
    )

# ---- Controls (weights are simple numeric inputs) ----
def weights_block(selected: List[str]) -> html.Div:
    default_map = dict(zip(DEFAULT_COINS, DEFAULT_WEIGHTS))
    n = len(selected) if selected else 0
    eq = (1.0 / n) if n > 0 else 0.0
    rows = []
    for c in selected or []:
        val = default_map.get(c, eq)
        rows.append(
            html.Div(
                className="d-flex align-items-center justify-content-between mb-2",
                children=[
                    html.Div(c, className="text-muted"),
                    dcc.Input(
                        id={"type": "w-input", "asset": c},
                        type="number",
                        min=0, step=0.0001,
                        value=val,
                        className="form-control bg-dark text-white",
                        style={"width": "140px"},
                    ),
                ],
            )
        )
    return html.Div(rows)

controls_card = html.Div(
    className="card bg-dark border-secondary shadow-sm",
    children=[
        html.Div(className="card-header bg-dark border-secondary fw-semibold", children="Inputs"),
        html.Div(
            className="card-body",
            children=[
                html.Div(className="mb-3", children=[
                    html.Label("Coins", className="form-label"),
                    dcc.Dropdown(
                        id="coins-dd",
                        options=[{"label": c, "value": c} for c in AVAILABLE_COINS],
                        value=[c for c in DEFAULT_COINS if c in AVAILABLE_COINS],
                        multi=True,
                        placeholder="Select coins...",
                        maxHeight=400,
                        className="cn-dark-dropdown", 
                    ),
                    html.Small("Only available tickers from betas panel are shown.", className="text-muted"),
                ]),
                html.Div(className="mb-2", children=[
                    html.Label("Weights (sum must equal 1.0000)", className="form-label"),
                    html.Div(id="weights-inputs", className="p-2 border rounded border-secondary",
                             style={"maxHeight": "360px", "overflowY": "auto"}),  # taller area
                    html.Div(id="weights-warning", className="form-text mt-2"),
                ]),
                html.Div(className="row g-2", children=[
                    html.Div(className="col-6", children=[
                        html.Label("Start date", className="form-label"),
                        dcc.Input(id="start-date", type="text", value=DEFAULT_START,
                                  placeholder="YYYY-MM-DD", className="form-control bg-dark text-white"),
                    ]),
                    html.Div(className="col-6", children=[
                        html.Label("End date", className="form-label"),
                        dcc.Input(id="end-date", type="text", value=DEFAULT_END,
                                  placeholder="YYYY-MM-DD", className="form-control bg-dark text-white"),
                    ]),
                ]),
                html.Div(className="d-flex gap-2 mt-3", children=[
                    html.Button("Run", id="run-btn", className="btn btn-primary", disabled=False),
                    html.Button("Export Exposures CSV", id="export-exposures", className="btn btn-outline-light"),
                    html.Button("Export Decomposition CSV", id="export-decomp", className="btn btn-outline-light"),
                    html.Button("Export Correlation CSV", id="export-corr", className="btn btn-outline-light"),
                ]),
                dcc.Download(id="dl-exposures"),
                dcc.Download(id="dl-decomp"),
                dcc.Download(id="dl-corr"),
                dcc.Store(id="store-exposures"),
                dcc.Store(id="store-decomp"),
                dcc.Store(id="store-corr"),
            ],
        ),
    ],
)

# ---- Results (hidden initially; revealed after first Run) ----
exposures_card = dark_card(
    "Portfolio Factor Loadings",
    [
        dcc.Graph(id="exposures-chart", config={"displayModeBar": True}),
        html.Div(className="mt-3", children=[
            dag.AgGrid(
                id="exposures-grid",
                className="ag-theme-quartz-dark",
                columnDefs=[],
                defaultColDef={"sortable": True, "filter": True, "floatingFilter": True, "resizable": True},
                rowData=[],
                dashGridOptions={"pagination": True, "paginationPageSize": 12, "ensureDomOrder": True, "enableCellTextSelection": True},
                style={"height": "320px", "width": "100%"},
            )
        ]),
    ],
    card_id="wrap-exposures",
)

decomp_card = dark_card(
    "Daily Factor Contributions (incl. Residual)",
    [
        dcc.Graph(id="decomp-chart", config={"displayModeBar": True}),
        html.Div(className="mt-3", children=[
            dag.AgGrid(
                id="decomp-grid",
                className="ag-theme-quartz-dark",
                columnDefs=[],
                defaultColDef={"sortable": True, "filter": True, "floatingFilter": True, "resizable": True},
                rowData=[],
                dashGridOptions={"pagination": True, "paginationPageSize": 12, "ensureDomOrder": True, "enableCellTextSelection": True, "rowSelection": "multiple"},
                style={"height": "320px", "width": "100%"},
            )
        ]),
    ],
    card_id="wrap-decomp",
)

corr_card = dark_card(
    "Factor Correlation Matrix",
    [
        dcc.Graph(id="corr-heatmap", config={"displayModeBar": True}),
        html.Div(className="mt-3", children=[
            dag.AgGrid(
                id="corr-grid",
                className="ag-theme-quartz-dark",
                columnDefs=[],
                defaultColDef={"sortable": True, "filter": True, "floatingFilter": True, "resizable": True},
                rowData=[],
                dashGridOptions={"pagination": True, "paginationPageSize": 12},
                style={"height": "320px", "width": "100%"},
            )
        ]),
    ],
    card_id="wrap-corr",
)

footer = html.Footer(
    className="container-fluid py-3 text-center",
    children=html.Small("© 2025 QuantNest • For research purposes only"),
)

# Wrap all results in a Loading overlay
results_column = dcc.Loading(
    id="loading",
    type="dot",
    color="#9ecbff",
    fullscreen=False,
    children=html.Div(children=[exposures_card, html.Div(className="my-3"), decomp_card, html.Div(className="my-3"), corr_card])
)

app.layout = html.Div(
    className="bg-dark text-light min-vh-100 d-flex flex-column",
    children=[
        brand_header(),
        html.Div(
            className="container-fluid my-3 flex-grow-1",
            children=[
                html.Div(className="row g-3", children=[
                    html.Div(className="col-lg-4", children=controls_card),
                    html.Div(className="col-lg-8", children=results_column),
                ])
            ],
        ),
        footer,  # <-- footer always at bottom
    ]
)

# =========================
# Helpers
# =========================
def df_to_aggrid(df: pd.DataFrame, digits: int = 6) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"columnDefs": [], "rowData": []}
    df_out = df.copy()
    if isinstance(df_out.index, pd.DatetimeIndex):
        df_out.index = df_out.index.strftime("%Y-%m-%d")
    df_out = df_out.reset_index().rename(columns={"index": "Date"})
    rowData = df_out.to_dict("records")
    columnDefs = [{"field": "Date", "pinned": "left"}]
    for c in df.columns:
        columnDefs.append({
            "field": str(c),
            "type": "rightAligned",
            "valueFormatter": {"function": f"d3.format('.{digits}f')(params.value)"},
            "filter": "agNumberColumnFilter",
        })
    return {"columnDefs": columnDefs, "rowData": rowData}

def matrix_to_aggrid(df: pd.DataFrame, digits: int = 3) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"columnDefs": [], "rowData": []}
    df_out = df.round(digits)
    df_out = df_out.reset_index().rename(columns={"index": "Factor"})
    rowData = df_out.to_dict("records")
    columnDefs = [{"field": "Factor", "pinned": "left"}]
    for c in df.columns:
        columnDefs.append({
            "field": str(c),
            "type": "rightAligned",
            "valueFormatter": {"function": f"d3.format('.{digits}f')(params.value)"},
            "filter": "agNumberColumnFilter",
        })
    return {"columnDefs": columnDefs, "rowData": rowData}

def line_figure(df: pd.DataFrame, title: str, ytitle: str) -> go.Figure:
    fig = go.Figure()
    if df is not None and not df.empty:
        for c in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=str(c)))
    fig.update_layout(
        title=title,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        xaxis_title="Date",
        yaxis_title=ytitle,
        template="plotly_dark",
        height=420,
    )
    return fig

def heatmap_figure(corr: pd.DataFrame, title: str) -> go.Figure:
    z = corr.values if corr is not None and not corr.empty else []
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=list(corr.columns) if not corr.empty else [],
            y=list(corr.index) if not corr.empty else [],
            colorbar=dict(title="ρ"),
            colorscale="Viridis",
            zmin=-1, zmax=1,
        )
    )
    fig.update_layout(
        title=title,
        margin=dict(l=60, r=20, t=50, b=60),
        template="plotly_dark",
        height=420,
    )
    return fig

# =========================
# Callbacks
# =========================
@app.callback(
    Output("weights-inputs", "children"),
    Input("coins-dd", "value"),
)
def build_weight_inputs(selected_coins):
    selected_coins = selected_coins or []
    return weights_block(selected_coins)

@app.callback(
    Output("weights-warning", "children"),
    Output("run-btn", "disabled"),
    Input({"type": "w-input", "asset": ALL}, "value"),
    State("coins-dd", "value"),
)
def validate_weights(values, coins):
    coins = coins or []
    values = values or []
    vals = [float(v) if v is not None else 0.0 for v in values]
    s = float(np.nansum(vals)) if len(vals) else 0.0
    if not coins:
        return html.Span("Select at least one coin.", className="text-warning"), True
    if np.isnan(s) or len(vals) != len(coins):
        return html.Span("Enter weights for all selected coins.", className="text-warning"), True
    diff = abs(s - 1.0)
    if diff <= 1e-6:
        return html.Span("Weights sum to 1.0000", className="text-success"), False
    else:
        return html.Span(f"Weights currently sum to {s:.6f}. They must sum to 1.0000.", className="text-danger"), True

@app.callback(
    Output("store-exposures", "data"),
    Output("store-decomp", "data"),
    Output("store-corr", "data"),
    Output("exposures-chart", "figure"),
    Output("exposures-grid", "columnDefs"),
    Output("exposures-grid", "rowData"),
    Output("decomp-chart", "figure"),
    Output("decomp-grid", "columnDefs"),
    Output("decomp-grid", "rowData"),
    Output("corr-heatmap", "figure"),
    Output("corr-grid", "columnDefs"),
    Output("corr-grid", "rowData"),
    Output("wrap-exposures", "style"),
    Output("wrap-decomp", "style"),
    Output("wrap-corr", "style"),
    Input("run-btn", "n_clicks"),
    State("coins-dd", "value"),
    State({"type": "w-input", "asset": ALL}, "value"),
    State("start-date", "value"),
    State("end-date", "value"),
    prevent_initial_call=True,
)
def run_model(_n, coins, weight_vals, start, end):
    coins = [c for c in (coins or []) if c in AVAILABLE_COINS]
    weight_vals = weight_vals or []
    if not coins or len(weight_vals) != len(coins):
        empty = line_figure(pd.DataFrame(), "Invalid inputs", "")
        hide = {"display": "none"}
        return None, None, None, empty, [], [], empty, [], [], heatmap_figure(pd.DataFrame(), "Correlation Matrix"), [], [], hide, hide, hide

    weights = [float(x) if x is not None else 0.0 for x in weight_vals]

    try:
        exposures = portfolio_factor_loadings(betas_panel, coins, weights, start, end, renormalize=True, include_alpha=False)
        decomp = portfolio_return_decomposition(betas_panel, factors_all, resids, coins, weights, start, end, include_alpha=False, renormalize_each_day=True)
        factors_window = factors_all.loc[start:end, exposures.columns.intersection(factors_all.columns)]
        corr = factors_window.corr().sort_index().loc[:, factors_window.columns.sort_values()]
    except Exception as e:
        msg = f"Error: {e}"
        empty = line_figure(pd.DataFrame(), msg, "")
        hide = {"display": "none"}
        return None, None, None, empty, [], [], empty, [], [], heatmap_figure(pd.DataFrame(), "Correlation Matrix"), [], [], hide, hide, hide

    fig_expo = line_figure(exposures, "Portfolio Factor Loadings", "Exposure")
    fig_decomp = line_figure(decomp[[c for c in decomp.columns if c != "TotalModel"]], "Daily Factor Contributions (incl. Residual)", "Daily Return")
    fig_corr = heatmap_figure(corr, "Factor Correlation Matrix")

    expo_grid = df_to_aggrid(exposures, digits=6)
    decomp_grid = df_to_aggrid(decomp, digits=6)
    corr_grid = matrix_to_aggrid(corr, digits=3)

    show = {"display": "block"}
    return (
        exposures.to_json(date_format="iso", orient="split"),
        decomp.to_json(date_format="iso", orient="split"),
        corr.to_json(date_format="iso", orient="split"),
        fig_expo,
        expo_grid["columnDefs"], expo_grid["rowData"],
        fig_decomp,
        decomp_grid["columnDefs"], decomp_grid["rowData"],
        fig_corr,
        corr_grid["columnDefs"], corr_grid["rowData"],
        show, show, show,
    )

# Export buttons
@app.callback(
    Output("dl-exposures", "data"),
    Input("export-exposures", "n_clicks"),
    State("store-exposures", "data"),
    prevent_initial_call=True,
)
def export_exposures(_n, data_json):
    if not data_json:
        return no_update
    df = pd.read_json(data_json, orient="split")
    return dcc.send_data_frame(df.to_csv, "exposures.csv")

@app.callback(
    Output("dl-decomp", "data"),
    Input("export-decomp", "n_clicks"),
    State("store-decomp", "data"),
    prevent_initial_call=True,
)
def export_decomp(_n, data_json):
    if not data_json:
        return no_update
    df = pd.read_json(data_json, orient="split")
    return dcc.send_data_frame(df.to_csv, "decomposition.csv")

@app.callback(
    Output("dl-corr", "data"),
    Input("export-corr", "n_clicks"),
    State("store-corr", "data"),
    prevent_initial_call=True,
)
def export_corr(_n, data_json):
    if not data_json:
        return no_update
    df = pd.read_json(data_json, orient="split")
    return dcc.send_data_frame(df.to_csv, "correlation_matrix.csv")

# =========================
# Main
# =========================
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))

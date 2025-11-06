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
import plotly.express as px

# =========================
# Config
# =========================
APP_TITLE = "CN — CryptoNest by QuantNest"
APP_BRAND = "CN"
DEFAULT_COINS = ["BTC/USD", "ETH/USD", "SOL/USD", "DAI/USD"]
DEFAULT_WEIGHTS = [0.5, 0.3, 0.1, 0.1]
DEFAULT_START = "2023-01-01"
DEFAULT_END = "2024-12-31"

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
    """Load betas panel CSV saved with MultiIndex columns (asset, factor)."""
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
    """Strict: all requested coins must exist; compute portfolio exposures over requested set."""
    if not isinstance(betas_panel.columns, pd.MultiIndex) or betas_panel.columns.nlevels != 2:
        raise ValueError("betas_panel must have MultiIndex columns (asset, factor).")

    b = betas_panel.loc[start:end] if (start or end) else betas_panel
    lvl1 = b.columns.get_level_values(1)
    beta_only = b.loc[:, lvl1 != "alpha"]
    alpha_df = b.xs("alpha", level=1, axis=1) if (include_alpha and "alpha" in set(lvl1)) else None

    available_assets = pd.Index(sorted(set(beta_only.columns.get_level_values(0))))
    requested_assets = pd.Index(coins, name="asset")
    missing = set(requested_assets) - set(available_assets)
    if missing:
        raise ValueError(f"The following tickers are not in betas_panel: {sorted(missing)}")
    use_assets = requested_assets

    w = pd.Series(weights, index=use_assets, dtype=float)
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
        B_t = beta_only.loc[[t]]  # keep as DataFrame
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

    assets_with_resids = r.columns.intersection(w_full.index)
    resid_contrib = []
    for t in common_idx:
        res_t = r.loc[t, assets_with_resids].dropna()
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
# App UI (DARK THEME + TABS)
# =========================
external_stylesheets = [
    "https://cdn.jsdelivr.net/npm/bootswatch@5.3.3/dist/darkly/bootstrap.min.css",
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

def small_card(title: str, children) -> html.Div:
    return html.Div(
        className="card bg-dark border-secondary shadow-sm",
        children=[
            html.Div(className="card-header bg-dark border-secondary fw-semibold", children=title),
            html.Div(className="card-body", children=children),
        ],
    )

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

# Controls card
controls_card = small_card(
    "Inputs",
    [
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
            html.Label("Weights (must sum to 1)", className="form-label"),
            html.Div(id="weights-inputs", className="p-2 border rounded border-secondary",
                     style={"maxHeight": "360px", "overflowY": "auto"}),
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
        ]),
    ],
)

# Export Results box
export_card = small_card(
    "Export Results",
    [
        html.Div(className="d-grid gap-2", children=[
            html.Button("Portfolio Factor Exposures", id="btn-exposures", className="btn btn-outline-light"),
            html.Button("Portfolio Total Return Decomposition", id="btn-decomp", className="btn btn-outline-light"),
            html.Button("Factor Correlations Matrix", id="btn-corr", className="btn btn-outline-light"),
        ]),
        dcc.Download(id="dl-exposures"),
        dcc.Download(id="dl-decomp"),
        dcc.Download(id="dl-corr"),
    ],
)

# Tabs content with proper structure
def exposures_tab():
    return html.Div(
        className="tab-content",
        children=[
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
        ]
    )

def decomp_tab():
    return html.Div(
        className="tab-content",
        children=[
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
        ]
    )

def factors_corr_tab():
    return html.Div(
        className="tab-content",
        children=[
            html.Div(
                className="factor-controls",
                children=[
                    html.H6("Factor Selection", className="text-light mb-3"),
                    html.Div(
                        className="row g-3",
                        children=[
                            html.Div(
                                className="col-md-8",
                                children=[
                                    dcc.Dropdown(
                                        id="factors-select",
                                        options=[{"label": f, "value": f} for f in FACTOR_LIST],
                                        value=FACTOR_LIST[:6],
                                        multi=True,
                                        placeholder="Select factors to analyze...",
                                        className="cn-dark-dropdown",
                                    ),
                                ]
                            ),
                            html.Div(
                                className="col-md-4",
                                children=[
                                    html.Button(
                                        "Update Factor Analysis",
                                        id="update-factors-btn",
                                        className="btn btn-primary w-100",
                                        n_clicks=0
                                    ),
                                ]
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="factor-grid",
                children=[
                    html.Div(
                        children=[
                            small_card("Factor Returns Time Series", [
                                dcc.Graph(id="factors-graph", config={"displayModeBar": True}),
                            ])
                        ],
                    ),
                    html.Div(
                        children=[
                            small_card("Factor Correlation Matrix", [
                                dcc.Graph(id="corr-heatmap", config={"displayModeBar": True}),
                            ])
                        ],
                    ),
                    html.Div(
                        children=[
                            small_card("Factor Returns Statistics", [
                                dag.AgGrid(
                                    id="factors-grid",
                                    className="ag-theme-quartz-dark",
                                    columnDefs=[],
                                    defaultColDef={
                                        "sortable": True,
                                        "filter": True,
                                        "floatingFilter": True,
                                        "resizable": True
                                    },
                                    rowData=[],
                                    dashGridOptions={
                                        "pagination": True,
                                        "paginationPageSize": 10,
                                        "ensureDomOrder": True,
                                        "enableCellTextSelection": True
                                    },
                                    style={"height": "300px", "width": "100%"},
                                )
                            ])
                        ],
                    ),
                    html.Div(
                        children=[
                            small_card("Correlation Matrix Data", [
                                dag.AgGrid(
                                    id="corr-grid",
                                    className="ag-theme-quartz-dark",
                                    columnDefs=[],
                                    defaultColDef={
                                        "sortable": True,
                                        "filter": True,
                                        "floatingFilter": True,
                                        "resizable": True
                                    },
                                    rowData=[],
                                    dashGridOptions={
                                        "pagination": True,
                                        "paginationPageSize": 10
                                    },
                                    style={"height": "300px", "width": "100%"},
                                )
                            ])
                        ],
                    ),
                ],
            ),
            html.Div(
                className="mt-3 d-grid gap-2 d-md-flex justify-content-md-end",
                children=[
                    html.Button(
                        "Download Factor Returns",
                        id="btn-factors",
                        className="btn btn-outline-light me-md-2"
                    ),
                    html.Button(
                        "Download Correlation Matrix",
                        id="btn-corr-matrix",
                        className="btn btn-outline-light"
                    ),
                    dcc.Download(id="dl-factors"),
                    dcc.Download(id="dl-corr-matrix"),
                ],
            ),
        ]
    )

results_tabs = html.Div(
    id="results-tabs-container",
    className="initially-hidden",
    children=[
        dcc.Tabs(
            id="result-tabs",
            value="tab-exposures",
            className="cn-tabs",       
            children=[
                dcc.Tab(label="Factor Exposures",
                        value="tab-exposures",
                        className="cn-tab",
                        selected_className="cn-tab--selected",
                        children=exposures_tab()),
                dcc.Tab(label="Return Decomposition",
                        value="tab-decomp",
                        className="cn-tab",
                        selected_className="cn-tab--selected",
                        children=decomp_tab()),
                dcc.Tab(label="Factor Analysis",
                        value="tab-factors",
                        className="cn-tab",
                        selected_className="cn-tab--selected",
                        children=factors_corr_tab()),
            ],
        )
    ]
)


footer = html.Footer(
    className="container-fluid py-3 text-center",
    children=html.Small("© 2025 QuantNest • For research purposes only"),
)

# Stores
stores = html.Div([
    dcc.Store(id="store-exposures"),
    dcc.Store(id="store-decomp"),
    dcc.Store(id="store-corr"),
    dcc.Store(id="store-factors"),
    dcc.Store(id="store-factor-data"),  # Store for raw factor data
])

app.layout = html.Div(
    className="bg-dark text-light min-vh-100 d-flex flex-column",
    children=[
        brand_header(),
        html.Div(
            className="container-fluid my-3 flex-grow-1",
            children=[
                html.Div(className="row g-3", children=[
                    html.Div(className="col-lg-4", children=[controls_card, html.Div(className="my-3"), export_card]),
                    html.Div(className="col-lg-8", children=[
                        dcc.Loading(
                            id="loading", 
                            type="dot", 
                            color="#9ecbff", 
                            children=results_tabs
                        )
                    ]),
                ]),
                stores,
            ],
        ),
        footer,
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

def stats_to_aggrid(stats_df: pd.DataFrame, digits: int = 6) -> Dict[str, Any]:
    if stats_df is None or stats_df.empty:
        return {"columnDefs": [], "rowData": []}
    df_out = stats_df.round(digits)
    df_out = df_out.reset_index().rename(columns={"index": "Statistic"})
    rowData = df_out.to_dict("records")
    columnDefs = [{"field": "Statistic", "pinned": "left"}]
    for c in stats_df.columns:
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
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=df[c], 
                mode="lines", 
                name=str(c),
                hovertemplate='<b>%{x}</b><br>%{y:.6f}<extra></extra>'
            ))
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        xaxis_title="Date",
        yaxis_title=ytitle,
        template="plotly_dark",
        height=500,
        hovermode='x unified'
    )
    return fig

def heatmap_figure(corr: pd.DataFrame, title: str) -> go.Figure:
    if corr is None or corr.empty:
        fig = go.Figure()
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center'),
            template="plotly_dark",
            height=500,
            xaxis={'visible': False},
            yaxis={'visible': False},
            annotations=[{
                'text': 'No correlation data available',
                'xref': 'paper',
                'yref': 'paper',
                'showarrow': False,
                'font': {'size': 16}
            }]
        )
        return fig

    z = corr.values
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            text=[[f'{val:.3f}' for val in row] for row in z],
            texttemplate='%{text}',
            textfont={"size": 12},
            colorscale="Viridis",
            zmin=-1, zmax=1,
            hoverongaps=False,
            hovertemplate='<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>'
        )
    )
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        margin=dict(l=80, r=20, t=60, b=80),
        template="plotly_dark",
        height=500,
        xaxis=dict(tickangle=-45),
    )
    return fig

def _format_sum(x: float) -> str:
    """Format a float without trailing zeros (e.g., 1.000000 -> '1')."""
    s = f"{x:.6f}".rstrip('0').rstrip('.')
    return s if s else "0"

def _normalize_date(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    try:
        return pd.to_datetime(s).strftime("%Y-%m-%d")
    except Exception:
        return None

def compute_factor_statistics(factors_df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive statistics for factors"""
    if factors_df.empty:
        return pd.DataFrame()
    
    stats = {
        'Mean': factors_df.mean(),
        'Std Dev': factors_df.std(),
        'Min': factors_df.min(),
        'Max': factors_df.max(),
        'Sharpe Ratio': factors_df.mean() / factors_df.std() * np.sqrt(252),
        'Skewness': factors_df.skew(),
        'Kurtosis': factors_df.kurtosis(),
    }
    return pd.DataFrame(stats).T

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
    Input({"type": "w-input", "asset": ALL}, "id"),
    State("coins-dd", "value"),
)
def validate_weights(values, ids, coins):
    coins = coins or []
    values = values or []
    ids = ids or []
    w_by_asset = {i.get("asset"): (float(v) if v is not None else 0.0)
                  for i, v in zip(ids, values)}
    vals = [w_by_asset.get(c, None) for c in coins]

    if not coins:
        return html.Span("Select at least one coin.", className="text-warning"), True
    if any(v is None for v in vals):
        return html.Span("Enter weights for all selected coins.", className="text-warning"), True

    s = float(np.nansum([v if v is not None else 0.0 for v in vals]))
    diff = abs(s - 1.0)
    if diff <= 1e-6:
        return html.Span("", className="text-success"), False
    else:
        return html.Span(f"Weights currently sum to {_format_sum(s)}. They must sum to 1.", className="text-danger"), True

@app.callback(
    Output("results-tabs-container", "className"),
    Output("store-exposures", "data"),
    Output("store-decomp", "data"),
    Output("store-factor-data", "data"),
    Output("exposures-chart", "figure"),
    Output("exposures-grid", "columnDefs"),
    Output("exposures-grid", "rowData"),
    Output("decomp-chart", "figure"),
    Output("decomp-grid", "columnDefs"),
    Output("decomp-grid", "rowData"),
    Input("run-btn", "n_clicks"),
    State("coins-dd", "value"),
    State({"type": "w-input", "asset": ALL}, "value"),
    State({"type": "w-input", "asset": ALL}, "id"),
    State("start-date", "value"),
    State("end-date", "value"),
    prevent_initial_call=True,
)
def run_model(_n, coins, weight_vals, weight_ids, start, end):
    # Show the results container
    results_class = "results-visible"
    
    coins = [c for c in (coins or []) if c in AVAILABLE_COINS]
    weight_vals = weight_vals or []
    weight_ids = weight_ids or []
    w_by_asset = {wid.get("asset"): (float(val) if val is not None else 0.0)
                  for wid, val in zip(weight_ids, weight_vals)}
    weights = [w_by_asset.get(c, 0.0) for c in coins]

    start = _normalize_date(start)
    end = _normalize_date(end)

    try:
        exposures = portfolio_factor_loadings(
            betas_panel, coins, weights, start, end, renormalize=True, include_alpha=False
        )
        decomp = portfolio_return_decomposition(
            betas_panel, factors_all, resids, coins, weights, start, end,
            include_alpha=False, renormalize_each_day=True
        )

        # Store all factor data for the factor analysis tab
        factor_data = factors_all.loc[start:end]

    except Exception as e:
        msg = f"Error: {e}"
        empty_line = line_figure(pd.DataFrame(), msg, "")
        return (
            results_class,
            None, None, None,
            empty_line, [], [],
            empty_line, [], [],
        )

    # Build figures
    fig_expo = line_figure(exposures, "Portfolio Factor Loadings", "Exposure")
    decomp_plot_cols = [c for c in decomp.columns if c != "TotalModel"]
    fig_decomp = line_figure(decomp[decomp_plot_cols], "Daily Factor Contributions (incl. Residual)", "Daily Return")

    # Grids
    expo_grid = df_to_aggrid(exposures, digits=6)
    decomp_grid = df_to_aggrid(decomp, digits=6)

    return (
        results_class,
        exposures.to_json(date_format="iso", orient="split"),
        decomp.to_json(date_format="iso", orient="split"),
        factor_data.to_json(date_format="iso", orient="split"),
        fig_expo,
        expo_grid["columnDefs"], expo_grid["rowData"],
        fig_decomp,
        decomp_grid["columnDefs"], decomp_grid["rowData"],
    )

@app.callback(
    Output("store-factors", "data"),
    Output("store-corr", "data"),
    Output("factors-graph", "figure"),
    Output("factors-grid", "columnDefs"),
    Output("factors-grid", "rowData"),
    Output("corr-heatmap", "figure"),
    Output("corr-grid", "columnDefs"),
    Output("corr-grid", "rowData"),
    Input("update-factors-btn", "n_clicks"),
    State("factors-select", "value"),
    State("store-factor-data", "data"),
    State("start-date", "value"),
    State("end-date", "value"),
    prevent_initial_call=True,
)
def update_factor_analysis(_n, selected_factors, factor_data_json, start, end):
    if not factor_data_json or not selected_factors:
        empty_fig = line_figure(pd.DataFrame(), "Select factors to analyze", "")
        empty_heatmap = heatmap_figure(pd.DataFrame(), "Factor Correlation Matrix")
        return (None, None, empty_fig, [], [], empty_heatmap, [], [])

    try:
        # Load factor data
        factor_data = pd.read_json(factor_data_json, orient="split")
        
        # Filter selected factors
        available_factors = [f for f in selected_factors if f in factor_data.columns]
        if not available_factors:
            empty_fig = line_figure(pd.DataFrame(), "No selected factors available in data", "")
            empty_heatmap = heatmap_figure(pd.DataFrame(), "Factor Correlation Matrix")
            return (None, None, empty_fig, [], [], empty_heatmap, [], [])
        
        factors_window = factor_data[available_factors]
        
        # Calculate correlation matrix
        corr = factors_window.corr(min_periods=10)  # Require at least 10 overlapping points
        
        # Calculate statistics
        stats_df = compute_factor_statistics(factors_window)
        
        # Create figures
        fig_factors = line_figure(factors_window, "Factor Returns Over Time", "Daily Return")
        fig_corr = heatmap_figure(corr, "Factor Correlation Matrix")
        
        # Create grids
        stats_grid = stats_to_aggrid(stats_df, digits=4)
        corr_grid = matrix_to_aggrid(corr, digits=3)
        
        return (
            factors_window.to_json(date_format="iso", orient="split"),
            corr.to_json(date_format="iso", orient="split"),
            fig_factors,
            stats_grid["columnDefs"], stats_grid["rowData"],
            fig_corr,
            corr_grid["columnDefs"], corr_grid["rowData"],
        )
        
    except Exception as e:
        error_fig = line_figure(pd.DataFrame(), f"Error: {str(e)}", "")
        error_heatmap = heatmap_figure(pd.DataFrame(), "Factor Correlation Matrix")
        return (None, None, error_fig, [], [], error_heatmap, [], [])

# Downloads
@app.callback(
    Output("dl-exposures", "data"),
    Input("btn-exposures", "n_clicks"),
    State("store-exposures", "data"),
    prevent_initial_call=True,
)
def download_exposures(_n, data_json):
    if not data_json:
        return no_update
    df = pd.read_json(data_json, orient="split")
    return dcc.send_data_frame(df.to_csv, "portfolio_exposures.csv")

@app.callback(
    Output("dl-decomp", "data"),
    Input("btn-decomp", "n_clicks"),
    State("store-decomp", "data"),
    prevent_initial_call=True,
)
def download_decomp(_n, data_json):
    if not data_json:
        return no_update
    df = pd.read_json(data_json, orient="split")
    return dcc.send_data_frame(df.to_csv, "return_decomposition.csv")

@app.callback(
    Output("dl-factors", "data"),
    Input("btn-factors", "n_clicks"),
    State("store-factors", "data"),
    prevent_initial_call=True,
)
def download_factors(_n, data_json):
    if not data_json:
        return no_update
    df = pd.read_json(data_json, orient="split")
    return dcc.send_data_frame(df.to_csv, "factor_returns.csv")

@app.callback(
    Output("dl-corr-matrix", "data"),
    Input("btn-corr-matrix", "n_clicks"),
    State("store-corr", "data"),
    prevent_initial_call=True,
)
def download_corr_matrix(_n, data_json):
    if not data_json:
        return no_update
    df = pd.read_json(data_json, orient="split")
    return dcc.send_data_frame(df.to_csv, "correlation_matrix.csv")

# =========================
# Main
# =========================
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
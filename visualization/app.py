"""
Visor 3D de detecciones multisensor y trayectorias de drones.

Uso:
    pip install -r requirements.txt
    python app.py                 # usa ./data
    python app.py --data RUTA     # carpeta con los CSV Delta_* y DroneTag_*
    python app.py --port 8051

    # Iteración original + tracks del tracker + comparación con el GT:
    python app.py --iteration "../data/raw/test/Scenario 1/Iteration 1" \
                  --interim ../data/interim --tracks ../reports/tracks.parquet
    # Comparar dos runs del tracker (antes / después de un cambio):
    python app.py ... --tracks ../reports/tracks.parquet --tracks-b otra/tracks.parquet

Abrir después http://127.0.0.1:8050 en el navegador.
"""
from __future__ import annotations

import argparse
import base64
import io
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, Patch, State, ctx, dcc, html, no_update
from plotly.subplots import make_subplots

import tracks_layer as TL
from data_loader import _read_drone, _read_sensor, bearing_rays, build_dataset, load_dataset, load_iteration

# --------------------------------------------------------------------------- #
# Configuración visual
# --------------------------------------------------------------------------- #
BG = "#0f1419"
PANEL = "#161d24"
TEXT = "#d9e1e8"
MUTED = "#8a97a3"

# Colores de sensor (evitan el rojo/azul puros, reservados a los drones)
SENSOR_COLORS = ["#f2d33c", "#5ee36a", "#b98cff", "#1fd1b5", "#e04dff",
                 "#c6d84a", "#d2b48c", "#eeeeee", "#80ffd4", "#ffd6a0"]
SENSOR_SYMBOLS = ["circle", "square", "diamond", "cross", "x",
                  "circle-open", "square-open", "diamond-open", "circle", "square"]
DRONE_COLORS = {"BLUE": ["#2f6bff", "#4fb6ff", "#7a8cff", "#1e90ff"],
                "RED": ["#ff2a2a", "#ff8a00", "#c00000", "#ff6f61", "#ffb347"]}
CONF_SCALE = "RdYlGn"   # escala global de confianza (rojo = baja, verde = alta)

PLAY_TICK_MS = 400


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _intensity_scale(hex_color, bg=BG, floor=0.18):
    """Escala de un solo tono: confianza 0 -> casi fondo, confianza 1 -> color pleno."""
    c = np.array(_hex_to_rgb(hex_color), float)
    b = np.array(_hex_to_rgb(bg), float)
    lo = b + (c - b) * floor
    mid = b + (c - b) * 0.55
    f = lambda v: "rgb(%d,%d,%d)" % tuple(np.clip(v, 0, 255))
    return [[0.0, f(lo)], [0.5, f(mid)], [1.0, f(c)]]


# --------------------------------------------------------------------------- #
# Datos
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description="Visor 3D multisensor")
parser.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
parser.add_argument("--port", type=int, default=8050)
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--iteration", default=None,
                    help="Carpeta de una iteración original (con SensorData/ y GroundTruthData/); sustituye a --data")
parser.add_argument("--interim", default=None, help="data/interim del ETL (origins, GT, detecciones, etiquetas)")
parser.add_argument("--split-group", default=None, help="p. ej. test:S1:I1; se deduce de --iteration si se omite")
parser.add_argument("--tracks", default=None, help="tracks.parquet del tracker (run A)")
parser.add_argument("--tracks-b", default=None, help="segundo tracks.parquet para comparar (run B)")
parser.add_argument("--config", default=None, help="config.yaml (corte GOSPA); por defecto el del repo")
args, _ = parser.parse_known_args()

DS = load_iteration(args.iteration) if args.iteration else load_dataset(args.data)
LAYER = None


def _cfg():
    path = args.config or os.path.join(TL.ROOT, "config.yaml")
    try:
        import yaml
        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def _load_layer():
    """Carga tracks, comparación y etiquetas si se pasó --interim (y --tracks para los tracks)."""
    global LAYER
    if not args.interim:
        LAYER = None
        return
    group = args.split_group or (TL.infer_split_group(args.iteration) if args.iteration else None)
    if not group:
        print("Aviso: no se pudo deducir --split-group; no se cargan tracks ni etiquetas.")
        LAYER = None
        return
    paths = {"A": args.tracks, "B": args.tracks_b}
    LAYER = TL.load_layer(DS, DS.t0, args.interim, group, {k: v for k, v in paths.items() if v}, _cfg())
    for name, run in LAYER.runs.items():
        it = run.compare.get("iterations") if run.compare else None
        if it is not None and len(it):
            r = it.iloc[0]
            print(f"Run {name}: {run.tracks.track_id.nunique()} tracks · GOSPA medio {r.gospa_mean:.1f} m · "
                  f"{r.pct_tracked:.0f}% del tiempo seguido · {int(r.id_switches)} cambios de track")
T0 = DS.t0
DURATION = 60
ALL_CLASSES = []
CLASS_COLORS = {}
XR = [-1000.0, 1000.0]
YR = [-1000.0, 1000.0]
MARGIN = 700


def _refresh_globals():
    global T0, DURATION, ALL_CLASSES, CLASS_COLORS, XR, YR
    T0 = DS.t0
    DURATION = max(1, int((DS.t1 - DS.t0).total_seconds()))

    ALL_CLASSES = sorted({str(c) for s in DS.sensors.values() for c in s.df["cls"].dropna().unique()})
    CLASS_COLORS = {}
    _c_idx = 0
    for _c in ALL_CLASSES:
        if _c.strip().lower() in ("unknown", "desconocido", "unk", "none", ""):
            CLASS_COLORS[_c] = "#555d66"
        else:
            CLASS_COLORS[_c] = SENSOR_COLORS[_c_idx % len(SENSOR_COLORS)]
            _c_idx += 1

    for i, s in enumerate(DS.sensors.values()):
        s.color = SENSOR_COLORS[i % len(SENSOR_COLORS)]
        s.symbol = SENSOR_SYMBOLS[i % len(SENSOR_SYMBOLS)]
        s.df["ts"] = (s.df["t"] - T0).dt.total_seconds()
        s.df["tlabel"] = s.df["t"].dt.strftime("%H:%M:%S.%f").str[:-3]

    _team_idx = {}
    for d in DS.drones.values():
        k = _team_idx.get(d.team, 0)
        d.color = DRONE_COLORS.get(d.team, ["#cccccc"])[k % len(DRONE_COLORS.get(d.team, ["#cccccc"]))]
        _team_idx[d.team] = k + 1
        d.df["ts"] = (d.df["t"] - T0).dt.total_seconds()
        d.df["tlabel"] = d.df["t"].dt.strftime("%H:%M:%S")

    e_series = [d.df["e"] for d in DS.drones.values()] + [pd.Series([s.pos_e]) for s in DS.sensors.values()]
    n_series = [d.df["n"] for d in DS.drones.values()] + [pd.Series([s.pos_n]) for s in DS.sensors.values()]
    # Detecciones con posición: en eval no hay drones GT y sin ellas los ejes quedaban ceñidos a los
    # sensores. Percentiles 2–98 y < 6 km (el GT de test vuela en ±4 km) para no estirar los ejes con
    # ruido lejano del radar (ecos hasta ~200 km).
    det_e, det_n = [], []
    for s in DS.sensors.values():
        if not s.bearing_only and {"e", "n"} <= set(s.df.columns):
            ok = np.hypot(s.df["e"], s.df["n"]) < 6000
            det_e.append(s.df.loc[ok, "e"])
            det_n.append(s.df.loc[ok, "n"])
    if det_e and pd.concat(det_e).notna().any():
        de, dn = pd.concat(det_e).dropna(), pd.concat(det_n).dropna()
        e_series.append(de.quantile([0.02, 0.98]))
        n_series.append(dn.quantile([0.02, 0.98]))
    if e_series and n_series:
        _e = pd.concat(e_series).dropna()
        _n = pd.concat(n_series).dropna()
        if len(_e) and len(_n):
            XR = [float(_e.min()) - MARGIN, float(_e.max()) + MARGIN]
            YR = [float(_n.min()) - MARGIN, float(_n.max()) + MARGIN]


_refresh_globals()
_load_layer()


def _drone_colors():
    return {k: d.color for k, d in DS.drones.items()}


# --------------------------------------------------------------------------- #
# Construcción de la figura
# --------------------------------------------------------------------------- #
def build_figure(t_now, sensors_on, drones_on, opts, color_mode, min_conf, ray_len,
                 z_max, play_mode, window_s, use_agl, tr_opts=(), run_name="A", ell_scale=1):
    fig = go.Figure()
    zc = "z_agl" if use_agl else "z_raw"
    show_rays = "rays" in opts
    show_pos = "sensor_pos" in opts
    size_by_conf = "size_conf" in opts
    t_lo = -1e9 if play_mode == "cum" else t_now - float(window_s or 30)

    global_scale_drawn = False
    
    # Leyenda de clases 
    if color_mode == "sensor":
        for cls_name, c_col in CLASS_COLORS.items():
            is_unk = cls_name.strip().lower() in ("unknown", "desconocido", "unk", "none", "")
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None], mode="markers", name=f"Clase: {cls_name}",
                legendgroup=f"cls_{cls_name}",
                marker=dict(size=8, color=c_col, symbol="circle", opacity=0.35 if is_unk else 0.9),
                hoverinfo="skip", showlegend=True))

    for key in sensors_on:
        s = DS.sensors[key]
        d = s.df
        m = (d["ts"] <= t_now) & (d["ts"] >= t_lo) & (d["conf"] >= min_conf)
        d = d[m]
        if not s.bearing_only:
            d = d[d[zc] <= z_max]

        for cls_name, d_cls in d.groupby("cls"):
            if d_cls.empty:
                continue
            if s.bearing_only:
                xs, ys, zs = bearing_rays(s, d_cls, ray_len, use_agl)
            else:
                xs, ys, zs = d_cls["e"].values, d_cls["n"].values, d_cls[zc].values

            conf = d_cls["conf"].values
            is_unk = str(cls_name).strip().lower() in ("unknown", "desconocido", "unk", "none", "")
            c_color = CLASS_COLORS.get(str(cls_name), "#555d66")

            if color_mode == "sensor":
                marker = dict(color=conf, colorscale=_intensity_scale(c_color, floor=0.08 if is_unk else 0.18),
                              cmin=0, cmax=1, symbol="circle", showscale=False)
            else:
                marker = dict(color=conf, colorscale=CONF_SCALE, cmin=0, cmax=1, symbol="circle",
                              showscale=not global_scale_drawn,
                              colorbar=dict(title=dict(text="Confianza", font=dict(color=TEXT)),
                                            tickfont=dict(color=TEXT), len=0.4, x=0.995, xanchor="right",
                                            y=0.99, yanchor="top", thickness=14))
                global_scale_drawn = True

            marker["size"] = (2 + 4.5 * conf) if size_by_conf else 3.5
            marker["line"] = dict(width=0)
            marker["opacity"] = 0.35 if is_unk else 0.9

            conf_txt = np.where(d_cls["conf_missing"].values, "n/d", np.round(conf, 3).astype(str))
            custom = np.stack([d_cls["tlabel"].values, d_cls["cls"].values, d_cls["subcls"].values, conf_txt,
                               d_cls["obj"].str[-8:].values, d_cls["state"].values], axis=-1)
            geo = ("az %{customdata[6]}°, el %{customdata[7]}° (sin distancia; punto a "
                   f"{ray_len:.0f} m)") if s.bearing_only else "E %{x:.0f} m · N %{y:.0f} m · z %{z:.0f} m"
            if s.bearing_only:
                custom = np.concatenate([custom, np.stack([np.round(d_cls["az"].values, 1).astype(str),
                                                           np.round(d_cls["el"].values, 1).astype(str)], -1)], 1)

            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="markers", name=str(cls_name), legendgroup=f"cls_{cls_name}",
                showlegend=False, marker=marker, customdata=custom,
                hovertemplate=(f"<b>Sensor: {s.key}</b><br>%{{customdata[0]}}<br>Clase: %{{customdata[1]}} "
                               "%{customdata[2]}<br>Confianza: %{customdata[3]}<br>Obj …%{customdata[4]} "
                               f"%{{customdata[5]}}<br>{geo}<extra></extra>"),
            ))

            # Rayos de marcación desde el sensor coloreados por clase
            if s.bearing_only and show_rays:
                n = len(xs)
                lx = np.empty(3 * n); ly = np.empty(3 * n); lz = np.empty(3 * n)
                z0 = s.pos_z_agl if use_agl else s.pos_z_raw
                lx[0::3], ly[0::3], lz[0::3] = s.pos_e, s.pos_n, z0
                lx[1::3], ly[1::3], lz[1::3] = xs, ys, zs
                lx[2::3] = ly[2::3] = lz[2::3] = np.nan
                lc = np.repeat(conf, 3)
                if color_mode == "sensor":
                    line = dict(color=lc, colorscale=_intensity_scale(c_color, floor=0.08 if is_unk else 0.18),
                                cmin=0, cmax=1, width=1.5)
                else:
                    line = dict(color=lc, colorscale=CONF_SCALE, cmin=0, cmax=1, width=1.5)
                fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode="lines", line=line,
                                           opacity=0.12 if is_unk else 0.25,
                                           name=f"{s.key} (rayos)", legendgroup=f"cls_{cls_name}", showlegend=False,
                                           hoverinfo="skip", connectgaps=False))

        if show_pos:
            z0 = s.pos_z_agl if use_agl else s.pos_z_raw
            fig.add_trace(go.Scatter3d(
                x=[s.pos_e], y=[s.pos_n], z=[z0], mode="markers",
                marker=dict(size=6, color="#8a97a3", symbol="diamond", line=dict(color="#000", width=1)),
                name=f"Sensor {s.node}", showlegend=False,
                hovertemplate=(f"<b>Sensor {s.key}</b><br>{s.file}<br>Confianza usada: {s.conf_source}"
                               f"<br>{s.z_note}<extra></extra>")))
    # Trayectorias de drones
    for key in drones_on:
        dr = DS.drones[key]
        d = dr.df[dr.df["ts"] <= t_now]
        if d.empty:
            continue
        custom = np.stack([d["tlabel"].values, np.round(d["geo_altitude"].values, 1),
                           np.round(d["height"].values, 1), np.round(d["speed"].values, 1)], -1)
        fig.add_trace(go.Scatter3d(
            x=d["e"], y=d["n"], z=d[zc], mode="lines", name=f"Drone {dr.key}",
            line=dict(color=dr.color, width=6), customdata=custom,
            hovertemplate=(f"<b>Drone {dr.key}</b><br>%{{customdata[0]}}<br>Alt. GNSS %{{customdata[1]}} m · "
                           "altura %{customdata[2]} m<br>Velocidad %{customdata[3]} m/s<extra></extra>")))
        last = d.iloc[-1]
        fig.add_trace(go.Scatter3d(
            x=[last["e"]], y=[last["n"]], z=[last[zc]], mode="markers+text", text=[dr.name],
            textposition="top center", textfont=dict(color=dr.color, size=12),
            marker=dict(size=8, color=dr.color, symbol="circle", line=dict(color="#fff", width=2)),
            showlegend=False, name=dr.key, hoverinfo="skip"))

    # Tracker: trayectorias estimadas, error real↔estimado, incertidumbre y etiquetas
    if LAYER is not None:
        if "labels" in tr_opts:
            for tr in TL.label_traces(go, LAYER, t_now, t_lo, zc, z_max, _drone_colors()):
                fig.add_trace(tr)
        for tr in TL.track_traces(go, LAYER, run_name, t_now, t_lo, zc, z_max, list(tr_opts), DS, _drone_colors(),
                                  ell_scale):
            fig.add_trace(tr)

    ztitle = "Altura aprox. sobre terreno (m)" if use_agl else "Cota original (m)"
    axis = dict(backgroundcolor=BG, gridcolor="#2a3540", zerolinecolor="#3a4855", color=MUTED,
                showspikes=False)
    zlo = -30 if use_agl else -30
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=TEXT, family="Inter, Segoe UI, sans-serif"),
        margin=dict(l=0, r=0, t=0, b=0), uirevision="keep",
        legend=dict(bgcolor="rgba(22,29,36,0.85)", bordercolor="#2a3540", borderwidth=1,
                    x=0.995, xanchor="right", y=0.01, yanchor="bottom", font=dict(size=10),
                    itemsizing="constant"),
        scene=dict(
            uirevision="keep",
            xaxis=dict(axis, title="Este (m)", range=XR),
            yaxis=dict(axis, title="Norte (m)", range=YR),
            zaxis=dict(axis, title=ztitle, range=[zlo, z_max]),
            aspectmode="manual",
            aspectratio=dict(x=1.0 * (XR[1] - XR[0]) / max(YR[1] - YR[0], 1), y=1.0, z=0.45),
            camera=dict(eye=dict(x=0.9, y=-1.5, z=0.9)),
        ),
    )
    return fig


# --------------------------------------------------------------------------- #
# Interfaz
# --------------------------------------------------------------------------- #
def swatch(color, label, sub=""):
    return html.Span([
        html.Span(style={"display": "inline-block", "width": "11px", "height": "11px",
                         "background": color, "borderRadius": "2px", "marginRight": "7px",
                         "verticalAlign": "middle"}),
        html.Span(label, style={"verticalAlign": "middle"}),
        html.Span(f"  {sub}", className="sub", style={"fontSize": "11px", "verticalAlign": "middle"}),
    ])


def section(title, *children):
    return html.Div([html.Div(title, className="sec-title"), *children], className="section")


def _get_sensor_options():
    return [{"label": html.Span([html.Span(s.key), html.Span(f"  {len(s.df)}" + (" · marcación" if s.bearing_only else ""), className="sub", style={"fontSize": "11px"})]),
            "value": k} for k, s in DS.sensors.items()]

def _get_drone_options():
    return [{"label": swatch(d.color, d.key, f"{d.df['t'].min():%H:%M:%S}–{d.df['t'].max():%H:%M:%S}"),
            "value": k} for k, d in DS.drones.items()]

def _get_marks():
    step = max(60, (DURATION // 10 // 60) * 60)
    return {int(s): (T0 + pd.Timedelta(seconds=int(s))).strftime("%H:%M") for s in range(0, DURATION + 1, step)}

def _tracker_section():
    if LAYER is None:
        return section("Tracker y comparación", html.Div(
            "Sin tracks cargados. Lanza con --interim (y --tracks) para ver las trayectorias estimadas, "
            "compararlas con las reales y colorear las detecciones por etiqueta.", className="help"),
            dcc.Checklist(id="tr-opts", options=[], value=[], style={"display": "none"}),
            dcc.RadioItems(id="run", options=[{"label": "A", "value": "A"}], value="A", style={"display": "none"}),
            dcc.Dropdown(id="ell-scale", options=[{"label": "real", "value": 1}], value=1, style={"display": "none"}))
    runs = list(LAYER.runs)
    opts = []
    if runs:
        opts += [{"label": "Trayectorias estimadas (discontinuas)", "value": "tracks"},
                 {"label": "Error real↔estimado (línea blanca)", "value": "errors"},
                 {"label": "Elipse de incertidumbre 2σ", "value": "ellipse"}]
    if not LAYER.labelled.empty:
        opts.append({"label": "Detecciones coloreadas por etiqueta", "value": "labels"})
    children = [html.Div(f"Iteración {LAYER.split_group}", className="help", style={"marginBottom": "4px"}),
                dcc.Checklist(id="tr-opts", options=opts, value=[o["value"] for o in opts if o["value"] in ("tracks", "errors")]),
                # La elipse 2σ real mide ~10 m: invisible en una escena de km. Se puede ampliar solo para verla.
                html.Div([html.Span("Tamaño de la elipse", className="help", style={"marginRight": "8px"}),
                          dcc.Dropdown(id="ell-scale", clearable=False, value=1, style={"width": "120px", "color": "#000"},
                                       options=[{"label": "real", "value": 1}, {"label": "×5 (ampliada)", "value": 5},
                                                {"label": "×10 (ampliada)", "value": 10}])],
                         style={"display": "flex", "alignItems": "center", "marginTop": "4px"})]
    children.append(html.Div([html.Span("Run", className="help", style={"marginRight": "8px"}),
                              dcc.RadioItems(id="run", value=runs[0] if runs else "A", inline=True,
                                             options=[{"label": f" {r} ", "value": r} for r in runs] or
                                             [{"label": " — ", "value": "A"}])],
                             style={"display": "flex" if len(runs) > 1 else "none", "alignItems": "center",
                                    "marginTop": "4px"}))
    children.append(html.Div("Al activar las etiquetas conviene ocultar los sensores para ver solo los colores "
                             "del etiquetado. Gris = ruido (no casa con ningún dron real).", className="help",
                             style={"marginTop": "4px"}))
    return section("Tracker y comparación", *children)


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return "–" if not np.isfinite(v) else f"{v:.1f}"
    return str(v)


def _table(df, highlight=None):
    head = html.Tr([html.Th(c) for c in df.columns])
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            style = {}
            if highlight and c in highlight and isinstance(r[c], (int, float, np.number)) and np.isfinite(r[c]) and r[c] != 0:
                better = (r[c] < 0) if highlight[c] == "low" else (r[c] > 0)
                style = {"color": "#5ee36a" if better else "#ff6f61", "fontWeight": 600}
            cells.append(html.Td(_fmt(r[c]), style=style))
        rows.append(html.Tr(cells))
    return html.Table([html.Thead(head), html.Tbody(rows)], className="tbl")


def _summary_children():
    if LAYER is None or not LAYER.runs:
        return [html.Div("Sin tracks cargados.", className="help")]
    tabs = TL.summary_tables(LAYER)
    if not tabs:
        return [html.Div("Esta iteración no tiene ground truth: solo se pueden ver los tracks, no compararlos.",
                         className="help")]
    out = []
    for name, t in tabs.items():
        it = t["iteration"]
        if len(it):
            r = it.iloc[0]
            out.append(html.Div([
                html.B(f"Run {name}: "),
                f"GOSPA medio {r.gospa_mean:.1f} m (posición {r.gospa_loc:.1f} · perdidos {r.gospa_missed:.1f} · "
                f"falsos {r.gospa_false:.1f}) · {r.pct_tracked:.0f}% del tiempo seguido · "
                f"{int(r.id_switches)} cambios de track · {int(r.tracks_never_matched)} tracks que nunca casan con un dron"],
                className="help", style={"margin": "6px 0", "color": TEXT}))
        out.append(_table(t["drones"]))
    if len(tabs) == 2:
        a, b = tabs["A"]["drones"], tabs["B"]["drones"]
        m = a.merge(b, on="Dron", suffixes=(" A", " B"))
        diff = pd.DataFrame({"Dron": m["Dron"]})
        for col, better in (("% seguido", "high"), ("Error medio (m)", "low"), ("Cambios de track", "low")):
            diff[f"{col} A"] = m[f"{col} A"]
            diff[f"{col} B"] = m[f"{col} B"]
            diff[f"Δ {col}"] = m[f"{col} B"] - m[f"{col} A"]
        out.insert(0, html.Div([html.B("A → B"), " (verde = B mejora, rojo = B empeora)"],
                               className="help", style={"margin": "6px 0", "color": TEXT}))
        out.insert(1, _table(diff, highlight={"Δ % seguido": "high", "Δ Error medio (m)": "low",
                                              "Δ Cambios de track": "low"}))
    return out


sensor_options = _get_sensor_options()
drone_options = _get_drone_options()
marks = _get_marks()

app = Dash(__name__, title="Visor 3D multisensor")
app.index_string = app.index_string.replace("</head>", f"""
<style>
 body {{ margin:0; background:{BG}; color:{TEXT}; font-family: Inter, 'Segoe UI', sans-serif; }}
 .sidebar {{ width:330px; min-width:330px; height:100vh; overflow-y:auto; background:{PANEL};
            padding:14px 16px; box-sizing:border-box; border-right:1px solid #24303a; font-size:13px; }}
 .section {{ margin-bottom:14px; }}
 .sec-title {{ font-size:11px; text-transform:uppercase; letter-spacing:.08em; color:{MUTED};
              margin-bottom:6px; font-weight:600; }}
 .btn {{ background:#24303a; color:{TEXT}; border:1px solid #33414d; border-radius:6px;
        padding:6px 12px; cursor:pointer; font-size:13px; margin-right:6px; }}
 .btn:hover {{ background:#2e3c48; }}
 .btn-play {{ background:#1f6f4a; border-color:#2a8a5e; min-width:86px; }}
 .mini {{ font-size:11px; color:{MUTED}; cursor:pointer; margin-right:10px; text-decoration:underline; }}
 label {{ display:block; margin:2px 0; cursor:pointer; }}
 input[type=checkbox], input[type=radio] {{ margin-right:6px; }}
 .Select-control, .Select-menu-outer {{ background:#24303a !important; color:{TEXT} !important; }}
 .rc-slider-mark-text {{ color:{MUTED} !important; font-size:10px; }}
 .help {{ color:{MUTED}; font-size:11px; line-height:1.4; }}
 .sidebar label, .sidebar label span {{ color:{TEXT} !important; }}
 .sidebar .sub {{ color:{MUTED} !important; }}
 .dash-slider-mark, .rc-slider-mark-text, [class*="mark"] {{ color:{MUTED} !important; font-size:10px; }}
 .bottom {{ height:420px; min-height:420px; border-top:1px solid #24303a; display:flex; flex-direction:column; }}
 .bottom .tab-content {{ flex:1; min-height:0; overflow:auto; }}
 .tbl {{ border-collapse:collapse; font-size:12px; margin:4px 0 12px; font-variant-numeric:tabular-nums; }}
 .tbl th, .tbl td {{ padding:4px 10px; border-bottom:1px solid #24303a; text-align:right; white-space:nowrap; }}
 .tbl th:first-child, .tbl td:first-child {{ text-align:left; }}
 .tbl th {{ color:{MUTED}; font-weight:600; font-size:11px; }}
</style></head>""")

app.layout = html.Div(style={"display": "flex", "height": "100vh"}, children=[
    html.Div(className="sidebar", children=[
        html.H3("Visor 3D multisensor", style={"margin": "0 0 4px 0", "fontSize": "17px"}),
    html.Div(f"{T0:%d/%m/%Y} · {DS.t0:%H:%M:%S}–{DS.t1:%H:%M:%S} UTC", id="date-range", className="help",
                 style={"marginBottom": "12px"}),

        section("Cargar datos",
                dcc.Upload(
                    id="upload-data",
                    children=html.Button("＋ Add sensor / drone CSV", className="btn", style={"width": "100%"}),
                    multiple=True,
                    accept=".csv"
                ),
                html.Div(id="upload-status", className="help", style={"marginTop": "4px"})),

        section("Reproducción",
                html.Div([
                    html.Button("▶ Play", id="btn-play", className="btn btn-play", n_clicks=0),
                    html.Button("⟲ Inicio", id="btn-reset", className="btn", n_clicks=0),
                    html.Button("Fin", id="btn-end", className="btn", n_clicks=0),
                ], style={"marginBottom": "8px"}),
                html.Div([html.Span("Velocidad", className="help", style={"marginRight": "8px"}),
                          dcc.Dropdown(id="speed", clearable=False, value=5, style={"width": "100px", "color": "#000"},
                                       options=[{"label": f"{v}×", "value": v} for v in (1, 2, 5, 10, 20, 40)])],
                         style={"display": "flex", "alignItems": "center", "marginBottom": "6px"}),
                dcc.RadioItems(id="play-mode", value="cum", options=[
                    {"label": "Acumulado (todo hasta t)", "value": "cum"},
                    {"label": "Ventana deslizante", "value": "win"}]),
                html.Div([html.Span("Ventana (s)", className="help", style={"marginRight": "8px"}),
                          dcc.Input(id="window", type="number", value=20, min=1, max=600, step=1,
                                    style={"width": "70px", "background": "#24303a", "color": TEXT, "border": "1px solid #33414d", "borderRadius": "4px", "padding": "3px 6px"})], style={"marginTop": "4px"}),
                ),

        section("Sensores",
                html.Div([html.Span("todos", id="s-all", className="mini"),
                          html.Span("ninguno", id="s-none", className="mini")]),
                # Con tracks cargados se arranca sin sensores: con todos y en modo acumulado cada paso
                # del Play enviaba hasta 4 MB al navegador. Se pueden activar a mano.
                html.Div("Con muchos sensores activos, usa la ventana deslizante para que el Play vaya fluido.",
                         className="help", style={"marginBottom": "4px"})
                if LAYER is not None and LAYER.runs else None,
                dcc.Checklist(id="sensors", options=sensor_options,
                              value=[] if LAYER is not None and LAYER.runs else list(DS.sensors.keys()))),

        section("Trayectorias de drones",
                html.Div([html.Span("todos", id="d-all", className="mini"),
                          html.Span("ninguno", id="d-none", className="mini")]),
                dcc.Checklist(id="drones", options=drone_options, value=list(DS.drones.keys()))),

        _tracker_section(),

        section("Codificación de la confianza",
                dcc.RadioItems(id="color-mode", value="sensor", options=[
                    {"label": "Color por clase · intensidad = confianza", "value": "sensor"},
                    {"label": "Escala global (rojo→verde)", "value": "global"}]),
                dcc.Checklist(id="opts", value=["rays", "sensor_pos", "size_conf", "agl"], options=[
                    {"label": "Tamaño del punto ∝ confianza", "value": "size_conf"},
                    {"label": "Rayos de marcación (cámara, acústico, RF)", "value": "rays"},
                    {"label": "Posición de los sensores", "value": "sensor_pos"},
                    {"label": "Altura sobre terreno (corregir datum vertical)", "value": "agl"}]),
                ),

        section("Filtros",
                html.Div("Confianza mínima", className="help"),
                dcc.Slider(id="min-conf", allow_direct_input=False, min=0, max=1, step=0.05, value=0,
                           marks={0: "0", 0.25: ".25", 0.5: ".5", 0.75: ".75", 1: "1"}),
                html.Div("Altura máxima mostrada (m)", className="help"),
                dcc.Slider(id="z-max", allow_direct_input=False, min=100, max=1500, step=50, value=500,
                           marks={100: "100", 500: "500", 1000: "1000", 1500: "1500"}),
                html.Div("Longitud de los rayos de marcación (m)", className="help"),
                dcc.Slider(id="ray-len", allow_direct_input=False, min=100, max=3000, step=50, value=1000,
                           marks={100: "100", 1000: "1000", 2000: "2000", 3000: "3000"}),
                ),

        html.Div(className="help", children=[
            "Los sensores de cámara, acústico y RF pasivo sólo dan azimut/elevación (sin distancia): "
            "se dibujan como rayos desde el sensor y un punto a la longitud indicada. ",
            "Pasa el ratón sobre la posición de cada sensor para ver qué campo de confianza se usa."]),
    ]),

    html.Div(style={"flex": "1", "display": "flex", "flexDirection": "column", "minWidth": 0}, children=[
        html.Div(style={"padding": "10px 18px 0 18px", "display": "flex", "alignItems": "center", "gap": "14px"},
                 children=[
                     html.Div(id="clock", style={"fontSize": "22px", "fontVariantNumeric": "tabular-nums",
                                                 "fontWeight": 600, "minWidth": "110px"}),
                     html.Div(id="counts", className="help"),
                 ]),
        html.Div(style={"padding": "4px 26px 0 18px"}, children=[
            dcc.Slider(id="time", min=0, max=DURATION, step=1, value=DURATION, marks=marks,
                       updatemode="drag", allow_direct_input=False)]),
        dcc.Graph(id="graph", style={"flex": "1", "minHeight": 0},
                  config={"displaylogo": False, "scrollZoom": True}),
        html.Div(className="bottom", style={"display": "flex" if LAYER is not None and LAYER.runs else "none"},
                 children=[
            dcc.Tabs(id="cmp-tabs", value="plots", style={"height": "34px"}, children=[
                dcc.Tab(label="Error y cobertura", value="plots",
                        style={"padding": "6px", "background": PANEL, "color": MUTED, "border": "none"},
                        selected_style={"padding": "6px", "background": BG, "color": TEXT, "border": "none"},
                        children=[dcc.Graph(id="cmp-graph", style={"height": "380px"},
                                            config={"displaylogo": False})]),
                dcc.Tab(label="Resumen por dron" + (" · A vs B" if LAYER is not None and len(LAYER.runs) > 1 else ""),
                        value="table",
                        style={"padding": "6px", "background": PANEL, "color": MUTED, "border": "none"},
                        selected_style={"padding": "6px", "background": BG, "color": TEXT, "border": "none"},
                        children=[html.Div(_summary_children(), style={"padding": "6px 18px", "overflow": "auto",
                                                                        "height": "380px"})]),
            ]),
        ]),
    ]),

    dcc.Interval(id="tick", interval=PLAY_TICK_MS, disabled=True),
    dcc.Store(id="playing", data=False),
])


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #
@app.callback(Output("sensors", "value"), Input("s-all", "n_clicks"), Input("s-none", "n_clicks"),
              prevent_initial_call=True)
def _sensors_all_none(_a, _b):
    return list(DS.sensors.keys()) if ctx.triggered_id == "s-all" else []


@app.callback(Output("drones", "value"), Input("d-all", "n_clicks"), Input("d-none", "n_clicks"),
              prevent_initial_call=True)
def _drones_all_none(_a, _b):
    return list(DS.drones.keys()) if ctx.triggered_id == "d-all" else []


@app.callback(Output("playing", "data"), Output("tick", "disabled"), Output("btn-play", "children"),
              Output("time", "value"),
              Input("btn-play", "n_clicks"), Input("btn-reset", "n_clicks"), Input("btn-end", "n_clicks"),
              Input("tick", "n_intervals"),
              State("playing", "data"), State("time", "value"), State("speed", "value"),
              prevent_initial_call=True)
def _player(_p, _r, _e, _n, playing, t, speed):
    trig = ctx.triggered_id
    step = (speed or 1) * PLAY_TICK_MS / 1000.0
    if trig == "btn-reset":
        return playing, not playing, ("⏸ Pausa" if playing else "▶ Play"), 0
    if trig == "btn-end":
        return False, True, "▶ Play", DURATION
    if trig == "btn-play":
        if playing:
            return False, True, "▶ Play", no_update
        start = 0 if t >= DURATION else t
        return True, False, "⏸ Pausa", start
    if trig == "tick" and playing:
        nt = min(DURATION, (t or 0) + step)
        if nt >= DURATION:
            return False, True, "▶ Play", DURATION
        return True, False, "⏸ Pausa", nt
    return no_update, no_update, no_update, no_update


@app.callback(Output("graph", "figure"), Output("clock", "children"), Output("counts", "children"),
              Input("time", "value"), Input("sensors", "value"), Input("drones", "value"),
              Input("opts", "value"), Input("color-mode", "value"), Input("min-conf", "value"),
              Input("ray-len", "value"), Input("z-max", "value"), Input("play-mode", "value"),
              Input("window", "value"), Input("tr-opts", "value"), Input("run", "value"), Input("ell-scale", "value"),
              State("graph", "relayoutData"))
def _update(t, sensors_on, drones_on, opts, color_mode, min_conf, ray_len, z_max, play_mode, window_s,
            tr_opts, run_name, ell_scale, relayout):
    t = float(t or 0)
    opts = opts or []
    fig = build_figure(t, sensors_on or [], drones_on or [], opts, color_mode, min_conf or 0,
                       ray_len or 1000, z_max or 500, play_mode, window_s, "agl" in opts,
                       tr_opts or [], run_name or "A", ell_scale or 1)
    if relayout and "scene.camera" in relayout:
        fig.layout.scene.camera = relayout["scene.camera"]
    n_pts = sum(len(tr.x) for tr in fig.data
                if tr.mode == "markers" and tr.x is not None and not tr.showlegend)
    clock = (T0 + pd.Timedelta(seconds=t)).strftime("%H:%M:%S")
    return fig, clock, f"{n_pts:,} detecciones visibles".replace(",", ".")


_CMP_CACHE = {}


def _now_shape(t):
    x = (DS.t0 + pd.Timedelta(seconds=float(t or 0))).isoformat()
    return {"type": "line", "xref": "x", "yref": "paper", "x0": x, "x1": x, "y0": 0, "y1": 1,
            "line": {"color": TEXT, "width": 1}}


@app.callback(Output("cmp-graph", "figure"), Input("time", "value"), Input("run", "value"))
def _update_cmp(t, run_name):
    """Panel inferior.

    La figura completa solo se envía al cambiar de run (o al cargar). Al moverse el tiempo
    se envía un parche con las shapes (la línea de "ahora"), no la figura entera:
    reenviarla en cada paso del Play era lo que hacía ir lento al visor.
    """
    if LAYER is None or not LAYER.runs:
        return go.Figure(layout=dict(paper_bgcolor=BG, plot_bgcolor=BG))
    run_name = run_name or "A"
    if run_name not in _CMP_CACHE:
        _CMP_CACHE[run_name] = TL.comparison_figure(go, make_subplots, LAYER, run_name, DS, None, _drone_colors(),
                                                    BG, TEXT, MUTED)
    base = _CMP_CACHE[run_name]
    shapes = [s.to_plotly_json() for s in base.layout.shapes] + [_now_shape(t)]
    if ctx.triggered_id == "time":
        patch = Patch()
        patch["layout"]["shapes"] = shapes
        return patch
    fig = go.Figure(base)
    fig.layout.shapes = shapes
    return fig

@app.callback(
    Output("sensors", "options"), Output("sensors", "value", allow_duplicate=True),
    Output("drones", "options"), Output("drones", "value", allow_duplicate=True),
    Output("time", "max"), Output("time", "marks"), Output("time", "value", allow_duplicate=True),
    Output("date-range", "children"), Output("upload-status", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    State("sensors", "value"), State("drones", "value"),
    prevent_initial_call=True
)
def _upload_csv(contents_list, filenames, current_sensors, current_drones):
    global DS
    if not contents_list:
        return (no_update,) * 9

    sensors_dict = dict(DS.sensors)
    drones_dict = dict(DS.drones)
    new_sensors, new_drones, errors = [], [], []

    for content, fname in zip(contents_list, filenames):
        try:
            _, content_string = content.split(",", 1)
            decoded = base64.b64decode(content_string).decode("utf-8", errors="replace")
            buf = io.StringIO(decoded)
            if fname.startswith("DroneTag_") or "geo_altitude" in decoded[:500]:
                dr = _read_drone(buf, filename=fname)
                drones_dict[dr.key] = dr
                new_drones.append(dr.key)
            else:
                s = _read_sensor(buf, filename=fname)
                sensors_dict[s.key] = s
                new_sensors.append(s.key)
        except Exception as e:
            errors.append(f"{fname}: {e}")

    DS = build_dataset(list(sensors_dict.values()), list(drones_dict.values()))
    _refresh_globals()
    # El origen del visor puede cambiar con los CSV nuevos: reproyectar tracks y etiquetas.
    _load_layer()
    _CMP_CACHE.clear()

    s_val = list(dict.fromkeys((current_sensors or []) + new_sensors))
    d_val = list(dict.fromkeys((current_drones or []) + new_drones))
    date_txt = f"{T0:%d/%m/%Y} · {DS.t0:%H:%M:%S}–{DS.t1:%H:%M:%S} UTC"
    status = f"Cargados: {len(new_sensors)} sensores, {len(new_drones)} drones."
    if errors:
        status += " Error en: " + "; ".join(errors)

    return (_get_sensor_options(), s_val, _get_drone_options(), d_val,
            DURATION, _get_marks(), DURATION, date_txt, status)
if __name__ == "__main__":
    print(f"Abrir http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)

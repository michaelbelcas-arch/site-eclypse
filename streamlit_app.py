# streamlit_app.py
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date
from pathlib import Path
import yaml, os
import hashlib
from utils_io import file_signature
from atomic_write import atomic_write_text
from streamlit_autorefresh import st_autorefresh
import time, threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

def add_temporal_analysis_tab():
    st.header("📈 Analyse Temporelle")

    matches = load_matches()
    if matches.empty:
        st.info("Aucun match enregistré pour l'analyse temporelle.")
        return

    # Filtres
    col1, col2, col3 = st.columns(3)
    with col1:
        period = st.selectbox("Période d'agrégation", ["jour", "semaine", "mois"], index=1)
    with col2:
        all_decks = sorted(set(matches["deck_a"].tolist() + matches["deck_b"].tolist()))
        selected_decks = st.multiselect("Decks à analyser", all_decks, default=all_decks[:5])
    with col3:
        min_matches = st.number_input("Matches minimum par période", min_value=1, value=3)

    if not selected_decks:
        st.warning("Sélectionnez au moins un deck pour l'analyse.")
        return

    # Préparation des données temporelles
    def prepare_temporal_data(matches_df, selected_decks, period_agg, min_matches_threshold):
        # Expansion des données : chaque match devient 2 lignes (une par deck)
        expanded_data = []

        for _, match in matches_df.iterrows():
            if pd.isna(match["timestamp"]):
                continue

            wins_a = int(match["score_a"])
            wins_b = int(match["score_b"])

            # Ligne pour deck_a
            if match["deck_a"] in selected_decks:
                expanded_data.append({
                    "timestamp": match["timestamp"],
                    "deck": match["deck_a"],
                    "wins": wins_a,        # Matchs gagnés par ce deck
                    "losses": wins_b,      # Matchs perdus par ce deck
                    "total_games": wins_a + wins_b  # Total des jeux dans ce match
                })

            # Ligne pour deck_b
            if match["deck_b"] in selected_decks:
                expanded_data.append({
                    "timestamp": match["timestamp"],
                    "deck": match["deck_b"],
                    "wins": wins_b,        # Matchs gagnés par ce deck
                    "losses": wins_a,      # Matchs perdus par ce deck
                    "total_games": wins_a + wins_b  # Total des jeux dans ce match
                })

        if not expanded_data:
            return pd.DataFrame()

        df_expanded = pd.DataFrame(expanded_data)

        # Agrégation temporelle
        freq_map = {"jour": "D", "semaine": "W-MON", "mois": "M"}
        df_expanded["period"] = df_expanded["timestamp"].dt.to_period(freq_map[period_agg])

        # Calcul du winrate par période et deck
        agg_data = df_expanded.groupby(["period", "deck"]).agg({
            "wins": "sum",        # Total des jeux gagnés
            "losses": "sum",      # Total des jeux perdus
            "total_games": "sum", # Total des jeux joués
        }).reset_index()

        # Calculer le winrate = jeux gagnés / total jeux
        agg_data["winrate"] = (agg_data["wins"] / agg_data["total_games"] * 100).round(1)
        agg_data["matches_count"] = df_expanded.groupby(["period", "deck"]).size().values

        # Filtrer par nombre minimum de matches
        agg_data = agg_data[agg_data["total_games"] >= min_matches_threshold]

        # Conversion de period en datetime pour les graphiques
        agg_data["date"] = agg_data["period"].dt.start_time

        return agg_data

    temporal_data = prepare_temporal_data(matches, selected_decks, period, min_matches)

    if temporal_data.empty:
        st.warning(f"Pas assez de données avec {min_matches} matches minimum par {period}.")
        return

    # --- Graphique 1: Évolution des winrates ---
    st.subheader("📈 Évolution des Winrates")

    fig_evolution = go.Figure()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    for i, deck in enumerate(selected_decks):
        deck_data = temporal_data[temporal_data["deck"] == deck].sort_values("date")
        if deck_data.empty:
            continue

        color = colors[i % len(colors)]

        fig_evolution.add_trace(go.Scatter(
            x=deck_data["date"],
            y=deck_data["winrate"],
            mode="lines+markers",
            name=deck,
            line=dict(color=color, width=2),
            marker=dict(color=color, size=6),
            hovertemplate=f"<b>{deck}</b><br>" +
                          "Date: %{x}<br>" +
                          "Winrate: %{y:.1f}%<br>" +
                          "W - L: %{customdata[0]} - %{customdata[1]}<br>" +
                          "Sessions jouées: %{customdata[2]}<extra></extra>",
            customdata=deck_data[["wins", "losses", "matches_count"]].values
        ))

    # Ligne de référence à 50%
    if not temporal_data.empty:
        fig_evolution.add_hline(y=50, line_dash="dash", line_color="gray",
                                annotation_text="Équilibre (50%)", annotation_position="bottom right")

    fig_evolution.update_layout(
        title=f"Évolution des Winrates par {period.title()}",
        xaxis_title="Période",
        yaxis_title="Winrate (%)",
        height=500,
        hovermode="closest"
    )

    st.plotly_chart(fig_evolution, use_container_width=True)

    # --- Graphique 2: Volume de matches ---
    st.subheader("📊 Volume de Matches")

    volume_data = temporal_data.groupby("date")["total_games"].sum().reset_index()

    fig_volume = go.Figure()
    fig_volume.add_trace(go.Bar(
        x=volume_data["date"],
        y=volume_data["total_games"],
        name="Matches",
        marker_color="lightblue",
        hovertemplate="Date: %{x}<br>Matches: %{y}<extra></extra>"
    ))

    fig_volume.update_layout(
        title=f"Volume de Matches par {period.title()}",
        xaxis_title="Période",
        yaxis_title="Nombre de matches",
        height=300
    )

    st.plotly_chart(fig_volume, use_container_width=True)

    # --- Heatmap temporelle ---
    if len(selected_decks) > 1:
        st.subheader("🌡️ Heatmap Temporelle")

        # Pivot pour la heatmap
        heatmap_data = temporal_data.pivot(index="deck", columns="date", values="winrate")

        if not heatmap_data.empty:
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=[d.strftime("%Y-%m-%d") for d in heatmap_data.columns],
                y=heatmap_data.index,
                colorscale="RdYlGn",
                zmid=50,
                colorbar=dict(title="Winrate (%)"),
                hoverongaps=False,
                hovertemplate="Deck: %{y}<br>Date: %{x}<br>Winrate: %{z:.1f}%<extra></extra>"
            ))

            fig_heatmap.update_layout(
                title=f"Heatmap des Winrates par {period.title()}",
                height=max(300, len(selected_decks) * 40),
                xaxis_title="Période",
                yaxis_title="Deck"
            )

            st.plotly_chart(fig_heatmap, use_container_width=True)

    # --- Statistiques récapitulatives ---
    st.subheader("📋 Statistiques de la Période")

    # Calcul des stats globales
    deck_stats = []
    for deck in selected_decks:
        deck_matches = temporal_data[temporal_data["deck"] == deck]
        if deck_matches.empty:
            continue

        total_wins = deck_matches["wins"].sum()
        total_losses = deck_matches["losses"].sum()
        total_games = total_wins + total_losses
        total_matches = deck_matches["matches_count"].sum()
        avg_winrate = (total_wins / total_games * 100) if total_games > 0 else 0

        deck_stats.append({
            "Deck": deck,
            "Sessions": total_matches,
            "Jeux Gagnés": total_wins,
            "Jeux Perdus": total_losses,
            "Total Jeux": total_games,
            "Winrate (%)": f"{avg_winrate:.1f}%",
            "Périodes Actives": len(deck_matches)
        })

    if deck_stats:
        stats_df = pd.DataFrame(deck_stats)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

class _Handler(FileSystemEventHandler):
    def __init__(self, target_path, on_change):
        self.target_path = os.path.abspath(target_path)
        self.on_change = on_change
    def on_modified(self, event): self._maybe(event)
    def on_created(self, event): self._maybe(event)
    def on_moved(self, event): self._maybe(event)
    def _maybe(self, event):
        if os.path.abspath(getattr(event, "src_path", "")) == self.target_path \
                or os.path.abspath(getattr(event, "dest_path", "")) == self.target_path:
            self.on_change()

@st.cache_resource
def start_watcher(path: str, _on_change):
    handler = _Handler(path, _on_change)
    obs = Observer()
    obs.schedule(handler, os.path.dirname(path) or ".", recursive=False)
    obs.start()
    return obs

st.set_page_config(page_title="Match Logger & Winrate Matrix", layout="wide")

DATA_DIR = Path(".")
MATCHES_CSV = DATA_DIR / "matches.csv"
CONFIG_YAML = DATA_DIR / "config.yaml"

# ---------------- Configuration (YAML) ----------------
DEFAULT_CONFIG = {
    "players": ["Alice", "Bob"],
    "decks": {
        "meta": ["MetaDeck1", "MetaDeck2"],
        "contenders": ["Contender1", "Contender2"],
    },
    "options": {
        "allow_draws": True,           # autoriser les égalités X=X
        "show_counts_on_matrix": True, # affiche W‑L dans les cellules
        "heatmap_midpoint": 50.0,      # centre visuel des couleurs (50%)
        "sort_rows_by_global_wr": True # trier les lignes par WR global décroissant
    },
}

#if "css_player_frames_done" not in st.session_state:

st.session_state["css_box_titles"] = True
st.markdown("""
    <style>
      .box-title { display:flex; align-items:center; gap:.5rem; margin:0 0 .5rem 0; }
      .pill { font-size:.80rem; font-weight:700; color:#fff; padding:.12rem .55rem; border-radius:999px; }
      .pill-a { background:#2b6cb0; }  /* bleu A */
      .pill-b { background:#c53030; }  /* rouge B */
      .vstripe { width:6px; align-self:stretch; border-radius:6px; }
      .vstripe-a { background:#2b6cb0; }
      .vstripe-b { background:#c53030; }
      .row { display:flex; gap:.75rem; }
      .col { flex:1; }
    </style>
    """, unsafe_allow_html=True)

# Tick léger pour vérifier le fichier (par ex. toutes les 2 s)
refreshes = st_autorefresh(interval=2000, key="refresh_hist_poll")

if "hist_sig" not in st.session_state:
    try:
        st.session_state.hist_sig = file_signature(MATCHES_CSV)
    except FileNotFoundError:
        st.session_state.hist_sig = None

def load_hist_df(path: str) -> pd.DataFrame:
    # Lecture "safe": si le writer remplace le fichier, open() + read sont atomiques pour le lecteur
    return pd.read_csv(path)

def maybe_reload():
    try:
        sig = file_signature(MATCHES_CSV)
    except FileNotFoundError:
        return None  # rien à faire
    if st.session_state.hist_sig != sig:
        st.session_state.hist_sig = sig
        # Recharge vos structures/matrices ici
        df = load_hist_df(MATCHES_CSV)
        st.session_state.hist_df = df
        #st.experimental_rerun()  # affiche immédiatement la nouvelle version
        st.rerun()

# Appel au début de chaque run
maybe_reload()

def _deep_merge(d, default):
    if not isinstance(d, dict): return default
    out = dict(default)
    for k, v in d.items():
        if isinstance(v, dict) and k in out and isinstance(out[k], dict):
            out[k] = _deep_merge(v, out[k])
        else:
            out[k] = v
    return out

def load_config() -> dict:
    if CONFIG_YAML.exists():
        try:
            with open(CONFIG_YAML, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}
    else:
        cfg = {}
    return _deep_merge(cfg, DEFAULT_CONFIG)

def save_config(cfg: dict):
    with open(CONFIG_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

CONFIG = load_config()

# Dans l'app:
def notify_change():
    # Marqueur pour forcer le rerun prochain tick
    st.session_state["hist_sig"] = None

obs = start_watcher(MATCHES_CSV, notify_change)

# ---------------- Données de matchs (CSV) ----------------
# Colonnes: timestamp, player_a, deck_a, score_a, player_b, deck_b, score_b
def load_matches() -> pd.DataFrame:
    # Ensuite, utilisez st.session_state.hist_df pour construire la matrice / historique
    df = st.session_state.get("hist_df")
    if df is None:
        try:
            df = load_hist_df(MATCHES_CSV)
            st.session_state.hist_df = df
        except FileNotFoundError:
            df = pd.DataFrame(columns=["timestamp","player_a","deck_a","score_a","player_b","deck_b","score_b"])

    # Nettoyage types
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for c in ["score_a","score_b"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    for c in ["player_a","player_b","deck_a","deck_b"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df

def save_matches(df: pd.DataFrame):
    df = df.copy()
    # Contrôle colonnes
    cols = ["timestamp","player_a","deck_a","score_a","player_b","deck_b","score_b"]
    missing = [c for c in cols if c not in df.columns]
    for c in missing:
        df[c] = np.nan
    # Types et valeurs cohérentes
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    for c in ["score_a","score_b"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    #df.to_csv(MATCHES_CSV, index=False)
    csv = df.to_csv(index=False)
    atomic_write_text(MATCHES_CSV, csv)

# ---------------- Utilitaires stats ----------------
def accumulate_points(df: pd.DataFrame) -> pd.DataFrame:
    # On accumule les points W/L par matchup deck_i vs deck_j
    rows = []
    for _, r in df.iterrows():
        a, b = r["deck_a"], r["deck_b"]
        sa, sb = int(r["score_a"]), int(r["score_b"])
        # A vs B
        rows.append({"deck_i": a, "deck_j": b, "W": sa, "L": sb})
        # B vs A
        rows.append({"deck_i": b, "deck_j": a, "W": sb, "L": sa})
    return pd.DataFrame(rows)

def wilson_ci_pct(w: int, n: int, z: float = 1.96):
    # intervalle Wilson en %
    if n == 0: return (np.nan, np.nan)
    p = w / n
    denom = 1 + z**2 / n
    center = (p + z**2/(2*n)) / denom
    margin = (z*np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom
    lo, hi = (center - margin)*100, (center + margin)*100
    return (max(0.0, lo), min(100.0, hi))

def compute_per_deck_global_wr(pt: pd.DataFrame) -> pd.DataFrame:
    # pt = DataFrame des points accumulés (deck_i, deck_j, W, L)
    grp = pt.groupby("deck_i", dropna=False).agg(W=("W","sum"), L=("L","sum"))
    grp["N"] = grp["W"] + grp["L"]
    grp["WR%"] = np.where(grp["N"]>0, grp["W"]/grp["N"]*100, np.nan)
    return grp

def filtered_df(matches: pd.DataFrame, date_from: date | None, date_to: date | None,
                show_meta: bool, show_cont: bool) -> pd.DataFrame:
    df = matches.copy()
    # Filtre période
    if date_from:
        df = df[df["timestamp"] >= pd.Timestamp(date_from)]
    if date_to:
        df = df[df["timestamp"] <= pd.Timestamp(date_to) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)]
    # Filtre groupes de decks
    decks_allowed = []
    if show_meta: decks_allowed += CONFIG["decks"]["meta"]
    if show_cont: decks_allowed += CONFIG["decks"]["contenders"]
    if decks_allowed:
        df = df[df["deck_a"].isin(decks_allowed) & df["deck_b"].isin(decks_allowed)]
    return df

# ---------------- UI ----------------
def ensure_list(x):
    """Garantit que x est une liste (sinon le transforme en liste)."""
    if isinstance(x, list):
        return x
    if x is None:
        return []
    # si c'est une string ou un nombre, on le met dans une liste
    return [x]

st.title("Match Logger & Winrate Matrix")

tab_matrix, tab_config, tab_data, tab_temporal = st.tabs(["🎯 Matrice","⚙️ Configuration", "💾 Données", "📈 Temporel"])
#tab_matrix, tab_config, tab_data = st.tabs(["Matrice", "Configuration", "Données"])

with tab_config:
    st.subheader("Decks")

    # On s'assure que ces valeurs sont toujours des listes
    meta_initial = ensure_list(CONFIG["decks"].get("meta", []))
    contenders_initial = ensure_list(CONFIG["decks"].get("contenders", []))
    players_initial = ensure_list(CONFIG.get("players", []))

    colM, colC = st.columns(2)
    with colM:
        meta = st.data_editor(
            pd.DataFrame({"meta": meta_initial}),
            num_rows="dynamic", use_container_width=True, key="meta_editor"
        )["meta"].dropna().astype(str).tolist()

    with colC:
        contenders = st.data_editor(
            pd.DataFrame({"contenders": contenders_initial}),
            num_rows="dynamic", use_container_width=True, key="cont_editor"
        )["contenders"].dropna().astype(str).tolist()

    st.subheader("Joueurs")
    players = st.data_editor(
        pd.DataFrame({"players": players_initial}),
        num_rows="dynamic", use_container_width=True, key="players_editor"
    )["players"].dropna().astype(str).tolist()

    st.subheader("Options")
    c1, c2, c3 = st.columns(3)
    with c1:
        allow_draws = st.checkbox("Autoriser les égalités (X = X)", value=CONFIG["options"]["allow_draws"])
    with c2:
        show_counts = st.checkbox("Afficher W‑L dans les cellules", value=CONFIG["options"]["show_counts_on_matrix"])
    with c3:
        midpoint = st.number_input("Centre heatmap (%)", value=float(CONFIG["options"]["heatmap_midpoint"]), min_value=0.0, max_value=100.0, step=1.0)

    if st.button("Sauvegarder la configuration", type="primary"):
        CONFIG["decks"]["meta"] = meta
        CONFIG["decks"]["contenders"] = contenders
        CONFIG["players"] = players
        CONFIG["options"]["allow_draws"] = bool(allow_draws)
        CONFIG["options"]["show_counts_on_matrix"] = bool(show_counts)
        CONFIG["options"]["heatmap_midpoint"] = float(midpoint)
        save_config(CONFIG)
        st.success("Configuration sauvegardée.")

with tab_matrix:
    st.subheader("Filtres")
    c1, c2, c3, c4 = st.columns([1,1,1,2])
    with c1:
        show_meta = st.checkbox("Meta", value=True)
    with c2:
        show_cont = st.checkbox("Contenders", value=False)
    with c3:
        date_from = st.date_input("Du", value=None)
    with c4:
        date_to = st.date_input("Au", value=None)

    # Chargement
    matches = load_matches()
    df_f = filtered_df(matches, date_from if date_from else None, date_to if date_to else None, show_meta, show_cont)

    # Accumulation points
    pt = accumulate_points(df_f) if not df_f.empty else pd.DataFrame(columns=["deck_i","deck_j","W","L"])

    # Ensemble de decks visibles
    decks_all = []
    if show_meta: decks_all += CONFIG["decks"]["meta"]
    if show_cont: decks_all += CONFIG["decks"]["contenders"]
    decks_all = list(dict.fromkeys(decks_all))  # unique et ordre

    # Stats globales par deck (WR global)
    per_deck = compute_per_deck_global_wr(pt).reindex(decks_all)
    hide_empty = st.checkbox("Masquer les decks sans matchs", value=True)
    if hide_empty:
        per_deck = per_deck[per_deck["N"] > 0]

    # Tri décroissant par WR%, puis par N (plus grand d'abord). NaN en bas.
    per_deck = per_deck.sort_values(
        by=["WR%", "N"],
        ascending=[True, False],
        na_position="first"
    )

    row_order = per_deck.index.tolist() if len(per_deck) else decks_all

    # Colonnes matchup (adversaires) dans l’ordre d’affichage: on conserve l’ordre d’origine pour les colonnes
    col_order = decks_all

    # Matrice W-L et %
    # values_df contient par case: "W-L (xx.x%)"
    # z_df contient le pourcentage (float) pour la heatmap
    idx_cols = pd.MultiIndex.from_product([row_order, col_order], names=["deck_i","deck_j"])
    tmp = pt.groupby(["deck_i","deck_j"]).agg(W=("W","sum"), L=("L","sum")).reindex(idx_cols).reset_index()
    tmp["N"] = (tmp["W"].fillna(0) + tmp["L"].fillna(0)).astype(int)
    tmp["WR%"] = np.where(tmp["N"]>0, tmp["W"]/tmp["N"]*100, np.nan)

    values_df = tmp.pivot(index="deck_i", columns="deck_j", values="WR%").reindex(index=row_order, columns=col_order)
    wl_pairs = tmp.assign(wl=lambda d: (d["W"].fillna(0).astype(int)).astype(str) + "-" + (d["L"].fillna(0).astype(int)).astype(str))
    text_df = wl_pairs.pivot(index="deck_i", columns="deck_j", values="wl").reindex(index=row_order, columns=col_order)

    # Colonne 0 = WR global
    global_wr_col = per_deck["WR%"].reindex(row_order)
    global_wl_col = (per_deck["W"].fillna(0).astype(int).astype(str) + "-" + per_deck["L"].fillna(0).astype(int).astype(str)).reindex(row_order)

    # Construit matrices finales avec colonne WR% (global) en première colonne
    z_df = pd.concat([global_wr_col.rename("WR% (global)"), values_df], axis=1)
    text_annot = pd.concat(
        [ (global_wr_col.map(lambda x: f"{x:.1f}%" if pd.notna(x) else "—") + "<br>(" + global_wl_col + ")").rename("WR% (global)"),
          (values_df.map(lambda x: f"{x:.1f}%" if pd.notna(x) else "—") + text_df.map(lambda x: f"<br>({x})" if x!= "0-0" else "")) ],
        axis=1,
    )

    # Heatmap (colonne 0 incluse)
    z = z_df.values.astype(float)
    with np.errstate(invalid="ignore"):
        z_masked = np.where(np.isfinite(z), z, np.nan)

    # Hover détaillé
    hover = []
    for i in z_df.index:
        row = []
        # Colonne globale
        w_global = int(per_deck.loc[i, "W"]) if i in per_deck.index and pd.notna(per_deck.loc[i,"W"]) else 0
        l_global = int(per_deck.loc[i, "L"]) if i in per_deck.index and pd.notna(per_deck.loc[i,"L"]) else 0
        n_global = max(w_global + l_global, 0)
        wr_global = per_deck.loc[i, "WR%"] if i in per_deck.index else np.nan
        if np.isnan(wr_global):
            row.append(f"{i} — global<br>—")
        else:
            lo, hi = wilson_ci_pct(w_global, n_global)
            row.append(f"{i} — global<br>{w_global}-{l_global} ({wr_global:.1f}%)<br>N={n_global} games<br>CI95%: {lo:.1f}–{hi:.1f}%")
        # Colonnes adversaires
        for j in values_df.columns:
            pct = values_df.loc[i, j]
            if np.isnan(pct):
                row.append(f"{i} vs {j}<br>—")
            else:
                w_l = text_df.loc[i, j]
                w, l = map(int, w_l.split("-"))
                n = max(w + l, 1)
                lo, hi = wilson_ci_pct(w, n)
                row.append(f"{i} vs {j}<br>{w_l} ({pct:.1f}%)<br>N={n} games<br>CI95%: {lo:.1f}–{hi:.1f}%")
        hover.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=z_masked,
        x=list(z_df.columns),
        y=list(z_df.index),
        colorscale="RdYlGn",
        zmin=0, zmax=80,
        zmid=float(CONFIG["options"]["heatmap_midpoint"]),
        hoverinfo="text",
        hovertext=hover,
        text=text_annot.values,
        texttemplate="%{text}",
        textfont={"size": 15},
        xgap=1, ygap=1,
        showscale=True,
        colorbar=dict(title="WR%", ticksuffix="%", thickness=14),
    ))

    # Axes & style
    fig.update_layout(
        xaxis=dict(
            side="top",
            tickangle=-45,                 # votre préférence
            ticklabelposition="outside top",
            tickfont=dict(size=15, color="#111", family="Segoe UI, sans-serif", weight="bold"),
            fixedrange=True,
        ),
        yaxis=dict(
            tickfont=dict(size=15, color="#111", family="Segoe UI, sans-serif", weight="bold"),
            fixedrange=True,
        ),
        yaxis_title="Deck (ligne = évalué)",
        xaxis_title="",
        margin=dict(l=10, r=10, t=70, b=10),
        height=680,
    )
    #Force carré
    #fig.update_yaxes(scaleanchor="x", scaleratio=1)
    # Contour des cellules pour lisibilité
    fig.update_traces(showscale=True)
    st.plotly_chart(fig, use_container_width=True)

with tab_data:
    st.subheader("Ajouter un match")
    matches = load_matches()

    # Toujours afficher du plus récent au plus ancien
    matches = matches.sort_values("timestamp", ascending=False, kind="mergesort")

    # Deux colonnes cartes + une colonne centrale "VS"
    colA, colVS, colB = st.columns([1, 0.12, 1])

    with colA:
        with st.container(border=True):
            st.markdown('<div class="row"><div class="vstripe vstripe-a"></div><div class="col">', unsafe_allow_html=True)
            st.markdown('<div class="box-title"><span class="pill pill-a">Joueur A</span></div>', unsafe_allow_html=True)

            player_a = st.selectbox(
                "player A",
                options=CONFIG["players"],
                index=0 if CONFIG["players"] else None,
                key="add_pa",
                label_visibility="hidden",
            )
            deck_a = st.selectbox(
                "Deck A",
                options=CONFIG["decks"]["meta"] + CONFIG["decks"]["contenders"],
                key="add_da",
            )
            score_a = st.number_input("Score A", min_value=0, value=2, step=1)
            st.markdown('</div></div>', unsafe_allow_html=True)  # ferme les deux div du header décoratif

    with colVS:
        st.markdown('<div style="display:flex;align-items:center;justify-content:center;height:100%;"><h3>VS</h3></div>', unsafe_allow_html=True)

    with colB:
        with st.container(border=True):
            st.markdown('<div class="row"><div class="vstripe vstripe-b"></div><div class="col">', unsafe_allow_html=True)
            st.markdown('<div class="box-title"><span class="pill pill-b">Joueur B</span></div>', unsafe_allow_html=True)

            player_b = st.selectbox(
                "player B",
                options=CONFIG["players"],
                index=1 if len(CONFIG["players"]) > 1 else 0,
                key="add_pb",
                label_visibility="hidden",
            )
            deck_b = st.selectbox(
                "Deck B",
                options=CONFIG["decks"]["meta"] + CONFIG["decks"]["contenders"],
                key="add_db",
            )
            score_b = st.number_input("Score B", min_value=0, value=1, step=1)
            st.markdown('</div></div>', unsafe_allow_html=True)

    if st.button("Enregistrer le match", type="primary"):
        new_row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "player_a": player_a, "deck_a": deck_a, "score_a": int(score_a),
            "player_b": player_b, "deck_b": deck_b, "score_b": int(score_b),
        }
        matches = pd.concat([matches, pd.DataFrame([new_row])], ignore_index=True)
        save_matches(matches)
        st.success("Match ajouté.")
        # Recharger trié
        matches = load_matches().sort_values("timestamp", ascending=False, kind="mergesort")

    st.markdown("Historique (éditable — sauvegarde automatique)")
    # Table éditable et sauvegarde auto
    def df_hash(d: pd.DataFrame) -> str:
        # hash simple pour détecter les changements
        return hashlib.md5(pd.util.hash_pandas_object(d.fillna(""), index=False).values).hexdigest()

    # Vue triée pour l'édition (timestamp string-friendly)
    matches_display = matches.copy()
    matches_display["timestamp"] = matches_display["timestamp"].apply(
        lambda x: x.strftime("%Y-%m-%d %H:%M:%S") if not pd.isna(x) and not isinstance(x, str) else x
    )
    edited = st.data_editor(
        matches_display,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "timestamp": st.column_config.TextColumn("timestamp (YYYY-MM-DD HH:MM:SS)", width="medium"),
            "player_a": st.column_config.SelectboxColumn("player_a", options=CONFIG["players"], width="small"),
            "deck_a":   st.column_config.SelectboxColumn("deck_a", options=CONFIG["decks"]["meta"] + CONFIG["decks"]["contenders"], width="small"),
            "score_a":  st.column_config.NumberColumn("score_a", min_value=0, step=1, width="small"),
            "player_b": st.column_config.SelectboxColumn("player_b", options=CONFIG["players"], width="small"),
            "deck_b":   st.column_config.SelectboxColumn("deck_b", options=CONFIG["decks"]["meta"] + CONFIG["decks"]["contenders"], width="small"),
            "score_b":  st.column_config.NumberColumn("score_b", min_value=0, step=1, width="small"),
        },
        key="editable_matches",
    )

    # Normalisation types avant comparaison
    edited_norm = edited.copy()
    edited_norm["timestamp"] = pd.to_datetime(edited_norm["timestamp"], errors="coerce")
    for c in ["score_a","score_b"]:
        edited_norm[c] = pd.to_numeric(edited_norm[c], errors="coerce").fillna(0).astype(int)
    for c in ["player_a","player_b","deck_a","deck_b"]:
        edited_norm[c] = edited_norm[c].astype(str)

    # Sauvegarde automatique si différence
    if not edited_norm.equals(matches.sort_values("timestamp", ascending=False)):
        save_matches(edited_norm)
        st.toast("Modifications sauvegardées.", icon="✅")
        # Recharger et réappliquer le tri pour la session courante
        matches = load_matches().sort_values("timestamp", ascending=False, kind="mergesort")

with tab_temporal:
    add_temporal_analysis_tab()

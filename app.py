
# streamlit_app.py
# Dashboard für Schulausfall-Daten (Region/Landkreis & Schularten)
# Start: streamlit run streamlit_app.py

import io
from pathlib import Path
import re
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

st.set_page_config(page_title="Schulausfall-Reporting", page_icon="📚", layout="wide")

# ---------- Load & Normalize ----------
@st.cache_data(show_spinner=False)
def load_file(uploaded_file):
    if uploaded_file is None:
        return None
    suf = Path(uploaded_file.name).suffix.lower()
    if suf in [".xlsx",".xls"]:
        return pd.read_excel(uploaded_file)
    if suf in [".csv",".txt"]:
        content = uploaded_file.getvalue()
        try:
            return pd.read_csv(io.BytesIO(content), sep=";", decimal=",")
        except Exception:
            return pd.read_csv(io.BytesIO(content), sep=",", decimal=".")
    raise ValueError("Nur CSV/XLSX werden unterstützt.")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for c in ["Region","Schulart","Name","Landkreis"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # parse numeric-like strings robustly
    num_cols = ["P_in_h","P_in_%","A_in_h","A_in_%","V_in_h","V_in_%","P+A_in_%","P+A+V_in_%"]
    def parse_num(val):
        if pd.isna(val): return np.nan
        if isinstance(val,(int,float,np.integer,np.floating)): return float(val)
        s = str(val).strip().replace("%","").replace("\u00A0","").replace(" ","")
        if s == "": return np.nan
        if re.fullmatch(r"[+-]?\d+", s): return float(s)
        has_comma, has_dot = ("," in s), ("." in s)
        if has_comma and has_dot:
            if s.rfind(",") > s.rfind("."):
                s = s.replace(".","").replace(",",".")
            else:
                s = s.replace(",","")
        elif has_comma:
            s = s.replace(",",".")
        try:
            return float(s)
        except Exception:
            return np.nan
    for c in num_cols:
        if c in df.columns:
            if df[c].dtype == "O":
                df[c] = df[c].map(parse_num)
            elif not pd.api.types.is_numeric_dtype(df[c]):
                df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def kpi_card(label, value, help_text=None):
    s = "–" if (value is None or pd.isna(value)) else f"{value:,.1f}%".replace(",", "X").replace(".", ",").replace("X", ".")
    st.metric(label, s, help=help_text)

def weighted_mean(values: pd.Series, weights: pd.Series):
    mask = (~values.isna()) & (~weights.isna()) & (weights>0)
    if mask.sum()==0: return np.nan
    v,w = values[mask], weights[mask]
    return float((v*w).sum()/w.sum())

def group_weighted_mean(df: pd.DataFrame, group_col: str, val_col: str, weight_col: str, weighted: bool):
    if group_col not in df.columns or val_col not in df.columns:
        return pd.DataFrame(columns=[group_col, val_col])
    if not weighted:
        out = df.groupby(group_col, as_index=False)[val_col].mean()
    else:
        def _wm(g): return weighted_mean(g[val_col], g[weight_col])
        out = df.groupby(group_col).apply(_wm).reset_index(name=val_col)
    return out

def df_download_button(df: pd.DataFrame, filename: str, label: str):
    csv = df.to_csv(index=False, sep=";", decimal=",").encode("utf-8")
    st.download_button(label, data=csv, file_name=filename, mime="text/csv")

# ---------- Denominator (Sollstunden) ----------
def estimate_denominator(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["P_in_h","A_in_h","V_in_h","P_in_%","A_in_%","V_in_%","P+A+V_in_%"]:
        if c not in df.columns: df[c] = np.nan
    total_h = df[["P_in_h","A_in_h","V_in_h"]].sum(axis=1, min_count=1)
    def candidates(row):
        cands = []
        for h_col,p_col in [("P_in_h","P_in_%"),("A_in_h","A_in_%"),("V_in_h","V_in_%")]:
            h,p = row[h_col], row[p_col]
            if pd.notna(h) and pd.notna(p) and p>0:
                cands.append(h/(p/100.0))
        pavp = row.get("P+A+V_in_%", np.nan)
        if pd.notna(total_h.loc[row.name]) and pd.notna(pavp) and pavp>0:
            cands.append(total_h.loc[row.name]/(pavp/100.0))
        return cands
    all_cands = df.apply(candidates, axis=1)
    est = all_cands.map(lambda xs: np.nan if len(xs)==0 else float(np.median(xs)))
    rel_range = []
    for xs in all_cands:
        if len(xs)>=2:
            mn,mx = float(np.min(xs)), float(np.max(xs))
            med = float(np.median(xs))
            rel_range.append((mx-mn)/med if med>0 else np.nan)
        else:
            rel_range.append(np.nan)
    df["Sollstunden_est"] = est
    df["Soll_kandidaten_anzahl"] = all_cands.map(len)
    df["Soll_rel_range"] = rel_range
    df["Soll_flag_konflikt"] = (df["Soll_rel_range"]>0.15)
    return df

def add_mix_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Ausfallstunden_total"] = df[["P_in_h","A_in_h","V_in_h"]].sum(axis=1, min_count=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["P_share_out"] = np.where(df["Ausfallstunden_total"]>0, df["P_in_h"]/df["Ausfallstunden_total"], np.nan)
        df["A_share_out"] = np.where(df["Ausfallstunden_total"]>0, df["A_in_h"]/df["Ausfallstunden_total"], np.nan)
        df["V_share_out"] = np.where(df["Ausfallstunden_total"]>0, df["V_in_h"]/df["Ausfallstunden_total"], np.nan)
        df["A_over_P_ratio"] = np.where(df["P_in_h"]>0, df["A_in_h"]/df["P_in_h"], np.nan)
    return df

# ---------- Story-Hooks helpers ----------
def de_pct(x, digits=1):
    if x is None or pd.isna(x): return "–"
    return f"{x:.{digits}f} %".replace(".", ",")

def de_int(x):
    try: x = int(round(float(x)))
    except Exception: return "–"
    return f"{x:,}".replace(",", "X").replace(".", ",").replace("X", ".")

def build_story_hooks_sections(fdf: pd.DataFrame, weighted_toggle: bool, min_soll: int, min_hours_hook: int, pareto_threshold: int, geo_col: str="Region"):
    if fdf is None or len(fdf)==0:
        return {"Überblick":["(Keine Daten im aktuellen Filter.)"]}
    weight_col = "Sollstunden_est" if weighted_toggle else None
    def agg_mean(series: pd.Series, weights: pd.Series|None):
        if weights is None: return float(np.nanmean(series)) if len(series) else np.nan
        m = (~series.isna()) & (~weights.isna()) & (weights>0)
        if m.sum()==0: return np.nan
        return float((series[m]*weights[m]).sum()/weights[m].sum())
    sections = {}
    # Überblick
    overall = agg_mean(fdf["P+A+V_in_%"], fdf[weight_col] if weight_col else None)
    sections["Überblick"] = [f"Gesamtausfall (P+A+V): **{de_pct(overall)}** ({'gewichtet' if weighted_toggle else 'ungewichtet'})."]
    # Gebiet hoch/tief
    if geo_col in fdf.columns and fdf[geo_col].nunique()>1:
        reg = fdf.groupby(geo_col).apply(lambda g: agg_mean(g["P+A+V_in_%"], g[weight_col] if weight_col else None)).dropna().sort_values(ascending=False)
        if len(reg)>=1:
            high_g, high_v = reg.index[0], float(reg.iloc[0])
            low_g, low_v   = reg.index[-1], float(reg.iloc[-1])
            if len(reg)>=2:
                sections["Gebiet (Region/Landkreis)"] = [f"Höchste Quote: **{high_g}** ({de_pct(high_v)}); niedrigste: **{low_g}** ({de_pct(low_v)}). Unterschied: **{de_pct(high_v-low_v)}**-Punkte."]
            else:
                sections["Gebiet (Region/Landkreis)"] = [f"{high_g}: **{de_pct(high_v)}**."]
    # Rankings (fair)
    ranked = fdf.copy()
    if "Sollstunden_est" in ranked.columns:
        ranked = ranked[(ranked["Sollstunden_est"].fillna(0) >= min_soll)]
    ranked = ranked.dropna(subset=["P+A+V_in_%"])
    if len(ranked)>=1:
        top = ranked.sort_values("P+A+V_in_%", ascending=False).head(3)
        flop = ranked.sort_values("P+A+V_in_%", ascending=True).head(3)
        def fmt_row(r):
            val = de_pct(float(r["P+A+V_in_%"]))
            size = f" · Soll ~ {de_int(r.get('Sollstunden_est', np.nan))}h" if pd.notna(r.get("Sollstunden_est", np.nan)) else ""
            where = ", ".join([str(r[c]) for c in ["Region","Schulart"] if c in r.index and pd.notna(r[c])])
            where = f" ({where})" if where else ""
            return f"**{r['Name']}**{where}: {val}{size}"
        sections["Top (höchste Quote)"] = [fmt_row(r) for _,r in top.iterrows()]
        sections["Positivliste (niedrigste Quote)"] = [fmt_row(r) for _,r in flop.iterrows()]
    # Ursachenmix nach Gebiet (mit 2.-bestem & Pareto)
    if all(c in fdf.columns for c in ["P_in_h","A_in_h","V_in_h"]) and geo_col in fdf.columns:
        mix_df = fdf.groupby(geo_col)[["P_in_h","A_in_h","V_in_h"]].sum(min_count=1).reset_index()
        mix_df["total"]   = mix_df[["P_in_h","A_in_h","V_in_h"]].sum(axis=1, min_count=1)
        mix_df["V_share"] = np.where(mix_df["total"]>0, mix_df["V_in_h"]/mix_df["total"], np.nan)
        mix_df["A_over_P"]= np.where(mix_df["P_in_h"]>0, mix_df["A_in_h"]/mix_df["P_in_h"], np.nan)
        cand = mix_df[mix_df["total"]>=float(min_hours_hook)].copy()
        if len(cand)==0: cand = mix_df.copy()
        lines = []
        cand_v = cand.dropna(subset=["V_share"]).sort_values("V_share", ascending=False)
        if len(cand_v)>=1:
            v1_g, v1 = cand_v.iloc[0][geo_col], float(cand_v.iloc[0]["V_share"])
            line = f"Höchster Anteil **fachfremder Vertretung** an Ausfallstunden: **{v1_g}** ({de_pct(v1*100)})."
            if len(cand_v)>=2:
                v2_g, v2 = cand_v.iloc[1][geo_col], float(cand_v.iloc[1]["V_share"])
                line += f" Zweithöchste: {v2_g} ({de_pct(v2*100)}), Abstand **{de_pct((v1-v2)*100)}**-Punkte."
            lines.append(line)
            # Pareto-Notiz
            reg_rows = fdf[fdf[geo_col] == v1_g].copy()
            by_school = reg_rows.groupby("Name")[["P_in_h","A_in_h","V_in_h"]].sum(min_count=1)
            by_school["sum"] = by_school.sum(axis=1, min_count=1)
            by_school = by_school.sort_values("sum", ascending=False)
            total = float(by_school["sum"].sum()) if by_school["sum"].notna().any() else 0.0
            if total>0 and len(by_school)>0:
                target = float(pareto_threshold)/100.0*total
                k = int((by_school["sum"].cumsum().to_numpy() >= target).argmax()) + 1
                n = int(len(by_school))
                share_sch = (k/n)*100.0
                lines.append(f"{pareto_threshold}% der Ausfallstunden in **{v1_g}** entfallen auf **{k}** von **{n}** Schulen (~{de_pct(share_sch)} der Schulen).")
        cand_aop = cand.dropna(subset=["A_over_P"]).sort_values("A_over_P", ascending=False)
        if len(cand_aop)>=1:
            a1_g, a1 = cand_aop.iloc[0][geo_col], float(cand_aop.iloc[0]["A_over_P"])
            line = f"Höchstes Verhältnis **A/P**: **{a1_g}** ({a1:.2f})."
            if len(cand_aop)>=2:
                a2_g, a2 = cand_aop.iloc[1][geo_col], float(cand_aop.iloc[1]["A_over_P"])
                line += f" Zweithöchste: {a2_g} ({a2:.2f}), Abstand **{(a1-a2):.2f}**."
            lines.append(line)
        if lines:
            sections[f"Ursachenmix ({'Landkreise' if geo_col=='Landkreis' else 'Regionen'})"] = lines
    # Schularten
    if "Schulart" in fdf.columns and fdf["Schulart"].nunique()>=1 and "P+A+V_in_%" in fdf.columns:
        def _agg_art(g): return weighted_mean(g["P+A+V_in_%"], g["Sollstunden_est"]) if ("Sollstunden_est" in g.columns and weighted_toggle) else float(np.nanmean(g["P+A+V_in_%"]))
        art = fdf.groupby("Schulart").apply(_agg_art).dropna().sort_values(ascending=False)
        if len(art)>=1:
            high_art, high_val = art.index[0], float(art.iloc[0])
            low_art, low_val   = art.index[-1], float(art.iloc[-1])
            overall2 = weighted_mean(fdf["P+A+V_in_%"], fdf["Sollstunden_est"]) if weighted_toggle else float(np.nanmean(fdf["P+A+V_in_%"]))
            lines2 = [f"Höchste Quote: **{high_art}** ({de_pct(high_val)})"]
            if len(art)>=2:
                lines2.append(f"Niedrigste Quote: **{low_art}** ({de_pct(low_val)}), Abstand **{de_pct(high_val-low_val)}**-Punkte.")
            if overall2==overall2:
                lines2.append(f"Gegenüber Gesamt (**{de_pct(overall2)}**) liegt **{high_art}** bei **+{de_pct(high_val-overall2)}**-Punkten.")
            if "Förderschule" in art.index:
                fs = float(art.loc["Förderschule"])
                diff = fs - overall2 if overall2==overall2 else np.nan
                lines2.append(f"**Förderschulen:** {de_pct(fs)}{(' (+' + de_pct(diff) + ' zum Gesamt)') if diff==diff else ''}.")
            sections["Schularten"] = lines2
    # Ausreißer
    if "Schulart" in fdf.columns and "P+A+V_in_%" in fdf.columns:
        def zflag(g):
            mu = g["P+A+V_in_%"].mean(); sd = g["P+A+V_in_%"].std(ddof=0)
            g = g.copy(); g["z"] = (g["P+A+V_in_%"]-mu)/sd if sd and sd>0 else np.nan; return g
        zdf = fdf.groupby("Schulart", group_keys=False).apply(zflag)
        outs = zdf[zdf["z"]>2].sort_values("z", ascending=False).head(5)
        if len(outs)>0:
            sections["Ausreißer (z>2)"] = [f"**{r['Name']}** ({r.get('Region','')}, {r['Schulart']}): {de_pct(r['P+A+V_in_%'])}, z={r['z']:.2f}" for _,r in outs.iterrows()]
    # Qualität
    qual = []
    if all(c in fdf.columns for c in ["P_in_%","A_in_%","V_in_%","P+A+V_in_%"]):
        tmp = fdf.copy()
        tmp["abweichung"] = np.abs(tmp["P_in_%"].fillna(0)+tmp["A_in_%"].fillna(0)+tmp["V_in_%"].fillna(0)-tmp["P+A+V_in_%"])
        qual.append(f"{(tmp['abweichung']>0.2).sum()} Summen-Abweichungen (>0,2 PP)")
    if "P+A+V_in_%" in fdf.columns:
        qual.append(f"{(fdf['P+A+V_in_%']>35).sum()} sehr hohe Quoten (>35 %)")
    if all(c in fdf.columns for c in ["P_in_h","A_in_h","V_in_h"]):
        zeros = ((fdf['P_in_h'].fillna(0)==0)&(fdf['A_in_h'].fillna(0)==0)&(fdf['V_in_h'].fillna(0)==0)).sum()
        qual.append(f"{zeros} 0-Stunden-Zeilen")
    if "Soll_flag_konflikt" in fdf.columns:
        qual.append(f"{(fdf['Soll_flag_konflikt']==True).sum()} Sollstunden-Konflikte")
    if qual: sections["Qualitätslage"] = [" · ".join(qual)]
    return sections

# ---------- UI ----------
st.title("📚 Schulausfall – Dashboard & Recherche")
st.caption("Region/Landkreis & Schularten – ohne Zeitreihen.")

st.sidebar.title("📥 Daten laden")
uploaded = st.sidebar.file_uploader("CSV oder XLSX hochladen", type=["csv","xlsx","xls","txt"])
df = load_file(uploaded)
if df is None:
    st.info("Lade eine Datei hoch, oder nutze die Demo-Daten unten.")
    if st.sidebar.button("Demo-Daten laden", key="demo_btn"):
        df = pd.DataFrame({
            "Region":["Bautzen","Bautzen","Bautzen","Chemnitz","Zwickau"],
            "Landkreis":["Bautzen","Görlitz","Bautzen","Chemnitz","Zwickau"],
            "Schulart":["Berufsschule","Berufsschule","Berufsschule","Gymnasium","Grundschule"],
            "Name":["BSZ Radeberg","BSZ Löbau","BSZ Weißwasser","Chemnitzer Gymnasium A","Grundschule Zwickau Mitte"],
            "P_in_h":[1088,4694,0,510,210],
            "P_in_%":[5.7,7.8,0.0,2.3,1.2],
            "A_in_h":[2445,2445,3617,670,320],
            "A_in_%":[8.0,4.1,11.3,3.0,1.9],
            "V_in_h":[0,1096,2177,420,140],
            "V_in_%":[0.0,1.8,6.8,1.8,0.9],
            "P+A_in_%":[13.7,11.9,11.3,5.3,3.1],
            "P+A+V_in_%":[13.7,13.7,18.1,7.1,4.0],
        })
else:
    df = normalize_columns(df)

if df is not None and len(df)>0:
    df = estimate_denominator(df)
    df = add_mix_metrics(df)

    with st.sidebar:
        st.subheader("🔎 Filter")
        geo_col = st.radio("Gebietsebene", ["Region", "Landkreis"] if "Landkreis" in df.columns else ["Region"], index=0, key="geo_col")
        regions = sorted(df["Region"].dropna().unique()) if "Region" in df.columns else []
        schularten = sorted(df["Schulart"].dropna().unique()) if "Schulart" in df.columns else []
        region_sel = st.multiselect("Region", regions, default=regions, key="region_sel")
        schulart_sel = st.multiselect("Schulart", schularten, default=schularten, key="schulart_sel")
        kreise = sorted(df["Landkreis"].dropna().unique()) if "Landkreis" in df.columns else []
        kreis_sel = st.multiselect("Landkreis", kreise, default=kreise, key="kreis_sel")

        default_highlight = ["Förderschule"] if "Förderschule" in schularten else []
        highlight_sel = st.multiselect("Hervorheben (Schularten)", schularten, default=default_highlight, key="highlight_sel")

        school_search = st.text_input("Schulsuche (Name, enthält)", key="school_search")
        top_n = st.slider("Top/Flop N für Rankings", min_value=5, max_value=50, value=15, step=5, key="top_n")

        st.divider()
        st.subheader("⚖️ Auswertung")
        weighted_toggle = st.checkbox("Gewichtete Auswertung (empfohlen)", value=True, key="weighted_toggle")
        min_soll = st.number_input("Mindest-Sollstunden für Rankings", min_value=0, value=500, step=100, key="min_soll")
        hide_soll_conflict = st.checkbox(
            "Schulen mit Sollstunden-Konflikt ausblenden",
            value=True, key="hide_soll_conflict"
        )

        st.divider()
        st.subheader("🔢 Anzeige")
        value_labels = st.checkbox("Werte auf Grafiken anzeigen", value=True, key="value_labels")
        label_threshold = st.slider("Schwelle für Stacked-Bar-Labels (%)", min_value=0, max_value=50, value=10, step=1, key="label_threshold")

        st.subheader("🧭 Story-Hooks – Optionen")
        min_hours_hook = st.number_input("Mindest-Ausfallstunden pro Gebiet für Hooks", min_value=0, value=500, step=100, key="min_hours_hook")
        pareto_threshold = st.slider("Pareto-Schwelle für Hooks (%)", min_value=10, max_value=90, value=50, step=5, key="pareto_threshold")

        st.divider()
        show_table = st.checkbox("Rohdaten-Tabelle anzeigen", value=False, key="show_table")

    # Apply filters
    fdf = df.copy()
    if "Region" in fdf.columns and region_sel:
        fdf = fdf[fdf["Region"].isin(region_sel)]
    if "Schulart" in fdf.columns and schulart_sel:
        fdf = fdf[fdf["Schulart"].isin(schulart_sel)]
    if "Landkreis" in fdf.columns and kreis_sel:
        fdf = fdf[fdf["Landkreis"].isin(kreis_sel)]
    if school_search:
        fdf = fdf[fdf["Name"].str.contains(school_search, case=False, na=False)]
    if "Soll_flag_konflikt" in fdf.columns and hide_soll_conflict:
        before = len(fdf)
        fdf = fdf[~fdf["Soll_flag_konflikt"].fillna(False)]
        st.caption(f"{before - len(fdf)} Schulen wegen Sollstunden-Konflikt ausgeblendet.")


    with st.expander("ℹ️ Interpretationshilfe & Methodik", expanded=True):
        st.markdown("""
        - **Gebietsebene** steuert, ob Vergleiche/Mix nach **Region** oder **Landkreis** laufen.
        - **Gewichtet** aggregiert nach Sollstunden (Systemlast), **ungewichtet** zeigt die typische Schule.
        - **Ursachenmix (Gebiet/Schulart)** basiert auf **Stundenanteilen**; Stacked-% der Schularten basiert auf **%-Punkten**.
        """)

    # KPIs
    st.subheader("Kennzahlen (Durchschnitt über gefilterte Schulen)")
    weight_col = "Sollstunden_est" if weighted_toggle else None
    def agg_metric(col):
        if col not in fdf.columns: return np.nan
        return weighted_mean(fdf[col], fdf[weight_col]) if weight_col else float(np.nanmean(fdf[col])) if len(fdf) else np.nan
    cols = st.columns(5)
    with cols[0]: kpi_card("Planmäßig (P)", agg_metric("P_in_%"))
    with cols[1]: kpi_card("Außerplanmäßig (A)", agg_metric("A_in_%"))
    with cols[2]: kpi_card("Vertretung fachfremd (V)", agg_metric("V_in_%"))
    with cols[3]: kpi_card("P + A", agg_metric("P+A_in_%"))
    with cols[4]: kpi_card("P + A + V", agg_metric("P+A+V_in_%"))
    st.caption("Aggregation: **{}**".format("gewichtet nach Sollstunden" if weighted_toggle else "ungewichtet (jede Schule zählt gleich)"))
    st.markdown("---")

    # Gebiet-Vergleich (Region/Landkreis)
    if "P+A+V_in_%" in fdf.columns:
        st.subheader(f"Vergleich nach {'Landkreis' if geo_col=='Landkreis' else 'Region'} (∅ P+A+V in %)")
        group_col = geo_col if geo_col in fdf.columns else "Region"
        reg_df = group_weighted_mean(fdf, group_col, "P+A+V_in_%", "Sollstunden_est", weighted_toggle).rename(columns={group_col:"Gebiet"}).sort_values("P+A+V_in_%", ascending=False)
        chart = alt.Chart(reg_df).mark_bar().encode(
            x=alt.X("Gebiet:N", sort="-y"),
            y=alt.Y("P+A+V_in_%:Q", title="Durchschnitt in %"),
            tooltip=[alt.Tooltip("Gebiet:N", title=("Landkreis" if geo_col=="Landkreis" else "Region")), alt.Tooltip("P+A+V_in_%:Q", format=".1f")]
        ).properties(height=300)
        if value_labels:
            text = alt.Chart(reg_df).mark_text(dy=-6).encode(x=alt.X("Gebiet:N", sort="-y"), y=alt.Y("P+A+V_in_%:Q"), text=alt.Text("P+A+V_in_%:Q", format=".1f"))
            chart = chart + text
        st.altair_chart(chart, use_container_width=True)

    # Schulart-Vergleich
    if "P+A+V_in_%" in fdf.columns:
        st.subheader("Vergleich nach Schulart (∅ P+A+V in %)")
        art_df = group_weighted_mean(fdf, "Schulart", "P+A+V_in_%", "Sollstunden_est", weighted_toggle).sort_values("P+A+V_in_%", ascending=False)
        chart_art = alt.Chart(art_df).mark_bar().encode(
            x=alt.X("Schulart:N", sort="-y"),
            y=alt.Y("P+A+V_in_%:Q", title="Durchschnitt in %"),
            tooltip=[alt.Tooltip("Schulart:N"), alt.Tooltip("P+A+V_in_%:Q", format=".1f")]
        ).properties(height=300)
        if value_labels:
            text_art = alt.Chart(art_df).mark_text(dy=-6).encode(x=alt.X("Schulart:N", sort="-y"), y=alt.Y("P+A+V_in_%:Q"), text=alt.Text("P+A+V_in_%:Q", format=".1f"))
            chart_art = chart_art + text_art
        st.altair_chart(chart_art, use_container_width=True)

    # Ursachenmix (Gebiet + Schulart)
    st.subheader("Ursachenmix der Ausfallstunden (P/A/V) – Stacked Bars")
    def mix_by(group_col):
        g = fdf.groupby(group_col, as_index=False)[["P_in_h","A_in_h","V_in_h"]].sum()
        g["total"] = g[["P_in_h","A_in_h","V_in_h"]].sum(axis=1, min_count=1)
        for col,new in [("P_in_h","P_share"),("A_in_h","A_share"),("V_in_h","V_share")]:
            g[new] = np.where(g["total"]>0, g[col]/g["total"], np.nan)
        long = g[[group_col,"P_share","A_share","V_share"]].melt(group_col, var_name="Typ", value_name="Anteil")
        long["Typ"] = long["Typ"].map({"P_share":"P (planmäßig)","A_share":"A (außerplanmäßig)","V_share":"V (fachfremde Vertretung)"})
        return long
    group_col = geo_col if geo_col in fdf.columns else "Region"
    mix_geo = mix_by(group_col).rename(columns={group_col:"Gebiet"})
    chart_mix_geo = alt.Chart(mix_geo).mark_bar().encode(
        x=alt.X("Gebiet:N"),
        y=alt.Y("Anteil:Q", axis=alt.Axis(format="%")),
        color=alt.Color("Typ:N", legend=alt.Legend(title="Anteil am Ausfall")),
        tooltip=["Gebiet","Typ",alt.Tooltip("Anteil:Q", format=".1%")]
    ).properties(height=320)
    if value_labels:
        thr = label_threshold/100.0
        label_geo = alt.Chart(mix_geo).transform_filter(alt.datum.Anteil >= thr).mark_text().encode(
            x=alt.X("Gebiet:N"), y=alt.Y("sum(Anteil):Q", stack="center"), detail="Typ:N", text=alt.Text("sum(Anteil):Q", format=".0%")
        )
        chart_mix_geo = chart_mix_geo + label_geo
    st.altair_chart(chart_mix_geo, use_container_width=True)

    mix_art = mix_by("Schulart")
    chart_mix_art = alt.Chart(mix_art).mark_bar().encode(
        x=alt.X("Schulart:N"), y=alt.Y("Anteil:Q", axis=alt.Axis(format="%")),
        color=alt.Color("Typ:N", legend=alt.Legend(title="Anteil am Ausfall")),
        tooltip=["Schulart","Typ",alt.Tooltip("Anteil:Q", format=".1%")]
    ).properties(height=320)
    if value_labels:
        thr = label_threshold/100.0
        label_art = alt.Chart(mix_art).transform_filter(alt.datum.Anteil >= thr).mark_text().encode(
            x=alt.X("Schulart:N"), y=alt.Y("sum(Anteil):Q", stack="center"), detail="Typ:N", text=alt.Text("sum(Anteil):Q", format=".0%")
        )
        chart_mix_art = chart_mix_art + label_art
    st.altair_chart(chart_mix_art, use_container_width=True)

    # Schularten – Stacked %
    st.subheader("Schularten – Zusammensetzung der Ausfallquote (P/A/V in %-Punkten)")
    def agg_pct_by_art(df_in):
        cols = ["P_in_%","A_in_%","V_in_%"]
        if not weighted_toggle:
            g = df_in.groupby("Schulart", as_index=False)[cols].mean()
        else:
            def _wm(g):
                return pd.Series({c: weighted_mean(g[c], g["Sollstunden_est"]) for c in cols})
            g = df_in.groupby("Schulart").apply(_wm).reset_index()
        return g
    art_pct = agg_pct_by_art(fdf)
    art_pct_long = art_pct.melt(id_vars=["Schulart"], var_name="Typ", value_name="Wert")
    art_pct_long["Typ"] = art_pct_long["Typ"].map({"P_in_%":"P (planmäßig)","A_in_%":"A (außerplanmäßig)","V_in_%":"V (fachfremd)"})
    art_pct_long["is_highlight"] = art_pct_long["Schulart"].isin(highlight_sel) if "highlight_sel" in locals() else False
    stack_chart = alt.Chart(art_pct_long).mark_bar().encode(
        x=alt.X("Schulart:N", sort="-y"),
        y=alt.Y("sum(Wert):Q", title="Summe P+A+V (in %-Punkten)"),
        color=alt.Color("Typ:N", legend=alt.Legend(title="Anteil an P+A+V")),
        opacity=alt.condition("datum.is_highlight", alt.value(1), alt.value(0.5)),
        tooltip=[alt.Tooltip("Schulart:N"), "Typ:N", alt.Tooltip("Wert:Q", format=".1f")]
    ).properties(height=330)
    if value_labels:
        thr_pp = label_threshold
        label_stack = alt.Chart(art_pct_long).transform_filter(f"datum.Wert >= {thr_pp}").mark_text().encode(
            x=alt.X("Schulart:N"), y=alt.Y("sum(Wert):Q", stack="center"), detail="Typ:N", text=alt.Text("sum(Wert):Q", format=".1f")
        )
        stack_chart = stack_chart + label_stack
    st.altair_chart(stack_chart, use_container_width=True)

    # Schularten – Index ggü. Gesamt (=100)
    st.subheader("Schularten – Index gegenüber Gesamt (=100)")
    overall_val = weighted_mean(fdf["P+A+V_in_%"], fdf["Sollstunden_est"]) if weighted_toggle else float(np.nanmean(fdf["P+A+V_in_%"]))
    idx_df = group_weighted_mean(fdf, "Schulart", "P+A+V_in_%", "Sollstunden_est", weighted_toggle)
    idx_df["Index"] = (idx_df["P+A+V_in_%"]/overall_val)*100.0 if overall_val==overall_val else np.nan
    idx_df["is_highlight"] = idx_df["Schulart"].isin(highlight_sel) if "highlight_sel" in locals() else False
    idx_chart = alt.Chart(idx_df).mark_bar().encode(
        x=alt.X("Schulart:N", sort="-y"),
        y=alt.Y("Index:Q", title="Index (Gesamt = 100)"),
        opacity=alt.condition("datum.is_highlight", alt.value(1), alt.value(0.5)),
        tooltip=["Schulart", alt.Tooltip("P+A+V_in_%:Q", format=".1f"), alt.Tooltip("Index:Q", format=".0f")]
    ).properties(height=300)
    baseline = alt.Chart(pd.DataFrame({"y":[100]})).mark_rule(strokeDash=[5,3]).encode(y="y:Q")
    if value_labels:
        idx_text = alt.Chart(idx_df).mark_text(dy=-6).encode(x=alt.X("Schulart:N", sort="-y"), y=alt.Y("Index:Q"), text=alt.Text("Index:Q", format=".0f"))
        idx_chart = idx_chart + idx_text
    st.altair_chart(idx_chart + baseline, use_container_width=True)

    # Rankings
    if "Name" in fdf.columns and "P+A+V_in_%" in fdf.columns:
        st.subheader(f"Top {top_n} – höchste Ausfallquote (P+A+V in %)")
        rank_df = fdf[(fdf["Sollstunden_est"].fillna(0) >= min_soll)].copy()
        top_df = rank_df.sort_values("P+A+V_in_%", ascending=False).head(top_n)
        st.dataframe(top_df[["Region","Landkreis","Schulart","Name","P_in_%","A_in_%","V_in_%","P+A_in_%","P+A+V_in_%","Sollstunden_est"]], use_container_width=True)
        df_download_button(top_df, "top_schulen.csv", "Top-Liste herunterladen")

        st.subheader(f"Flop {top_n} – niedrigste Ausfallquote (P+A+V in %)")
        flop_df = rank_df.sort_values("P+A+V_in_%", ascending=True).head(top_n)
        st.dataframe(flop_df[["Region","Landkreis","Schulart","Name","P_in_%","A_in_%","V_in_%","P+A_in_%","P+A+V_in_%","Sollstunden_est"]], use_container_width=True)
        df_download_button(flop_df, "flop_schulen.csv", "Flop-Liste herunterladen")

    # Peer-Vergleich
    if "Schulart" in fdf.columns and "P+A+V_in_%" in fdf.columns:
        st.subheader("Peer-Vergleich innerhalb der Schulart (Perzentile & z-Scores)")
        bp = alt.Chart(fdf).mark_boxplot().encode(
            x=alt.X("Schulart:N"),
            y=alt.Y("P+A+V_in_%:Q", title="P+A+V in %"),
            tooltip=[alt.Tooltip("Schulart:N")]
        ).properties(height=320)
        if value_labels:
            med = alt.Chart(fdf).transform_aggregate(median_val="median(P+A+V_in_%)", groupby=["Schulart"]).mark_text(dy=-8).encode(
                x=alt.X("Schulart:N"), y=alt.Y("median_val:Q"), text=alt.Text("median_val:Q", format=".1f")
            )
            bp = bp + med
        st.altair_chart(bp, use_container_width=True)

        def add_percentile(g):
            q = g["P+A+V_in_%"].rank(pct=True)
            z = (g["P+A+V_in_%"] - g["P+A+V_in_%"].mean()) / g["P+A+V_in_%"].std(ddof=0)
            g = g.copy(); g["Perzentil_in_Schulart"]=(q*100).round(1); g["z_Score_in_Schulart"]=z.round(2); return g
        peer = fdf.groupby("Schulart", group_keys=False).apply(add_percentile)
        st.dataframe(peer[["Region","Landkreis","Schulart","Name","P+A+V_in_%","Perzentil_in_Schulart","z_Score_in_Schulart","Sollstunden_est"]].sort_values(["Schulart","P+A+V_in_%"], ascending=[True, False]), use_container_width=True)
        df_download_button(peer, "peer_vergleich.csv", "Peer-Vergleich herunterladen")

    # Story-Hooks
    st.subheader("🧭 Story-Hooks (automatisch)")
    st.caption("Filter setzen → Darstellung wählen → Download.")
    mode = st.radio("Darstellung", ["Bulletpoints kompakt", "Erklärung in einfacher Sprache"], horizontal=True, key="story_mode")
    sections = build_story_hooks_sections(fdf, weighted_toggle, min_soll, min_hours_hook, pareto_threshold, geo_col)

    if mode == "Bulletpoints kompakt":
        for title, items in sections.items():
            if not items: continue
            st.markdown(f"**{title}**")
            st.markdown("\\n".join([f"- {it}" for it in items])); st.markdown("")
    else:
        for title, items in sections.items():
            if not items: continue
            st.markdown(f"**{title}**")
            if title == "Überblick":
                st.write("So hoch ist die durchschnittliche Ausfallquote im aktuellen Filter.")
            elif title == "Gebiet (Region/Landkreis)":
                st.write("Vergleicht Regionen oder Landkreise – je nach gewählter Gebietsebene.")
            elif title.startswith("Top"):
                st.write("Höchste Quoten: Ansatzpunkte für Nachfragen.")
            elif title.startswith("Positivliste"):
                st.write("Niedrigste Quoten: mögliche Good-Practice-Beispiele.")
            elif title.startswith("Ursachenmix"):
                st.write("Was dominiert den Ausfall: planmäßig, außerplanmäßig oder fachfremd?")
            elif title == "Schularten":
                st.write("Schularten im Vergleich zum Gesamtniveau.")
            elif title.startswith("Ausreißer"):
                st.write("Deutlich herausstechende Schulen (z > 2) innerhalb ihrer Schulart.")
            elif title == "Qualitätslage":
                st.write("Auffälligkeiten und Datenqualität.")
            for it in items: st.markdown(f"- {it}")
            st.markdown("")
    # Download hooks
    md_lines = []
    for title, items in sections.items():
        if not items: continue
        md_lines.append(f"## {title}"); md_lines += [f"- {it}" for it in items]; md_lines.append("")
        story_md = "\\n".join(md_lines)
    st.download_button("Story-Hooks herunterladen (Markdown)", data=story_md.encode("utf-8"), file_name="story_hooks.md", mime="text/markdown")

    # Qualitätschecks
    st.subheader("Qualitätschecks")
    issues = []
    if all(c in fdf.columns for c in ["P_in_%","A_in_%","V_in_%","P+A+V_in_%"]):
        tmp = fdf.copy()
        tmp["abweichung"] = np.abs(tmp["P_in_%"].fillna(0)+tmp["A_in_%"].fillna(0)+tmp["V_in_%"].fillna(0)-tmp["P+A+V_in_%"])
        bad = tmp[tmp["abweichung"]>0.2]
        if len(bad)>0:
            issues.append(f"🔴 {len(bad)} Zeilen mit auffälliger Summenabweichung (>0,2 %-Punkte).")
            st.dataframe(bad[["Region","Landkreis","Schulart","Name","P_in_%","A_in_%","V_in_%","P+A+V_in_%","abweichung"]], use_container_width=True)
    if "P+A+V_in_%" in fdf.columns:
        high = fdf[fdf["P+A+V_in_%"]>35]
        if len(high)>0:
            issues.append(f"🟠 {len(high)} Zeilen mit sehr hoher Ausfallquote (>35%).")
            with st.expander("Zeigen: sehr hohe Werte"):
                st.dataframe(high[["Region","Landkreis","Schulart","Name","P+A+V_in_%"]], use_container_width=True)
    if all(c in fdf.columns for c in ["P_in_h","A_in_h","V_in_h"]):
        zeros = fdf[(fdf["P_in_h"].fillna(0)==0) & (fdf["A_in_h"].fillna(0)==0) & (fdf["V_in_h"].fillna(0)==0)]
        if len(zeros)>0:
            issues.append(f"🟡 {len(zeros)} Zeilen ohne Stundenangaben (alle 0).")
            with st.expander("Zeigen: 0-Stunden-Zeilen"):
                st.dataframe(zeros[["Region","Landkreis","Schulart","Name","P_in_h","A_in_h","V_in_h"]], use_container_width=True)
    if "Soll_flag_konflikt" in fdf.columns:
        conflict = fdf[fdf["Soll_flag_konflikt"]==True]
        if len(conflict)>0:
            issues.append(f"⚠️ {len(conflict)} Zeilen mit widersprüchlicher Sollstunden-Schätzung (>15% Spannweite).")
            with st.expander("Zeigen: Sollstunden-Konflikte"):
                st.dataframe(conflict[["Region","Landkreis","Schulart","Name","Sollstunden_est","Soll_kandidaten_anzahl","Soll_rel_range"]], use_container_width=True)
    if not issues: st.success("Keine Auffälligkeiten gefunden.")
    else:
        for msg in issues: st.write(msg)

    if show_table:
        st.subheader("Rohdaten (gefiltert)")
        st.dataframe(fdf, use_container_width=True)
        df_download_button(fdf, "rohdaten_gefiltert.csv", "Gefilterte Rohdaten herunterladen")
else:
    st.warning("Keine Daten geladen.")

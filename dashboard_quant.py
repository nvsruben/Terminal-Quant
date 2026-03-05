import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# --- CONFIGURATION STREAMLIT ---
st.set_page_config(page_title="Terminal Quantitatif V12", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# 🔒 SYSTÈME DE SÉCURITÉ
# ==========================================
if "authentifie" not in st.session_state:
    st.session_state.authentifie = False

if not st.session_state.authentifie:
    st.title("🛡️ Terminal Quantitatif Privé")
    st.markdown("Veuillez vous identifier pour accéder au moteur d'allocation.")
    MOT_DE_PASSE_SECRET = "evalyn" 
    mdp_saisi = st.text_input("Mot de passe", type="password")
    if st.button("Déverrouiller le Terminal"):
        if mdp_saisi == MOT_DE_PASSE_SECRET:
            st.session_state.authentifie = True
            st.rerun()
        else:
            st.error("Accès refusé.")
    st.stop()

# ==========================================
# ⚙️ LE MOTEUR QUANTITATIF & MACRO
# ==========================================

# Tes actifs actuels
MES_FAVORIS = {
    "Bitcoin": "BTC-EUR", "Ethereum": "ETH-EUR", "Solana": "SOL-EUR",
    "NVIDIA": "NVDA", "Tesla": "TSLA", "Apple": "AAPL", "Alphabet (A)": "GOOGL", "AMD": "AMD", "Palantir": "PLTR",
    "Dassault Systèmes": "DSY.PA", "ASML": "ASML.AS", "TSMC": "TSM",
    "Lockheed Martin": "LMT", "Rheinmetall": "RHM.DE", "Thales": "HO.PA", "Airbus": "AIR.PA",
    "LVMH": "MC.PA", "TotalEnergies": "TTE.PA", "Air Liquide": "AI.PA",
    "Or Physique": "IGLN.L", "Argent Physique": "PHAG.L", "Copper": "CPER", 
    "Uranium USD": "URNM", "Rare Earth": "REMX", "Gold Producers": "GDX", "Copper Miners": "COPX",
    "Core S&P 500": "CSPX.AS", "Core MSCI World": "IWDA.AS", "MSCI EM": "EMIM.L", 
    "Cyber Security": "ISPY.L", "Defense USD": "DFNS.L"
}

# L'univers du Screener (Actifs hors de ta liste pour dénicher de l'Alpha)
UNIVERS_SCREENER = {
    "Berkshire Hathaway": "BRK-B", "JPMorgan": "JPM", "Visa": "V", "Eli Lilly": "LLY", 
    "Novo Nordisk": "NVO", "Broadcom": "AVGO", "S&P 500 Tech ETF": "IUIT.L", 
    "S&P 500 Health ETF": "IUHC.L", "MSCI India ETF": "NDIA.L", "Obligations US 20+ Yrs": "TLT"
}

@st.cache_data(ttl=3600)
def telecharger_donnees():
    # ^VIX = Peur du marché | ^TNX = Taux à 10 ans de la FED
    tickers_complets = list(MES_FAVORIS.values()) + list(UNIVERS_SCREENER.values()) + ['^VIX', '^TNX']
    df = yf.download(tickers_complets, period="2y", progress=False)['Close']
    df = df.ffill().bfill()
    return df

# Générateur CSV spécial Excel Français (Séparateur ; et virgules pour décimales)
def generer_csv_europe(allocations, budget_total, reserve_cash, vix_actuel, taux_fed):
    date_jour = datetime.now().strftime("%d/%m/%Y")
    en_tetes = "Date;Actif_Achete;Montant_Alloue_EUR;Infos_Macro\n"
    
    # On remplace les points par des virgules pour Excel France
    lignes = f"{date_jour};RESERVE CASH TR;{str(round(reserve_cash, 2)).replace('.', ',')};VIX={vix_actuel:.1f} FED={taux_fed:.2f}%\n"
    
    for actif, montant in allocations.items():
        if montant > 0:
            montant_str = str(round(montant, 2)).replace('.', ',')
            lignes += f"{date_jour};{actif};{montant_str};DCA Mensuel\n"
            
    # L'encodage 'utf-8-sig' force Excel à lire correctement les caractères spéciaux (é, è)
    return (en_tetes + lignes).encode('utf-8-sig')

df_brut = telecharger_donnees()
vix_actuel = float(df_brut['^VIX'].iloc[-1])
taux_fed_10y = float(df_brut['^TNX'].iloc[-1]) # Le taux d'intérêt sans risque

# Séparation des données
df_actifs = df_brut[list(MES_FAVORIS.values())].copy()
df_actifs.columns = list(MES_FAVORIS.keys())

df_screener = df_brut[list(UNIVERS_SCREENER.values())].copy()
df_screener.columns = list(UNIVERS_SCREENER.keys())

# --- CALCULS RISK PARITY SUR FAVORIS ---
rendements = np.log(df_actifs / df_actifs.shift(1)).dropna(how='all')
volatilite_ewma = rendements.ewm(span=60).std().iloc[-1] * np.sqrt(252)

rendements_negatifs = rendements.copy()
rendements_negatifs[rendements_negatifs > 0] = 0
downside_vol = rendements_negatifs.std() * np.sqrt(252)
sortino = (rendements.mean() * 252) / downside_vol.replace(0, np.nan)

rendements_cumules = (1 + rendements).cumprod()
sommet_historique = rendements_cumules.cummax()
drawdown = (rendements_cumules - sommet_historique) / sommet_historique
max_dd = drawdown.min()
correlation = rendements.ewm(span=60).corr().loc[rendements.index[-1]]

# --- BARRE LATÉRALE ---
st.sidebar.title("⚙️ Paramètres de Risque")
budget = st.sidebar.number_input("Budget DCA (€)", min_value=10.0, value=950.0, step=10.0)
seuil_vix = st.sidebar.slider("Seuil Panique VIX (Cash)", 15, 40, 20)
vol_max = st.sidebar.slider("Rejet : Volatilité EWMA Max (%)", 20, 150, 45) / 100.0
dd_max = st.sidebar.slider("Rejet : Max Drawdown pire (%)", -80, -10, -35) / 100.0
correl_max = st.sidebar.slider("Filtre Anti-Doublon (%)", 50, 95, 70) / 100.0

# Logique Macro : Si le VIX est haut ET que les taux sont hauts (>4.5%), on augmente le cash à 30%
if vix_actuel > seuil_vix and taux_fed_10y > 4.5:
    pourcentage_cash = 0.30
elif vix_actuel > seuil_vix:
    pourcentage_cash = 0.20
else:
    pourcentage_cash = 0.0

reserve_cash = budget * pourcentage_cash
budget_investissable = budget - reserve_cash

# Moteur de sélection
actifs_eligibles = []
raisons = {}
for actif in MES_FAVORIS.keys():
    vol = volatilite_ewma[actif]
    dd = max_dd[actif]
    if vol > vol_max: raisons[actif] = f"Rejeté (Volatilité > {vol_max*100:.0f}%)"
    elif dd < dd_max: raisons[actif] = f"Rejeté (Max DD < {dd_max*100:.0f}%)"
    elif pd.isna(sortino[actif]): raisons[actif] = "Données insuffisantes"
    else: actifs_eligibles.append(actif)

sortino_eligibles = sortino[actifs_eligibles].sort_values(ascending=False)
candidats = sortino_eligibles.index.tolist()
top_5_actifs = []

for candidat in candidats:
    if len(top_5_actifs) >= 5:
        raisons[candidat] = "Hors Top 5"
        continue
    trop_correle = False
    for selectionne in top_5_actifs:
        if correlation.loc[candidat, selectionne] > correl_max:
            trop_correle = True
            raisons[candidat] = f"Doublon avec {selectionne}"
            break
    if not trop_correle:
        top_5_actifs.append(candidat)

if len(top_5_actifs) > 0:
    vol_top5 = volatilite_ewma[top_5_actifs]
    inv_vol = 1 / vol_top5
    poids = inv_vol / inv_vol.sum()
    allocations = poids * budget_investissable
else:
    allocations = pd.Series(dtype=float)

# --- BOUTONS ---
if st.sidebar.button("Se déconnecter"):
    st.session_state.authentifie = False
    st.rerun()

st.sidebar.markdown("---")
if len(top_5_actifs) > 0:
    csv_data = generer_csv_europe(allocations, budget, reserve_cash, vix_actuel, taux_fed_10y)
    date_fichier = datetime.now().strftime("%Y_%m_%d")
    st.sidebar.download_button(
        label="💾 Exporter Excel (Format FR)",
        data=csv_data,
        file_name=f"Ordre_DCA_FR_{date_fichier}.csv",
        mime="text/csv"
    )

# --- INTERFACE PRINCIPALE ---
st.title("🛡️ Terminal Quant & Macro (V12)")

# Tableau de bord Macro
col1, col2, col3, col4 = st.columns(4)
col1.metric("VIX (Peur)", f"{vix_actuel:.1f}", delta="Panique" if vix_actuel > seuil_vix else "Calme", delta_color="inverse")
col2.metric("Taux US 10 Ans (FED)", f"{taux_fed_10y:.2f}%", delta="Surchauffe" if taux_fed_10y > 4.5 else "Normal", delta_color="inverse")
col3.metric("Capital à Investir", f"{budget_investissable:.2f} €")
col4.metric("Cash Défensif", f"{reserve_cash:.2f} €", delta=f"{pourcentage_cash*100}% du budget")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Plan d'Épargne", "🔗 Matrice & Visuels", "📈 Backtest 2 Ans", "🔍 Screener Opportunités"])

with tab1:
    donnees_tableau = []
    for actif in MES_FAVORIS.keys():
        statut = "✅ SÉLECTIONNÉ" if actif in top_5_actifs else raisons.get(actif, "Ignoré")
        mnt = allocations[actif] if actif in top_5_actifs else 0.0
        donnees_tableau.append({
            "Actif": actif, "Statut": statut, "Sortino": sortino[actif], 
            "Max DD (%)": max_dd[actif]*100, "Vol. EWMA (%)": volatilite_ewma[actif]*100, 
            "Montant (€)": mnt
        })
    df_affichage = pd.DataFrame(donnees_tableau).sort_values(by="Montant (€)", ascending=False)
    st.dataframe(df_affichage.style.format({"Sortino": "{:.2f}", "Max DD (%)": "{:.1f}%", "Vol. EWMA (%)": "{:.1f}%", "Montant (€)": "{:.2f} €"}).applymap(lambda x: 'background-color: #004d00' if x == '✅ SÉLECTIONNÉ' else '', subset=['Statut']), use_container_width=True, height=350)

with tab2:
    col_pie, col_heat = st.columns(2)
    with col_pie:
        labels_pie = top_5_actifs + (["Cash (TR)"] if reserve_cash > 0 else [])
        valeurs_pie = list(allocations.values) + ([reserve_cash] if reserve_cash > 0 else [])
        fig_pie = px.pie(names=labels_pie, values=valeurs_pie, hole=0.4, color_discrete_sequence=px.colors.sequential.Teal)
        fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_heat:
        top_15 = sortino_eligibles.head(15).index.tolist()
        if len(top_15) > 1:
            fig_heat = px.imshow(correlation.loc[top_15, top_15], text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            st.plotly_chart(fig_heat, use_container_width=True)

with tab3:
    if len(top_5_actifs) > 0:
        ret_2y = df_actifs.pct_change().dropna()
        poids_backtest = pd.Series(allocations) / budget_investissable
        frais = (len(top_5_actifs) * 12) / (budget * 12) 
        ret_portefeuille = (ret_2y[top_5_actifs] * poids_backtest.values).sum(axis=1) - (frais / 252)
        croissance_pf = (1 + ret_portefeuille).cumprod() * 100
        croissance_sp = (1 + ret_2y["Core S&P 500"]).cumprod() * 100
        df_backtest = pd.DataFrame({"Ton Algo Quant": croissance_pf, "S&P 500": croissance_sp})
        st.plotly_chart(px.line(df_backtest, labels={"value": "Croissance (Base 100)", "Date": "Date"}), use_container_width=True)

with tab4:
    st.subheader("Radar à Alpha (Recherche de nouveaux actifs)")
    st.markdown("L'algorithme a scanné un univers d'actifs hors de tes favoris pour identifier ceux qui offrent le meilleur rendement ajusté au risque (Sortino) actuellement.")
    
    rend_screen = np.log(df_screener / df_screener.shift(1)).dropna(how='all')
    rend_neg_screen = rend_screen.copy()
    rend_neg_screen[rend_neg_screen > 0] = 0
    sortino_screen = (rend_screen.mean() * 252) / (rend_neg_screen.std() * np.sqrt(252))
    vol_screen = rend_screen.std() * np.sqrt(252)
    
    df_resultats_screener = pd.DataFrame({
        "Ratio Sortino": sortino_screen,
        "Volatilité": vol_screen * 100
    }).sort_values(by="Ratio Sortino", ascending=False).head(5)
    
    st.dataframe(df_resultats_screener.style.format({"Ratio Sortino": "{:.2f}", "Volatilité": "{:.1f}%"}), use_container_width=True)
    st.info("💡 Idée : Si un actif ci-dessus a un Sortino supérieur à tes favoris actuels, ajoute-le à ton code !")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# --- CONFIGURATION STREAMLIT ---
st.set_page_config(page_title="Terminal Quantitatif V11", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# 🔒 SYSTÈME DE SÉCURITÉ (LOGIN)
# ==========================================
if "authentifie" not in st.session_state:
    st.session_state.authentifie = False

if not st.session_state.authentifie:
    st.title("🛡️ Terminal Quantitatif Privé")
    st.markdown("Veuillez vous identifier pour accéder au moteur d'allocation.")
    
    # CHANGER LE MOT DE PASSE ICI :
    MOT_DE_PASSE_SECRET = "evalyn" 
    
    mdp_saisi = st.text_input("Mot de passe", type="password")
    if st.button("Déverrouiller le Terminal"):
        if mdp_saisi == MOT_DE_PASSE_SECRET:
            st.session_state.authentifie = True
            st.rerun() # Recharge la page en mode déverrouillé
        else:
            st.error("Accès refusé. Mot de passe incorrect.")
    st.stop() # Bloque l'exécution de tout le code en dessous si pas connecté

# ==========================================
# ⚙️ LE MOTEUR QUANTITATIF (Une fois connecté)
# ==========================================

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

@st.cache_data(ttl=3600)
def telecharger_donnees():
    tickers_complets = list(MES_FAVORIS.values()) + ['^VIX']
    df = yf.download(tickers_complets, period="2y", progress=False)['Close']
    df = df.ffill().bfill()
    return df

# Générateur de fichier CSV Cloud-Native
def generer_csv(allocations, budget_total, reserve_cash, vix_actuel):
    date_jour = datetime.now().strftime("%Y-%m-%d %H:%M")
    en_tetes = "Date,Regime_VIX,Budget_Total,Reserve_Cash,Actif_1,Montant_1,Actif_2,Montant_2,Actif_3,Montant_3,Actif_4,Montant_4,Actif_5,Montant_5\n"
    ligne = f"{date_jour},{vix_actuel:.1f},{budget_total:.2f},{reserve_cash:.2f}"
    
    for actif, montant in allocations.items():
        if montant > 0:
            ligne += f",{actif},{montant:.2f}"
            
    return (en_tetes + ligne).encode('utf-8')

df_brut = telecharger_donnees()
vix_actuel = float(df_brut['^VIX'].iloc[-1])

df_actifs = df_brut[list(MES_FAVORIS.values())].copy()
df_actifs.columns = list(MES_FAVORIS.keys())

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

st.sidebar.title("⚙️ Paramètres de Risque")
budget = st.sidebar.number_input("Budget DCA (€)", min_value=10.0, value=950.0, step=10.0)
seuil_vix = st.sidebar.slider("Seuil Panique VIX (Cash)", 15, 40, 20)
vol_max = st.sidebar.slider("Rejet : Volatilité EWMA Max (%)", 20, 150, 45) / 100.0
dd_max = st.sidebar.slider("Rejet : Max Drawdown pire (%)", -80, -10, -35) / 100.0
correl_max = st.sidebar.slider("Filtre Anti-Doublon (%)", 50, 95, 70) / 100.0

pourcentage_cash = 0.20 if vix_actuel > seuil_vix else 0.0
reserve_cash = budget * pourcentage_cash
budget_investissable = budget - reserve_cash

actifs_eligibles = []
raisons = {}

for actif in MES_FAVORIS.keys():
    vol = volatilite_ewma[actif]
    dd = max_dd[actif]
    if vol > vol_max: raisons[actif] = f"Rejeté (Volatilité EWMA > {vol_max*100:.0f}%)"
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

# --- BOUTON DE TÉLÉCHARGEMENT CLOUD ---
if st.sidebar.button("Se déconnecter"):
    st.session_state.authentifie = False
    st.rerun()

st.sidebar.markdown("---")
if len(top_5_actifs) > 0:
    csv_data = generer_csv(allocations, budget, reserve_cash, vix_actuel)
    date_fichier = datetime.now().strftime("%Y_%m_%d")
    st.sidebar.download_button(
        label="💾 Télécharger l'Ordre (CSV)",
        data=csv_data,
        file_name=f"Ordre_DCA_{date_fichier}.csv",
        mime="text/csv"
    )

st.title("🛡️ Hedge Fund Personnel (Moteur EWMA)")

col1, col2, col3 = st.columns(3)
col1.metric("VIX Actuel", f"{vix_actuel:.1f}", delta="Panique" if vix_actuel > seuil_vix else "Calme", delta_color="inverse")
col2.metric("Capital Investi", f"{budget_investissable:.2f} €", delta=f"Sélection : {len(top_5_actifs)} actifs")
col3.metric("Réserve Cash (Trade Republic)", f"{reserve_cash:.2f} €", delta="Sécurité Activée" if reserve_cash > 0 else "")

tab1, tab2, tab3 = st.tabs(["📊 Allocation DCA", "🔗 Matrice & Visuels", "📈 Backtest 2 Ans"])

with tab1:
    donnees_tableau = []
    for actif in MES_FAVORIS.keys():
        if actif in top_5_actifs:
            statut = "✅ SÉLECTIONNÉ"
            mnt = allocations[actif]
        else:
            statut = raisons.get(actif, "Ignoré")
            mnt = 0.0
            
        donnees_tableau.append({
            "Actif": actif, "Statut": statut, "Sortino": sortino[actif], 
            "Max DD (%)": max_dd[actif]*100, "Vol. EWMA (%)": volatilite_ewma[actif]*100, 
            "Montant (€)": mnt
        })
        
    df_affichage = pd.DataFrame(donnees_tableau).sort_values(by="Montant (€)", ascending=False)
    st.dataframe(df_affichage.style.format({
        "Sortino": "{:.2f}", "Max DD (%)": "{:.1f}%", "Vol. EWMA (%)": "{:.1f}%", "Montant (€)": "{:.2f} €"
    }).applymap(lambda x: 'background-color: #004d00' if x == '✅ SÉLECTIONNÉ' else '', subset=['Statut']), 
    use_container_width=True, height=350)

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
            corr_reduite = correlation.loc[top_15, top_15]
            fig_heat = px.imshow(corr_reduite, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            st.plotly_chart(fig_heat, use_container_width=True)

with tab3:
    st.markdown("Simulation de la Forteresse (Top 5 actuel pondéré EWMA) vs S&P 500 sur 24 mois.")
    if len(top_5_actifs) > 0:
        ret_2y = df_actifs.pct_change().dropna()
        poids_backtest = pd.Series(allocations) / budget_investissable
        frais_annuels_estimes = (len(top_5_actifs) * 12) / (budget * 12) 
        ret_portefeuille = (ret_2y[top_5_actifs] * poids_backtest.values).sum(axis=1) - (frais_annuels_estimes / 252)
        ret_sp500 = ret_2y["Core S&P 500"]
        
        croissance_pf = (1 + ret_portefeuille).cumprod() * 100
        croissance_sp = (1 + ret_sp500).cumprod() * 100
        
        df_backtest = pd.DataFrame({"Portefeuille Quant (Net de frais)": croissance_pf, "S&P 500": croissance_sp})
        fig_line = px.line(df_backtest, labels={"value": "Croissance (Base 100)", "Date": "Date"})
        st.plotly_chart(fig_line, use_container_width=True)

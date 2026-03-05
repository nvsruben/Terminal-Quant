import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURATION DE LA PAGE STREAMLIT ---
st.set_page_config(page_title="Terminal Quantitatif V9", layout="wide", initial_sidebar_state="expanded")

# --- UNIVERS D'INVESTISSEMENT ---
MES_FAVORIS = {
    "Bitcoin": "BTC-EUR", "Ethereum": "ETH-EUR", "Solana": "SOL-EUR",
    "NVIDIA": "NVDA", "Tesla": "TSLA", "Apple": "AAPL", "Alphabet (A)": "GOOGL", "AMD": "AMD", "Palantir": "PLTR",
    "Dassault Systèmes": "DSY.PA", "ASML": "ASML.AS", "TSMC": "TSM",
    "Lockheed Martin": "LMT", "Rheinmetall": "RHM.DE", "Thales": "HO.PA", "Airbus": "AIR.PA", "Dassault Aviation": "AM.PA",
    "LVMH": "MC.PA", "TotalEnergies": "TTE.PA", "Air Liquide": "AI.PA",
    "Or Physique": "IGLN.L", "Argent Physique": "PHAG.L", "Copper": "CPER", 
    "Uranium USD": "URNM", "Rare Earth": "REMX", "Gold Producers": "GDX", "Copper Miners": "COPX",
    "Core S&P 500": "CSPX.AS", "Core MSCI World": "IWDA.AS", "MSCI EM": "EMIM.L", 
    "Cyber Security": "ISPY.L", "Defense USD": "DFNS.L"
}

# --- TÉLÉCHARGEMENT AVEC MISE EN CACHE (Ultra Rapide) ---
@st.cache_data(ttl=3600) # Garde en mémoire pendant 1 heure
def telecharger_donnees():
    tickers_complets = list(MES_FAVORIS.values()) + ['^VIX']
    # On télécharge 2 ans pour pouvoir faire le mini-backtest
    df = yf.download(tickers_complets, period="2y", progress=False)['Close']
    df = df.ffill().bfill()
    return df

df_brut = telecharger_donnees()

# Extraction VIX
vix_actuel = float(df_brut['^VIX'].iloc[-1])

# Isolation des actifs (sur les 252 derniers jours pour les calculs actuels)
df_actifs = df_brut[list(MES_FAVORIS.values())].copy()
df_actifs.columns = list(MES_FAVORIS.keys())
df_1y = df_actifs.tail(252)

# --- CALCULS QUANTITATIFS ---
rendements = np.log(df_1y / df_1y.shift(1)).dropna(how='all')
volatilite = rendements.std() * np.sqrt(252)

rendements_negatifs = rendements.copy()
rendements_negatifs[rendements_negatifs > 0] = 0
downside_vol = rendements_negatifs.std() * np.sqrt(252)
sortino = (rendements.mean() * 252) / downside_vol.replace(0, np.nan)

rendements_cumules = (1 + rendements).cumprod()
sommet_historique = rendements_cumules.cummax()
drawdown = (rendements_cumules - sommet_historique) / sommet_historique
max_dd = drawdown.min()
correlation = rendements.corr()

# --- BARRE LATÉRALE (PANNEAU DE CONTRÔLE INTERACTIF) ---
st.sidebar.title("⚙️ Paramètres de Risque")
st.sidebar.markdown("Ajustez l'algorithme en temps réel.")

budget = st.sidebar.number_input("Budget DCA (€)", min_value=10.0, value=950.0, step=10.0)
seuil_vix = st.sidebar.slider("Seuil Panique VIX (Cash)", 15, 40, 20)
vol_max = st.sidebar.slider("Rejet : Volatilité Max (%)", 20, 150, 50) / 100.0
dd_max = st.sidebar.slider("Rejet : Max Drawdown pire (%)", -80, -10, -40) / 100.0
correl_max = st.sidebar.slider("Filtre Anti-Doublon (%)", 50, 95, 75) / 100.0

# --- MOTEUR DE DÉCISION ---
pourcentage_cash = 0.20 if vix_actuel > seuil_vix else 0.0
reserve_cash = budget * pourcentage_cash
budget_investissable = budget - reserve_cash

actifs_eligibles = []
raisons = {}

for actif in MES_FAVORIS.keys():
    vol = volatilite[actif]
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

vol_top5 = volatilite[top_5_actifs]
inv_vol = 1 / vol_top5
poids = inv_vol / inv_vol.sum()
allocations = poids * budget_investissable

# --- INTERFACE PRINCIPALE ---
st.title("🛡️ Terminal Quantitatif & Gestion des Risques")

# 1. Métriques Macro (KPIs)
col1, col2, col3 = st.columns(3)
col1.metric("VIX Actuel", f"{vix_actuel:.1f}", delta="Panique" if vix_actuel > seuil_vix else "Calme", delta_color="inverse")
col2.metric("Capital Investi", f"{budget_investissable:.2f} €", delta=f"Sélection : {len(top_5_actifs)} actifs")
col3.metric("Réserve Cash (Trade Republic)", f"{reserve_cash:.2f} €", delta="Sécurité Activée" if reserve_cash > 0 else "")

# --- ONGLETS (TABS) ---
tab1, tab2, tab3 = st.tabs(["📊 Allocation du Mois", "🔗 Matrice & Visuels", "📈 Backtest Historique"])

with tab1:
    st.subheader("Plan d'Épargne Recommandé")
    
    # Création d'un beau DataFrame pour Streamlit
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
            "Max DD (%)": max_dd[actif]*100, "Volatilité (%)": volatilite[actif]*100, 
            "Montant (€)": mnt
        })
        
    df_affichage = pd.DataFrame(donnees_tableau).sort_values(by="Montant (€)", ascending=False)
    
    # Formatage des colonnes
    st.dataframe(df_affichage.style.format({
        "Sortino": "{:.2f}", "Max DD (%)": "{:.1f}%", "Volatilité (%)": "{:.1f}%", "Montant (€)": "{:.2f} €"
    }).applymap(lambda x: 'background-color: #004d00' if x == '✅ SÉLECTIONNÉ' else '', subset=['Statut']), 
    use_container_width=True, height=350)

with tab2:
    col_pie, col_heat = st.columns(2)
    
    with col_pie:
        st.subheader("Répartition du Portefeuille")
        # Données pour le camembert (incluant le cash)
        labels_pie = top_5_actifs + (["Cash (TR)"] if reserve_cash > 0 else [])
        valeurs_pie = list(allocations.values) + ([reserve_cash] if reserve_cash > 0 else [])
        
        fig_pie = px.pie(names=labels_pie, values=valeurs_pie, hole=0.4, color_discrete_sequence=px.colors.sequential.Teal)
        fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_heat:
        st.subheader("Heatmap Anti-Doublon (Corrélation)")
        # Heatmap uniquement sur le Top 15 pour que ce soit lisible
        top_15 = sortino_eligibles.head(15).index.tolist()
        corr_reduite = correlation.loc[top_15, top_15]
        
        fig_heat = px.imshow(corr_reduite, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        st.plotly_chart(fig_heat, use_container_width=True)

with tab3:
    st.subheader("Backtest (Simulation 2 Ans)")
    st.markdown("Si vous aviez acheté et conservé la **sélection d'aujourd'hui** (en pondération Risk Parity) par rapport au S&P 500 sur les 2 dernières années.")
    
    if len(top_5_actifs) > 0:
        # Extraction de l'historique sur 2 ans
        df_2y = df_actifs.copy()
        
        # Rendements quotidiens
        ret_2y = df_2y.pct_change().dropna()
        
        # Calcul du portefeuille simulé (Poids fixes initiaux)
        poids_backtest = pd.Series(allocations) / budget_investissable
        ret_portefeuille = (ret_2y[top_5_actifs] * poids_backtest.values).sum(axis=1)
        
        # Rendement S&P 500
        ret_sp500 = ret_2y["Core S&P 500"]
        
        # Calcul de la courbe de croissance (Base 100)
        croissance_pf = (1 + ret_portefeuille).cumprod() * 100
        croissance_sp = (1 + ret_sp500).cumprod() * 100
        
        df_backtest = pd.DataFrame({"Ton Portefeuille Défensif": croissance_pf, "S&P 500 (Benchmark)": croissance_sp})
        
        fig_line = px.line(df_backtest, labels={"value": "Croissance (Base 100)", "Date": "Date"})
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.warning("Aucun actif sélectionné (Filtres trop stricts).")
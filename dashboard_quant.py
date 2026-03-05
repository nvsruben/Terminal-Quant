import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import requests

# --- CONFIGURATION STREAMLIT ---
st.set_page_config(page_title="Terminal Quantitatif V18", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# SYSTÈME D'AUTHENTIFICATION STRICTE
# ==========================================
if "authentifie" not in st.session_state:
    st.session_state.authentifie = False

if not st.session_state.authentifie:
    st.title("QUANTITATIVE ALLOCATION TERMINAL")
    st.markdown("AUTHENTIFICATION REQUISE. ACCÈS RESTREINT.")
    MOT_DE_PASSE_SECRET = "evalyn" 
    mdp_saisi = st.text_input("Passkey", type="password")
    if st.button("INITIALISER LA SESSION"):
        if mdp_saisi == MOT_DE_PASSE_SECRET:
            st.session_state.authentifie = True
            st.rerun()
        else:
            st.error("ACCÈS REFUSÉ.")
    st.stop()

# ==========================================
# UNIVERS D'INVESTISSEMENT (DICTIONNAIRE COMPLET)
# ==========================================

MES_FAVORIS = {
    "Bitcoin": {"ticker": "BTC-EUR", "nom": "Bitcoin (Crypto)"},
    "Ethereum": {"ticker": "ETH-EUR", "nom": "Ethereum (Crypto)"},
    "Solana": {"ticker": "SOL-EUR", "nom": "Solana (Crypto)"},
    "Or Physique": {"ticker": "IGLN.L", "nom": "iShares Physical Gold ETC"},
    "Argent Physique": {"ticker": "PHAG.L", "nom": "WisdomTree Physical Silver"},
    "Copper": {"ticker": "CPER", "nom": "US Copper Index Fund"},
    "Uranium USD": {"ticker": "URNM", "nom": "Sprott Uranium Miners ETF"},
    "Rare Earth": {"ticker": "REMX", "nom": "VanEck Rare Earth/Strategic Metals"},
    "Gold Producers": {"ticker": "GDX", "nom": "VanEck Gold Miners ETF"},
    "Copper Miners": {"ticker": "COPX", "nom": "Global X Copper Miners ETF"},
    "Core S&P 500": {"ticker": "CSPX.AS", "nom": "iShares Core S&P 500 UCITS"},
    "Core MSCI World": {"ticker": "IWDA.AS", "nom": "iShares Core MSCI World"},
    "MSCI EM": {"ticker": "EMIM.L", "nom": "iShares Core MSCI EM IMI"},
    "MSCI Japan": {"ticker": "SJPA.L", "nom": "iShares MSCI Japan"},
    "MSCI Korea": {"ticker": "CSKR.L", "nom": "iShares MSCI Korea"},
    "MSCI India": {"ticker": "NDIA.L", "nom": "iShares MSCI India"},
    "Cyber Security": {"ticker": "ISPY.L", "nom": "L&G Cyber Security UCITS"},
    "Defense USD": {"ticker": "DFNS.L", "nom": "VanEck Defense UCITS"},
    "Dassault Systèmes": {"ticker": "DSY.PA", "nom": "Dassault Systèmes SE"},
    "ASML": {"ticker": "ASML.AS", "nom": "ASML Holding N.V."},
    "TSMC": {"ticker": "TSM", "nom": "Taiwan Semiconductor Mfg."},
    "Lockheed Martin": {"ticker": "LMT", "nom": "Lockheed Martin Corp."},
    "Rheinmetall": {"ticker": "RHM.DE", "nom": "Rheinmetall AG"},
    "Thales": {"ticker": "HO.PA", "nom": "Thales S.A."},
    "Airbus": {"ticker": "AIR.PA", "nom": "Airbus SE"},
    "LVMH": {"ticker": "MC.PA", "nom": "LVMH Moët Hennessy"},
    "TotalEnergies": {"ticker": "TTE.PA", "nom": "TotalEnergies SE"},
    "Air Liquide": {"ticker": "AI.PA", "nom": "Air Liquide S.A."}
}

@st.cache_data(ttl=86400)
def aspirer_le_marche_sp500():
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        html = requests.get(url, headers=headers).text
        table = pd.read_html(html)[0]
        tickers = table['Symbol'].tolist()
        noms = table['Security'].tolist()
        
        dictionnaire_sp500 = {}
        for t, n in zip(tickers, noms):
            ticker_propre = t.replace('.', '-')
            dictionnaire_sp500[f"S&P500: {ticker_propre}"] = {"ticker": ticker_propre, "nom": n}
        return dictionnaire_sp500
    except Exception as e:
        return {}

univers_etudie = MES_FAVORIS.copy()
mega_dict = aspirer_le_marche_sp500()

for cle, donnees in mega_dict.items():
    tickers_actuels = [v["ticker"] for v in univers_etudie.values()]
    if donnees["ticker"] not in tickers_actuels:
        univers_etudie[cle] = donnees

@st.cache_data(ttl=3600)
def telecharger_donnees(liste_tickers):
    # Ajout du ^GSPC (S&P 500 Index) pour le calcul de la SMA 200
    tickers_complets = liste_tickers + ['^VIX', '^TNX', '^GSPC']
    df = yf.download(tickers_complets, period="3y", progress=False)['Close']
    df = df.ffill().bfill()
    return df

def generer_csv_europe(allocations, budget_total, reserve_cash, regime_txt):
    date_jour = datetime.now().strftime("%d/%m/%Y")
    en_tetes = "Date;Instrument;Ticker;Allocation_EUR;Notes\n"
    lignes = f"{date_jour};RESERVE CASH TR;-;{str(round(reserve_cash, 2)).replace('.', ',')};{regime_txt}\n"
    
    for actif, montant in allocations.items():
        if montant > 0:
            montant_str = str(round(montant, 2)).replace('.', ',')
            ticker = univers_etudie[actif]["ticker"]
            nom = univers_etudie[actif]["nom"]
            lignes += f"{date_jour};{nom};{ticker};{montant_str};DCA\n"
    return (en_tetes + lignes).encode('utf-8-sig')

# --- BARRE LATÉRALE ---
st.sidebar.title("DATA FEED & SYSTEM")
if st.sidebar.button("Force Real-Time Refresh"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.title("MACRO RISK PARAMETERS")
budget = st.sidebar.number_input("Capital Allocation (EUR)", min_value=10.0, value=950.0, step=10.0)
seuil_vix = st.sidebar.slider("VIX Threshold (Cash Trigger)", 15, 40, 22)
vol_max = st.sidebar.slider("Max Weekly Volatility (%)", 30, 150, 60) / 100.0
dd_max = st.sidebar.slider("Max Historical Drawdown (%)", -80, -10, -45) / 100.0
correl_max = st.sidebar.slider("Covariance Rejection Limit (%)", 50, 95, 75) / 100.0

# --- TÉLÉCHARGEMENT MASSIF ---
with st.spinner(f'Processing {len(univers_etudie)} market instruments & running Monte Carlo...'):
    liste_tickers_bruts = [v["ticker"] for k, v in univers_etudie.items()]
    df_brut = telecharger_donnees(liste_tickers_bruts)

vix_actuel = float(df_brut['^VIX'].iloc[-1])
taux_fed_10y = float(df_brut['^TNX'].iloc[-1])

# --- NOUVEAU : FILTRE DE TENDANCE ABSOLUE (SMA 200) ---
sp500_close = float(df_brut['^GSPC'].iloc[-1])
sp500_sma200 = float(df_brut['^GSPC'].tail(200).mean())
regime_marche = "BULL MARKET" if sp500_close > sp500_sma200 else "BEAR MARKET"

# Logique de protection Cash agressive
if regime_marche == "BEAR MARKET":
    pourcentage_cash = 0.80 # 80% en cash si le marché s'effondre sous la SMA 200
elif vix_actuel > seuil_vix and taux_fed_10y > 4.5: 
    pourcentage_cash = 0.30
elif vix_actuel > seuil_vix: 
    pourcentage_cash = 0.20
else: 
    pourcentage_cash = 0.0

reserve_cash = budget * pourcentage_cash
budget_investissable = budget - reserve_cash

df_hebdo = df_brut.resample('W-FRI').last()
df_actifs = df_hebdo[liste_tickers_bruts].copy()

inv_map = {v["ticker"]: k for k, v in univers_etudie.items()}
df_actifs.rename(columns=inv_map, inplace=True)

# --- CALCULS QUANTITATIFS ---
rendements_hebdo = np.log(df_actifs / df_actifs.shift(1)).dropna(how='all')
volatilite = rendements_hebdo.rolling(window=52).std().iloc[-1] * np.sqrt(52)

rendements_negatifs = rendements_hebdo.copy()
rendements_negatifs[rendements_negatifs > 0] = 0
downside_vol = rendements_negatifs.std() * np.sqrt(52)
sortino_brut = (rendements_hebdo.mean() * 52) / downside_vol.replace(0, np.nan)

rendements_cumules = (1 + rendements_hebdo).cumprod()
sommet_historique = rendements_cumules.cummax()
drawdown = (rendements_cumules - sommet_historique) / sommet_historique
max_dd = drawdown.min()
correlation = rendements_hebdo.corr()

# --- NOUVEAU : TURNOVER PENALTY (HURDLE RATE 15%) ---
sortino_ajuste = sortino_brut.copy()
for actif in sortino_ajuste.index:
    if "S&P500:" in actif:
        # Les actions scannées subissent une pénalité de 15% pour compenser les frais
        sortino_ajuste[actif] = sortino_ajuste[actif] * 0.85 

# --- MOTEUR DE SÉLECTION ---
actifs_eligibles = []
raisons = {}
for actif in univers_etudie.keys():
    if actif not in volatilite.index or actif not in max_dd.index:
        raisons[actif] = "ERREUR DATA"
        continue
    if pd.isna(volatilite[actif]) or pd.isna(max_dd[actif]):
        raisons[actif] = "DATA INSUFFISANTE"
        continue
        
    vol = volatilite[actif]
    dd = max_dd[actif]
    if vol > vol_max: raisons[actif] = f"REJETÉ (Volatilité > {vol_max*100:.0f}%)"
    elif dd < dd_max: raisons[actif] = f"REJETÉ (Max DD < {dd_max*100:.0f}%)"
    elif pd.isna(sortino_ajuste[actif]): raisons[actif] = "DATA INSUFFISANTE"
    else: actifs_eligibles.append(actif)

sortino_eligibles = sortino_ajuste[actifs_eligibles].sort_values(ascending=False)
candidats = sortino_eligibles.index.tolist()
top_5_actifs = []

for candidat in candidats:
    if len(top_5_actifs) >= 5:
        raisons[candidat] = "FILTRÉ (Hors Top 5 Ajusté)"
        continue
    trop_correle = False
    for selectionne in top_5_actifs:
        if correlation.loc[candidat, selectionne] > correl_max:
            trop_correle = True
            raisons[candidat] = f"FILTRÉ (Corrélé à {selectionne})"
            break
    if not trop_correle:
        top_5_actifs.append(candidat)

if len(top_5_actifs) > 0:
    vol_top5 = volatilite[top_5_actifs]
    inv_vol = 1 / vol_top5
    poids = inv_vol / inv_vol.sum()
    allocations = poids * budget_investissable
else:
    allocations = pd.Series(dtype=float)

# --- BOUTONS EXPORT / DÉCONNEXION ---
st.sidebar.markdown("---")
if len(top_5_actifs) > 0:
    csv_data = generer_csv_europe(allocations, budget, reserve_cash, regime_marche)
    st.sidebar.download_button(label="Export Execution Order (CSV)", data=csv_data, file_name=f"Ordre_DCA_V18.csv", mime="text/csv")
if st.sidebar.button("Terminate Session"):
    st.session_state.authentifie = False
    st.rerun()

# --- INTERFACE PRINCIPALE ---
st.title("QUANTITATIVE ALLOCATION TERMINAL")

col1, col2, col3, col4 = st.columns(4)
couleur_regime = "normal" if regime_marche == "BULL MARKET" else "inverse"
col1.metric("Global Trend (SMA 200)", regime_marche, delta="Risk-On" if regime_marche == "BULL MARKET" else "Risk-Off", delta_color=couleur_regime)
col2.metric("Implied Volatility (VIX)", f"{vix_actuel:.1f}", delta="Alert" if vix_actuel > seuil_vix else "Stable", delta_color="inverse")
col3.metric("Processed Universe", f"{len(univers_etudie)} inst.", delta="Hurdle Rate 15% Active")
col4.metric("Defensive Cash Reserve", f"{reserve_cash:.2f} EUR", delta=f"{pourcentage_cash*100}% exposure")

tab1, tab2, tab3 = st.tabs(["ALLOCATION MATRIX", "MONTE CARLO (VaR 95%)", "COVARIANCE HEATMAP"])

with tab1:
    donnees_tableau = []
    actifs_a_afficher = list(dict.fromkeys(top_5_actifs + candidats[:45])) 
    
    for actif in actifs_a_afficher:
        statut = "ALLOUÉ" if actif in top_5_actifs else raisons.get(actif, "IGNORÉ")
        mnt = allocations[actif] if actif in top_5_actifs else 0.0
        
        nom_precis = univers_etudie[actif]["nom"]
        ticker = univers_etudie[actif]["ticker"]
        instrument_str = f"{nom_precis} [{ticker}]"
        
        donnees_tableau.append({
            "Instrument (Ticker)": instrument_str, 
            "Statut": statut, 
            "Sortino (Ajusté)": sortino_ajuste[actif], 
            "Max Drawdown": max_dd[actif]*100, 
            "Volatilité": volatilite[actif]*100, 
            "Allocation (EUR)": mnt
        })
        
    df_affichage = pd.DataFrame(donnees_tableau).sort_values(by="Allocation (EUR)", ascending=False)
    st.dataframe(df_affichage.style.format({"Sortino (Ajusté)": "{:.2f}", "Max Drawdown": "{:.1f}%", "Volatilité": "{:.1f}%", "Allocation (EUR)": "{:.2f}"}).applymap(lambda x: 'background-color: #1a4222; color: #ffffff;' if x == 'ALLOUÉ' else '', subset=['Statut']), use_container_width=True, height=500)

with tab2:
    st.markdown("### STOCHASTIC SIMULATION (1000 PATHS)")
    st.write("Simulation de Monte Carlo projetant la valeur de l'allocation mensuelle (Top 5) sur les 12 prochains mois (252 jours de bourse) en se basant sur la matrice de covariance historique.")
    
    if len(top_5_actifs) > 0 and budget_investissable > 0:
        # Variables pour Monte Carlo
        jours_simules = 252
        simulations = 1000
        
        # Rendements journaliers pour extraire mu et cov
        rendements_jour = np.log(df_brut[liste_tickers_bruts].tail(252) / df_brut[liste_tickers_bruts].tail(252).shift(1)).dropna()
        rendements_jour.rename(columns=inv_map, inplace=True)
        
        mu = rendements_jour[top_5_actifs].mean().values
        cov = rendements_jour[top_5_actifs].cov().values
        poids_mc = (allocations[top_5_actifs] / budget_investissable).values
        
        # Rendement et Volatilité du Portefeuille simulé
        port_mu = np.dot(poids_mc, mu)
        port_vol = np.sqrt(np.dot(poids_mc.T, np.dot(cov, poids_mc)))
        
        # Génération stochastique
        simulated_paths = np.zeros((jours_simules, simulations))
        simulated_paths[0] = budget_investissable
        
        for t in range(1, jours_simules):
            choc_aleatoire = np.random.normal(port_mu, port_vol, simulations)
            simulated_paths[t] = simulated_paths[t-1] * np.exp(choc_aleatoire)
            
        # Extraction des quantiles
        valeur_finale = simulated_paths[-1]
        var_95 = np.percentile(valeur_finale, 5) # Le pire des 5% (Value at Risk)
        mediane = np.percentile(valeur_finale, 50)
        
        # Création du graphique (On affiche 100 chemins pour la lisibilité)
        fig_mc = go.Figure()
        for i in range(100):
            fig_mc.add_trace(go.Scatter(y=simulated_paths[:, i], mode='lines', line=dict(color='rgba(100, 100, 100, 0.1)'), showlegend=False))
            
        # Lignes de probabilités
        fig_mc.add_hline(y=var_95, line_dash="dash", line_color="#ff4b4b", annotation_text=f"VaR 95% (Pire Scénario) : {var_95:.2f} €")
        fig_mc.add_hline(y=mediane, line_dash="dash", line_color="#4a90e2", annotation_text=f"Médiane attendue : {mediane:.2f} €")
        
        fig_mc.update_layout(title="Future Equity Distribution", xaxis_title="Jours (T+252)", yaxis_title="Valeur du Capital (EUR)", template="plotly_dark", height=500)
        st.plotly_chart(fig_mc, use_container_width=True)
        
        # Interprétation stricte de la VaR
        st.info(f"**Value at Risk (VaR 95%) :** Il y a 95% de probabilité mathématique que votre capital investi ({budget_investissable:.2f} €) ne descende pas en dessous de **{var_95:.2f} €** d'ici un an, en supposant que la volatilité du marché reste normale.")
    else:
        st.warning("Simulation impossible : 100% Cash ou données manquantes.")

with tab3:
    col_heat1, col_heat2 = st.columns([1, 5]) # Centrage
    with col_heat2:
        top_15 = sortino_eligibles.head(15).index.tolist()
        if len(top_15) > 1:
            fig_heat = px.imshow(correlation.loc[top_15, top_15], text_auto=".2f", color_continuous_scale="Greys", zmin=-1, zmax=1)
            st.plotly_chart(fig_heat, use_container_width=True)

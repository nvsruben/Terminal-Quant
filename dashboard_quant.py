import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import requests
from scipy.linalg import inv

# --- CONFIGURATION STREAMLIT ---
st.set_page_config(page_title="Terminal Quantitatif V19", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# SYSTÈME D'AUTHENTIFICATION STRICTE
# ==========================================
if "authentifie" not in st.session_state:
    st.session_state.authentifie = False

if not st.session_state.authentifie:
    st.title("QUANTITATIVE ALLOCATION TERMINAL")
    st.markdown("AUTHENTIFICATION REQUISE. ACCÈS RESTREINT (V19 - ML & Smart Beta).")
    MOT_DE_PASSE_SECRET = "BTSCG2026" 
    mdp_saisi = st.text_input("Passkey", type="password")
    if st.button("INITIALISER LA SESSION"):
        if mdp_saisi == MOT_DE_PASSE_SECRET:
            st.session_state.authentifie = True
            st.rerun()
        else:
            st.error("ACCÈS REFUSÉ.")
    st.stop()

# ==========================================
# UNIVERS D'INVESTISSEMENT
# ==========================================

MES_FAVORIS = {
    "Bitcoin": {"ticker": "BTC-EUR", "nom": "Bitcoin (Crypto)"},
    "Ethereum": {"ticker": "ETH-EUR", "nom": "Ethereum (Crypto)"},
    "Solana": {"ticker": "SOL-EUR", "nom": "Solana (Crypto)"},
    "Or Physique": {"ticker": "IGLN.L", "nom": "iShares Physical Gold ETC"},
    "Argent Physique": {"ticker": "PHAG.L", "nom": "WisdomTree Physical Silver"},
    "Copper": {"ticker": "CPER", "nom": "US Copper Index Fund"},
    "Uranium USD": {"ticker": "URNM", "nom": "Sprott Uranium Miners ETF"},
    "Defense USD": {"ticker": "DFNS.L", "nom": "VanEck Defense UCITS"},
    "Core S&P 500": {"ticker": "CSPX.AS", "nom": "iShares Core S&P 500 UCITS"},
    "Dassault Systèmes": {"ticker": "DSY.PA", "nom": "Dassault Systèmes SE"},
    "TotalEnergies": {"ticker": "TTE.PA", "nom": "TotalEnergies SE"},
    "LVMH": {"ticker": "MC.PA", "nom": "LVMH Moët Hennessy"}
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
with st.spinner(f'Processing {len(univers_etudie)} market instruments & running ML...'):
    liste_tickers_bruts = [v["ticker"] for k, v in univers_etudie.items()]
    df_brut = telecharger_donnees(liste_tickers_bruts)

vix_actuel = float(df_brut['^VIX'].iloc[-1])
taux_fed_10y = float(df_brut['^TNX'].iloc[-1])

sp500_close = float(df_brut['^GSPC'].iloc[-1])
sp500_sma200 = float(df_brut['^GSPC'].tail(200).mean())
regime_marche = "BULL MARKET" if sp500_close > sp500_sma200 else "BEAR MARKET"

if regime_marche == "BEAR MARKET": pourcentage_cash = 0.80 
elif vix_actuel > seuil_vix and taux_fed_10y > 4.5: pourcentage_cash = 0.30
elif vix_actuel > seuil_vix: pourcentage_cash = 0.20
else: pourcentage_cash = 0.0

reserve_cash = budget * pourcentage_cash
budget_investissable = budget - reserve_cash

df_hebdo = df_brut.resample('W-FRI').last()
df_actifs = df_hebdo[liste_tickers_bruts].copy()

inv_map = {v["ticker"]: k for k, v in univers_etudie.items()}
df_actifs.rename(columns=inv_map, inplace=True)

# --- CALCULS QUANTITATIFS (NIVEAU 1) ---
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

sortino_ajuste = sortino_brut.copy()
for actif in sortino_ajuste.index:
    if "S&P500:" in actif: sortino_ajuste[actif] = sortino_ajuste[actif] * 0.85 

# --- PRE-FILTRAGE MATHÉMATIQUE ---
actifs_pre_eligibles = []
raisons = {}
for actif in univers_etudie.keys():
    if actif not in volatilite.index or pd.isna(volatilite[actif]): continue
    vol = volatilite[actif]
    dd = max_dd[actif]
    if vol > vol_max: raisons[actif] = f"REJETÉ (Volatilité > {vol_max*100:.0f}%)"
    elif dd < dd_max: raisons[actif] = f"REJETÉ (Max DD < {dd_max*100:.0f}%)"
    else: actifs_pre_eligibles.append(actif)

# On isole les 15 meilleurs pour l'inspection fondamentale (Smart Beta)
top_15_candidats = sortino_ajuste[actifs_pre_eligibles].sort_values(ascending=False).head(15).index.tolist()
actifs_eligibles_finaux = []

# --- FILTRE FONDAMENTAL (SMART BETA) ---
with st.spinner('Running Fundamental Due Diligence on Top Candidates...'):
    for candidat in top_15_candidats:
        ticker_str = univers_etudie[candidat]["ticker"]
        # Les ETF et Cryptos n'ont pas de PER, on les laisse passer
        if "Crypto" in univers_etudie[candidat]["nom"] or "ETF" in univers_etudie[candidat]["nom"] or "UCITS" in univers_etudie[candidat]["nom"] or "ETC" in univers_etudie[candidat]["nom"] or "Fund" in univers_etudie[candidat]["nom"]:
            actifs_eligibles_finaux.append(candidat)
            continue
            
        try:
            info = yf.Ticker(ticker_str).info
            pe_ratio = info.get('trailingPE', 15) # 15 par défaut si introuvable
            
            if pe_ratio is None or pe_ratio < 0:
                raisons[candidat] = "FILTRÉ FONDAMENTAL (Entreprise en perte / PER négatif)"
            elif pe_ratio > 100:
                raisons[candidat] = f"FILTRÉ FONDAMENTAL (Bulle de valorisation : PER > 100)"
            else:
                actifs_eligibles_finaux.append(candidat)
        except:
            actifs_eligibles_finaux.append(candidat) # Sécurité anti-crash API

# --- MOTEUR DE SÉLECTION FINAL ---
top_5_actifs = []
for candidat in actifs_eligibles_finaux:
    if len(top_5_actifs) >= 5:
        raisons[candidat] = "FILTRÉ (Hors Top 5 Qualité)"
        continue
    trop_correle = False
    for selectionne in top_5_actifs:
        if correlation.loc[candidat, selectionne] > correl_max:
            trop_correle = True
            raisons[candidat] = f"FILTRÉ (Corrélé à {selectionne})"
            break
    if not trop_correle:
        top_5_actifs.append(candidat)

# --- NOUVEAU : ALLOCATION PAR INVERSION DE MATRICE (MINIMUM VARIANCE) ---
if len(top_5_actifs) > 0:
    try:
        # On extrait la matrice de covariance de nos 5 vainqueurs
        cov_matrix = rendements_hebdo[top_5_actifs].cov().values * 52
        inv_cov = inv(cov_matrix)
        ones = np.ones(len(top_5_actifs))
        # Formule de Markowitz pour minimiser la variance du portefeuille
        poids_optimaux = np.dot(inv_cov, ones) / np.dot(ones.T, np.dot(inv_cov, ones))
        
        # Sécurité : Si l'inversion donne des poids négatifs (Short), on repasse en Risk Parity standard
        if any(w < 0 for w in poids_optimaux):
            vol_top5 = volatilite[top_5_actifs]
            inv_vol = 1 / vol_top5
            poids_optimaux = inv_vol / inv_vol.sum()
    except:
        vol_top5 = volatilite[top_5_actifs]
        inv_vol = 1 / vol_top5
        poids_optimaux = inv_vol / inv_vol.sum()
        
    allocations = pd.Series(poids_optimaux * budget_investissable, index=top_5_actifs)
else:
    allocations = pd.Series(dtype=float)

# --- BOUTONS EXPORT / DÉCONNEXION ---
st.sidebar.markdown("---")
if len(top_5_actifs) > 0:
    csv_data = generer_csv_europe(allocations, budget, reserve_cash, regime_marche)
    st.sidebar.download_button(label="Export Execution Order (CSV)", data=csv_data, file_name=f"Ordre_DCA_V19.csv", mime="text/csv")
if st.sidebar.button("Terminate Session"):
    st.session_state.authentifie = False
    st.rerun()

# --- INTERFACE PRINCIPALE ---
st.title("QUANTITATIVE ALLOCATION TERMINAL")

col1, col2, col3, col4 = st.columns(4)
couleur_regime = "normal" if regime_marche == "BULL MARKET" else "inverse"
col1.metric("Global Trend (SMA 200)", regime_marche, delta="Risk-On" if regime_marche == "BULL MARKET" else "Risk-Off", delta_color=couleur_regime)
col2.metric("Implied Volatility (VIX)", f"{vix_actuel:.1f}", delta="Alert" if vix_actuel > seuil_vix else "Stable", delta_color="inverse")
col3.metric("Processed Universe", f"{len(univers_etudie)} inst.", delta="Smart Beta Filter Active")
col4.metric("Defensive Cash Reserve", f"{reserve_cash:.2f} EUR", delta=f"{pourcentage_cash*100}% exposure")

tab1, tab2, tab3 = st.tabs(["ALLOCATION MATRIX", "MONTE CARLO (VaR 95%)", "COVARIANCE HEATMAP"])

with tab1:
    st.markdown("*L'allocation est désormais calculée via l'inversion de la matrice de covariance (Markowitz Minimum Variance).*")
    donnees_tableau = []
    actifs_a_afficher = list(dict.fromkeys(top_5_actifs + actifs_pre_eligibles[:30])) 
    
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
    st.dataframe(df_affichage.style.format({"Sortino (Ajusté)": "{:.2f}", "Max Drawdown": "{:.1f}%", "Volatilité": "{:.1f}%", "Allocation (EUR)": "{:.2f}"}).applymap(lambda x: 'background-color: #1a4222; color: #ffffff;' if x == 'ALLOUÉ' else ('color: #ff4b4b;' if 'FONDAMENTAL' in str(x) else ''), subset=['Statut']), use_container_width=True, height=500)

with tab2:
    if len(top_5_actifs) > 0 and budget_investissable > 0:
        jours_simules = 252
        simulations = 1000
        rendements_jour = np.log(df_brut[liste_tickers_bruts].tail(252) / df_brut[liste_tickers_bruts].tail(252).shift(1)).dropna()
        rendements_jour.rename(columns=inv_map, inplace=True)
        mu = rendements_jour[top_5_actifs].mean().values
        cov = rendements_jour[top_5_actifs].cov().values
        poids_mc = (allocations[top_5_actifs] / budget_investissable).values
        
        port_mu = np.dot(poids_mc, mu)
        port_vol = np.sqrt(np.dot(poids_mc.T, np.dot(cov, poids_mc)))
        
        simulated_paths = np.zeros((jours_simules, simulations))
        simulated_paths[0] = budget_investissable
        for t in range(1, jours_simules):
            choc_aleatoire = np.random.normal(port_mu, port_vol, simulations)
            simulated_paths[t] = simulated_paths[t-1] * np.exp(choc_aleatoire)
            
        valeur_finale = simulated_paths[-1]
        var_95 = np.percentile(valeur_finale, 5) 
        mediane = np.percentile(valeur_finale, 50)
        
        fig_mc = go.Figure()
        for i in range(100): fig_mc.add_trace(go.Scatter(y=simulated_paths[:, i], mode='lines', line=dict(color='rgba(100, 100, 100, 0.1)'), showlegend=False))
        fig_mc.add_hline(y=var_95, line_dash="dash", line_color="#ff4b4b", annotation_text=f"VaR 95% : {var_95:.2f} EUR")
        fig_mc.add_hline(y=mediane, line_dash="dash", line_color="#4a90e2", annotation_text=f"Médiane : {mediane:.2f} EUR")
        fig_mc.update_layout(title="Stochastic Future Equity Distribution", xaxis_title="Jours (T+252)", yaxis_title="Capital (EUR)", template="plotly_dark", height=500)
        st.plotly_chart(fig_mc, use_container_width=True)
    else:
        st.warning("Simulation impossible : 100% Cash.")

with tab3:
    col_heat1, col_heat2 = st.columns([1, 5]) 
    with col_heat2:
        top_15 = sortino_eligibles.head(15).index.tolist()
        if len(top_15) > 1:
            fig_heat = px.imshow(correlation.loc[top_15, top_15], text_auto=".2f", color_continuous_scale="Greys", zmin=-1, zmax=1)
            st.plotly_chart(fig_heat, use_container_width=True)

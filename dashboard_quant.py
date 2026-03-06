import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import requests
from scipy.linalg import inv
from sklearn.covariance import ledoit_wolf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION STREAMLIT ---
st.set_page_config(page_title="QUANTITATIVE TERMINAL V28", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# SYSTEM AUTHENTICATION
# ==========================================
if "authentifie" not in st.session_state:
    st.session_state.authentifie = False

if not st.session_state.authentifie:
    st.markdown("<h1 style='text-align: center; color: #ffffff; font-family: monospace;'>QUANTITATIVE ALLOCATION DESK</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888888; font-family: monospace;'>RESTRICTED ACCESS. V28 (NLP SENTIMENT ENGINE).</p>", unsafe_allow_html=True)
    
    col_login1, col_login2, col_login3 = st.columns([1, 1, 1])
    with col_login2:
        MOT_DE_PASSE_SECRET = "BTSCG2026" 
        mdp_saisi = st.text_input("Enter Passkey", type="password")
        if st.button("INITIALIZE SESSION", use_container_width=True):
            if mdp_saisi == MOT_DE_PASSE_SECRET:
                st.session_state.authentifie = True
                st.rerun()
            else:
                st.error("ACCESS DENIED.")
    st.stop()

# ==========================================
# INVESTMENT UNIVERSE & NLP ENGINE
# ==========================================
MES_FAVORIS = {
    "Bitcoin": {"ticker": "BTC-EUR", "nom": "Bitcoin (Crypto)"},
    "Ethereum": {"ticker": "ETH-EUR", "nom": "Ethereum (Crypto)"},
    "Or Physique": {"ticker": "IGLN.L", "nom": "iShares Physical Gold ETC"},
    "Argent Physique": {"ticker": "PHAG.L", "nom": "WisdomTree Physical Silver"},
    "US Treasuries 20Y+": {"ticker": "TLT", "nom": "iShares 20+ Year Treasury Bond"},
    "Defense USD": {"ticker": "DFNS.L", "nom": "VanEck Defense UCITS"},
    "Rheinmetall": {"ticker": "RHM.DE", "nom": "Rheinmetall AG"},
    "Palantir": {"ticker": "PLTR", "nom": "Palantir Technologies"},
    "Uranium USD": {"ticker": "URNM", "nom": "Sprott Uranium Miners ETF"},
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
    tickers_complets = liste_tickers + ['^VIX', '^TNX', '^GSPC', '^IRX', 'HYG', 'IEF', 'GLD']
    df = yf.download(tickers_complets, period="10y", progress=False)['Close']
    df = df.ffill().bfill()
    return df

@st.cache_data(ttl=3600)
def analyser_sentiment_nlp():
    try:
        analyzer = SentimentIntensityAnalyzer()
        tickers_macro = ['^GSPC', 'TLT', 'GLD'] 
        toutes_les_news = []
        
        for t in tickers_macro:
            news = yf.Ticker(t).news
            if news: toutes_les_news.extend(news)
            
        if not toutes_les_news:
            return 0.0, pd.DataFrame()
            
        toutes_les_news = sorted(toutes_les_news, key=lambda x: x.get('providerPublishTime', 0), reverse=True)[:20]
        
        lignes_news = []
        scores_vader = []
        for item in toutes_les_news:
            titre = item.get('title', '')
            score = analyzer.polarity_scores(titre)['compound'] 
            scores_vader.append(score)
            
            timestamp = item.get('providerPublishTime', 0)
            date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M') if timestamp > 0 else "N/A"
            
            lignes_news.append({
                "Date": date_str,
                "Headline": titre,
                "VADER Score": score
            })
            
        sentiment_moyen = np.mean(scores_vader)
        return sentiment_moyen, pd.DataFrame(lignes_news)
    except Exception as e:
        return 0.0, pd.DataFrame()

def generer_csv_europe(df_ordres):
    date_jour = datetime.now().strftime("%d/%m/%Y")
    en_tetes = "Date;Instrument;Action_Requise;Montant_EUR\n"
    lignes = ""
    for _, row in df_ordres.iterrows():
        montant_str = str(round(abs(row["Order Delta (EUR)"]), 2)).replace('.', ',')
        lignes += f"{date_jour};{row['Instrument']};{row['Action']};{montant_str}\n"
    return (en_tetes + lignes).encode('utf-8-sig')

def calculate_z_score(series):
    if series.std() == 0: return np.zeros(len(series))
    return (series - series.mean()) / series.std()

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.markdown("<h3 style='font-family: monospace;'>SYSTEM CONTROLS</h3>", unsafe_allow_html=True)
if st.sidebar.button("FORCE REAL-TIME REFRESH", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='font-family: monospace;'>PORTFOLIO LIMITS</h3>", unsafe_allow_html=True)
budget = st.sidebar.number_input("Total Capital (EUR)", min_value=10.0, value=950.0, step=10.0)
seuil_vix = st.sidebar.slider("VIX Threshold", 15, 40, 22) # CORRECTIF : Remis en place !
target_volatility = st.sidebar.slider("Target Annual Volatility (%)", 5, 25, 12) / 100.0
min_trade_size = st.sidebar.slider("Minimum Trade Size (EUR)", 10, 100, 50)
turnover_penalty = st.sidebar.slider("Hurdle Rate (Turnover Penalty %)", 5, 30, 15) / 100.0
correl_max = st.sidebar.slider("Base Covariance Limit (%)", 50, 95, 75) / 100.0
max_weight_limit = 0.25

# --- INIT PORTFOLIO STATE ---
if 'mon_portefeuille' not in st.session_state:
    st.session_state.mon_portefeuille = pd.DataFrame({
        "Actif": ["Core S&P 500", "Bitcoin", "Or Physique", "Palantir", "Argent Physique", "Uranium USD", "Rheinmetall"],
        "Valeur (EUR)": [401.43, 295.83, 114.86, 55.27, 42.54, 10.78, 9.47],
        "🔒 Core (Ne pas vendre)": [True, True, True, False, False, False, False]
    })

# --- CORE ENGINE EXECUTION ---
with st.spinner(f'Executing Alternative Data Models (NLP & K-Means)...'):
    liste_tickers_bruts = [v["ticker"] for k, v in univers_etudie.items()]
    df_brut = telecharger_donnees(liste_tickers_bruts)
    avg_nlp_score, df_headlines = analyser_sentiment_nlp()

# --- 1. AI REGIME & MACRO DETECTION ---
df_ml = pd.DataFrame({
    'VIX': df_brut['^VIX'],
    'Yield_Spread': df_brut['^TNX'] - df_brut['^IRX'],
    'Credit_Spread': df_brut['HYG'] / df_brut['IEF'],
    'SP500_Ret': df_brut['^GSPC'].pct_change()
}).dropna()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_ml)
kmeans = KMeans(n_clusters=3, random_state=42).fit(scaled_data)
df_ml['Cluster'] = kmeans.labels_

cluster_vix_means = df_ml.groupby('Cluster')['VIX'].mean()
bull_cluster = cluster_vix_means.idxmin()
bear_cluster = cluster_vix_means.idxmax()
current_cluster = kmeans.predict(scaled_data[-1].reshape(1, -1))[0]

# Initialisation du Risk Score via le NLP
risk_score = 0
if avg_nlp_score < -0.15: risk_score += 20 # Malus Panique Médiatique
elif avg_nlp_score > 0.15: risk_score -= 10 # Bonus Euphorie

vix_actuel = float(df_brut['^VIX'].iloc[-1])
sp500_close = float(df_brut['^GSPC'].iloc[-1])
sp500_sma200 = float(df_brut['^GSPC'].tail(200).mean())
if sp500_close < sp500_sma200: risk_score += 40
if float(df_brut['^TNX'].iloc[-1]) - float(df_brut['^IRX'].iloc[-1]) < 0.0: risk_score += 30
if float(df_brut['HYG'].iloc[-1]/df_brut['IEF'].iloc[-1]) < float((df_brut['HYG']/df_brut['IEF']).tail(50).mean()): risk_score += 30
if vix_actuel > seuil_vix: risk_score += 20

risk_score = max(0, min(100, risk_score)) # Bloquer entre 0 et 100

if risk_score >= 70 or current_cluster == bear_cluster:
    regime_marche = "CRITICAL BEAR (AI & NLP)"
    tail_hedge_active = True
elif risk_score >= 40:
    regime_marche = "DEFENSIVE REGIME"
    tail_hedge_active = False
else:
    regime_marche = "BULL REGIME"
    tail_hedge_active = False

dynamic_correl_max = correl_max
if tail_hedge_active or risk_score >= 40:
    dynamic_correl_max = min(correl_max, 0.55) 

# --- DATA PREPARATION ---
df_hebdo = df_brut.tail(252*3).resample('W-FRI').last() 
df_actifs = df_hebdo[liste_tickers_bruts].copy()
inv_map = {v["ticker"]: k for k, v in univers_etudie.items()}
df_actifs.rename(columns=inv_map, inplace=True)

# --- QUANTITATIVE CALCULATIONS ---
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

rendements_propres = rendements_hebdo.dropna()
lw_cov_brut, shrinkage_penalty = ledoit_wolf(rendements_propres)
correlation = pd.DataFrame(lw_cov_brut, index=rendements_propres.columns, columns=rendements_propres.columns).corr()

sortino_ajuste = sortino_brut.copy()
for actif in sortino_ajuste.index:
    if "S&P500:" in actif: sortino_ajuste[actif] = sortino_ajuste[actif] * (1.0 - turnover_penalty) 

# --- PRE-FILTERING ---
actifs_pre_eligibles = []
raisons = {}
for actif in univers_etudie.keys():
    if actif not in volatilite.index or pd.isna(volatilite[actif]): continue
    if actif == "US Treasuries 20Y+": continue 
    vol = volatilite[actif]
    dd = max_dd[actif]
    if vol > 0.60: raisons[actif] = f"REJECTED (Vol > 60%)"
    elif dd < -0.45: raisons[actif] = f"REJECTED (Drawdown < -45%)"
    else: actifs_pre_eligibles.append(actif)

top_20_candidats = sortino_ajuste[actifs_pre_eligibles].sort_values(ascending=False).head(20).index.tolist()

# --- 2. MULTI-FACTOR Z-SCORE ENGINE ---
fundamentals_data = []
for candidat in top_20_candidats:
    ticker_str = univers_etudie[candidat]["ticker"]
    is_etf_or_crypto = any(kw in univers_etudie[candidat]["nom"] for kw in ["Crypto", "ETF", "UCITS", "ETC", "Fund", "AG"])
    
    prix_actuel = float(df_brut[ticker_str].iloc[-1]) if ticker_str in df_brut.columns else 0
    sma_200 = float(df_brut[ticker_str].tail(200).mean()) if ticker_str in df_brut.columns else prix_actuel
    trend_ratio = (prix_actuel / sma_200) - 1.0 if sma_200 > 0 else 0
    
    if is_etf_or_crypto:
        fundamentals_data.append({"Actif": candidat, "PE": 15.0, "ROE": 0.15, "Consensus": 3.0, "Sortino": sortino_ajuste[candidat], "Trend": trend_ratio})
        continue
    try:
        info = yf.Ticker(ticker_str).info
        pe = info.get('trailingPE', 15)
        roe = info.get('returnOnEquity', 0.15)
        consensus = info.get('recommendationMean', 3.0)
        
        if pe is None or pe < 0 or pe > 100:
            raisons[candidat] = "FUNDAMENTAL REJECT (Negative/Bubble PE)"
            continue
        if roe is None: roe = 0.10
        if consensus is None: consensus = 3.0
            
        fundamentals_data.append({"Actif": candidat, "PE": pe, "ROE": roe, "Consensus": consensus, "Sortino": sortino_ajuste[candidat], "Trend": trend_ratio})
    except:
        fundamentals_data.append({"Actif": candidat, "PE": 15.0, "ROE": 0.10, "Consensus": 3.0, "Sortino": sortino_ajuste[candidat], "Trend": trend_ratio})

df_zscore = pd.DataFrame(fundamentals_data)
if not df_zscore.empty:
    df_zscore['Z_PE'] = -calculate_z_score(df_zscore['PE']) 
    df_zscore['Z_ROE'] = calculate_z_score(df_zscore['ROE']) 
    df_zscore['Z_Sortino'] = calculate_z_score(df_zscore['Sortino'])
    df_zscore['Z_Consensus'] = -calculate_z_score(df_zscore['Consensus'])
    df_zscore['Z_Trend'] = calculate_z_score(df_zscore['Trend'])
    df_zscore['Global_Score'] = df_zscore['Z_PE'].fillna(0) + df_zscore['Z_ROE'].fillna(0) + df_zscore['Z_Sortino'].fillna(0) + df_zscore['Z_Consensus'].fillna(0) + df_zscore['Z_Trend'].fillna(0)
    actifs_eligibles_finaux = df_zscore.sort_values(by="Global_Score", ascending=False)['Actif'].tolist()
else:
    actifs_eligibles_finaux = []

top_5_actifs = []
for candidat in actifs_eligibles_finaux:
    if len(top_5_actifs) >= 5: break
    trop_correle = False
    for selectionne in top_5_actifs:
        if correlation.loc[candidat, selectionne] > dynamic_correl_max:
            trop_correle = True
            raisons[candidat] = f"FILTERED (Adaptive Covariance w/ {selectionne})"
            break
    if not trop_correle: top_5_actifs.append(candidat)

# --- 3. CORE-SATELLITE & BLACK-LITTERMAN ALLOCATION ---
capital_verrouille = 0.0
actifs_verrouilles = []
if not st.session_state.mon_portefeuille.empty:
    for _, row in st.session_state.mon_portefeuille.iterrows():
        if row.get("🔒 Core (Ne pas vendre)", False) and row["Actif"] in univers_etudie:
            capital_verrouille += row["Valeur (EUR)"]
            actifs_verrouilles.append(row["Actif"])

budget_satellite = budget - capital_verrouille

port_vol_initial = 0.0
expected_portfolio_return = 0.0
if len(top_5_actifs) > 0 and budget_satellite > 0:
    try:
        tau = 0.05
        rendements_top5 = rendements_propres[top_5_actifs]
        lw_cov_top5, _ = ledoit_wolf(rendements_top5)
        cov_matrix = lw_cov_top5 * 52
        
        inv_vol_bl = 1 / volatilite[top_5_actifs].values
        w_eq = inv_vol_bl / inv_vol_bl.sum()
        Pi = 2.5 * np.dot(cov_matrix, w_eq)
        rendements_3m = np.log(df_actifs[top_5_actifs].iloc[-1] / df_actifs[top_5_actifs].iloc[-13]).values
        P = np.eye(len(top_5_actifs))
        Q = rendements_3m * 4 
        Omega = np.diag(np.diag(cov_matrix)) * tau
        
        inv_tau_cov = inv(tau * cov_matrix)
        inv_Omega = inv(Omega)
        term1 = inv(inv_tau_cov + np.dot(np.dot(P.T, inv_Omega), P))
        term2 = np.dot(inv_tau_cov, Pi) + np.dot(np.dot(P.T, inv_Omega), Q)
        BL_returns = np.dot(term1, term2)
        
        poids_optimaux = np.dot(inv(cov_matrix), BL_returns)
        poids_optimaux = np.clip(poids_optimaux, 0, None)
        if poids_optimaux.sum() == 0: poids_optimaux = w_eq
        else: poids_optimaux = poids_optimaux / poids_optimaux.sum()

        while any(poids_optimaux > max_weight_limit + 1e-5):
            excess = sum(poids_optimaux[poids_optimaux > max_weight_limit] - max_weight_limit)
            poids_optimaux[poids_optimaux > max_weight_limit] = max_weight_limit
            mask = poids_optimaux < max_weight_limit
            if sum(mask) > 0: poids_optimaux[mask] += excess * (poids_optimaux[mask] / sum(poids_optimaux[mask]))
            else: break
                
    except Exception as e:
        vol_top5 = volatilite[top_5_actifs]
        poids_optimaux = (1/vol_top5) / (1/vol_top5).sum()
        lw_cov_top5, _ = ledoit_wolf(rendements_propres[top_5_actifs])
        cov_matrix = lw_cov_top5 * 52
        
    port_vol_initial = np.sqrt(np.dot(poids_optimaux.T, np.dot(cov_matrix, poids_optimaux)))
    expected_portfolio_return = np.dot(poids_optimaux, (rendements_hebdo[top_5_actifs].mean() * 52).values)
    
    exposure_factor = target_volatility / port_vol_initial if port_vol_initial > 0 else 1.0
    exposure_factor = min(1.0, exposure_factor) 
    
    pourcentage_cash = 1.0 - exposure_factor
    reserve_cash = budget * pourcentage_cash
    budget_satellite_ajuste = max(0, budget_satellite - reserve_cash)
    
    if tail_hedge_active:
        budget_tail_risk = budget_satellite_ajuste * 0.30
        budget_satellite_ajuste -= budget_tail_risk
        
    allocations_satellite = pd.Series(poids_optimaux * budget_satellite_ajuste, index=top_5_actifs)
else:
    allocations_satellite = pd.Series(dtype=float)
    reserve_cash = budget_satellite
    budget_tail_risk = 0

allocations = pd.Series(dtype=float)
for _, row in st.session_state.mon_portefeuille.iterrows():
    if row.get("🔒 Core (Ne pas vendre)", False) and row["Actif"] in univers_etudie:
        allocations[row["Actif"]] = row["Valeur (EUR)"]
for actif, val in allocations_satellite.items():
    allocations[actif] = allocations.get(actif, 0.0) + val
if tail_hedge_active:
    allocations["US Treasuries 20Y+"] = allocations.get("US Treasuries 20Y+", 0.0) + budget_tail_risk

# --- MAIN DASHBOARD ---
st.markdown("<h2 style='font-family: monospace; border-bottom: 1px solid #444; padding-bottom: 10px;'>PRIME BROKER DESK (NLP + AI)</h2>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("SYSTEMIC RISK SCORE", f"{risk_score}/100", delta=regime_marche, delta_color="normal" if risk_score < 40 else "inverse")

sentiment_label = "Neutral"
sentiment_color = "off"
if avg_nlp_score > 0.15: 
    sentiment_label = "Euphoria"
    sentiment_color = "normal"
elif avg_nlp_score < -0.15: 
    sentiment_label = "Fear/Panic"
    sentiment_color = "inverse"

col2.metric("NLP NEWS SENTIMENT", f"{avg_nlp_score:.2f}", delta=sentiment_label, delta_color=sentiment_color)
col3.metric("TACTICAL SATELLITE", f"{budget_satellite:.2f} EUR", delta=f"Covariance Limit: {dynamic_correl_max*100:.0f}%", delta_color="normal")
col4.metric("VOL-TARGETED CASH", f"{reserve_cash:.2f} EUR", delta=f"{(reserve_cash/budget)*100:.1f}% dynamic weight", delta_color="off")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["ORDER MANAGEMENT SYSTEM (OMS)", "TARGET ALLOCATION MATRIX", "MACRO & NLP SENTIMENT", "FACTOR EXPOSURE", "HISTORICAL STRESS TESTS"])

with tab1:
    st.markdown("<h4 style='font-family: monospace;'>SMART ORDER MANAGEMENT SYSTEM</h4>", unsafe_allow_html=True)
    col_ed1, col_ed2 = st.columns([1, 1.5])
    with col_ed1:
        st.markdown("**1. Vos Positions Actuelles**")
        config_colonnes = {
            "Actif": st.column_config.SelectboxColumn("Instrument Détenu", options=list(univers_etudie.keys()), required=True),
            "Valeur (EUR)": st.column_config.NumberColumn("Valeur Actuelle (€)", min_value=0.0, step=10.0, format="%.2f"),
            "🔒 Core (Ne pas vendre)": st.column_config.CheckboxColumn("🔒 Core (Ne pas vendre)", default=False)
        }
        df_edited = st.data_editor(st.session_state.mon_portefeuille, column_config=config_colonnes, num_rows="dynamic", use_container_width=True, key="editor_portefeuille")
        st.session_state.mon_portefeuille = df_edited

    with col_ed2:
        st.markdown("**2. Ticket d'Exécution Généré (Optimisé Frais)**")
        if len(allocations) > 0:
            dict_actuel = {}
            for _, row in df_edited.iterrows():
                if pd.notna(row["Actif"]) and row["Actif"] != "":
                    dict_actuel[row["Actif"]] = dict_actuel.get(row["Actif"], 0.0) + row["Valeur (EUR)"]
            
            tous_les_actifs = list(set(allocations[allocations > 0].index.tolist() + list(dict_actuel.keys())))
            lignes_ordres = []
            for a in tous_les_actifs:
                val_cible = allocations.get(a, 0.0)
                val_actuelle = dict_actuel.get(a, 0.0)
                delta = val_cible - val_actuelle
                
                if abs(delta) >= min_trade_size: 
                    action = "BUY" if delta > 0 else "SELL"
                    nom_complet = univers_etudie[a]["nom"] if a in univers_etudie else a
                    ticker = univers_etudie[a]["ticker"] if a in univers_etudie else ""
                    lignes_ordres.append({"Instrument": f"{nom_complet} [{ticker}]", "Target": val_cible, "Current": val_actuelle, "Order Delta (EUR)": delta, "Action": action})
                elif delta != 0:
                    nom_complet = univers_etudie[a]["nom"] if a in univers_etudie else a
                    ticker = univers_etudie[a]["ticker"] if a in univers_etudie else ""
                    lignes_ordres.append({"Instrument": f"{nom_complet} [{ticker}]", "Target": val_cible, "Current": val_actuelle, "Order Delta (EUR)": delta, "Action": "HOLD (Too Small)"})
            
            if len(lignes_ordres) > 0:
                df_ordres = pd.DataFrame(lignes_ordres).sort_values(by="Order Delta (EUR)", ascending=False)
                st.dataframe(df_ordres.style.format({"Target": "{:.2f} €", "Current": "{:.2f} €", "Order Delta (EUR)": "{:+.2f} €"}).applymap(lambda x: 'color: #00cc00; font-weight: bold;' if x == 'BUY' else ('color: #ff4b4b; font-weight: bold;' if x == 'SELL' else 'color: #888888;'), subset=['Action']), use_container_width=True, height=400)
            else:
                st.success("✅ Portefeuille aligné. Aucun ordre majeur requis.")

with tab2:
    st.markdown("<div style='background-color: #1e1e1e; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>", unsafe_allow_html=True)
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    expected_sharpe = expected_portfolio_return / port_vol_initial if port_vol_initial > 0 else 0
    col_kpi1.metric("Expected Annual Return", f"{expected_portfolio_return*100:.1f}%")
    col_kpi2.metric("Portfolio Sharpe Ratio", f"{expected_sharpe:.2f}")
    col_kpi3.metric("Global Z-Score Avg", f"{np.mean([df_zscore.loc[df_zscore['Actif'] == a, 'Global_Score'].values[0] for a in top_5_actifs if a in df_zscore['Actif'].values]):.2f}" if len(top_5_actifs)>0 and not df_zscore.empty else "N/A")
    col_kpi4.metric("Assets Allocated", f"{len(allocations[allocations > 0])}")
    st.markdown("</div>", unsafe_allow_html=True)

    donnees_tableau = []
    actifs_a_afficher = list(dict.fromkeys(list(allocations.index) + top_20_candidats[:15])) 
    
    for actif in actifs_a_afficher:
        if actif in allocations.index and allocations[actif] > 0:
            if actif in actifs_verrouilles: statut = "LOCKED (CORE)"
            else: statut = "ALLOCATED (SATELLITE)"
        else: statut = raisons.get(actif, "REJECTED")
        
        mnt = allocations.get(actif, 0.0)
        instrument_str = f"{univers_etudie[actif]['nom']} [{univers_etudie[actif]['ticker']}]"
        z_val = df_zscore.loc[df_zscore['Actif'] == actif, 'Global_Score'].values[0] if not df_zscore.empty and actif in df_zscore['Actif'].values else 0
        
        donnees_tableau.append({
            "Instrument (Ticker)": instrument_str, "Status": statut, "Z-Score": f"{z_val:.2f}" if z_val != 0 else "N/A",
            "Max Drawdown": f"{max_dd.get(actif, 0)*100:.1f}%", "Volatility": f"{volatilite.get(actif, 0)*100:.1f}%", "Target Capital": mnt
        })
        
    df_affichage = pd.DataFrame(donnees_tableau).sort_values(by="Target Capital", ascending=False)
    
    def styler_status(val):
        if 'LOCKED' in str(val): return 'background-color: #2c3e50; color: #ffffff;'
        elif 'ALLOCATED' in str(val): return 'background-color: #1a4222; color: #ffffff;'
        elif 'REJECT' in str(val): return 'color: #8b0000;'
        return ''
        
    st.dataframe(df_affichage.style.format({"Target Capital": "{:.2f} €"}).applymap(styler_status, subset=['Status']), use_container_width=True, height=450)

with tab3:
    st.markdown("<h4 style='font-family: monospace;'>ALTERNATIVE DATA & MACRO MONITORS</h4>", unsafe_allow_html=True)
    
    if not df_headlines.empty:
        st.markdown("**Real-Time Natural Language Processing (News Sentiment)**")
        st.write("Le moteur VADER analyse les derniers titres financiers mondiaux pour détecter la panique ou l'euphorie.")
        
        def styler_vader(val):
            if isinstance(val, float):
                if val > 0.15: return 'color: #00cc00;'
                elif val < -0.15: return 'color: #ff4b4b;'
            return 'color: #888888;'
            
        st.dataframe(df_headlines.style.applymap(styler_vader, subset=['VADER Score']).format({"VADER Score": "{:.2f}"}), use_container_width=True, height=250)
    else:
        st.info("Aucune actualité macroéconomique détectée dans l'heure.")

    st.markdown("---")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("<span style='color: #888;'>Yield Curve (10Y - 3M Spread)</span>", unsafe_allow_html=True)
        df_yield = pd.DataFrame({"Spread (%)": (df_brut['^TNX'] - df_brut['^IRX']).tail(252)})
        st.line_chart(df_yield, color="#8b0000" if curve_inverted else "#4a90e2")
    with col_m2:
        st.markdown("<span style='color: #888;'>Interbank Credit Stress (HYG/IEF)</span>", unsafe_allow_html=True)
        df_credit = pd.DataFrame({"Ratio": df_brut['Credit_Spread'].tail(252)})
        st.line_chart(df_credit, color="#8b0000" if credit_stress else "#4a90e2")

with tab4:
    betas = {"Equity Beta (GSPC)": 0.0, "Duration Beta (IEF)": 0.0, "Commodity Beta (GLD)": 0.0}
    if len(allocations) > 0:
        ret_market = df_brut['^GSPC'].pct_change().dropna()
        ret_bonds = df_brut['IEF'].pct_change().dropna()
        ret_gold = df_brut['GLD'].pct_change().dropna()
        poids_beta = (allocations / budget).fillna(0)
        ret_port_beta = (df_brut[ [univers_etudie[a]["ticker"] for a in allocations.index] ].pct_change().dropna() * poids_beta.values).sum(axis=1)
        df_beta = pd.DataFrame({"Portfolio": ret_port_beta, "Market": ret_market, "Bonds": ret_bonds, "Gold": ret_gold}).dropna()
        if not df_beta.empty:
            betas["Equity Beta (GSPC)"] = df_beta["Portfolio"].cov(df_beta["Market"]) / df_beta["Market"].var()
            betas["Duration Beta (IEF)"] = df_beta["Portfolio"].cov(df_beta["Bonds"]) / df_beta["Bonds"].var()
            betas["Commodity Beta (GLD)"] = df_beta["Portfolio"].cov(df_beta["Gold"]) / df_beta["Gold"].var()

    col_beta1, col_beta2 = st.columns([1, 2])
    with col_beta1:
        st.metric("Equity Beta (vs S&P 500)", f"{betas['Equity Beta (GSPC)']:.2f}")
        st.metric("Duration Beta (vs US Bonds)", f"{betas['Duration Beta (IEF)']:.2f}")
        st.metric("Commodity Beta (vs Gold)", f"{betas['Commodity Beta (GLD)']:.2f}")
    with col_beta2:
        df_factors = pd.DataFrame({"Factor": ["Global Equities (Risk-On)", "US Treasury Bonds (Rates)", "Physical Gold (Inflation)"], "Beta Exposure": [betas['Equity Beta (GSPC)'], betas['Duration Beta (IEF)'], betas['Commodity Beta (GLD)']]})
        fig_bar = px.bar(df_factors, x="Beta Exposure", y="Factor", orientation='h', color="Beta Exposure", color_continuous_scale="RdBu", range_color=[-1, 1])
        fig_bar.update_layout(template="plotly_dark", margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_bar, use_container_width=True)

with tab5:
    if len(allocations) > 0:
        poids_test = (pd.Series(allocations) / budget).fillna(0)
        actifs_testes = poids_test[poids_test > 0].index.tolist()
        if len(actifs_testes) > 0:
            df_history = df_brut.copy()
            
            df_covid_brut = df_history.loc['2020-02-15':'2020-04-30']
            actifs_valides_covid = [a for a in actifs_testes if df_covid_brut[univers_etudie[a]["ticker"]].isna().sum() < 5]
            poids_covid = poids_test[actifs_valides_covid]
            if len(actifs_valides_covid) > 0 and poids_covid.sum() > 0:
                poids_covid = poids_covid / poids_covid.sum() 
                colonnes_covid = [univers_etudie[a]["ticker"] for a in actifs_valides_covid] + ['^GSPC']
                ret_covid = df_covid_brut[colonnes_covid].pct_change().dropna()
                port_covid = (ret_covid[ [univers_etudie[a]["ticker"] for a in actifs_valides_covid] ] * poids_covid.values).sum(axis=1)
                sp_covid = ret_covid['^GSPC']
                croissance_port_covid = (1 + port_covid).cumprod() * 100
                croissance_sp_covid = (1 + sp_covid).cumprod() * 100
                df_graph_covid = pd.DataFrame({"Proprietary Engine": croissance_port_covid, "S&P 500 Benchmark": croissance_sp_covid})
                col_st1, col_st2 = st.columns(2)
                with col_st1:
                    st.markdown("**SCENARIO A: COVID-19 Liquidity Crisis (Feb-Apr 2020)**")
                    fig_covid = px.line(df_graph_covid, color_discrete_sequence=['#4a90e2', '#444444'])
                    fig_covid.update_layout(template="plotly_dark", margin=dict(t=10, b=10, l=10, r=10), xaxis_title="", yaxis_title="Base 100")
                    st.plotly_chart(fig_covid, use_container_width=True)
                    st.caption(f"Max Drawdown Engine: **{(croissance_port_covid.min() - 100):.1f}%**")
            else:
                col_st1, col_st2 = st.columns(2)
                with col_st1: st.warning("Insufficient historical data for 2020.")
                    
            df_inf_brut = df_history.loc['2022-01-01':'2022-10-31']
            actifs_valides_inf = [a for a in actifs_testes if df_inf_brut[univers_etudie[a]["ticker"]].isna().sum() < 5]
            poids_inf = poids_test[actifs_valides_inf]
            if len(actifs_valides_inf) > 0 and poids_inf.sum() > 0:
                poids_inf = poids_inf / poids_inf.sum()
                colonnes_inf = [univers_etudie[a]["ticker"] for a in actifs_valides_inf] + ['^GSPC']
                ret_inf = df_inf_brut[colonnes_inf].pct_change().dropna()
                port_inf = (ret_inf[ [univers_etudie[a]["ticker"] for a in actifs_valides_inf] ] * poids_inf.values).sum(axis=1)
                sp_inf = ret_inf['^GSPC']
                croissance_port_inf = (1 + port_inf).cumprod() * 100
                croissance_sp_inf = (1 + sp_inf).cumprod() * 100
                df_graph_inf = pd.DataFrame({"Proprietary Engine": croissance_port_inf, "S&P 500 Benchmark": croissance_sp_inf})
                with col_st2:
                    st.markdown("**SCENARIO B: Interest Rate Shock (Jan-Oct 2022)**")
                    fig_inf = px.line(df_graph_inf, color_discrete_sequence=['#4a90e2', '#444444'])
                    fig_inf.update_layout(template="plotly_dark", margin=dict(t=10, b=10, l=10, r=10), xaxis_title="", yaxis_title="")
                    st.plotly_chart(fig_inf, use_container_width=True)
                    st.caption(f"Max Drawdown Engine: **{(croissance_port_inf.min() - 100):.1f}%**")
            else:
                with col_st2: st.warning("Insufficient historical data for 2022.")

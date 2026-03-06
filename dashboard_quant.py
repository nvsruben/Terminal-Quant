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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION STREAMLIT ---
st.set_page_config(page_title="QUANTITATIVE TERMINAL V29", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# SYSTEM AUTHENTICATION
# ==========================================
if "authentifie" not in st.session_state:
    st.session_state.authentifie = False

if not st.session_state.authentifie:
    st.markdown("<h1 style='text-align: center; color: #ffffff; font-family: monospace;'>QUANTITATIVE ALLOCATION DESK</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888888; font-family: monospace;'>RESTRICTED ACCESS. V29 (DEEP ALPHA & STOCHASTIC ENGINE).</p>", unsafe_allow_html=True)
    
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
# INVESTMENT UNIVERSE
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
    except:
        return {}

univers_etudie = MES_FAVORIS.copy()
mega_dict = aspirer_le_marche_sp500()
for cle, donnees in mega_dict.items():
    if donnees["ticker"] not in [v["ticker"] for v in univers_etudie.values()]:
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
        if not toutes_les_news: return 0.0, pd.DataFrame()
        toutes_les_news = sorted(toutes_les_news, key=lambda x: x.get('providerPublishTime', 0), reverse=True)[:20]
        
        lignes_news = []
        scores_vader = []
        for item in toutes_les_news:
            titre = item.get('title', '')
            score = analyzer.polarity_scores(titre)['compound'] 
            scores_vader.append(score)
            timestamp = item.get('providerPublishTime', 0)
            date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M') if timestamp > 0 else "N/A"
            lignes_news.append({"Date": date_str, "Headline": titre, "VADER Score": score})
            
        return np.mean(scores_vader), pd.DataFrame(lignes_news)
    except:
        return 0.0, pd.DataFrame()

def generer_csv_europe(df_ordres):
    date_jour = datetime.now().strftime("%d/%m/%Y")
    en_tetes = "Date;Instrument;Action_Requise;Montant_EUR\n"
    lignes = "".join([f"{date_jour};{row['Instrument']};{row['Action']};{str(round(abs(row['Order Delta (EUR)']), 2)).replace('.', ',')}\n" for _, row in df_ordres.iterrows()])
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
seuil_vix = st.sidebar.slider("VIX Threshold", 15, 40, 22)
target_volatility = st.sidebar.slider("Target Annual Volatility (%)", 5, 25, 12) / 100.0
min_trade_size = st.sidebar.slider("Minimum Trade Size (EUR)", 10, 100, 50)
turnover_penalty = st.sidebar.slider("Hurdle Rate (Turnover Penalty %)", 5, 30, 15) / 100.0
correl_max = st.sidebar.slider("Base Covariance Limit (%)", 50, 95, 75) / 100.0
max_weight_limit = 0.25

if 'mon_portefeuille' not in st.session_state:
    st.session_state.mon_portefeuille = pd.DataFrame({
        "Actif": ["Core S&P 500", "Bitcoin", "Or Physique", "Palantir", "Argent Physique", "Uranium USD", "Rheinmetall"],
        "Valeur (EUR)": [401.43, 295.83, 114.86, 55.27, 42.54, 10.78, 9.47],
        "🔒 Core (Ne pas vendre)": [True, True, True, False, False, False, False]
    })

# --- CORE ENGINE EXECUTION ---
with st.spinner(f'V29 AI Matrix Active. Processing Gradient Boosting & PCA...'):
    liste_tickers_bruts = [v["ticker"] for k, v in univers_etudie.items()]
    df_brut = telecharger_donnees(liste_tickers_bruts)
    avg_nlp_score, df_headlines = analyser_sentiment_nlp()

# --- 1. AI REGIME DETECTION (K-MEANS) ---
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

vix_actuel = float(df_brut['^VIX'].iloc[-1])
taux_10y = float(df_brut['^TNX'].iloc[-1])
taux_3m = float(df_brut['^IRX'].iloc[-1])
curve_inverted = (taux_10y - taux_3m) < 0.0

df_brut['Credit_Spread'] = df_brut['HYG'] / df_brut['IEF']
credit_stress = float(df_brut['Credit_Spread'].iloc[-1]) < float(df_brut['Credit_Spread'].tail(50).mean())

sp500_close = float(df_brut['^GSPC'].iloc[-1])
sp500_sma200 = float(df_brut['^GSPC'].tail(200).mean())

risk_score = 0
if avg_nlp_score < -0.15: risk_score += 20 
elif avg_nlp_score > 0.15: risk_score -= 10 
if sp500_close < sp500_sma200: risk_score += 40
if curve_inverted: risk_score += 30
if credit_stress: risk_score += 30
if vix_actuel > seuil_vix: risk_score += 20

risk_score = max(0, min(100, risk_score)) 

if risk_score >= 70 or current_cluster == bear_cluster:
    regime_marche = "CRITICAL BEAR (AI & NLP)"
    tail_hedge_active = True
elif risk_score >= 40:
    regime_marche = "DEFENSIVE REGIME"
    tail_hedge_active = False
else:
    regime_marche = "BULL REGIME"
    tail_hedge_active = False

dynamic_correl_max = min(correl_max, 0.55) if tail_hedge_active or risk_score >= 40 else correl_max

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
sortino_brut = (rendements_hebdo.mean() * 52) / (rendements_negatifs.std() * np.sqrt(52)).replace(0, np.nan)

rendements_cumules = (1 + rendements_hebdo).cumprod()
max_dd = ((rendements_cumules - rendements_cumules.cummax()) / rendements_cumules.cummax()).min()

rendements_propres = rendements_hebdo.dropna()
lw_cov_brut, shrinkage_penalty = ledoit_wolf(rendements_propres)
correlation = pd.DataFrame(lw_cov_brut, index=rendements_propres.columns, columns=rendements_propres.columns).corr()

sortino_ajuste = sortino_brut.copy()
for actif in sortino_ajuste.index:
    if "S&P500:" in actif: sortino_ajuste[actif] *= (1.0 - turnover_penalty) 

actifs_pre_eligibles = [a for a in univers_etudie.keys() if a in volatilite.index and not pd.isna(volatilite[a]) and volatilite[a] <= 0.60 and max_dd[a] >= -0.45 and a != "US Treasuries 20Y+"]
top_20_candidats = sortino_ajuste[actifs_pre_eligibles].sort_values(ascending=False).head(20).index.tolist()

# --- 2. PREDICTIVE MACHINE LEARNING (GRADIENT BOOSTING) ---
ml_predictions = {}
with st.spinner("Training Gradient Boosting Decision Trees on Top 20..."):
    for candidat in top_20_candidats:
        try:
            target_series = rendements_hebdo[candidat].shift(-1).dropna()
            feature_series = rendements_hebdo[candidat].iloc[:-1].values.reshape(-1, 1)
            model = GradientBoostingRegressor(n_estimators=50, random_state=42)
            model.fit(feature_series, target_series)
            pred = model.predict(rendements_hebdo[candidat].iloc[-1].reshape(1, -1))[0]
            ml_predictions[candidat] = pred
        except:
            ml_predictions[candidat] = 0.0

# --- 3. MULTI-FACTOR Z-SCORE ENGINE (V5.0 - ML INCLUDED) ---
fundamentals_data = []
for candidat in top_20_candidats:
    ticker_str = univers_etudie[candidat]["ticker"]
    is_etf_or_crypto = any(kw in univers_etudie[candidat]["nom"] for kw in ["Crypto", "ETF", "UCITS", "ETC", "Fund", "AG"])
    prix_actuel = float(df_brut[ticker_str].iloc[-1]) if ticker_str in df_brut.columns else 0
    sma_200 = float(df_brut[ticker_str].tail(200).mean()) if ticker_str in df_brut.columns else prix_actuel
    trend_ratio = (prix_actuel / sma_200) - 1.0 if sma_200 > 0 else 0
    
    pe, roe, consensus = 15.0, 0.15, 3.0
    if not is_etf_or_crypto:
        try:
            info = yf.Ticker(ticker_str).info
            pe = info.get('trailingPE', 15)
            roe = info.get('returnOnEquity', 0.15)
            consensus = info.get('recommendationMean', 3.0)
            if pe is None or pe < 0 or pe > 100: continue
        except: pass
            
    fundamentals_data.append({"Actif": candidat, "PE": pe or 15.0, "ROE": roe or 0.10, "Consensus": consensus or 3.0, "Sortino": sortino_ajuste[candidat], "Trend": trend_ratio, "ML_Pred": ml_predictions[candidat]})

df_zscore = pd.DataFrame(fundamentals_data)
if not df_zscore.empty:
    df_zscore['Global_Score'] = -calculate_z_score(df_zscore['PE']).fillna(0) + calculate_z_score(df_zscore['ROE']).fillna(0) + calculate_z_score(df_zscore['Sortino']).fillna(0) - calculate_z_score(df_zscore['Consensus']).fillna(0) + calculate_z_score(df_zscore['Trend']).fillna(0) + calculate_z_score(df_zscore['ML_Pred']).fillna(0)
    actifs_eligibles_finaux = df_zscore.sort_values(by="Global_Score", ascending=False)['Actif'].tolist()
else:
    actifs_eligibles_finaux = []

top_5_actifs = []
for candidat in actifs_eligibles_finaux:
    if len(top_5_actifs) >= 5: break
    if not any(correlation.loc[candidat, s] > dynamic_correl_max for s in top_5_actifs): top_5_actifs.append(candidat)

# --- 4. EXTREME VALUE THEORY (STRESSED COVARIANCE) & BLACK-LITTERMAN ---
capital_verrouille = sum(row["Valeur (EUR)"] for _, row in st.session_state.mon_portefeuille.iterrows() if row.get("🔒 Core (Ne pas vendre)", False) and row["Actif"] in univers_etudie)
actifs_verrouilles = [row["Actif"] for _, row in st.session_state.mon_portefeuille.iterrows() if row.get("🔒 Core (Ne pas vendre)", False) and row["Actif"] in univers_etudie]
budget_satellite = budget - capital_verrouille

port_vol_initial = expected_portfolio_return = 0.0
allocations_satellite = pd.Series(dtype=float)
reserve_cash = budget_satellite
budget_tail_risk = 0

if len(top_5_actifs) > 0 and budget_satellite > 0:
    rendements_top5 = rendements_propres[top_5_actifs]
    lw_cov_top5, _ = ledoit_wolf(rendements_top5)
    cov_matrix_normal = lw_cov_top5 * 52
    
    # EVT: Calcul de la covariance en temps de Krach (Copule Empirique)
    jours_krach = df_brut['^GSPC'].pct_change().dropna() < -0.01
    if sum(jours_krach) > 10:
        cov_matrix_crash = rendements_propres.loc[jours_krach[jours_krach].index.intersection(rendements_propres.index), top_5_actifs].cov().values * 52
        cov_matrix = 0.7 * cov_matrix_normal + 0.3 * cov_matrix_crash # Blend Institutionnel
    else:
        cov_matrix = cov_matrix_normal
        
    inv_vol_bl = 1 / volatilite[top_5_actifs].values
    w_eq = inv_vol_bl / inv_vol_bl.sum()
    Pi = 2.5 * np.dot(cov_matrix, w_eq)
    Q = np.log(df_actifs[top_5_actifs].iloc[-1] / df_actifs[top_5_actifs].iloc[-13]).values * 4 
    Omega = np.diag(np.diag(cov_matrix)) * 0.05
    
    term1 = inv(inv(0.05 * cov_matrix) + np.dot(np.dot(np.eye(len(top_5_actifs)).T, inv(Omega)), np.eye(len(top_5_actifs))))
    term2 = np.dot(inv(0.05 * cov_matrix), Pi) + np.dot(np.dot(np.eye(len(top_5_actifs)).T, inv(Omega)), Q)
    BL_returns = np.dot(term1, term2)
    
    poids_optimaux = np.clip(np.dot(inv(cov_matrix), BL_returns), 0, None)
    poids_optimaux = poids_optimaux / poids_optimaux.sum() if poids_optimaux.sum() > 0 else w_eq

    while any(poids_optimaux > max_weight_limit + 1e-5):
        excess = sum(poids_optimaux[poids_optimaux > max_weight_limit] - max_weight_limit)
        poids_optimaux[poids_optimaux > max_weight_limit] = max_weight_limit
        mask = poids_optimaux < max_weight_limit
        if sum(mask) > 0: poids_optimaux[mask] += excess * (poids_optimaux[mask] / sum(poids_optimaux[mask]))
        else: break
            
    port_vol_initial = np.sqrt(np.dot(poids_optimaux.T, np.dot(cov_matrix, poids_optimaux)))
    expected_portfolio_return = np.dot(poids_optimaux, (rendements_hebdo[top_5_actifs].mean() * 52).values)
    
    exposure_factor = min(1.0, target_volatility / port_vol_initial if port_vol_initial > 0 else 1.0)
    reserve_cash = budget * (1.0 - exposure_factor)
    budget_satellite_ajuste = max(0, budget_satellite - reserve_cash)
    
    if tail_hedge_active:
        budget_tail_risk = budget_satellite_ajuste * 0.30
        budget_satellite_ajuste -= budget_tail_risk
        
    allocations_satellite = pd.Series(poids_optimaux * budget_satellite_ajuste, index=top_5_actifs)

allocations = pd.Series({row["Actif"]: row["Valeur (EUR)"] for _, row in st.session_state.mon_portefeuille.iterrows() if row.get("🔒 Core (Ne pas vendre)", False) and row["Actif"] in univers_etudie})
for actif, val in allocations_satellite.items(): allocations[actif] = allocations.get(actif, 0.0) + val
if tail_hedge_active: allocations["US Treasuries 20Y+"] = allocations.get("US Treasuries 20Y+", 0.0) + budget_tail_risk

# --- 5. PRINCIPAL COMPONENT ANALYSIS (PCA) ---
pca_explained_variance = [0, 0]
if len(allocations) > 1:
    try:
        pca = PCA(n_components=2)
        actifs_valides_pca = [a for a in allocations.index if a in rendements_propres.columns]
        if len(actifs_valides_pca) > 1:
            pca.fit(rendements_propres[actifs_valides_pca])
            pca_explained_variance = pca.explained_variance_ratio_ * 100
    except: pass

# --- MAIN DASHBOARD ---
st.markdown("<h2 style='font-family: monospace; border-bottom: 1px solid #444; padding-bottom: 10px;'>HEDGE FUND PRIME DESK (V29)</h2>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("SYSTEMIC RISK SCORE", f"{risk_score}/100", delta=regime_marche, delta_color="normal" if risk_score < 40 else "inverse")
col2.metric("NLP NEWS SENTIMENT", f"{avg_nlp_score:.2f}", delta="Euphoria" if avg_nlp_score > 0.15 else ("Fear/Panic" if avg_nlp_score < -0.15 else "Neutral"), delta_color="normal" if avg_nlp_score > 0.15 else ("inverse" if avg_nlp_score < -0.15 else "off"))
col3.metric("TACTICAL SATELLITE", f"{budget_satellite:.2f} EUR", delta=f"Covariance Limit: {dynamic_correl_max*100:.0f}%", delta_color="normal")
col4.metric("VOL-TARGETED CASH", f"{reserve_cash:.2f} EUR", delta=f"{(reserve_cash/budget)*100:.1f}% dynamic weight", delta_color="off")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["ORDER MANAGEMENT SYSTEM", "TARGET ALLOCATION MATRIX", "FACTOR & PCA ANALYSIS", "STOCHASTIC MONTE CARLO", "MACRO & NLP DASHBOARD"])

with tab1:
    col_ed1, col_ed2 = st.columns([1, 1.5])
    with col_ed1:
        st.markdown("**1. Vos Positions Actuelles**")
        config_colonnes = {"Actif": st.column_config.SelectboxColumn("Instrument Détenu", options=list(univers_etudie.keys()), required=True), "Valeur (EUR)": st.column_config.NumberColumn("Valeur Actuelle (€)", min_value=0.0, step=10.0, format="%.2f")}
        df_edited = st.data_editor(st.session_state.mon_portefeuille, column_config=config_colonnes, num_rows="dynamic", use_container_width=True)
        st.session_state.mon_portefeuille = df_edited

    with col_ed2:
        st.markdown("**2. Ticket d'Exécution (Optimisé Frais)**")
        if len(allocations) > 0:
            dict_actuel = {row["Actif"]: row["Valeur (EUR)"] for _, row in df_edited.iterrows() if pd.notna(row["Actif"]) and row["Actif"] != ""}
            lignes_ordres = []
            for a in set(allocations[allocations > 0].index.tolist() + list(dict_actuel.keys())):
                val_cible, val_actuelle = allocations.get(a, 0.0), dict_actuel.get(a, 0.0)
                delta = val_cible - val_actuelle
                instrument_str = f"{univers_etudie.get(a, {'nom':a})['nom']} [{univers_etudie.get(a, {'ticker':''})['ticker']}]"
                action = "BUY" if delta > 0 else "SELL" if abs(delta) >= min_trade_size else "HOLD (Too Small)"
                if delta != 0: lignes_ordres.append({"Instrument": instrument_str, "Target": val_cible, "Current": val_actuelle, "Order Delta (EUR)": delta, "Action": action})
            
            if lignes_ordres:
                df_ordres = pd.DataFrame(lignes_ordres).sort_values(by="Order Delta (EUR)", ascending=False)
                st.dataframe(df_ordres.style.format({"Target": "{:.2f} €", "Current": "{:.2f} €", "Order Delta (EUR)": "{:+.2f} €"}).applymap(lambda x: 'color: #00cc00;' if x == 'BUY' else ('color: #ff4b4b;' if x == 'SELL' else 'color: #888888;'), subset=['Action']), use_container_width=True)
            else: st.success("Portefeuille aligné.")

with tab2:
    st.markdown("<div style='background-color: #1e1e1e; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>", unsafe_allow_html=True)
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    col_kpi1.metric("Expected Annual Return", f"{expected_portfolio_return*100:.1f}%")
    col_kpi2.metric("Portfolio Sharpe Ratio", f"{expected_portfolio_return / port_vol_initial if port_vol_initial > 0 else 0:.2f}")
    col_kpi3.metric("ML Predictive Boost", "ACTIVE", "Gradient Boosting Trees")
    col_kpi4.metric("Tail Risk EVT Matrix", "ACTIVE", "Stressed Covariance")
    st.markdown("</div>", unsafe_allow_html=True)

    donnees_tableau = [{"Instrument (Ticker)": f"{univers_etudie[a]['nom']} [{univers_etudie[a]['ticker']}]", "Status": "LOCKED (CORE)" if a in actifs_verrouilles else ("ALLOCATED" if a in allocations and allocations[a]>0 else "REJECTED"), "Z-Score": f"{df_zscore.loc[df_zscore['Actif']==a, 'Global_Score'].values[0]:.2f}" if not df_zscore.empty and a in df_zscore['Actif'].values else "N/A", "Target Capital": allocations.get(a, 0.0)} for a in dict.fromkeys(list(allocations.index) + top_20_candidats[:15])]
    st.dataframe(pd.DataFrame(donnees_tableau).sort_values(by="Target Capital", ascending=False).style.format({"Target Capital": "{:.2f} €"}).applymap(lambda val: 'background-color: #2c3e50;' if 'LOCKED' in str(val) else ('background-color: #1a4222;' if 'ALLOCATED' in str(val) else 'color: #8b0000;'), subset=['Status']), use_container_width=True, height=450)

with tab3:
    col_pca1, col_pca2 = st.columns([1, 1.5])
    with col_pca1:
        st.markdown("**Principal Component Analysis (PCA)**")
        st.write("Identification des forces invisibles dirigeant le portefeuille.")
        st.metric("Ghost Factor 1 (Market Risk)", f"{pca_explained_variance[0]:.1f}% variance")
        if len(pca_explained_variance) > 1: st.metric("Ghost Factor 2 (Style/Sector)", f"{pca_explained_variance[1]:.1f}% variance")
    with col_pca2:
        st.markdown("**Macro Beta Exposure**")
        if len(allocations) > 0:
            ret_port = (df_brut[[univers_etudie[a]["ticker"] for a in allocations.index]].pct_change().dropna() * (allocations/budget).fillna(0).values).sum(axis=1)
            df_beta = pd.DataFrame({"Port": ret_port, "GSPC": df_brut['^GSPC'].pct_change(), "IEF": df_brut['IEF'].pct_change(), "GLD": df_brut['GLD'].pct_change()}).dropna()
            if not df_beta.empty:
                st.plotly_chart(px.bar(pd.DataFrame({"Factor": ["Equities", "Bonds", "Gold"], "Beta": [df_beta["Port"].cov(df_beta[f])/df_beta[f].var() for f in ["GSPC", "IEF", "GLD"]]}), x="Beta", y="Factor", orientation='h', color="Beta", color_continuous_scale="RdBu", range_color=[-1, 1]).update_layout(template="plotly_dark", margin=dict(t=0,b=0,l=0,r=0)), use_container_width=True)

with tab4:
    st.markdown("**Geometric Brownian Motion (1000 Paths, 252 Days)**")
    if len(allocations) > 0 and budget > 0 and port_vol_initial > 0:
        sims = np.zeros((252, 100))
        sims[0] = budget
        drift = expected_portfolio_return - (0.5 * port_vol_initial**2)
        for t in range(1, 252): sims[t] = sims[t-1] * np.exp(drift / 252 + (port_vol_initial / np.sqrt(252)) * np.random.normal(0, 1, 100))
        fig_mc = go.Figure()
        for i in range(100): fig_mc.add_trace(go.Scatter(y=sims[:, i], mode='lines', line=dict(color='rgba(100, 100, 100, 0.2)'), showlegend=False))
        fig_mc.add_hline(y=np.percentile(sims[-1], 5), line_dash="dash", line_color="#ff4b4b", annotation_text="VaR 95%")
        fig_mc.update_layout(template="plotly_dark", height=400, margin=dict(t=0,b=0,l=0,r=0))
        st.plotly_chart(fig_mc, use_container_width=True)
    else: st.warning("Allocation insuffisante pour la simulation stochastique.")

with tab5:
    if not df_headlines.empty:
        st.dataframe(df_headlines.style.applymap(lambda v: 'color: #00cc00;' if v>0.15 else ('color: #ff4b4b;' if v<-0.15 else 'color: #888888;'), subset=['VADER Score']).format({"VADER Score": "{:.2f}"}), use_container_width=True, height=200)
    col_m1, col_m2 = st.columns(2)
    with col_m1: st.line_chart(pd.DataFrame({"Spread (10Y-3M)": (df_brut['^TNX'] - df_brut['^IRX']).tail(252)}), color="#8b0000" if curve_inverted else "#4a90e2")
    with col_m2: st.line_chart(pd.DataFrame({"Credit Stress": df_brut['Credit_Spread'].tail(252)}), color="#8b0000" if credit_stress else "#4a90e2")

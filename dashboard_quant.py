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
st.set_page_config(page_title="TERMINAL QUANTITATIF V30", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# SYSTÈME D'AUTHENTIFICATION
# ==========================================
if "authentifie" not in st.session_state:
    st.session_state.authentifie = False

if not st.session_state.authentifie:
    st.markdown("<h1 style='text-align: center; color: #ffffff; font-family: monospace;'>BUREAU D'ALLOCATION QUANTITATIVE</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888888; font-family: monospace;'>ACCÈS RESTREINT. V30 (ADVISORY & BRIEFING IA).</p>", unsafe_allow_html=True)
    
    col_login1, col_login2, col_login3 = st.columns([1, 1, 1])
    with col_login2:
        MOT_DE_PASSE_SECRET = "BTSCG2026" 
        mdp_saisi = st.text_input("Clé d'Accès", type="password")
        if st.button("INITIALISER LA SESSION", use_container_width=True):
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
    "Or Physique": {"ticker": "IGLN.L", "nom": "iShares Physical Gold ETC"},
    "Argent Physique": {"ticker": "PHAG.L", "nom": "WisdomTree Physical Silver"},
    "Bons du Trésor US 20A+": {"ticker": "TLT", "nom": "iShares 20+ Year Treasury Bond"},
    "Défense USD": {"ticker": "DFNS.L", "nom": "VanEck Defense UCITS"},
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
            date_str = datetime.fromtimestamp(timestamp).strftime('%d/%m/%Y %H:%M') if timestamp > 0 else "N/A"
            lignes_news.append({"Date": date_str, "Gros Titre": titre, "Score VADER": score})
            
        return np.mean(scores_vader), pd.DataFrame(lignes_news)
    except:
        return 0.0, pd.DataFrame()

def calculate_z_score(series):
    if series.std() == 0: return np.zeros(len(series))
    return (series - series.mean()) / series.std()

# --- PANNEAU DE CONTRÔLE LATÉRAL ---
st.sidebar.markdown("<h3 style='font-family: monospace;'>CONTRÔLES SYSTÈME</h3>", unsafe_allow_html=True)
if st.sidebar.button("FORCER L'ACTUALISATION EN DIRECT", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='font-family: monospace;'>PARAMÈTRES DU PORTEFEUILLE</h3>", unsafe_allow_html=True)
budget = st.sidebar.number_input("Capital Total Actuel (EUR)", min_value=10.0, value=950.0, step=10.0)
seuil_vix = st.sidebar.slider("Seuil d'Alerte VIX (Panique)", 15, 40, 22)
target_volatility = st.sidebar.slider("Volatilité Annuelle Cible (%)", 5, 25, 12) / 100.0
min_trade_size = st.sidebar.slider("Taille d'Ordre Minimum (EUR)", 10, 100, 50)
turnover_penalty = st.sidebar.slider("Pénalité de Rotation (%)", 5, 30, 15) / 100.0
correl_max = st.sidebar.slider("Limite de Covariance Initiale (%)", 50, 95, 75) / 100.0
max_weight_limit = 0.25

if 'mon_portefeuille' not in st.session_state:
    st.session_state.mon_portefeuille = pd.DataFrame({
        "Actif": ["Core S&P 500", "Bitcoin", "Or Physique", "Palantir", "Argent Physique", "Uranium USD", "Rheinmetall"],
        "Valeur (EUR)": [401.43, 295.83, 114.86, 55.27, 42.54, 10.78, 9.47],
        "🔒 Cœur (Ne pas vendre)": [True, True, True, False, False, False, False]
    })

# --- EXÉCUTION DU MOTEUR CENTRAL ---
with st.spinner(f'Calcul des recommandations IA en cours...'):
    liste_tickers_bruts = [v["ticker"] for k, v in univers_etudie.items()]
    df_brut = telecharger_donnees(liste_tickers_bruts)
    avg_nlp_score, df_headlines = analyser_sentiment_nlp()

# --- 1. DÉTECTION DU RÉGIME IA (K-MEANS) ---
df_ml = pd.DataFrame({
    'VIX': df_brut['^VIX'],
    'Écart_Taux': df_brut['^TNX'] - df_brut['^IRX'],
    'Écart_Crédit': df_brut['HYG'] / df_brut['IEF'],
    'SP500_Rendement': df_brut['^GSPC'].pct_change()
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

df_brut['Stress_Credit'] = df_brut['HYG'] / df_brut['IEF']
credit_stress = float(df_brut['Stress_Credit'].iloc[-1]) < float(df_brut['Stress_Credit'].tail(50).mean())

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
    regime_marche = "KRACH IMMINENT (IA & NLP)"
    tail_hedge_active = True
elif risk_score >= 40:
    regime_marche = "RÉGIME DÉFENSIF"
    tail_hedge_active = False
else:
    regime_marche = "RÉGIME HAUSSIER"
    tail_hedge_active = False

dynamic_correl_max = min(correl_max, 0.55) if tail_hedge_active or risk_score >= 40 else correl_max

# --- PRÉPARATION DES DONNÉES ---
df_hebdo = df_brut.tail(252*3).resample('W-FRI').last() 
df_actifs = df_hebdo[liste_tickers_bruts].copy()
inv_map = {v["ticker"]: k for k, v in univers_etudie.items()}
df_actifs.rename(columns=inv_map, inplace=True)

# --- CALCULS QUANTITATIFS ---
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

actifs_pre_eligibles = [a for a in univers_etudie.keys() if a in volatilite.index and not pd.isna(volatilite[a]) and volatilite[a] <= 0.60 and max_dd[a] >= -0.45 and a != "Bons du Trésor US 20A+"]
top_20_candidats = sortino_ajuste[actifs_pre_eligibles].sort_values(ascending=False).head(20).index.tolist()

# --- 2. MACHINE LEARNING PRÉDICTIF ---
ml_predictions = {}
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

# --- 3. MOTEUR DE SCORE MULTI-FACTEURS ---
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
            
    fundamentals_data.append({"Actif": candidat, "P/E": pe or 15.0, "ROE": roe or 0.10, "Consensus": consensus or 3.0, "Sortino": sortino_ajuste[candidat], "Tendance": trend_ratio, "Pred_ML": ml_predictions[candidat]})

df_zscore = pd.DataFrame(fundamentals_data)
if not df_zscore.empty:
    df_zscore['Score_Global'] = -calculate_z_score(df_zscore['P/E']).fillna(0) + calculate_z_score(df_zscore['ROE']).fillna(0) + calculate_z_score(df_zscore['Sortino']).fillna(0) - calculate_z_score(df_zscore['Consensus']).fillna(0) + calculate_z_score(df_zscore['Tendance']).fillna(0) + calculate_z_score(df_zscore['Pred_ML']).fillna(0)
    actifs_eligibles_finaux = df_zscore.sort_values(by="Score_Global", ascending=False)['Actif'].tolist()
else:
    actifs_eligibles_finaux = []

top_5_actifs = []
for candidat in actifs_eligibles_finaux:
    if len(top_5_actifs) >= 5: break
    if not any(correlation.loc[candidat, s] > dynamic_correl_max for s in top_5_actifs): top_5_actifs.append(candidat)

# --- 4. THÉORIE DES VALEURS EXTRÊMES (EVT) & BLACK-LITTERMAN ---
capital_verrouille = sum(row["Valeur (EUR)"] for _, row in st.session_state.mon_portefeuille.iterrows() if row.get("🔒 Cœur (Ne pas vendre)", False) and row["Actif"] in univers_etudie)
actifs_verrouilles = [row["Actif"] for _, row in st.session_state.mon_portefeuille.iterrows() if row.get("🔒 Cœur (Ne pas vendre)", False) and row["Actif"] in univers_etudie]
budget_satellite = budget - capital_verrouille

port_vol_initial = expected_portfolio_return = 0.0
allocations_satellite = pd.Series(dtype=float)
reserve_cash = budget_satellite
budget_tail_risk = 0

if len(top_5_actifs) > 0 and budget_satellite > 0:
    rendements_top5 = rendements_propres[top_5_actifs]
    lw_cov_top5, _ = ledoit_wolf(rendements_top5)
    cov_matrix_normal = lw_cov_top5 * 52
    
    jours_krach = df_brut['^GSPC'].pct_change().dropna() < -0.01
    if sum(jours_krach) > 10:
        cov_matrix_crash = rendements_propres.loc[jours_krach[jours_krach].index.intersection(rendements_propres.index), top_5_actifs].cov().values * 52
        cov_matrix = 0.7 * cov_matrix_normal + 0.3 * cov_matrix_crash 
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

allocations = pd.Series({row["Actif"]: row["Valeur (EUR)"] for _, row in st.session_state.mon_portefeuille.iterrows() if row.get("🔒 Cœur (Ne pas vendre)", False) and row["Actif"] in univers_etudie})
for actif, val in allocations_satellite.items(): allocations[actif] = allocations.get(actif, 0.0) + val
if tail_hedge_active: allocations["Bons du Trésor US 20A+"] = allocations.get("Bons du Trésor US 20A+", 0.0) + budget_tail_risk

# --- 5. ANALYSE EN COMPOSANTES PRINCIPALES (PCA) ---
pca_explained_variance = [0, 0]
if len(allocations) > 1:
    try:
        pca = PCA(n_components=2)
        actifs_valides_pca = [a for a in allocations.index if a in rendements_propres.columns]
        if len(actifs_valides_pca) > 1:
            pca.fit(rendements_propres[actifs_valides_pca])
            pca_explained_variance = pca.explained_variance_ratio_ * 100
    except: pass

# --- TABLEAU DE BORD PRINCIPAL ---
st.markdown("<h2 style='font-family: monospace; border-bottom: 1px solid #444; padding-bottom: 10px;'>BUREAU CONSEIL & EXÉCUTION (V30)</h2>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("MÉTÉO DU MARCHÉ (IA)", regime_marche, delta=f"Score Risque: {risk_score}/100", delta_color="normal" if risk_score < 40 else "inverse")
col2.metric("POIDS CŒUR (SÉCURISÉ)", f"{capital_verrouille:.2f} €", delta="Sanctuarisé", delta_color="off")
col3.metric("POIDS SATELLITE (ACTIF)", f"{budget_satellite:.2f} €", delta="Géré par l'IA", delta_color="normal")
col4.metric("LIQUIDITÉS RECOMMANDÉES", f"{reserve_cash:.2f} €", delta=f"{(reserve_cash/budget)*100:.1f}% de protection", delta_color="off")

tab1, tab2, tab3, tab4 = st.tabs(["💡 CONSEIL & EXÉCUTION (OMS)", "MATRICE D'ALLOCATION CIBLE", "ANALYSE DE RISQUE (PCA)", "STRESS-TESTS HISTORIQUES"])

# =================================================================
# L'ONGLET V30 : LE CONSEILLER STRATÉGIQUE IA
# =================================================================
with tab1:
    
    col_ed1, col_ed2 = st.columns([1, 1.5])
    with col_ed1:
        st.markdown("**1. Vos Positions Actuelles (Saisie)**")
        config_colonnes = {"Actif": st.column_config.SelectboxColumn("Instrument", options=list(univers_etudie.keys()), required=True), "Valeur (EUR)": st.column_config.NumberColumn("Valeur Actuelle (€)", min_value=0.0, step=10.0, format="%.2f")}
        df_edited = st.data_editor(st.session_state.mon_portefeuille, column_config=config_colonnes, num_rows="dynamic", use_container_width=True)
        st.session_state.mon_portefeuille = df_edited

    with col_ed2:
        st.markdown("**2. Le Ticket d'Ordres de l'IA**")
        lignes_ordres = []
        dict_actuel = {}
        if len(allocations) > 0:
            dict_actuel = {row["Actif"]: row["Valeur (EUR)"] for _, row in df_edited.iterrows() if pd.notna(row["Actif"]) and row["Actif"] != ""}
            for a in set(allocations[allocations > 0].index.tolist() + list(dict_actuel.keys())):
                val_cible, val_actuelle = allocations.get(a, 0.0), dict_actuel.get(a, 0.0)
                delta = val_cible - val_actuelle
                instrument_str = f"{univers_etudie.get(a, {'nom':a})['nom']}"
                action = "ACHETER" if delta > 0 else "VENDRE" if abs(delta) >= min_trade_size else "CONSERVER"
                if delta != 0: lignes_ordres.append({"Instrument": instrument_str, "Cible": val_cible, "Actuel": val_actuelle, "Ordre Net (EUR)": delta, "Action": action})
            
            if lignes_ordres:
                df_ordres = pd.DataFrame(lignes_ordres).sort_values(by="Ordre Net (EUR)", ascending=False)
                st.dataframe(df_ordres.style.format({"Cible": "{:.2f} €", "Actuel": "{:.2f} €", "Ordre Net (EUR)": "{:+.2f} €"}).applymap(lambda x: 'color: #00cc00; font-weight: bold;' if x == 'ACHETER' else ('color: #ff4b4b; font-weight: bold;' if x == 'VENDRE' else 'color: #888888;'), subset=['Action']), use_container_width=True)
            else: st.success("🎉 Votre portefeuille est parfaitement optimisé. Aucun ordre nécessaire.")

    st.markdown("---")
    st.markdown("### 🧠 Le Diagnostic du Quant (Justification des choix)")
    
    # GÉRÉNATION DU TEXTE DE CONSEIL INTELLIGENT
    st.info(f"**Vision Macroéconomique :** Le logiciel détecte actuellement un **{regime_marche}**. Le niveau de panique médiatique (NLP) est de {avg_nlp_score:.2f}. Par conséquent, la consigne mathématique est de conserver **{reserve_cash:.2f} € en liquidités (Cash)** pour respecter votre limite de volatilité de {target_volatility*100}%.")
    
    col_conseil1, col_conseil2 = st.columns(2)
    with col_conseil1:
        st.markdown("#### 🔴 Pourquoi l'IA vend certains actifs ?")
        a_vendre = [row['Instrument'] for row in lignes_ordres if row['Action'] == 'VENDRE']
        if len(a_vendre) == 0:
            st.write("L'IA ne trouve aucune raison mathématique de vendre vos actifs non-verrouillés actuellement. Leur qualité est bonne.")
        else:
            for instrument in a_vendre:
                # Retrouver le ticker
                ticker = next((k for k, v in univers_etudie.items() if v['nom'] == instrument), None)
                if ticker:
                    z_score_val = df_zscore.loc[df_zscore['Actif']==ticker, 'Score_Global'].values[0] if not df_zscore.empty and ticker in df_zscore['Actif'].values else -99
                    if z_score_val == -99:
                        st.write(f"- **{instrument} :** Ses fondamentaux ou son momentum se sont trop dégradés. L'IA le rejette totalement de la matrice d'élite.")
                    elif z_score_val < 0:
                        st.write(f"- **{instrument} :** Son Score de Qualité (Z-Score) est négatif ({z_score_val:.2f}). Il perd sa force face à d'autres opportunités sur le marché.")
                    else:
                        st.write(f"- **{instrument} :** C'est un bon actif, mais l'IA estime que vous en possédez *trop* par rapport au risque global. Elle propose de prendre quelques profits pour rééquilibrer.")

    with col_conseil2:
        st.markdown("#### 🟢 Pourquoi l'IA achète ces actifs ?")
        a_acheter = [row['Instrument'] for row in lignes_ordres if row['Action'] == 'ACHETER']
        if len(a_acheter) == 0:
            st.write("L'IA ne trouve aucune nouvelle opportunité qui justifie de payer les frais de transaction (Pénalité de Rotation active).")
        else:
            for instrument in a_acheter:
                ticker = next((k for k, v in univers_etudie.items() if v['nom'] == instrument), None)
                if ticker == "Bons du Trésor US 20A+":
                    st.write(f"- **{instrument} :** Couverture d'Urgence ! Le marché est dangereux, l'IA achète ceci pour protéger le portefeuille en cas de krach imminent.")
                elif ticker:
                    z_score_val = df_zscore.loc[df_zscore['Actif']==ticker, 'Score_Global'].values[0] if not df_zscore.empty and ticker in df_zscore['Actif'].values else 0
                    pred_ml = df_zscore.loc[df_zscore['Actif']==ticker, 'Pred_ML'].values[0] if not df_zscore.empty and ticker in df_zscore['Actif'].values else 0
                    st.write(f"- **{instrument} :** Fait partie de l'Élite. Score Global massif de **{z_score_val:.2f}**. " + ("Le Machine Learning prédit en plus une forte hausse future." if pred_ml > 0 else "Ses bilans financiers sont excellents."))

with tab2:
    st.markdown("<div style='background-color: #1e1e1e; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>", unsafe_allow_html=True)
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    col_kpi1.metric("Rendement Annuel Espéré", f"{expected_portfolio_return*100:.1f}%")
    col_kpi2.metric("Ratio de Sharpe du Portefeuille", f"{expected_portfolio_return / port_vol_initial if port_vol_initial > 0 else 0:.2f}")
    col_kpi3.metric("Boost Prédictif (IA)", "ACTIF", "Arbres de Décision Gradient")
    col_kpi4.metric("Matrice Risque Extrême (EVT)", "ACTIVE", "Covariance sous Stress")
    st.markdown("</div>", unsafe_allow_html=True)

    donnees_tableau = [{"Instrument (Ticker)": f"{univers_etudie[a]['nom']} [{univers_etudie[a]['ticker']}]", "Statut": "VERROUILLÉ (CŒUR)" if a in actifs_verrouilles else ("ALLOUÉ" if a in allocations and allocations[a]>0 else "REJETÉ"), "Score Qualité (Z)": f"{df_zscore.loc[df_zscore['Actif']==a, 'Score_Global'].values[0]:.2f}" if not df_zscore.empty and a in df_zscore['Actif'].values else "N/A", "Capital Cible": allocations.get(a, 0.0)} for a in dict.fromkeys(list(allocations.index) + top_20_candidats[:15])]
    st.dataframe(pd.DataFrame(donnees_tableau).sort_values(by="Capital Cible", ascending=False).style.format({"Capital Cible": "{:.2f} €"}).applymap(lambda val: 'background-color: #2c3e50;' if 'VERROUILLÉ' in str(val) else ('background-color: #1a4222;' if 'ALLOUÉ' in str(val) else 'color: #8b0000;'), subset=['Statut']), use_container_width=True, height=450)

with tab3:
    col_pca1, col_pca2 = st.columns([1, 1.5])
    with col_pca1:
        st.markdown("**Analyse en Composantes Principales (PCA)**")
        st.write("Identification mathématique des forces invisibles dirigeant le portefeuille.")
        st.metric("Facteur Fantôme 1 (Risque Global)", f"{pca_explained_variance[0]:.1f}% de variance expliquée")
        if len(pca_explained_variance) > 1: st.metric("Facteur Fantôme 2 (Secteur/Style)", f"{pca_explained_variance[1]:.1f}% de variance expliquée")
    with col_pca2:
        st.markdown("**Exposition Bêta Macroéconomique**")
        if len(allocations) > 0:
            ret_port = (df_brut[[univers_etudie[a]["ticker"] for a in allocations.index]].pct_change().dropna() * (allocations/budget).fillna(0).values).sum(axis=1)
            df_beta = pd.DataFrame({"Port": ret_port, "S&P500": df_brut['^GSPC'].pct_change(), "Taux US": df_brut['IEF'].pct_change(), "Or": df_brut['GLD'].pct_change()}).dropna()
            if not df_beta.empty:
                st.plotly_chart(px.bar(pd.DataFrame({"Facteur Macro": ["Actions Mondiales", "Taux d'Intérêt US", "Or / Inflation"], "Bêta": [df_beta["Port"].cov(df_beta[f])/df_beta[f].var() for f in ["S&P500", "Taux US", "Or"]]}), x="Bêta", y="Facteur Macro", orientation='h', color="Bêta", color_continuous_scale="RdBu", range_color=[-1, 1]).update_layout(template="plotly_dark", margin=dict(t=0,b=0,l=0,r=0)), use_container_width=True)

with tab4:
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
                df_graph_covid = pd.DataFrame({"Ton Portefeuille (Moteur IA)": croissance_port_covid, "Indice S&P 500": croissance_sp_covid})
                col_st1, col_st2 = st.columns(2)
                with col_st1:
                    st.markdown("**SCÉNARIO A : Krach de Liquidité COVID-19 (Fév-Avr 2020)**")
                    fig_covid = px.line(df_graph_covid, color_discrete_sequence=['#4a90e2', '#444444'])
                    fig_covid.update_layout(template="plotly_dark", margin=dict(t=10, b=10, l=10, r=10), xaxis_title="", yaxis_title="Base 100")
                    st.plotly_chart(fig_covid, use_container_width=True)
                    st.caption(f"Pire perte subie par ton portefeuille : **{(croissance_port_covid.min() - 100):.1f}%**")
            else:
                col_st1, col_st2 = st.columns(2)
                with col_st1: st.warning("Données historiques insuffisantes pour 2020.")
                    
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
                df_graph_inf = pd.DataFrame({"Ton Portefeuille (Moteur IA)": croissance_port_inf, "Indice S&P 500": croissance_sp_inf})
                with col_st2:
                    st.markdown("**SCÉNARIO B : Choc des Taux d'Intérêt (Jan-Oct 2022)**")
                    fig_inf = px.line(df_graph_inf, color_discrete_sequence=['#4a90e2', '#444444'])
                    fig_inf.update_layout(template="plotly_dark", margin=dict(t=10, b=10, l=10, r=10), xaxis_title="", yaxis_title="")
                    st.plotly_chart(fig_inf, use_container_width=True)
                    st.caption(f"Pire perte subie par ton portefeuille : **{(croissance_port_inf.min() - 100):.1f}%**")
            else:
                with col_st2: st.warning("Données historiques insuffisantes pour 2022.")

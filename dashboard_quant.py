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
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging
import warnings

warnings.filterwarnings('ignore')

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION STREAMLIT ---
st.set_page_config(page_title="TERMINAL QUANTITATIF V33", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# CONSTANTES TRADE REPUBLIC
# ==========================================
FRAIS_ORDRE_TR = 1.0        # 1€ par ordre sur Trade Republic
SPREAD_ESTIME_TR = 0.0015   # ~0.15% spread estimé moyen
SEUIL_ORDRE_RENTABLE = 50   # Sous 50€, les frais mangent trop de rendement

# ==========================================
# SYSTÈME D'AUTHENTIFICATION
# ==========================================
if "authentifie" not in st.session_state:
    st.session_state.authentifie = False

if not st.session_state.authentifie:
    st.markdown(
        "<h1 style='text-align: center; color: #ffffff; font-family: monospace;'>"
        "BUREAU D'ALLOCATION QUANTITATIVE</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; color: #888888; font-family: monospace;'>"
        "ACCÈS RESTREINT. V33 — MODE DCA · TRADE REPUBLIC.</p>",
        unsafe_allow_html=True,
    )
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
# DEVISES — Conversion FX vers EUR
# ==========================================
DEVISE_PAR_TICKER = {
    "BTC-EUR": "EUR", "ETH-EUR": "EUR",
    "IGLN.L": "GBP", "PHAG.L": "GBP",
    "TLT": "USD", "DFNS.L": "GBP",
    "RHM.DE": "EUR", "PLTR": "USD", "URNM": "USD",
    "CSPX.AS": "EUR", "DSY.PA": "EUR", "TTE.PA": "EUR", "MC.PA": "EUR",
    "HYG": "USD", "IEF": "USD", "GLD": "USD",
    "^VIX": "USD", "^TNX": "USD", "^GSPC": "USD", "^IRX": "USD",
}


@st.cache_data(ttl=3600)
def obtenir_taux_change():
    try:
        fx = yf.download(["EURUSD=X", "EURGBP=X"], period="5d", progress=False)["Close"]
        fx = fx.ffill().bfill()
        return {
            "USD": 1.0 / float(fx["EURUSD=X"].iloc[-1]),
            "GBP": 1.0 / float(fx["EURGBP=X"].iloc[-1]),
            "EUR": 1.0,
        }
    except Exception as e:
        logger.warning(f"FX fallback 1:1 — {e}")
        return {"USD": 1.0, "GBP": 1.0, "EUR": 1.0}


def devise_ticker(ticker: str) -> str:
    return DEVISE_PAR_TICKER.get(ticker, "USD")


def prix_en_eur(prix: float, ticker: str, taux_fx: dict) -> float:
    return prix * taux_fx.get(devise_ticker(ticker), 1.0)


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
    "LVMH": {"ticker": "MC.PA", "nom": "LVMH Moët Hennessy"},
}

MOTS_CLES_ETF_CRYPTO = [
    "Crypto", "ETF", "UCITS", "ETC", "Fund",
    "iShares", "WisdomTree", "VanEck", "Sprott", "SPDR",
    "Vanguard", "Amundi", "Lyxor",
]


def est_etf_ou_crypto(nom_instrument: str) -> bool:
    return any(kw.lower() in nom_instrument.lower() for kw in MOTS_CLES_ETF_CRYPTO)


@st.cache_data(ttl=86400)
def aspirer_le_marche_sp500():
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        html = requests.get(url, headers=headers, timeout=15).text
        table = pd.read_html(html)[0]
        tickers = table["Symbol"].tolist()
        noms = table["Security"].tolist()
        d = {}
        for t, n in zip(tickers, noms):
            tp = t.replace(".", "-")
            d[f"S&P500: {tp}"] = {"ticker": tp, "nom": n}
        if len(d) < 400:
            logger.error(f"Scraping S&P 500 incomplet ({len(d)} titres)")
        return d
    except Exception as e:
        logger.error(f"Échec scraping S&P 500 : {e}")
        return {}


univers_etudie = MES_FAVORIS.copy()
mega_dict = aspirer_le_marche_sp500()
for cle, donnees in mega_dict.items():
    if donnees["ticker"] not in [v["ticker"] for v in univers_etudie.values()]:
        univers_etudie[cle] = donnees


# ==========================================
# CHARGEMENT DES DONNÉES
# ==========================================
TICKERS_MACRO = ["^VIX", "^TNX", "^GSPC", "^IRX", "HYG", "IEF", "GLD"]


@st.cache_data(ttl=3600)
def telecharger_donnees(liste_tickers):
    tickers_complets = list(set(liste_tickers + TICKERS_MACRO))
    df = yf.download(tickers_complets, period="6y", progress=False)["Close"]
    df = df.ffill().bfill()
    return df


@st.cache_data(ttl=86400)
def obtenir_fondamentaux(ticker_str: str) -> dict:
    try:
        info = yf.Ticker(ticker_str).info
        return {
            "trailingPE": info.get("trailingPE"),
            "returnOnEquity": info.get("returnOnEquity"),
            "recommendationMean": info.get("recommendationMean"),
        }
    except Exception as e:
        logger.warning(f"Fondamentaux indisponibles pour {ticker_str}: {e}")
        return {}


@st.cache_data(ttl=3600)
def analyser_sentiment_nlp():
    try:
        analyzer = SentimentIntensityAnalyzer()
        tickers_macro = ["^GSPC", "TLT", "GLD"]
        toutes_les_news = []
        for t in tickers_macro:
            news = yf.Ticker(t).news
            if news:
                toutes_les_news.extend(news)
        if not toutes_les_news:
            return 0.0, pd.DataFrame()
        toutes_les_news = sorted(
            toutes_les_news, key=lambda x: x.get("providerPublishTime", 0), reverse=True
        )[:20]
        lignes_news, scores_vader = [], []
        for item in toutes_les_news:
            titre = item.get("title", "")
            score = analyzer.polarity_scores(titre)["compound"]
            scores_vader.append(score)
            ts = item.get("providerPublishTime", 0)
            date_str = datetime.fromtimestamp(ts).strftime("%d/%m/%Y %H:%M") if ts > 0 else "N/A"
            lignes_news.append({"Date": date_str, "Gros Titre": titre, "Score VADER": score})
        return float(np.mean(scores_vader)), pd.DataFrame(lignes_news)
    except Exception as e:
        logger.warning(f"Analyse NLP échouée : {e}")
        return 0.0, pd.DataFrame()


def calculate_z_score(series):
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


# ==========================================
# PORTEFEUILLE PAR DÉFAUT (valeurs réelles Trade Republic)
# ==========================================
if "mon_portefeuille" not in st.session_state:
    st.session_state.mon_portefeuille = pd.DataFrame({
        "Actif": [
            "Core S&P 500", "Bitcoin", "Or Physique", "Palantir",
            "Argent Physique", "Uranium USD", "Rheinmetall",
        ],
        "Valeur (EUR)": [401.00, 295.00, 114.00, 55.00, 42.00, 10.00, 9.00],
        "🔒 Cœur (Ne pas vendre)": [True, True, True, False, False, False, False],
    })

# ==========================================
# PANNEAU DE CONTRÔLE LATÉRAL
# ==========================================
st.sidebar.markdown("<h3 style='font-family: monospace;'>CONTRÔLES SYSTÈME</h3>", unsafe_allow_html=True)
if st.sidebar.button("FORCER L'ACTUALISATION EN DIRECT", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='font-family: monospace;'>PARAMÈTRES DU PORTEFEUILLE</h3>", unsafe_allow_html=True)

# [V33] Le capital est calculé automatiquement depuis les positions
st.sidebar.markdown("*Le capital est calculé depuis vos positions.*")

seuil_vix = st.sidebar.slider("Seuil d'Alerte VIX (Panique)", 15, 40, 22)
target_volatility = st.sidebar.slider("Volatilité Annuelle Cible (%)", 5, 25, 12) / 100.0
min_trade_size = st.sidebar.slider("Taille d'Ordre Minimum (EUR)", 10, 100, 50,
                                    help=f"Sous {SEUIL_ORDRE_RENTABLE}€, les frais TR de {FRAIS_ORDRE_TR}€ grignotent trop le rendement.")
turnover_penalty = st.sidebar.slider("Pénalité de Rotation (%)", 5, 30, 15) / 100.0
correl_max = st.sidebar.slider("Limite de Corrélation Max (%)", 50, 95, 75) / 100.0
max_weight_limit = 0.25

# [V33] Validation du portefeuille AVANT tout calcul (fixe le bug V31)
df_port = st.session_state.mon_portefeuille.copy()
df_port = df_port.dropna(subset=["Actif"])
df_port = df_port[df_port["Actif"].isin(univers_etudie.keys())]
df_port = df_port.drop_duplicates(subset=["Actif"], keep="first")
df_port["Valeur (EUR)"] = df_port["Valeur (EUR)"].clip(lower=0).fillna(0)
st.session_state.mon_portefeuille = df_port.reset_index(drop=True)

# Capital calculé automatiquement
budget = float(df_port["Valeur (EUR)"].sum())
st.sidebar.metric("Capital Total Calculé", f"{budget:.2f} €")


# ==========================================
# EXÉCUTION DU MOTEUR CENTRAL
# ==========================================
with st.spinner("Calcul des recommandations IA en cours..."):
    liste_tickers_bruts = [v["ticker"] for v in univers_etudie.values()]
    df_brut = telecharger_donnees(liste_tickers_bruts)
    taux_fx = obtenir_taux_change()
    avg_nlp_score, df_headlines = analyser_sentiment_nlp()


# ==========================================
# 1. DÉTECTION DU RÉGIME IA (K-MEANS)
# ==========================================
df_ml = pd.DataFrame({
    "VIX": df_brut["^VIX"],
    "Écart_Taux": df_brut["^TNX"] - df_brut["^IRX"],
    "Écart_Crédit": df_brut["HYG"] / df_brut["IEF"],
    "SP500_Rendement": df_brut["^GSPC"].pct_change(),
}).dropna()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_ml)

initial_centers = np.array([
    [15.0, 1.5, 1.05, 0.002],
    [22.0, 0.5, 1.00, -0.001],
    [35.0, -0.5, 0.95, -0.005],
])
initial_centers_scaled = scaler.transform(initial_centers)
kmeans = KMeans(n_clusters=3, init=initial_centers_scaled, n_init=1, random_state=42)
kmeans.fit(scaled_data)
df_ml["Cluster"] = kmeans.labels_

cluster_vix_means = df_ml.groupby("Cluster")["VIX"].mean()
bull_cluster = cluster_vix_means.idxmin()
bear_cluster = cluster_vix_means.idxmax()
current_cluster = kmeans.predict(scaled_data[-1].reshape(1, -1))[0]

vix_actuel = float(df_brut["^VIX"].iloc[-1])
taux_10y = float(df_brut["^TNX"].iloc[-1])
taux_3m = float(df_brut["^IRX"].iloc[-1])
curve_inverted = (taux_10y - taux_3m) < 0.0

df_brut["Stress_Credit"] = df_brut["HYG"] / df_brut["IEF"]
credit_stress = float(df_brut["Stress_Credit"].iloc[-1]) < float(df_brut["Stress_Credit"].tail(50).mean())

sp500_close = float(df_brut["^GSPC"].iloc[-1])
sp500_sma200 = float(df_brut["^GSPC"].tail(200).mean())

risk_score = 0
if avg_nlp_score < -0.15:
    risk_score += 20
elif avg_nlp_score > 0.15:
    risk_score -= 10
if sp500_close < sp500_sma200:
    risk_score += 40
if curve_inverted:
    risk_score += 30
if credit_stress:
    risk_score += 30
if vix_actuel > seuil_vix:
    risk_score += 20

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

# En mode défensif on resserre la corrélation mais pas au point de tout rejeter
if tail_hedge_active:
    dynamic_correl_max = min(correl_max, 0.60)
elif risk_score >= 40:
    dynamic_correl_max = min(correl_max, 0.70)
else:
    dynamic_correl_max = correl_max


# ==========================================
# PRÉPARATION DES DONNÉES QUANTITATIVES
# ==========================================
df_hebdo = df_brut.tail(252 * 3).resample("W-FRI").last().dropna(how="all")
colonnes_actifs_dispo = [t for t in liste_tickers_bruts if t in df_hebdo.columns]
df_actifs = df_hebdo[colonnes_actifs_dispo].copy()
inv_map = {v["ticker"]: k for k, v in univers_etudie.items()}
df_actifs.rename(columns=inv_map, inplace=True)

rendements_hebdo = np.log(df_actifs / df_actifs.shift(1)).dropna(how="all")
volatilite = rendements_hebdo.rolling(window=52).std().iloc[-1] * np.sqrt(52)
rendements_negatifs = rendements_hebdo.copy()
rendements_negatifs[rendements_negatifs > 0] = 0
sortino_brut = (rendements_hebdo.mean() * 52) / (
    rendements_negatifs.std() * np.sqrt(52)
).replace(0, np.nan)

rendements_cumules = (1 + rendements_hebdo).cumprod()
max_dd = ((rendements_cumules - rendements_cumules.cummax()) / rendements_cumules.cummax()).min()

# Garder les colonnes avec assez de données (au moins 80% de valeurs)
rendements_propres = rendements_hebdo.dropna(axis=1, thresh=int(len(rendements_hebdo) * 0.8))
rendements_propres = rendements_propres.dropna(axis=0)
if rendements_propres.shape[0] > 10 and rendements_propres.shape[1] > 1:
    lw_cov_brut, shrinkage_penalty = ledoit_wolf(rendements_propres)
    correlation = pd.DataFrame(lw_cov_brut, index=rendements_propres.columns, columns=rendements_propres.columns).corr()
else:
    correlation = rendements_propres.corr()

sortino_ajuste = sortino_brut.copy()
for actif in sortino_ajuste.index:
    if "S&P500:" in actif:
        sortino_ajuste[actif] *= 1.0 - turnover_penalty

actifs_pre_eligibles = [
    a for a in univers_etudie.keys()
    if a in volatilite.index
    and not pd.isna(volatilite.get(a, np.nan))
    and volatilite[a] <= 0.60
    and max_dd.get(a, -1.0) >= -0.45
    and a != "Bons du Trésor US 20A+"
]
top_20_candidats = sortino_ajuste.reindex(actifs_pre_eligibles).dropna().sort_values(ascending=False).head(20).index.tolist()


# ==========================================
# 2. MACHINE LEARNING PRÉDICTIF
# ==========================================
def construire_features_ml(candidat, rendements, df_macro):
    ret = rendements.get(candidat)
    if ret is None:
        return pd.DataFrame()
    features = pd.DataFrame({
        "ret_lag1": ret,
        "ret_lag2": ret.shift(1),
        "ret_lag4": ret.shift(3),
        "vol_12w": ret.rolling(12).std(),
        "momentum_13w": ret.rolling(13).sum(),
    }, index=ret.index)
    for col in ["VIX", "Écart_Taux", "Écart_Crédit", "SP500_Rendement"]:
        if col in df_macro.columns:
            features[col] = df_macro[col].reindex(features.index, method="ffill")
    features["target"] = ret.shift(-1)
    features = features.dropna()
    return features


ml_predictions = {}
ml_cv_scores = {}

for candidat in top_20_candidats:
    try:
        features_df = construire_features_ml(candidat, rendements_hebdo, df_ml)
        if len(features_df) < 60:
            ml_predictions[candidat] = 0.0
            continue
        X = features_df.drop(columns=["target"]).values
        y = features_df["target"].values
        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42
        )
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_squared_error")
        ml_cv_scores[candidat] = float(np.mean(cv_scores))
        model.fit(X, y)
        pred = model.predict(X[-1].reshape(1, -1))[0]
        ml_predictions[candidat] = pred
    except Exception as e:
        logger.warning(f"ML échoué pour {candidat}: {e}")
        ml_predictions[candidat] = 0.0


# ==========================================
# 3. MOTEUR DE SCORE MULTI-FACTEURS
# ==========================================
fundamentals_data = []

for candidat in top_20_candidats:
    ticker_str = univers_etudie[candidat]["ticker"]
    nom_instrument = univers_etudie[candidat]["nom"]
    is_etf = est_etf_ou_crypto(nom_instrument)

    if ticker_str not in df_brut.columns:
        continue

    prix_brut = float(df_brut[ticker_str].iloc[-1])
    sma_200_brut = float(df_brut[ticker_str].tail(200).mean())
    trend_ratio = (prix_brut / sma_200_brut) - 1.0 if sma_200_brut > 0 else 0

    pe, roe, consensus = 15.0, 0.15, 3.0
    if not is_etf:
        fondamentaux = obtenir_fondamentaux(ticker_str)
        pe_raw = fondamentaux.get("trailingPE")
        roe_raw = fondamentaux.get("returnOnEquity")
        cons_raw = fondamentaux.get("recommendationMean")
        if pe_raw is not None and 0 < pe_raw < 100:
            pe = pe_raw
        elif pe_raw is not None and (pe_raw < 0 or pe_raw > 100):
            continue
        if roe_raw is not None:
            roe = roe_raw
        if cons_raw is not None:
            consensus = cons_raw

    fundamentals_data.append({
        "Actif": candidat,
        "P/E": pe, "ROE": roe, "Consensus": consensus,
        "Sortino": sortino_ajuste.get(candidat, 0.0),
        "Tendance": trend_ratio,
        "Pred_ML": ml_predictions.get(candidat, 0.0),
    })

df_zscore = pd.DataFrame(fundamentals_data)
if not df_zscore.empty:
    df_zscore["Score_Global"] = (
        -calculate_z_score(df_zscore["P/E"]).fillna(0)
        + calculate_z_score(df_zscore["ROE"]).fillna(0)
        + calculate_z_score(df_zscore["Sortino"]).fillna(0)
        - calculate_z_score(df_zscore["Consensus"]).fillna(0)
        + calculate_z_score(df_zscore["Tendance"]).fillna(0)
        + calculate_z_score(df_zscore["Pred_ML"]).fillna(0)
    )
    actifs_eligibles_finaux = df_zscore.sort_values("Score_Global", ascending=False)["Actif"].tolist()
else:
    actifs_eligibles_finaux = []

# ==========================================
# 4. BLACK-LITTERMAN & ALLOCATION
# ==========================================
actifs_verrouilles = [
    row["Actif"] for _, row in st.session_state.mon_portefeuille.iterrows()
    if row.get("🔒 Cœur (Ne pas vendre)", False) and row["Actif"] in univers_etudie
]
capital_verrouille = sum(
    row["Valeur (EUR)"] for _, row in st.session_state.mon_portefeuille.iterrows()
    if row.get("🔒 Cœur (Ne pas vendre)", False) and row["Actif"] in univers_etudie
)
budget_satellite = max(0, budget - capital_verrouille)

# Nombre max d'actifs satellite adapté au budget disponible
max_satellite_actifs = max(1, min(8, int(budget_satellite / min_trade_size))) if budget_satellite > 0 else 0

top_satellite_actifs = []
for candidat in actifs_eligibles_finaux:
    if len(top_satellite_actifs) >= max_satellite_actifs:
        break
    if candidat not in correlation.index:
        continue
    trop_correle = False
    for s in top_satellite_actifs:
        if s in correlation.index and candidat in correlation.columns:
            if abs(correlation.loc[candidat, s]) > dynamic_correl_max:
                trop_correle = True
                break
    if not trop_correle:
        top_satellite_actifs.append(candidat)


port_vol_initial = 0.0
expected_portfolio_return = 0.0
allocations_satellite = pd.Series(dtype=float)
budget_tail_risk = 0

# Cash minimum obligatoire selon le régime de marché
if tail_hedge_active or risk_score >= 70:
    min_cash_pct = 0.15  # 15% en cash minimum si krach
elif risk_score >= 40:
    min_cash_pct = 0.05  # 5% minimum en défensif
else:
    min_cash_pct = 0.0   # 0% en haussier

reserve_cash_pct = min_cash_pct

if len(top_satellite_actifs) > 0 and budget_satellite > 0:
    actifs_bl = [a for a in top_satellite_actifs if a in rendements_propres.columns]
    if len(actifs_bl) > 0:
        rendements_top5 = rendements_propres[actifs_bl]
        
        # Cas spécial : 1 seul actif satellite → pas besoin de BL
        if len(actifs_bl) == 1:
            budget_invest = budget_satellite * (1.0 - reserve_cash_pct)
            if tail_hedge_active:
                budget_tail_risk = budget_invest * 0.30
                budget_invest -= budget_tail_risk
            allocations_satellite = pd.Series({actifs_bl[0]: budget_invest})
            ret_a = rendements_top5.mean().values[0] * 52
            vol_a = rendements_top5.std().values[0] * np.sqrt(52)
            expected_portfolio_return = float(ret_a) if not np.isnan(ret_a) else 0.0
            port_vol_initial = float(vol_a) if not np.isnan(vol_a) else 0.0
        
        # Cas normal : 2+ actifs → Black-Litterman complet
        elif len(actifs_bl) >= 2:
            try:
                lw_cov_top5, _ = ledoit_wolf(rendements_top5)
                cov_matrix_normal = lw_cov_top5 * 52

                # Covariance sous stress
                jours_krach = df_brut["^GSPC"].pct_change().dropna() < -0.01
                idx_crash = jours_krach[jours_krach].index.intersection(rendements_propres.index)
                if len(idx_crash) > 10:
                    cov_crash = rendements_propres.loc[idx_crash, actifs_bl].cov().values * 52
                    if cov_crash.shape == cov_matrix_normal.shape:
                        cov_matrix = 0.7 * cov_matrix_normal + 0.3 * cov_crash
                    else:
                        cov_matrix = cov_matrix_normal
                else:
                    cov_matrix = cov_matrix_normal

                n_a = len(actifs_bl)
                inv_vol_bl = 1.0 / np.clip(volatilite[actifs_bl].values, 0.01, None)
                w_eq = inv_vol_bl / inv_vol_bl.sum()
                Pi = 2.5 * np.dot(cov_matrix, w_eq)

                # Vues ML avec confiance calibrée
                Q = np.array([ml_predictions.get(a, 0.0) * 52 for a in actifs_bl])
                omega_diag = [max(abs(ml_cv_scores.get(a, -0.01)) * 52, 0.01) for a in actifs_bl]
                Omega = np.diag(omega_diag)
                P = np.eye(n_a)
                tau = 0.05

                term1 = inv(inv(tau * cov_matrix) + P.T @ inv(Omega) @ P)
                term2 = inv(tau * cov_matrix) @ Pi + P.T @ inv(Omega) @ Q
                BL_returns = term1 @ term2

                poids_optimaux = np.clip(inv(cov_matrix) @ BL_returns, 0, None)
                if poids_optimaux.sum() > 0:
                    poids_optimaux = poids_optimaux / poids_optimaux.sum()
                else:
                    poids_optimaux = w_eq

                # Plafonner les poids
                for _ in range(10):
                    if not any(poids_optimaux > max_weight_limit + 1e-5):
                        break
                    excess = (poids_optimaux[poids_optimaux > max_weight_limit] - max_weight_limit).sum()
                    poids_optimaux[poids_optimaux > max_weight_limit] = max_weight_limit
                    mask = poids_optimaux < max_weight_limit
                    if mask.sum() > 0:
                        poids_optimaux[mask] += excess * (poids_optimaux[mask] / poids_optimaux[mask].sum())
                    else:
                        break

                port_vol_initial = float(np.sqrt(poids_optimaux.T @ cov_matrix @ poids_optimaux))
                ret_annuel = rendements_top5.mean() * 52
                ret_annuel = ret_annuel.fillna(0)
                expected_portfolio_return = float(np.dot(poids_optimaux, ret_annuel.values))
                if np.isnan(expected_portfolio_return) or np.isinf(expected_portfolio_return):
                    expected_portfolio_return = 0.0

                exposure_factor = min(1.0, target_volatility / port_vol_initial if port_vol_initial > 0 else 1.0)
                vol_cash_pct = 1.0 - exposure_factor
                reserve_cash_pct = max(min_cash_pct, vol_cash_pct)
                budget_invest = budget_satellite * (1.0 - reserve_cash_pct)

                if tail_hedge_active:
                    budget_tail_risk = budget_invest * 0.30
                    budget_invest -= budget_tail_risk

                allocations_satellite = pd.Series(poids_optimaux * budget_invest, index=actifs_bl)
                
                # Éliminer les allocations trop petites (pas rentable avec frais TR)
                for _ in range(3):
                    trop_petit = allocations_satellite[allocations_satellite < min_trade_size]
                    assez_grand = allocations_satellite[allocations_satellite >= min_trade_size]
                    if len(trop_petit) == 0 or len(assez_grand) == 0:
                        break
                    montant_redistribue = trop_petit.sum()
                    allocations_satellite = assez_grand.copy()
                    allocations_satellite *= (1 + montant_redistribue / assez_grand.sum())
                if (allocations_satellite < min_trade_size).all() and len(allocations_satellite) > 0:
                    meilleur = allocations_satellite.idxmax()
                    allocations_satellite = pd.Series({meilleur: budget_invest})

            except Exception as e:
                logger.error(f"Optimisation BL échouée : {e}")

reserve_cash = budget_satellite * reserve_cash_pct

# Agréger allocations cœur + satellite
allocations = pd.Series(dtype=float)
for _, row in st.session_state.mon_portefeuille.iterrows():
    actif = row["Actif"]
    if row.get("🔒 Cœur (Ne pas vendre)", False) and actif in univers_etudie:
        allocations[actif] = row["Valeur (EUR)"]

for actif, val in allocations_satellite.items():
    allocations[actif] = allocations.get(actif, 0.0) + val
if tail_hedge_active:
    allocations["Bons du Trésor US 20A+"] = allocations.get("Bons du Trésor US 20A+", 0.0) + budget_tail_risk


# ==========================================
# 5. PCA
# ==========================================
pca_explained_variance = [0, 0]
if len(allocations) > 1:
    try:
        actifs_pca = [a for a in allocations.index if a in rendements_propres.columns]
        if len(actifs_pca) > 1:
            pca = PCA(n_components=min(2, len(actifs_pca)))
            pca.fit(rendements_propres[actifs_pca])
            pca_explained_variance = list(pca.explained_variance_ratio_ * 100)
    except Exception as e:
        logger.warning(f"PCA échouée : {e}")


# ==========================================
# 6. BILAN DE SANTÉ PAR POSITION (nouveau V33)
# ==========================================
def calculer_sante_position(actif_nom, rendements, vol, mdd, corr_matrix, df_prix, taux, all_actifs_port):
    """Calcule un diagnostic complet pour une position du portefeuille."""
    if actif_nom not in univers_etudie:
        return None
    ticker = univers_etudie[actif_nom]["ticker"]
    if ticker not in df_prix.columns:
        return None

    resultats = {"actif": actif_nom, "ticker": ticker}

    # Momentum
    prix = df_prix[ticker].dropna()
    if len(prix) > 200:
        sma50 = prix.tail(50).mean()
        sma200 = prix.tail(200).mean()
        prix_actuel = float(prix.iloc[-1])
        resultats["au_dessus_sma50"] = prix_actuel > sma50
        resultats["au_dessus_sma200"] = prix_actuel > sma200
        resultats["distance_sma200_pct"] = ((prix_actuel / sma200) - 1) * 100
    else:
        resultats["au_dessus_sma50"] = None
        resultats["au_dessus_sma200"] = None
        resultats["distance_sma200_pct"] = 0

    # Volatilité
    resultats["volatilite_annualisee"] = float(vol.get(actif_nom, 0)) if actif_nom in vol.index else 0

    # Max drawdown
    resultats["max_drawdown"] = float(mdd.get(actif_nom, 0)) if actif_nom in mdd.index else 0

    # Sortino
    resultats["sortino"] = float(sortino_brut.get(actif_nom, 0)) if actif_nom in sortino_brut.index else 0

    # Corrélation moyenne avec le reste du portefeuille
    autres = [a for a in all_actifs_port if a != actif_nom and a in corr_matrix.index]
    if actif_nom in corr_matrix.index and len(autres) > 0:
        corr_moyenne = corr_matrix.loc[actif_nom, autres].mean()
        resultats["correlation_moyenne"] = float(corr_moyenne)
    else:
        resultats["correlation_moyenne"] = 0.0

    # Score de santé global (0-100)
    score = 50  # Base neutre
    if resultats["au_dessus_sma200"]:
        score += 15
    elif resultats["au_dessus_sma200"] is False:
        score -= 20
    if resultats["au_dessus_sma50"]:
        score += 10
    elif resultats["au_dessus_sma50"] is False:
        score -= 10
    if resultats["sortino"] > 1.0:
        score += 15
    elif resultats["sortino"] < 0:
        score -= 15
    if resultats["volatilite_annualisee"] < 0.20:
        score += 10
    elif resultats["volatilite_annualisee"] > 0.40:
        score -= 10
    if resultats["correlation_moyenne"] < 0.3:
        score += 10  # Bonne diversification
    elif resultats["correlation_moyenne"] > 0.7:
        score -= 10

    resultats["score_sante"] = max(0, min(100, score))

    # Recommandation
    if resultats["score_sante"] >= 70:
        resultats["recommandation"] = "🟢 RENFORCER"
    elif resultats["score_sante"] >= 40:
        resultats["recommandation"] = "🟡 CONSERVER"
    else:
        resultats["recommandation"] = "🔴 ALLÉGER"

    return resultats


# Calculer la santé de chaque position
actifs_portefeuille = st.session_state.mon_portefeuille["Actif"].tolist()
sante_positions = []
for actif in actifs_portefeuille:
    s = calculer_sante_position(
        actif, rendements_hebdo, volatilite, max_dd, correlation,
        df_brut, taux_fx, actifs_portefeuille
    )
    if s:
        sante_positions.append(s)

# Score de diversification global
def calculer_score_diversification(actifs_port, corr_matrix):
    """Score de 0 (tout corrélé) à 100 (parfaitement diversifié)."""
    valides = [a for a in actifs_port if a in corr_matrix.index]
    if len(valides) < 2:
        return 50
    sous_matrice = corr_matrix.loc[valides, valides]
    n = len(valides)
    # Moyenne des corrélations hors diagonale
    masque = ~np.eye(n, dtype=bool)
    corr_moy = abs(sous_matrice.values[masque]).mean()
    # Score inversé : moins c'est corrélé, mieux c'est
    score = int((1 - corr_moy) * 100)
    return max(0, min(100, score))

score_diversification = calculer_score_diversification(actifs_portefeuille, correlation)


# ==========================================
# INTERFACE PRINCIPALE
# ==========================================
st.markdown(
    "<h2 style='font-family: monospace; border-bottom: 1px solid #444; "
    "padding-bottom: 10px;'>BUREAU CONSEIL & EXÉCUTION (V33)</h2>",
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "MÉTÉO DU MARCHÉ (IA)", regime_marche,
    delta=f"Score Risque: {risk_score}/100",
    delta_color="normal" if risk_score < 40 else "inverse",
)
col2.metric(
    "POIDS CŒUR (SÉCURISÉ)", f"{capital_verrouille:.2f} €",
    delta="Sanctuarisé", delta_color="off",
)
col3.metric(
    "POIDS SATELLITE (ACTIF)", f"{budget_satellite:.2f} €",
    delta="Géré par l'IA", delta_color="normal",
)
reserve_pct_display = (reserve_cash / budget * 100) if budget > 0 else 0
col4.metric(
    "LIQUIDITÉS RECOMMANDÉES", f"{reserve_cash:.2f} €",
    delta=f"{reserve_pct_display:.1f}% de protection", delta_color="off",
)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "💡 CONSEIL & EXÉCUTION (OMS)",
    "MATRICE D'ALLOCATION CIBLE",
    "🩺 BILAN DE SANTÉ",
    "🔗 DIVERSIFICATION",
    "🔮 PROJECTIONS MONTE CARLO",
    "ANALYSE DE RISQUE (PCA)",
    "STRESS-TESTS HISTORIQUES",
])


# =================================================================
# ONGLET 1 : CONSEIL & EXÉCUTION
# =================================================================
with tab1:
    col_ed1, col_ed2 = st.columns([1, 1.5])
    with col_ed1:
        st.markdown("**1. Vos Positions Actuelles (Trade Republic)**")
        config_colonnes = {
            "Actif": st.column_config.SelectboxColumn("Instrument", options=list(univers_etudie.keys()), required=True),
            "Valeur (EUR)": st.column_config.NumberColumn("Valeur Actuelle (€)", min_value=0.0, step=1.0, format="%.2f"),
        }
        df_edited = st.data_editor(
            st.session_state.mon_portefeuille,
            column_config=config_colonnes, num_rows="dynamic", use_container_width=True,
        )
        # Appliquer les éditions pour le prochain rerun
        df_edited_clean = df_edited.dropna(subset=["Actif"])
        df_edited_clean = df_edited_clean[df_edited_clean["Actif"].isin(univers_etudie.keys())]
        df_edited_clean = df_edited_clean.drop_duplicates(subset=["Actif"], keep="first")
        df_edited_clean["Valeur (EUR)"] = df_edited_clean["Valeur (EUR)"].clip(lower=0).fillna(0)
        st.session_state.mon_portefeuille = df_edited_clean.reset_index(drop=True)

    with col_ed2:
        st.markdown("**2. Le Ticket d'Ordres de l'IA**")
        lignes_ordres = []
        dict_actuel = {
            row["Actif"]: row["Valeur (EUR)"]
            for _, row in st.session_state.mon_portefeuille.iterrows()
            if pd.notna(row["Actif"])
        }
        if len(allocations) > 0:
            tous_actifs_ordres = set(allocations[allocations > 0].index) | set(dict_actuel.keys())
            for a in tous_actifs_ordres:
                if a not in univers_etudie:
                    continue
                val_cible_raw = allocations.get(a, 0.0)
                val_cible = float(val_cible_raw) if pd.notna(val_cible_raw) else 0.0
                val_actuelle = float(dict_actuel.get(a, 0.0))
                delta = val_cible - val_actuelle

                # Frais TR : ordre rentable seulement si |delta| > seuil
                frais_total = FRAIS_ORDRE_TR + abs(delta) * SPREAD_ESTIME_TR
                if abs(delta) < min_trade_size:
                    action = "CONSERVER"
                elif delta > 0:
                    action = "ACHETER"
                else:
                    action = "VENDRE"

                if abs(delta) > 0.01:
                    lignes_ordres.append({
                        "Instrument": univers_etudie[a]["nom"],
                        "Cible": val_cible,
                        "Actuel": val_actuelle,
                        "Ordre Net (EUR)": delta,
                        "Frais TR": frais_total if action != "CONSERVER" else 0.0,
                        "Action": action,
                    })

            if lignes_ordres:
                df_ordres = pd.DataFrame(lignes_ordres).sort_values("Ordre Net (EUR)", ascending=False)
                total_frais = sum(r["Frais TR"] for r in lignes_ordres if r["Action"] != "CONSERVER")
                st.dataframe(
                    df_ordres.style.format({
                        "Cible": "{:.2f} €", "Actuel": "{:.2f} €",
                        "Ordre Net (EUR)": "{:+.2f} €", "Frais TR": "{:.2f} €",
                    }).map(
                        lambda x: (
                            "color: #00cc00; font-weight: bold;" if x == "ACHETER"
                            else ("color: #ff4b4b; font-weight: bold;" if x == "VENDRE" else "color: #888888;")
                        ),
                        subset=["Action"],
                    ),
                    use_container_width=True,
                )
                if total_frais > 0:
                    st.caption(f"💰 Coût total estimé des ordres sur Trade Republic : **{total_frais:.2f} €** (frais 1€/ordre + spread ~0.15%)")
            else:
                st.success("🎉 Portefeuille optimisé. Aucun ordre nécessaire.")

    st.markdown("---")
    st.markdown("### 🧠 Le Diagnostic du Quant (Justification des choix)")

    st.info(
        f"**Vision Macroéconomique :** Le logiciel détecte un **{regime_marche}**. "
        f"Score de panique médiatique (NLP) : {avg_nlp_score:.2f}. "
        f"Consigne : conserver **{reserve_cash:.2f} € en liquidités** "
        f"pour respecter votre cible de volatilité de {target_volatility * 100:.0f}%."
    )

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("#### 🔴 Pourquoi l'IA vend certains actifs ?")
        a_vendre = [r["Instrument"] for r in lignes_ordres if r["Action"] == "VENDRE"]
        if not a_vendre:
            st.write("Aucune vente recommandée. Vos actifs non-verrouillés restent pertinents.")
        else:
            for instr in a_vendre:
                ticker = next((k for k, v in univers_etudie.items() if v["nom"] == instr), None)
                if ticker:
                    z = df_zscore.loc[df_zscore["Actif"] == ticker, "Score_Global"].values
                    z_val = z[0] if len(z) > 0 else -99
                    if z_val == -99:
                        st.write(f"- **{instr} :** Fondamentaux ou momentum trop dégradés.")
                    elif z_val < 0:
                        st.write(f"- **{instr} :** Score Qualité négatif ({z_val:.2f}). D'autres actifs offrent un meilleur ratio risque/rendement.")
                    else:
                        st.write(f"- **{instr} :** Surpondéré par rapport au risque global. Prise de profits recommandée.")

    with col_c2:
        st.markdown("#### 🟢 Pourquoi l'IA achète ces actifs ?")
        a_acheter = [r["Instrument"] for r in lignes_ordres if r["Action"] == "ACHETER"]
        if not a_acheter:
            st.write("Aucun achat recommandé. Le portefeuille actuel est bien positionné.")
        else:
            for instr in a_acheter:
                ticker = next((k for k, v in univers_etudie.items() if v["nom"] == instr), None)
                if ticker == "Bons du Trésor US 20A+":
                    st.write(f"- **{instr} :** Couverture d'urgence anti-krach activée.")
                elif ticker:
                    z = df_zscore.loc[df_zscore["Actif"] == ticker, "Score_Global"].values
                    z_val = z[0] if len(z) > 0 else 0
                    pred = ml_predictions.get(ticker, 0)
                    ml_txt = "ML prédit une hausse." if pred > 0 else "Fondamentaux solides."
                    st.write(f"- **{instr} :** Score Global **{z_val:.2f}**. {ml_txt}")


# =================================================================
# ONGLET 2 : MATRICE D'ALLOCATION
# =================================================================
with tab2:
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    ret_display = f"{expected_portfolio_return * 100:.1f}%" if not np.isnan(expected_portfolio_return) else "N/A"
    sharpe_display = f"{expected_portfolio_return / port_vol_initial:.2f}" if port_vol_initial > 0 else "N/A"
    col_k1.metric("Rendement Annuel Espéré", ret_display)
    col_k2.metric("Ratio de Sharpe", sharpe_display)
    ml_actif = any(v != 0 for v in ml_predictions.values())
    col_k3.metric("Boost Prédictif (IA)", "ACTIF" if ml_actif else "INACTIF", "Gradient Boosting (9 features)")
    col_k4.metric("Covariance sous Stress", "ACTIVE", "Blend Normal/Crash 70/30")

    tous_actifs_tab = list(dict.fromkeys(list(allocations.index) + top_20_candidats[:15]))
    donnees_tab = []
    for a in tous_actifs_tab:
        if a not in univers_etudie:
            continue
        statut = (
            "VERROUILLÉ (CŒUR)" if a in actifs_verrouilles
            else ("ALLOUÉ" if a in allocations and allocations.get(a, 0) > 0 else "REJETÉ")
        )
        z_str = "N/A"
        if not df_zscore.empty and a in df_zscore["Actif"].values:
            z_str = f"{df_zscore.loc[df_zscore['Actif'] == a, 'Score_Global'].values[0]:.2f}"
        # Conversion robuste : jamais de None/NaN
        cap_cible_raw = allocations.get(a, 0.0)
        cap_cible = float(cap_cible_raw) if pd.notna(cap_cible_raw) else 0.0
        donnees_tab.append({
            "Instrument (Ticker)": f"{univers_etudie[a]['nom']} [{univers_etudie[a]['ticker']}]",
            "Statut": statut,
            "Score Qualité (Z)": z_str,
            "Capital Cible": cap_cible,
        })
    if donnees_tab:
        st.dataframe(
            pd.DataFrame(donnees_tab).sort_values("Capital Cible", ascending=False)
            .style.format({"Capital Cible": "{:.2f} €"})
            .map(
                lambda v: (
                    "background-color: #2c3e50;" if "VERROUILLÉ" in str(v)
                    else ("background-color: #1a4222;" if "ALLOUÉ" in str(v) else "color: #8b0000;")
                ),
                subset=["Statut"],
            ),
            use_container_width=True, height=450,
        )


# =================================================================
# ONGLET 3 : BILAN DE SANTÉ (nouveau V33)
# =================================================================
with tab3:
    st.markdown("### 🩺 Diagnostic individuel de chaque position")
    st.markdown(
        f"**Score de diversification global : {score_diversification}/100** "
        + ("🟢 Excellent" if score_diversification >= 70 else "🟡 Correct" if score_diversification >= 40 else "🔴 Insuffisant")
    )
    st.markdown("---")

    if sante_positions:
        for sp in sorted(sante_positions, key=lambda x: x["score_sante"], reverse=True):
            valeur_pos = dict_actuel.get(sp["actif"], 0)
            poids_pct = (valeur_pos / budget * 100) if budget > 0 else 0

            with st.expander(f"{sp['recommandation']}  **{sp['actif']}** — Santé: {sp['score_sante']}/100 — {poids_pct:.1f}% du portefeuille", expanded=False):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Score Santé", f"{sp['score_sante']}/100")
                c2.metric("Volatilité", f"{sp['volatilite_annualisee']*100:.1f}%")
                c3.metric("Max Drawdown", f"{sp['max_drawdown']*100:.1f}%")
                c4.metric("Sortino", f"{sp['sortino']:.2f}")

                c5, c6, c7, c8 = st.columns(4)
                sma200_txt = "✅ Oui" if sp["au_dessus_sma200"] else ("❌ Non" if sp["au_dessus_sma200"] is False else "N/A")
                sma50_txt = "✅ Oui" if sp["au_dessus_sma50"] else ("❌ Non" if sp["au_dessus_sma50"] is False else "N/A")
                c5.metric("Au-dessus SMA 200j", sma200_txt)
                c6.metric("Au-dessus SMA 50j", sma50_txt)
                c7.metric("Distance SMA 200j", f"{sp['distance_sma200_pct']:+.1f}%")
                c8.metric("Corrél. moy. portefeuille", f"{sp['correlation_moyenne']:.2f}")

                # Graphique de l'actif (6 mois)
                ticker = sp["ticker"]
                if ticker in df_brut.columns:
                    prix_6m = df_brut[ticker].tail(126).dropna()
                    if len(prix_6m) > 10:
                        fig_prix = go.Figure()
                        fig_prix.add_trace(go.Scatter(x=prix_6m.index, y=prix_6m.values, mode="lines", name="Prix", line=dict(color="#4a90e2")))
                        if len(df_brut[ticker].dropna()) >= 200:
                            sma200_series = df_brut[ticker].rolling(200).mean().tail(126)
                            fig_prix.add_trace(go.Scatter(x=sma200_series.index, y=sma200_series.values, mode="lines", name="SMA 200", line=dict(color="#ff6b6b", dash="dot")))
                        fig_prix.update_layout(template="plotly_dark", height=250, margin=dict(t=10, b=10, l=10, r=10), showlegend=True)
                        st.plotly_chart(fig_prix, use_container_width=True)
    else:
        st.warning("Aucune position à analyser.")


# =================================================================
# ONGLET 4 : DIVERSIFICATION & CORRÉLATION (nouveau V33)
# =================================================================
with tab4:
    st.markdown("### 🔗 Matrice de Corrélation de votre Portefeuille")

    actifs_corr = [a for a in actifs_portefeuille if a in correlation.index]
    if len(actifs_corr) > 1:
        corr_sub = correlation.loc[actifs_corr, actifs_corr]

        fig_heatmap = go.Figure(data=go.Heatmap(
            z=corr_sub.values,
            x=corr_sub.columns.tolist(),
            y=corr_sub.index.tolist(),
            colorscale="RdBu_r",
            zmin=-1, zmax=1,
            text=np.round(corr_sub.values, 2),
            texttemplate="%{text}",
            textfont={"size": 11},
        ))
        fig_heatmap.update_layout(
            template="plotly_dark", height=500,
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

        st.markdown(f"**Score de Diversification : {score_diversification}/100**")
        st.markdown(
            "Un score élevé signifie que vos actifs bougent indépendamment les uns des autres, "
            "ce qui réduit le risque qu'ils chutent tous en même temps."
        )

        # Paires les plus corrélées
        st.markdown("#### ⚠️ Paires les plus corrélées (risque de concentration)")
        paires = []
        for i, a1 in enumerate(actifs_corr):
            for a2 in actifs_corr[i+1:]:
                paires.append({"Actif 1": a1, "Actif 2": a2, "Corrélation": corr_sub.loc[a1, a2]})
        if paires:
            df_paires = pd.DataFrame(paires).sort_values("Corrélation", ascending=False)
            alertes = df_paires[df_paires["Corrélation"] > 0.5]
            if not alertes.empty:
                st.dataframe(alertes.style.format({"Corrélation": "{:.2f}"}), use_container_width=True)
            else:
                st.success("Aucune paire n'a une corrélation supérieure à 0.50. Votre diversification est bonne.")
    else:
        st.warning("Il faut au moins 2 positions pour analyser la diversification.")


# =================================================================
# ONGLET 5 : PROJECTIONS MONTE CARLO (nouveau V33)
# =================================================================
with tab5:
    st.markdown("### 🔮 Projections Monte Carlo de votre portefeuille")
    st.markdown("*Simulation de 1000 trajectoires possibles sur 1 an, basée sur les rendements historiques de vos actifs.*")

    actifs_mc = [a for a in actifs_portefeuille if a in rendements_hebdo.columns]
    if len(actifs_mc) > 0 and budget > 0:
        # Poids du portefeuille actuel
        poids_mc = np.array([dict_actuel.get(a, 0) for a in actifs_mc])
        if poids_mc.sum() > 0:
            poids_mc = poids_mc / poids_mc.sum()
        else:
            poids_mc = np.ones(len(actifs_mc)) / len(actifs_mc)

        ret_mc = rendements_hebdo[actifs_mc].dropna()
        mu_hebdo = ret_mc.mean().values
        sigma_hebdo = ret_mc.cov().values

        n_simulations = 1000
        n_semaines = 52
        np.random.seed(42)

        trajectoires = np.zeros((n_simulations, n_semaines + 1))
        trajectoires[:, 0] = budget

        for sim in range(n_simulations):
            val = budget
            for sem in range(n_semaines):
                rendements_sim = np.random.multivariate_normal(mu_hebdo, sigma_hebdo)
                ret_port = np.dot(poids_mc, rendements_sim)
                val *= (1 + ret_port)
                trajectoires[sim, sem + 1] = val

        # Percentiles
        p5 = np.percentile(trajectoires, 5, axis=0)
        p25 = np.percentile(trajectoires, 25, axis=0)
        p50 = np.percentile(trajectoires, 50, axis=0)
        p75 = np.percentile(trajectoires, 75, axis=0)
        p95 = np.percentile(trajectoires, 95, axis=0)

        semaines = list(range(n_semaines + 1))

        fig_mc = go.Figure()
        fig_mc.add_trace(go.Scatter(x=semaines, y=p95, mode="lines", name="Optimiste (95e)", line=dict(color="#00cc00", dash="dot")))
        fig_mc.add_trace(go.Scatter(x=semaines, y=p75, mode="lines", name="Favorable (75e)", line=dict(color="#66cc66")))
        fig_mc.add_trace(go.Scatter(x=semaines, y=p50, mode="lines", name="Médian (50e)", line=dict(color="#4a90e2", width=3)))
        fig_mc.add_trace(go.Scatter(x=semaines, y=p25, mode="lines", name="Défavorable (25e)", line=dict(color="#cc6666")))
        fig_mc.add_trace(go.Scatter(x=semaines, y=p5, mode="lines", name="Pessimiste (5e)", line=dict(color="#ff4b4b", dash="dot")))
        fig_mc.add_hline(y=budget, line_dash="dash", line_color="white", annotation_text="Capital initial")
        fig_mc.update_layout(
            template="plotly_dark", height=500,
            xaxis_title="Semaines", yaxis_title="Valeur du portefeuille (€)",
            margin=dict(t=10, b=40, l=40, r=10),
        )
        st.plotly_chart(fig_mc, use_container_width=True)

        # Statistiques clés
        val_finales = trajectoires[:, -1]
        proba_perte = (val_finales < budget).mean() * 100
        proba_perte_10 = (val_finales < budget * 0.90).mean() * 100
        gain_median = ((p50[-1] / budget) - 1) * 100
        pire_cas = ((p5[-1] / budget) - 1) * 100
        meilleur_cas = ((p95[-1] / budget) - 1) * 100

        col_mc1, col_mc2, col_mc3, col_mc4, col_mc5 = st.columns(5)
        col_mc1.metric("Probabilité de perte", f"{proba_perte:.1f}%",
                       delta="⚠️ Attention" if proba_perte > 50 else "✅ Favorable",
                       delta_color="inverse" if proba_perte > 50 else "normal")
        col_mc2.metric("Prob. perte > 10%", f"{proba_perte_10:.1f}%")
        col_mc3.metric("Scénario médian (1 an)", f"{gain_median:+.1f}%")
        col_mc4.metric("Pire scénario (5e %)", f"{pire_cas:+.1f}%")
        col_mc5.metric("Meilleur scénario (95e %)", f"{meilleur_cas:+.1f}%")

        st.markdown("---")
        st.markdown("#### 📊 Distribution des rendements à 1 an")
        fig_hist = px.histogram(
            x=((val_finales / budget) - 1) * 100,
            nbins=50,
            labels={"x": "Rendement (%)"},
            color_discrete_sequence=["#4a90e2"],
        )
        fig_hist.add_vline(x=0, line_dash="dash", line_color="white", annotation_text="Seuil 0%")
        fig_hist.update_layout(template="plotly_dark", height=300, margin=dict(t=10, b=40, l=40, r=10), showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

        # Value at Risk
        var_95 = np.percentile(val_finales - budget, 5)
        cvar_95 = val_finales[val_finales < budget + var_95].mean() - budget if any(val_finales < budget + var_95) else var_95
        st.markdown(f"**Value at Risk (95%) sur 1 an :** Vous avez 5% de chances de perdre plus de **{abs(var_95):.2f} €** ({abs(var_95)/budget*100:.1f}% du capital).")
        st.markdown(f"**Conditional VaR (95%) :** Dans les 5% pires scénarios, la perte moyenne serait de **{abs(cvar_95):.2f} €**.")
    else:
        st.warning("Données insuffisantes pour les projections.")


# =================================================================
# ONGLET 6 : ANALYSE DE RISQUE (PCA)
# =================================================================
with tab6:
    col_pca1, col_pca2 = st.columns([1, 1.5])
    with col_pca1:
        st.markdown("**Analyse en Composantes Principales (PCA)**")
        st.write("Identification des forces invisibles dirigeant le portefeuille.")
        st.metric("Facteur Fantôme 1 (Risque Global)", f"{pca_explained_variance[0]:.1f}% de variance expliquée")
        if len(pca_explained_variance) > 1:
            st.metric("Facteur Fantôme 2 (Secteur/Style)", f"{pca_explained_variance[1]:.1f}% de variance expliquée")
    with col_pca2:
        st.markdown("**Exposition Bêta Macroéconomique**")
        if len(allocations) > 0:
            tickers_alloc = [
                univers_etudie[a]["ticker"] for a in allocations.index
                if a in univers_etudie and univers_etudie[a]["ticker"] in df_brut.columns
            ]
            poids_alloc = np.array([
                float(allocations[a]) / budget if pd.notna(allocations[a]) else 0.0
                for a in allocations.index
                if a in univers_etudie and univers_etudie[a]["ticker"] in df_brut.columns
            ])
            if len(tickers_alloc) > 0 and poids_alloc.sum() > 0:
                ret_port = (df_brut[tickers_alloc].pct_change().dropna() * poids_alloc).sum(axis=1)
                df_beta = pd.DataFrame({
                    "Port": ret_port,
                    "S&P500": df_brut["^GSPC"].pct_change(),
                    "Taux US": df_brut["IEF"].pct_change(),
                    "Or": df_brut["GLD"].pct_change(),
                }).dropna()
                if len(df_beta) > 10:
                    betas = []
                    for f in ["S&P500", "Taux US", "Or"]:
                        var_f = df_beta[f].var()
                        betas.append(df_beta["Port"].cov(df_beta[f]) / var_f if var_f > 0 else 0)
                    fig_beta = px.bar(
                        pd.DataFrame({
                            "Facteur Macro": ["Actions Mondiales", "Taux d'Intérêt US", "Or / Inflation"],
                            "Bêta": betas,
                        }),
                        x="Bêta", y="Facteur Macro", orientation="h",
                        color="Bêta", color_continuous_scale="RdBu", range_color=[-1, 1],
                    )
                    fig_beta.update_layout(template="plotly_dark", margin=dict(t=0, b=0, l=0, r=0))
                    st.plotly_chart(fig_beta, use_container_width=True)


# =================================================================
# ONGLET 7 : STRESS-TESTS HISTORIQUES
# =================================================================
with tab7:
    if len(allocations) > 0:
        poids_test = (pd.Series(allocations) / budget).fillna(0)
        actifs_testes = poids_test[poids_test > 0].index.tolist()

        def run_stress_test(df_period, actifs, poids):
            actifs_valides = [
                a for a in actifs
                if a in univers_etudie
                and univers_etudie[a]["ticker"] in df_period.columns
                and df_period[univers_etudie[a]["ticker"]].isna().sum() < 5
            ]
            poids_v = poids.reindex(actifs_valides).dropna()
            if len(actifs_valides) == 0 or poids_v.sum() == 0:
                return None, None, []
            poids_r = poids_v / poids_v.sum()
            tickers_v = [univers_etudie[a]["ticker"] for a in actifs_valides]
            ret = df_period[tickers_v + ["^GSPC"]].pct_change().dropna()
            port_ret = (ret[tickers_v] * poids_r.values).sum(axis=1)
            sp_ret = ret["^GSPC"]
            crois_p = (1 + port_ret).cumprod() * 100
            crois_sp = (1 + sp_ret).cumprod() * 100
            df_g = pd.DataFrame({"Ton Portefeuille (Moteur IA)": crois_p, "Indice S&P 500": crois_sp})
            perte = crois_p.min() - 100
            exclus = [a for a in actifs if a not in actifs_valides]
            return df_g, perte, exclus

        if len(actifs_testes) > 0:
            col_st1, col_st2 = st.columns(2)

            with col_st1:
                st.markdown("**SCÉNARIO A : Krach COVID-19 (Fév-Avr 2020)**")
                g_covid, p_covid, e_covid = run_stress_test(
                    df_brut.loc["2020-02-15":"2020-04-30"], actifs_testes, poids_test
                )
                if g_covid is not None:
                    fig = px.line(g_covid, color_discrete_sequence=["#4a90e2", "#444444"])
                    fig.update_layout(template="plotly_dark", margin=dict(t=10, b=10, l=10, r=10), xaxis_title="", yaxis_title="Base 100")
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"Pire perte : **{p_covid:.1f}%**")
                    if e_covid:
                        st.warning(f"Actifs exclus (données manquantes) : {', '.join(e_covid)}")
                else:
                    st.warning("Données insuffisantes pour 2020.")

            with col_st2:
                st.markdown("**SCÉNARIO B : Choc des Taux (Jan-Oct 2022)**")
                g_inf, p_inf, e_inf = run_stress_test(
                    df_brut.loc["2022-01-01":"2022-10-31"], actifs_testes, poids_test
                )
                if g_inf is not None:
                    fig = px.line(g_inf, color_discrete_sequence=["#4a90e2", "#444444"])
                    fig.update_layout(template="plotly_dark", margin=dict(t=10, b=10, l=10, r=10), xaxis_title="", yaxis_title="")
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"Pire perte : **{p_inf:.1f}%**")
                    if e_inf:
                        st.warning(f"Actifs exclus (données manquantes) : {', '.join(e_inf)}")
                else:
                    st.warning("Données insuffisantes pour 2022.")


# --- Footer ---
st.markdown("---")
st.caption(
    f"Terminal Quantitatif V33 (Mode DCA · Trade Republic) — "
    f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} — "
    f"Taux FX : 1 EUR = {1/taux_fx.get('USD', 1):.4f} USD, "
    f"1 EUR = {1/taux_fx.get('GBP', 1):.4f} GBP — "
    f"Frais TR : {FRAIS_ORDRE_TR}€/ordre"
)

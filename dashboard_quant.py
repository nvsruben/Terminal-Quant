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
import logging
import warnings

warnings.filterwarnings('ignore')

HMM_DISPONIBLE = False
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_DISPONIBLE = True
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(page_title="TERMINAL QUANTITATIF V37", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# CSS PROFESSIONNEL
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');
    .stApp { font-family: 'IBM Plex Sans', sans-serif; }
    h1, h2, h3, h4, h5, h6 { font-family: 'JetBrains Mono', monospace !important; letter-spacing: 0.05em !important; text-transform: uppercase !important; }
    [data-testid="stSidebar"] { background-color: #0a0a0a !important; border-right: 1px solid #1a1a2e !important; }
    [data-testid="stSidebar"] h3 { font-size: 0.75rem !important; color: #6c7293 !important; letter-spacing: 0.15em !important; }
    [data-testid="stMetric"] { background-color: #0d0d1a; border: 1px solid #1a1a2e; border-radius: 2px; padding: 12px 16px; }
    [data-testid="stMetricLabel"] { font-family: 'JetBrains Mono', monospace !important; font-size: 0.65rem !important; letter-spacing: 0.12em !important; color: #6c7293 !important; text-transform: uppercase !important; }
    [data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; font-weight: 500 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 0px; border-bottom: 1px solid #1a1a2e; }
    .stTabs [data-baseweb="tab"] { font-family: 'JetBrains Mono', monospace !important; font-size: 0.7rem !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; padding: 8px 16px !important; color: #6c7293 !important; border-radius: 0 !important; }
    .stTabs [aria-selected="true"] { color: #ffffff !important; border-bottom: 2px solid #4a90e2 !important; background-color: transparent !important; }
    .stDataFrame { border: 1px solid #1a1a2e; border-radius: 2px; }
    .streamlit-expanderHeader { font-family: 'IBM Plex Sans', sans-serif !important; font-size: 0.85rem !important; border: 1px solid #1a1a2e !important; border-radius: 2px !important; background-color: #0d0d1a !important; }
    .stAlert { border-radius: 2px !important; border-left: 3px solid #4a90e2 !important; }
    .section-header { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; letter-spacing: 0.15em; color: #6c7293; text-transform: uppercase; border-bottom: 1px solid #1a1a2e; padding-bottom: 4px; margin: 16px 0 12px 0; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# CONSTANTES
# ==========================================
FRAIS_ORDRE_TR = 1.0
SPREAD_ESTIME_TR = 0.0015
FLAT_TAX_FR = 0.30
TAUX_CASH_TR_BRUT = 0.0375
TAUX_CASH_TR_NET = TAUX_CASH_TR_BRUT * (1 - FLAT_TAX_FR)
TICKER_MONETAIRE = "XEON.DE"
HURDLE_RATE = 2.0  # Un nouveau candidat doit avoir un Z-Score supérieur de 2.0 pour justifier un swap

# ==========================================
# AUTHENTIFICATION
# ==========================================
if "authentifie" not in st.session_state:
    st.session_state.authentifie = False
if not st.session_state.authentifie:
    st.markdown("<h1 style='text-align:center;color:#fff;font-family:JetBrains Mono,monospace;letter-spacing:0.1em;font-weight:300;'>BUREAU D'ALLOCATION QUANTITATIVE</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#6c7293;font-family:JetBrains Mono,monospace;font-size:0.75rem;letter-spacing:0.15em;'>V37 &mdash; MODE FORTERESSE &mdash; SYSTEME CLOS</p>", unsafe_allow_html=True)
    _, c, _ = st.columns([1, 1, 1])
    with c:
        mdp = st.text_input("Cle d'Acces", type="password")
        if st.button("INITIALISER", use_container_width=True):
            if mdp == "BTSCG2026":
                st.session_state.authentifie = True
                st.rerun()
            else:
                st.error("ACCES REFUSE.")
    st.stop()

# ==========================================
# FX
# ==========================================
DEVISE_PAR_TICKER = {
    "BTC-EUR": "EUR", "ETH-EUR": "EUR", "IGLN.L": "GBP", "PHAG.L": "GBP",
    "TLT": "USD", "DFNS.L": "GBP", "RHM.DE": "EUR", "PLTR": "USD", "URNM": "USD",
    "CSPX.AS": "EUR", "DSY.PA": "EUR", "TTE.PA": "EUR", "MC.PA": "EUR",
    "HYG": "USD", "IEF": "USD", "GLD": "USD",
    "^VIX": "USD", "^TNX": "USD", "^GSPC": "USD", "^IRX": "USD",
}

@st.cache_data(ttl=3600)
def obtenir_taux_change():
    try:
        fx = yf.download(["EURUSD=X", "EURGBP=X"], period="5d", progress=False)["Close"]
        fx = fx.ffill().bfill()
        return {"USD": 1.0 / float(fx["EURUSD=X"].iloc[-1]), "GBP": 1.0 / float(fx["EURGBP=X"].iloc[-1]), "EUR": 1.0}
    except Exception as e:
        logger.warning(f"FX fallback — {e}")
        return {"USD": 1.0, "GBP": 1.0, "EUR": 1.0}

def devise_ticker(t): return DEVISE_PAR_TICKER.get(t, "USD")
def prix_en_eur(prix, ticker, fx): return prix * fx.get(devise_ticker(ticker), 1.0)

# ==========================================
# UNIVERS D'INVESTISSEMENT (MULTI-MARCHES)
# ==========================================
MES_FAVORIS = {
    "Bitcoin": {"ticker": "BTC-EUR", "nom": "Bitcoin (Crypto)"},
    "Ethereum": {"ticker": "ETH-EUR", "nom": "Ethereum (Crypto)"},
    "Or Physique": {"ticker": "IGLN.L", "nom": "iShares Physical Gold ETC"},
    "Argent Physique": {"ticker": "PHAG.L", "nom": "WisdomTree Physical Silver"},
    "Bons du Tresor US 20A+": {"ticker": "TLT", "nom": "iShares 20+ Year Treasury Bond"},
    "Defense USD": {"ticker": "DFNS.L", "nom": "VanEck Defense UCITS"},
    "Rheinmetall": {"ticker": "RHM.DE", "nom": "Rheinmetall AG"},
    "Palantir": {"ticker": "PLTR", "nom": "Palantir Technologies"},
    "Uranium USD": {"ticker": "URNM", "nom": "Sprott Uranium Miners ETF"},
    "Core S&P 500": {"ticker": "CSPX.AS", "nom": "iShares Core S&P 500 UCITS"},
    "Dassault Systemes": {"ticker": "DSY.PA", "nom": "Dassault Systemes SE"},
    "TotalEnergies": {"ticker": "TTE.PA", "nom": "TotalEnergies SE"},
    "LVMH": {"ticker": "MC.PA", "nom": "LVMH Moet Hennessy"},
}

MOTS_CLES_ETF_CRYPTO = ["Crypto", "ETF", "UCITS", "ETC", "Fund", "iShares", "WisdomTree", "VanEck", "Sprott", "SPDR", "Vanguard", "Amundi", "Lyxor"]
def est_etf_ou_crypto(nom): return any(kw.lower() in nom.lower() for kw in MOTS_CLES_ETF_CRYPTO)

def _scrape_index(url, sym_col, name_col, prefix, ticker_transform=None):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        html = requests.get(url, headers=headers, timeout=15).text
        table = pd.read_html(html)[0]
        d = {}
        for _, row in table.iterrows():
            t = str(row[sym_col]).strip().replace(".", "-")
            if ticker_transform:
                t = ticker_transform(t)
            n = str(row[name_col]).strip()
            if t and n and len(t) < 12:
                d[f"{prefix}: {t}"] = {"ticker": t, "nom": n}
        return d
    except Exception as e:
        logger.warning(f"Scraping {prefix} echoue : {e}")
        return {}

@st.cache_data(ttl=86400)
def aspirer_sp500():
    return _scrape_index("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol", "Security", "S&P500")

@st.cache_data(ttl=86400)
def aspirer_nasdaq100():
    return _scrape_index("https://en.wikipedia.org/wiki/Nasdaq-100", "Ticker", "Company", "NDX100")

@st.cache_data(ttl=86400)
def aspirer_eurostoxx50():
    return _scrape_index("https://en.wikipedia.org/wiki/EURO_STOXX_50", "Ticker", "Company", "STOXX50")

# Construire l'univers mondial
univers_etudie = MES_FAVORIS.copy()
tickers_existants = set(v["ticker"] for v in univers_etudie.values())
for scraper in [aspirer_sp500, aspirer_nasdaq100, aspirer_eurostoxx50]:
    for cle, donnees in scraper().items():
        if donnees["ticker"] not in tickers_existants:
            univers_etudie[cle] = donnees
            tickers_existants.add(donnees["ticker"])

# ==========================================
# DATA LOADING
# ==========================================
TICKERS_MACRO = ["^VIX", "^TNX", "^GSPC", "^IRX", "HYG", "IEF", "GLD"]

@st.cache_data(ttl=3600)
def telecharger_donnees(liste_tickers):
    tickers_complets = list(set(liste_tickers + TICKERS_MACRO))
    df = yf.download(tickers_complets, period="6y", progress=False)["Close"]
    return df.ffill().bfill()

@st.cache_data(ttl=86400)
def obtenir_fondamentaux(ticker_str):
    try:
        info = yf.Ticker(ticker_str).info
        return {"trailingPE": info.get("trailingPE"), "returnOnEquity": info.get("returnOnEquity"), "recommendationMean": info.get("recommendationMean")}
    except Exception as e:
        logger.debug(f"Fondamentaux indisponibles: {ticker_str}: {e}")
        return {}

@st.cache_data(ttl=3600)
def analyser_sentiment_nlp():
    try:
        analyzer = SentimentIntensityAnalyzer()
        toutes = []
        for t in ["^GSPC", "TLT", "GLD"]:
            news = yf.Ticker(t).news
            if news: toutes.extend(news)
        if not toutes: return 0.0
        toutes = sorted(toutes, key=lambda x: x.get("providerPublishTime", 0), reverse=True)[:20]
        return float(np.mean([analyzer.polarity_scores(i.get("title", ""))["compound"] for i in toutes]))
    except: return 0.0

def calculate_z_score(s):
    std = s.std()
    return (s - s.mean()) / std if std > 0 and not pd.isna(std) else pd.Series(0, index=s.index)

# ==========================================
# PORTEFEUILLE — SYSTEME CLOS
# ==========================================
if "mon_portefeuille" not in st.session_state or "Valeur (EUR)" in getattr(st.session_state.get("mon_portefeuille", pd.DataFrame()), "columns", []):
    st.session_state.mon_portefeuille = pd.DataFrame({
        "Actif": ["Core S&P 500", "Bitcoin", "Or Physique", "Palantir", "Argent Physique", "Uranium USD", "Rheinmetall"],
        "Quantite": [0.72, 0.0035, 4.5, 0.65, 1.8, 0.55, 0.015],
        "PRU / Part": [557.0, 84285.0, 22.4, 84.6, 27.8, 19.6, 633.0],
        "Coeur": [True, True, True, False, False, False, False],
    })

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.markdown("<h3 style='font-family:monospace;'>CONTROLES</h3>", unsafe_allow_html=True)
if st.sidebar.button("FORCER L'ACTUALISATION", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='font-family:monospace;'>PARAMETRES</h3>", unsafe_allow_html=True)
seuil_vix = st.sidebar.slider("Seuil VIX (Panique)", 15, 40, 22)
target_volatility = st.sidebar.slider("Volatilite Cible (%)", 5, 25, 12) / 100.0
correl_max = st.sidebar.slider("Correlation Max (%)", 50, 95, 75) / 100.0
cash_tr = st.sidebar.number_input("Cash sur TR (EUR)", min_value=0.0, value=5.0, step=1.0)

# Validation portefeuille
df_port = st.session_state.mon_portefeuille.copy()
df_port = df_port.dropna(subset=["Actif"])
df_port = df_port[df_port["Actif"].isin(univers_etudie.keys())].drop_duplicates(subset=["Actif"], keep="first")
df_port["Quantite"] = pd.to_numeric(df_port["Quantite"], errors="coerce").clip(lower=0).fillna(0)
df_port["PRU / Part"] = pd.to_numeric(df_port["PRU / Part"], errors="coerce").clip(lower=0).fillna(0)
if "Coeur" not in df_port.columns: df_port["Coeur"] = False
st.session_state.mon_portefeuille = df_port.reset_index(drop=True)

# ==========================================
# MOTEUR
# ==========================================
with st.spinner("Analyse en cours..."):
    liste_tickers = [v["ticker"] for v in univers_etudie.values()]
    df_brut = telecharger_donnees(liste_tickers)
    taux_fx = obtenir_taux_change()
    avg_nlp = analyser_sentiment_nlp()

# Mark-to-Market
def valoriser(actif, qty, df_prix, fx):
    if actif not in univers_etudie or qty <= 0: return 0.0
    t = univers_etudie[actif]["ticker"]
    if t not in df_prix.columns: return 0.0
    try:
        p = float(df_prix[t].iloc[-1])
        return qty * prix_en_eur(p, t, fx) if not pd.isna(p) and p > 0 else 0.0
    except: return 0.0

dict_actuel, dict_pru, dict_qty = {}, {}, {}
for _, r in st.session_state.mon_portefeuille.iterrows():
    a = r["Actif"]
    if pd.isna(a) or a not in univers_etudie: continue
    q = float(r.get("Quantite", 0))
    dict_actuel[a] = valoriser(a, q, df_brut, taux_fx)
    dict_pru[a] = q * float(r.get("PRU / Part", 0))
    dict_qty[a] = q

budget_actions = sum(dict_actuel.values())
budget = budget_actions + cash_tr
st.sidebar.markdown("---")
st.sidebar.metric("Actions (Live)", f"{budget_actions:.2f} EUR")
st.sidebar.metric("Cash TR", f"{cash_tr:.2f} EUR")
st.sidebar.metric("Capital Total", f"{budget:.2f} EUR")

# ==========================================
# REGIME K-MEANS
# ==========================================
df_macro = pd.DataFrame({
    "VIX": df_brut["^VIX"], "Spread": df_brut["^TNX"] - df_brut["^IRX"],
    "Credit": df_brut["HYG"] / df_brut["IEF"], "SP_Ret": df_brut["^GSPC"].pct_change(),
}).dropna()
scaler = StandardScaler()
scaled = scaler.fit_transform(df_macro)
init_c = scaler.transform(np.array([[15,1.5,1.05,0.002],[22,0.5,1.0,-0.001],[35,-0.5,0.95,-0.005]]))
kmeans = KMeans(n_clusters=3, init=init_c, n_init=1, random_state=42).fit(scaled)
cluster_vix = df_macro.assign(C=kmeans.labels_).groupby("C")["VIX"].mean()
bear_c = cluster_vix.idxmax()
current_c = kmeans.predict(scaled[-1].reshape(1,-1))[0]

vix = float(df_brut["^VIX"].iloc[-1])
taux_10y = float(df_brut["^TNX"].iloc[-1])
taux_3m = float(df_brut["^IRX"].iloc[-1])
curve_inv = (taux_10y - taux_3m) < 0
credit_stress = float(df_brut["HYG"].iloc[-1]/df_brut["IEF"].iloc[-1]) < float((df_brut["HYG"]/df_brut["IEF"]).tail(50).mean())
sp_close = float(df_brut["^GSPC"].iloc[-1])
sp_sma200 = float(df_brut["^GSPC"].tail(200).mean())

risk_score = 0
if avg_nlp < -0.15: risk_score += 20
elif avg_nlp > 0.15: risk_score -= 10
if sp_close < sp_sma200: risk_score += 40
if curve_inv: risk_score += 30
if credit_stress: risk_score += 30
if vix > seuil_vix: risk_score += 20
risk_score = max(0, min(100, risk_score))

if risk_score >= 70 or current_c == bear_c:
    regime = "KRACH IMMINENT"
    tail_hedge = True
elif risk_score >= 40:
    regime = "DEFENSIF"
    tail_hedge = False
else:
    regime = "HAUSSIER"
    tail_hedge = False

# HMM
hmm_proba_krach, hmm_transmat, hmm_regime = 0.0, None, -1
hmm_names = {0: "BULL", 1: "NEUTRE", 2: "BEAR"}
if HMM_DISPONIBLE:
    try:
        hf = df_macro.resample("W-FRI").last().dropna()
        hs = StandardScaler()
        hd = hs.fit_transform(hf)
        hm = GaussianHMM(n_components=3, covariance_type="full", n_iter=200, random_state=42, tol=0.01).fit(hd)
        states = hm.predict(hd)
        svix = hf.iloc[:len(states)].assign(S=states).groupby("S")["VIX"].mean()
        bear_s = svix.idxmax(); bull_s = svix.idxmin()
        neut_s = [s for s in range(3) if s != bear_s and s != bull_s][0]
        hmm_names = {bull_s: "BULL", neut_s: "NEUTRE", bear_s: "BEAR"}
        hmm_regime = states[-1]
        hmm_transmat = hm.transmat_
        hmm_proba_krach = float(hmm_transmat[hmm_regime, bear_s])
    except Exception as e:
        logger.warning(f"HMM: {e}")

# ==========================================
# QUANTITATIF
# ==========================================
df_hebdo = df_brut.tail(252*3).resample("W-FRI").last().dropna(how="all")
inv_map = {v["ticker"]: k for k, v in univers_etudie.items()}
df_actifs = df_hebdo[[t for t in liste_tickers if t in df_hebdo.columns]].rename(columns=inv_map)
ret_hebdo = np.log(df_actifs / df_actifs.shift(1)).dropna(how="all")
vol = ret_hebdo.rolling(52).std().iloc[-1] * np.sqrt(52)
ret_neg = ret_hebdo.copy(); ret_neg[ret_neg > 0] = 0
sortino = (ret_hebdo.mean() * 52) / (ret_neg.std() * np.sqrt(52)).replace(0, np.nan)
ret_cum = (1 + ret_hebdo).cumprod()
max_dd = ((ret_cum - ret_cum.cummax()) / ret_cum.cummax()).min()
ret_propres = ret_hebdo.dropna(axis=1, thresh=int(len(ret_hebdo)*0.8)).dropna(axis=0)
if ret_propres.shape[0] > 10 and ret_propres.shape[1] > 1:
    correlation = pd.DataFrame(ledoit_wolf(ret_propres)[0], index=ret_propres.columns, columns=ret_propres.columns).corr()
else:
    correlation = ret_propres.corr()

# ==========================================
# Z-SCORE FORTERESSE (sans ML)
# ==========================================
actifs_portefeuille = st.session_state.mon_portefeuille["Actif"].tolist()
actifs_verrouilles = [r["Actif"] for _, r in st.session_state.mon_portefeuille.iterrows() if r.get("Coeur", False)]

eligibles = [a for a in univers_etudie if a in vol.index and not pd.isna(vol.get(a, np.nan)) and vol[a] <= 0.60 and max_dd.get(a, -1) >= -0.45 and a != "Bons du Tresor US 20A+"]
top_candidats = sortino.reindex(eligibles).dropna().sort_values(ascending=False).head(30).index.tolist()

fund_data = []
for c in top_candidats:
    t = univers_etudie[c]["ticker"]
    if t not in df_brut.columns: continue
    p_brut = float(df_brut[t].iloc[-1])
    sma200 = float(df_brut[t].tail(200).mean())
    trend = (p_brut / sma200 - 1) if sma200 > 0 else 0
    pe, roe, cons = 15.0, 0.15, 3.0
    is_etf = est_etf_ou_crypto(univers_etudie[c]["nom"])
    if not is_etf:
        f = obtenir_fondamentaux(t)
        pe_r, roe_r, cons_r = f.get("trailingPE"), f.get("returnOnEquity"), f.get("recommendationMean")
        # FORTERESSE: rejeter PE negatif ou ROE negatif
        if pe_r is not None and pe_r < 0: continue
        if roe_r is not None and roe_r < 0: continue
        if pe_r is not None and 0 < pe_r < 100: pe = pe_r
        elif pe_r is not None and pe_r > 100: continue
        if roe_r is not None: roe = roe_r
        if cons_r is not None: cons = cons_r
    fund_data.append({"Actif": c, "P/E": pe, "ROE": roe, "Consensus": cons, "Sortino": sortino.get(c, 0), "Tendance": trend})

df_z = pd.DataFrame(fund_data)
if not df_z.empty:
    df_z["Score_Global"] = (
        -calculate_z_score(df_z["P/E"]).fillna(0)
        + calculate_z_score(df_z["ROE"]).fillna(0) * 1.5  # Surponderer la qualite
        + calculate_z_score(df_z["Sortino"]).fillna(0)
        - calculate_z_score(df_z["Consensus"]).fillna(0)
        + calculate_z_score(df_z["Tendance"]).fillna(0)
    )
    top_ranked = df_z.sort_values("Score_Global", ascending=False)["Actif"].tolist()
else:
    top_ranked = []

# ==========================================
# SMART SWAP OMS (systeme clos)
# ==========================================
capital_coeur = sum(dict_actuel.get(a, 0) for a in actifs_verrouilles)
capital_satellite = max(0, budget - capital_coeur)

# Trouver le meilleur candidat externe (pas deja en portefeuille)
meilleur_candidat = None
meilleur_score = -np.inf
for c in top_ranked:
    if c in actifs_portefeuille: continue
    if c not in correlation.index: continue
    sc = df_z.loc[df_z["Actif"] == c, "Score_Global"].values
    if len(sc) > 0 and sc[0] > meilleur_score:
        meilleur_score = sc[0]
        meilleur_candidat = c

# Trouver la position satellite la plus faible
pire_satellite = None
pire_score = np.inf
for a in actifs_portefeuille:
    if a in actifs_verrouilles: continue
    if dict_actuel.get(a, 0) <= 0: continue
    sc = df_z.loc[df_z["Actif"] == a, "Score_Global"].values
    a_score = sc[0] if len(sc) > 0 else 0
    if a_score < pire_score:
        pire_score = a_score
        pire_satellite = a

# Decision de swap
swap_recommande = False
swap_vente = None
swap_achat = None
swap_raison = ""
if meilleur_candidat and pire_satellite:
    ecart = meilleur_score - pire_score
    val_vente = dict_actuel.get(pire_satellite, 0)
    frais_swap = 2 * FRAIS_ORDRE_TR + val_vente * SPREAD_ESTIME_TR * 2
    pnl_vente = val_vente - dict_pru.get(pire_satellite, val_vente)
    eco_fiscale = abs(pnl_vente) * FLAT_TAX_FR if pnl_vente < 0 else 0
    flat_tax_vente = max(0, pnl_vente) * FLAT_TAX_FR

    if ecart >= HURDLE_RATE and val_vente >= 50:
        swap_recommande = True
        swap_vente = pire_satellite
        swap_achat = meilleur_candidat
        swap_raison = f"Ecart Z-Score de {ecart:.2f} (seuil: {HURDLE_RATE})"

# Sante des positions
def calc_sante(actif):
    if actif not in univers_etudie: return None
    t = univers_etudie[actif]["ticker"]
    if t not in df_brut.columns: return None
    prix = df_brut[t].dropna()
    if len(prix) < 200: return {"actif": actif, "score": 50, "sma200": None, "sma50": None, "vol": 0, "dd": 0, "sortino": 0, "corr_moy": 0}
    p = float(prix.iloc[-1])
    s200 = float(prix.tail(200).mean())
    s50 = float(prix.tail(50).mean())
    v = float(vol.get(actif, 0)) if actif in vol.index else 0
    dd = float(max_dd.get(actif, 0)) if actif in max_dd.index else 0
    so = float(sortino.get(actif, 0)) if actif in sortino.index else 0
    autres = [x for x in actifs_portefeuille if x != actif and x in correlation.index]
    cm = float(correlation.loc[actif, autres].mean()) if actif in correlation.index and autres else 0
    sc = 50
    if p > s200: sc += 15
    else: sc -= 20
    if p > s50: sc += 10
    else: sc -= 10
    if so > 1: sc += 15
    elif so < 0: sc -= 15
    if v < 0.20: sc += 10
    elif v > 0.40: sc -= 10
    if cm < 0.3: sc += 10
    elif cm > 0.7: sc -= 10
    return {"actif": actif, "score": max(0, min(100, sc)), "sma200": p > s200, "sma50": p > s50, "dist_sma200": (p/s200-1)*100, "vol": v, "dd": dd, "sortino": so, "corr_moy": cm}

sante = [s for s in (calc_sante(a) for a in actifs_portefeuille) if s]
score_div = 50
valides_corr = [a for a in actifs_portefeuille if a in correlation.index]
if len(valides_corr) > 1:
    sub = correlation.loc[valides_corr, valides_corr]
    mask = ~np.eye(len(valides_corr), dtype=bool)
    score_div = int((1 - abs(sub.values[mask]).mean()) * 100)

# ==========================================
# ALERTES SIDEBAR
# ==========================================
st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='font-family:monospace;'>ALERTES</h3>", unsafe_allow_html=True)
alertes = []
if vix > seuil_vix: alertes.append(("CRITIQUE", f"VIX {vix:.0f} > {seuil_vix}"))
if risk_score >= 70: alertes.append(("CRITIQUE", f"Score risque {risk_score}/100"))
elif risk_score >= 40: alertes.append(("ATTENTION", f"Score risque {risk_score}/100"))
if curve_inv: alertes.append(("ATTENTION", "Courbe inversee"))
if sp_close < sp_sma200: alertes.append(("ATTENTION", "S&P sous SMA200"))
if swap_recommande: alertes.append(("INFO", f"Swap recommande: {swap_vente} -> {swap_achat}"))
for s in sante:
    if s["score"] < 30: alertes.append(("CRITIQUE", f"{s['actif']}: sante {s['score']}/100"))
if alertes:
    for n, m in alertes:
        (st.sidebar.error if n == "CRITIQUE" else st.sidebar.warning if n == "ATTENTION" else st.sidebar.info)(f"[{n}] {m}")
else:
    st.sidebar.success("Portefeuille sain. Ne touchez a rien.")

# ==========================================
# INTERFACE
# ==========================================
st.markdown("<h2 style='font-family:JetBrains Mono,monospace;border-bottom:1px solid #1a1a2e;padding-bottom:10px;letter-spacing:0.08em;font-weight:400;'>BUREAU CONSEIL &amp; EXECUTION</h2>", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("REGIME", regime, delta=f"Risque: {risk_score}/100", delta_color="inverse" if risk_score >= 40 else "normal")
c2.metric("HMM KRACH", f"{hmm_proba_krach:.0%}" if HMM_DISPONIBLE else "N/A", delta=hmm_names.get(hmm_regime, "?") if HMM_DISPONIBLE else "")
c3.metric("COEUR", f"{capital_coeur:.0f} EUR")
c4.metric("SATELLITE", f"{capital_satellite:.0f} EUR")
c5.metric("DIVERSIFICATION", f"{score_div}/100")

tab1, tab2, tab3, tab4 = st.tabs(["GESTION & ARBITRAGE", "RADAR A PEPITES", "SANTE DES FORTERESSES", "MACRO & IA"])

# ==========================================
# TAB 1 : OMS SMART SWAP
# ==========================================
with tab1:
    st.markdown("<p class='section-header'>POSITIONS — MARK-TO-MARKET</p>", unsafe_allow_html=True)
    df_disp = st.session_state.mon_portefeuille.copy()
    df_disp["Valeur Live"] = df_disp["Actif"].map(lambda a: round(dict_actuel.get(a, 0), 2))
    df_disp["P&L"] = df_disp.apply(lambda r: round(dict_actuel.get(r["Actif"], 0) - r.get("Quantite", 0) * r.get("PRU / Part", 0), 2), axis=1)
    df_disp["Poids %"] = df_disp["Valeur Live"].apply(lambda v: round(v/budget*100, 1) if budget > 0 else 0)

    cfg = {
        "Actif": st.column_config.SelectboxColumn("Instrument", options=list(univers_etudie.keys()), required=True),
        "Quantite": st.column_config.NumberColumn("Parts", min_value=0.0, step=0.001, format="%.4f"),
        "PRU / Part": st.column_config.NumberColumn("PRU/Part", min_value=0.0, step=0.1, format="%.2f"),
        "Coeur": st.column_config.CheckboxColumn("Coeur"),
        "Valeur Live": st.column_config.NumberColumn("Live (EUR)", format="%.2f"),
        "P&L": st.column_config.NumberColumn("P&L", format="%.2f"),
        "Poids %": st.column_config.NumberColumn("%", format="%.1f"),
    }
    df_ed = st.data_editor(df_disp, column_config=cfg, use_container_width=True, num_rows="dynamic", disabled=["Valeur Live", "P&L", "Poids %"], key="ed")
    df_cl = df_ed.dropna(subset=["Actif"])
    df_cl = df_cl[df_cl["Actif"].isin(univers_etudie.keys())].drop_duplicates(subset=["Actif"], keep="first")
    df_cl["Quantite"] = pd.to_numeric(df_cl["Quantite"], errors="coerce").clip(lower=0).fillna(0)
    df_cl["PRU / Part"] = pd.to_numeric(df_cl["PRU / Part"], errors="coerce").clip(lower=0).fillna(0)
    if "Coeur" not in df_cl.columns: df_cl["Coeur"] = False
    st.session_state.mon_portefeuille = df_cl[["Actif", "Quantite", "PRU / Part", "Coeur"]].reset_index(drop=True)

    total_pnl = sum(dict_actuel.get(a, 0) - dict_pru.get(a, 0) for a in dict_actuel)
    pc = "#00c853" if total_pnl >= 0 else "#ff1744"
    st.markdown(f"<div style='display:flex;gap:40px;padding:8px 0;'><span style='font-family:JetBrains Mono;font-size:0.8rem;color:#6c7293;'>INVESTI: <b style='color:white'>{sum(dict_pru.values()):.2f} EUR</b></span><span style='font-family:JetBrains Mono;font-size:0.8rem;color:#6c7293;'>LIVE: <b style='color:white'>{budget_actions:.2f} EUR</b></span><span style='font-family:JetBrains Mono;font-size:0.8rem;color:#6c7293;'>P&L: <b style='color:{pc}'>{total_pnl:+.2f} EUR</b></span></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p class='section-header'>RECOMMANDATION SMART SWAP</p>", unsafe_allow_html=True)

    if swap_recommande:
        val_v = dict_actuel.get(swap_vente, 0)
        pnl_v = val_v - dict_pru.get(swap_vente, val_v)
        ftax = max(0, pnl_v) * FLAT_TAX_FR
        frais = 2 * FRAIS_ORDRE_TR
        montant_net = val_v - ftax - frais
        z_v = df_z.loc[df_z["Actif"] == swap_vente, "Score_Global"].values
        z_a = df_z.loc[df_z["Actif"] == swap_achat, "Score_Global"].values

        col_sw1, col_sw2, col_sw3 = st.columns(3)
        col_sw1.error(f"VENDRE: {swap_vente}\nValeur: {val_v:.2f} EUR\nP&L: {pnl_v:+.2f} EUR\nZ-Score: {z_v[0]:.2f}" if len(z_v) > 0 else f"VENDRE: {swap_vente}")
        col_sw2.success(f"ACHETER: {swap_achat}\nMontant: {montant_net:.2f} EUR\nZ-Score: {z_a[0]:.2f}" if len(z_a) > 0 else f"ACHETER: {swap_achat}")
        col_sw3.info(f"Frais: {frais:.2f} EUR\nFlat Tax: {ftax:.2f} EUR\n{'Eco fiscale (MV): ' + str(round(abs(pnl_v)*FLAT_TAX_FR,2)) + ' EUR' if pnl_v < 0 else ''}\nRaison: {swap_raison}")
    else:
        st.success("PORTEFEUILLE SAIN — AUCUN ARBITRAGE NECESSAIRE. Le radar n'a identifie aucune opportunite justifiant un swap avec le hurdle rate actuel.")

# ==========================================
# TAB 2 : RADAR
# ==========================================
with tab2:
    st.markdown("<p class='section-header'>TOP 20 — CLASSEMENT MONDIAL PAR Z-SCORE FORTERESSE</p>", unsafe_allow_html=True)
    if not df_z.empty:
        top20 = df_z.sort_values("Score_Global", ascending=False).head(20)
        tab_data = []
        for _, r in top20.iterrows():
            a = r["Actif"]
            en_ptf = "OUI" if a in actifs_portefeuille else ""
            coeur = "COEUR" if a in actifs_verrouilles else ""
            tab_data.append({
                "Instrument": f"{univers_etudie[a]['nom']} [{univers_etudie[a]['ticker']}]",
                "Z-Score": round(r["Score_Global"], 2),
                "P/E": round(r["P/E"], 1), "ROE": f"{r['ROE']*100:.1f}%",
                "Sortino": round(r["Sortino"], 2),
                "En portefeuille": en_ptf, "Statut": coeur,
            })
        st.dataframe(pd.DataFrame(tab_data), use_container_width=True, height=500)
    else:
        st.warning("Aucune donnee disponible.")

# ==========================================
# TAB 3 : SANTE DES FORTERESSES
# ==========================================
with tab3:
    st.markdown(f"<p class='section-header'>DIAGNOSTIC — SCORE DIVERSIFICATION: {score_div}/100</p>", unsafe_allow_html=True)
    if sante:
        for sp in sorted(sante, key=lambda x: x["score"], reverse=True):
            val = dict_actuel.get(sp["actif"], 0)
            poids = val / budget * 100 if budget > 0 else 0
            reco = "RENFORCER" if sp["score"] >= 70 else ("CONSERVER" if sp["score"] >= 40 else "ALLEGER")
            icon = "▲" if reco == "RENFORCER" else ("■" if reco == "CONSERVER" else "▼")
            with st.expander(f"{icon} {reco}  **{sp['actif']}** — Sante: {sp['score']}/100 — {poids:.1f}%", expanded=False):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Score", f"{sp['score']}/100")
                c2.metric("Volatilite", f"{sp['vol']*100:.1f}%")
                c3.metric("Max DD", f"{sp['dd']*100:.1f}%")
                c4.metric("Sortino", f"{sp['sortino']:.2f}")
                c5, c6, c7, c8 = st.columns(4)
                c5.metric("SMA 200", "OUI" if sp.get("sma200") else ("NON" if sp.get("sma200") is False else "N/A"))
                c6.metric("SMA 50", "OUI" if sp.get("sma50") else ("NON" if sp.get("sma50") is False else "N/A"))
                c7.metric("Dist. SMA200", f"{sp.get('dist_sma200', 0):+.1f}%")
                c8.metric("Correl. moy.", f"{sp['corr_moy']:.2f}")
                t = univers_etudie[sp["actif"]]["ticker"]
                if t in df_brut.columns:
                    p6m = df_brut[t].tail(126).dropna()
                    if len(p6m) > 10:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=p6m.index, y=p6m.values, mode="lines", name="Prix", line=dict(color="#4a90e2")))
                        if len(df_brut[t].dropna()) >= 200:
                            sma = df_brut[t].rolling(200).mean().tail(126)
                            fig.add_trace(go.Scatter(x=sma.index, y=sma.values, mode="lines", name="SMA200", line=dict(color="#ff6b6b", dash="dot")))
                        fig.update_layout(template="plotly_dark", height=220, margin=dict(t=5,b=5,l=5,r=5))
                        st.plotly_chart(fig, use_container_width=True)

    # Heatmap de correlation (deplacee depuis l'ancien onglet Diversification)
    st.markdown("<p class='section-header'>MATRICE DE CORRELATION</p>", unsafe_allow_html=True)
    vc = [a for a in actifs_portefeuille if a in correlation.index]
    if len(vc) > 1:
        cs = correlation.loc[vc, vc]
        fig_h = go.Figure(data=go.Heatmap(z=cs.values, x=vc, y=vc, colorscale="RdBu_r", zmin=-1, zmax=1, text=np.round(cs.values, 2), texttemplate="%{text}"))
        fig_h.update_layout(template="plotly_dark", height=450, margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig_h, use_container_width=True)

# ==========================================
# TAB 4 : MACRO & IA
# ==========================================
with tab4:
    st.markdown("<p class='section-header'>MODELE DE MARKOV CACHE (HMM)</p>", unsafe_allow_html=True)
    if HMM_DISPONIBLE and hmm_transmat is not None:
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.metric("Regime HMM", hmm_names.get(hmm_regime, "?"))
            st.metric("P(Krach semaine prochaine)", f"{hmm_proba_krach:.1%}")
            labels = [hmm_names.get(i, f"E{i}") for i in range(3)]
            st.dataframe(pd.DataFrame(hmm_transmat, index=labels, columns=labels).style.format("{:.1%}"), use_container_width=True)
        with c2:
            labels = [hmm_names.get(i, f"E{i}") for i in range(3)]
            fig_t = go.Figure(data=go.Heatmap(z=hmm_transmat, x=labels, y=labels, colorscale="RdYlGn_r", zmin=0, zmax=1, text=np.round(hmm_transmat*100, 1), texttemplate="%{text}%", textfont={"size": 14}))
            fig_t.update_layout(template="plotly_dark", height=350, margin=dict(t=10,b=10,l=10,r=10), xaxis_title="Vers", yaxis_title="Depuis")
            st.plotly_chart(fig_t, use_container_width=True)
    else:
        st.warning("HMM inactif. Ajoutez hmmlearn dans requirements.txt.")

    st.markdown("---")
    st.markdown("<p class='section-header'>SMART CASH</p>", unsafe_allow_html=True)
    if cash_tr > 0:
        try:
            xeon = yf.download(TICKER_MONETAIRE, period="1y", progress=False)["Close"]
            if not xeon.empty and len(xeon) > 20:
                xv = xeon.values.flatten() if xeon.ndim > 1 else xeon.values
                rdt_brut = float(xv[-1]/xv[0]-1) if len(xv) > 20 else 0
                rdt_net = rdt_brut * (1 - FLAT_TAX_FR)
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("Cash TR", f"{cash_tr:.2f} EUR")
                sc2.metric(f"{TICKER_MONETAIRE} net", f"{rdt_net*100:.2f}%")
                sc3.metric("Cash TR net", f"{TAUX_CASH_TR_NET*100:.2f}%")
                if cash_tr >= 500:
                    if rdt_net > TAUX_CASH_TR_NET:
                        st.success(f"Placer {cash_tr:.0f} EUR sur {TICKER_MONETAIRE} (rendement superieur de {(rdt_net-TAUX_CASH_TR_NET)*100:.2f}%)")
                    else:
                        st.info("Cash TR plus avantageux. Gardez vos liquidites.")
                else:
                    st.info(f"Avec {cash_tr:.0f} EUR, les frais d'ordre rendent le placement non rentable.")
        except: st.info("Donnees XEON indisponibles.")
    else:
        st.info("Aucune liquidite a optimiser.")

    st.markdown("---")
    st.markdown("<p class='section-header'>SENTINEL — TELEGRAM</p>", unsafe_allow_html=True)
    ct1, ct2 = st.columns(2)
    tg_tok = ct1.text_input("Bot Token", type="password", key="tg1")
    tg_cid = ct2.text_input("Chat ID", key="tg2")
    if st.button("POUSSER ALERTE", key="tg3"):
        if tg_tok and tg_cid:
            msg = f"[TERMINAL V37]\nRegime: {regime}\nRisque: {risk_score}/100\nVIX: {vix:.1f}\nHMM: {hmm_proba_krach:.0%}\nCapital: {budget:.0f}EUR"
            if swap_recommande: msg += f"\nSWAP: {swap_vente} -> {swap_achat}"
            try:
                r = requests.get(f"https://api.telegram.org/bot{tg_tok}/sendMessage", params={"chat_id": tg_cid, "text": msg}, timeout=10)
                st.success("Envoye.") if r.status_code == 200 else st.error(f"Erreur {r.status_code}")
            except Exception as e: st.error(f"Echec: {e}")
        else: st.warning("Token et Chat ID requis.")

# Footer
st.markdown("---")
st.caption(f"Terminal V37 — Mode Forteresse — {datetime.now().strftime('%d/%m/%Y %H:%M')} — FX: 1EUR={1/taux_fx.get('USD',1):.4f}USD — Hurdle Rate: {HURDLE_RATE}")

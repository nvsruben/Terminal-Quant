import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import requests
from scipy.linalg import inv
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION STREAMLIT ---
st.set_page_config(page_title="QUANTITATIVE TERMINAL V22", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# SYSTEM AUTHENTICATION
# ==========================================
if "authentifie" not in st.session_state:
    st.session_state.authentifie = False

if not st.session_state.authentifie:
    st.markdown("<h1 style='text-align: center; color: #ffffff; font-family: monospace;'>QUANTITATIVE ALLOCATION DESK</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888888; font-family: monospace;'>RESTRICTED ACCESS. V22 PROPRIETARY ENGINE.</p>", unsafe_allow_html=True)
    
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
# INVESTMENT UNIVERSE & TAIL RISK ASSETS
# ==========================================

MES_FAVORIS = {
    "Bitcoin": {"ticker": "BTC-EUR", "nom": "Bitcoin (Crypto)"},
    "Ethereum": {"ticker": "ETH-EUR", "nom": "Ethereum (Crypto)"},
    "Or Physique": {"ticker": "IGLN.L", "nom": "iShares Physical Gold ETC"},
    "US Treasuries 20Y+": {"ticker": "TLT", "nom": "iShares 20+ Year Treasury Bond"},
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
    tickers_complets = liste_tickers + ['^VIX', '^TNX', '^GSPC', '^IRX', 'HYG', 'IEF']
    df = yf.download(tickers_complets, period="5y", progress=False)['Close']
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

def calculate_z_score(series):
    if series.std() == 0: return np.zeros(len(series))
    return (series - series.mean()) / series.std()

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.markdown("<h3 style='font-family: monospace;'>SYSTEM CONTROLS</h3>", unsafe_allow_html=True)
if st.sidebar.button("FORCE REAL-TIME REFRESH", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='font-family: monospace;'>RISK PARAMETERS</h3>", unsafe_allow_html=True)
budget = st.sidebar.number_input("Capital Allocation (EUR)", min_value=10.0, value=950.0, step=10.0)
seuil_vix = st.sidebar.slider("VIX Threshold", 15, 40, 22)
vol_max = st.sidebar.slider("Max Weekly Volatility (%)", 30, 150, 60) / 100.0
dd_max = st.sidebar.slider("Max Historical Drawdown (%)", -80, -10, -45) / 100.0
correl_max = st.sidebar.slider("Covariance Rejection Limit (%)", 50, 95, 75) / 100.0
# New Limit Constraint
max_weight_limit = 0.25

# --- CORE ENGINE EXECUTION ---
with st.spinner(f'Processing {len(univers_etudie)} instruments. Running Analyst Consensus Scoring...'):
    liste_tickers_bruts = [v["ticker"] for k, v in univers_etudie.items()]
    df_brut = telecharger_donnees(liste_tickers_bruts)

# --- 1. MACRO REGIME DETECTION & TAIL RISK HEDGING ---
vix_actuel = float(df_brut['^VIX'].iloc[-1])
taux_10y = float(df_brut['^TNX'].iloc[-1])
taux_3m = float(df_brut['^IRX'].iloc[-1])

sp500_close = float(df_brut['^GSPC'].iloc[-1])
sp500_sma200 = float(df_brut['^GSPC'].tail(200).mean())
trend_bear = sp500_close < sp500_sma200

yield_curve_spread = taux_10y - taux_3m
curve_inverted = yield_curve_spread < 0.0

df_brut['Credit_Spread'] = df_brut['HYG'] / df_brut['IEF']
credit_spread_actuel = float(df_brut['Credit_Spread'].iloc[-1])
credit_spread_sma50 = float(df_brut['Credit_Spread'].tail(50).mean())
credit_stress = credit_spread_actuel < credit_spread_sma50

risk_score = 0
if trend_bear: risk_score += 40
if curve_inverted: risk_score += 30
if credit_stress: risk_score += 30
if vix_actuel > seuil_vix: risk_score = min(100, risk_score + 20)

tail_hedge_active = False
if risk_score >= 70:
    regime_marche = "CRITICAL BEAR MARKET"
    pourcentage_cash = 0.40 
    tail_hedge_active = True 
elif risk_score >= 40:
    regime_marche = "DEFENSIVE REGIME"
    pourcentage_cash = 0.40
elif vix_actuel > seuil_vix:
    regime_marche = "HIGH VOLATILITY"
    pourcentage_cash = 0.20
else:
    regime_marche = "BULL MARKET"
    pourcentage_cash = 0.0

reserve_cash = budget * pourcentage_cash
budget_investissable = budget - reserve_cash

if tail_hedge_active:
    budget_tail_risk = budget * 0.40
    budget_investissable = budget - reserve_cash - budget_tail_risk 

# --- DATA PREPARATION ---
df_hebdo = df_brut.resample('W-FRI').last()
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
correlation = rendements_hebdo.corr()

sortino_ajuste = sortino_brut.copy()
for actif in sortino_ajuste.index:
    if "S&P500:" in actif: sortino_ajuste[actif] = sortino_ajuste[actif] * 0.85 

# --- PRE-FILTERING ---
actifs_pre_eligibles = []
raisons = {}
for actif in univers_etudie.keys():
    if actif not in volatilite.index or pd.isna(volatilite[actif]): continue
    if actif == "US Treasuries 20Y+": continue 
    
    vol = volatilite[actif]
    dd = max_dd[actif]
    if vol > vol_max: raisons[actif] = f"REJECTED (Vol > {vol_max*100:.0f}%)"
    elif dd < dd_max: raisons[actif] = f"REJECTED (Drawdown < {dd_max*100:.0f}%)"
    else: actifs_pre_eligibles.append(actif)

top_20_candidats = sortino_ajuste[actifs_pre_eligibles].sort_values(ascending=False).head(20).index.tolist()

# --- 2. MULTI-FACTOR Z-SCORE ENGINE (SMART BETA 3.0 + CONSENSUS) ---
fundamentals_data = []
with st.spinner('Compiling Analyst Consensus and Value Metrics...'):
    for candidat in top_20_candidats:
        ticker_str = univers_etudie[candidat]["ticker"]
        is_etf_or_crypto = any(kw in univers_etudie[candidat]["nom"] for kw in ["Crypto", "ETF", "UCITS", "ETC", "Fund"])
        
        if is_etf_or_crypto:
            fundamentals_data.append({"Actif": candidat, "PE": 15.0, "ROE": 0.15, "Consensus": 3.0, "Sortino": sortino_ajuste[candidat]})
            continue
            
        try:
            info = yf.Ticker(ticker_str).info
            pe = info.get('trailingPE', 15)
            roe = info.get('returnOnEquity', 0.15)
            consensus = info.get('recommendationMean', 3.0) # 1=Strong Buy, 5=Strong Sell
            
            if pe is None or pe < 0 or pe > 100:
                raisons[candidat] = "FUNDAMENTAL REJECT (Negative/Bubble PE)"
                continue
            if roe is None: roe = 0.10
            if consensus is None: consensus = 3.0
                
            fundamentals_data.append({"Actif": candidat, "PE": pe, "ROE": roe, "Consensus": consensus, "Sortino": sortino_ajuste[candidat]})
        except:
            fundamentals_data.append({"Actif": candidat, "PE": 15.0, "ROE": 0.10, "Consensus": 3.0, "Sortino": sortino_ajuste[candidat]})

# Z-Score Calculation (4 Factors)
df_zscore = pd.DataFrame(fundamentals_data)
if not df_zscore.empty:
    df_zscore['Z_PE'] = -calculate_z_score(df_zscore['PE']) 
    df_zscore['Z_ROE'] = calculate_z_score(df_zscore['ROE']) 
    df_zscore['Z_Sortino'] = calculate_z_score(df_zscore['Sortino'])
    df_zscore['Z_Consensus'] = -calculate_z_score(df_zscore['Consensus']) # Inverse car bas = achat fort
    df_zscore['Global_Score'] = df_zscore['Z_PE'].fillna(0) + df_zscore['Z_ROE'].fillna(0) + df_zscore['Z_Sortino'].fillna(0) + df_zscore['Z_Consensus'].fillna(0)
    
    actifs_eligibles_finaux = df_zscore.sort_values(by="Global_Score", ascending=False)['Actif'].tolist()
else:
    actifs_eligibles_finaux = []

# --- FINAL SELECTION ---
top_5_actifs = []
for candidat in actifs_eligibles_finaux:
    if len(top_5_actifs) >= 5:
        raisons[candidat] = "FILTERED (Below Top 5 Z-Score)"
        continue
    trop_correle = False
    for selectionne in top_5_actifs:
        if correlation.loc[candidat, selectionne] > correl_max:
            trop_correle = True
            raisons[candidat] = f"FILTERED (Covariance w/ {selectionne})"
            break
    if not trop_correle:
        top_5_actifs.append(candidat)

# --- BLACK-LITTERMAN ALLOCATION WITH HARD CONCENTRATION LIMITS ---
if len(top_5_actifs) > 0:
    try:
        tau = 0.05
        cov_matrix = rendements_hebdo[top_5_actifs].cov().values * 52
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

        # APPLY HARD LIMITS (25% MAX)
        while any(poids_optimaux > max_weight_limit + 1e-5):
            excess = sum(poids_optimaux[poids_optimaux > max_weight_limit] - max_weight_limit)
            poids_optimaux[poids_optimaux > max_weight_limit] = max_weight_limit
            mask = poids_optimaux < max_weight_limit
            if sum(mask) > 0:
                poids_optimaux[mask] += excess * (poids_optimaux[mask] / sum(poids_optimaux[mask]))
            else:
                break
                
    except Exception as e:
        vol_top5 = volatilite[top_5_actifs]
        poids_optimaux = (1/vol_top5) / (1/vol_top5).sum()
        
    allocations = pd.Series(poids_optimaux * budget_investissable, index=top_5_actifs)
else:
    allocations = pd.Series(dtype=float)

if tail_hedge_active:
    allocations["US Treasuries 20Y+"] = budget_tail_risk

# --- CSV EXPORT ---
st.sidebar.markdown("---")
if len(allocations) > 0:
    csv_data = generer_csv_europe(allocations, budget, reserve_cash, regime_marche)
    st.sidebar.download_button(label="EXPORT EXECUTION ORDER (.CSV)", data=csv_data, file_name=f"Execution_V22.csv", mime="text/csv")
if st.sidebar.button("TERMINATE SESSION", use_container_width=True):
    st.session_state.authentifie = False
    st.rerun()

# --- MAIN DASHBOARD (INSTITUTIONAL UI) ---
st.markdown("<h2 style='font-family: monospace; border-bottom: 1px solid #444; padding-bottom: 10px;'>ALLOCATION DESK DASHBOARD</h2>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("SYSTEMIC RISK SCORE", f"{risk_score}/100", delta=regime_marche, delta_color="normal" if risk_score < 40 else "inverse")
col2.metric("MULTI-FACTOR ENGINE", "ACTIVE", delta="Val/Qual/Mom/Consensus", delta_color="normal")
col3.metric("TAIL RISK HEDGE", "DEPLOYED" if tail_hedge_active else "STANDBY", delta="TLT 20Y+ Treasuries", delta_color="normal" if tail_hedge_active else "off")
col4.metric("DEFENSIVE CASH", f"{reserve_cash:.2f} EUR", delta=f"{pourcentage_cash*100}% exposure", delta_color="off")

tab1, tab2, tab3 = st.tabs(["ALLOCATION MATRIX", "HISTORICAL STRESS TESTS", "MACRO MONITORS"])

with tab1:
    donnees_tableau = []
    actifs_a_afficher = list(dict.fromkeys(top_5_actifs + top_20_candidats[:15])) 
    if tail_hedge_active: actifs_a_afficher.insert(0, "US Treasuries 20Y+")
    
    for actif in actifs_a_afficher:
        if actif in allocations.index and allocations[actif] > 0: statut = "ALLOCATED"
        else: statut = raisons.get(actif, "REJECTED")
        
        mnt = allocations.get(actif, 0.0)
        nom_precis = univers_etudie[actif]["nom"]
        ticker = univers_etudie[actif]["ticker"]
        instrument_str = f"{nom_precis} [{ticker}]"
        
        z_score_display = "N/A"
        rec_display = "N/A"
        if not df_zscore.empty and actif in df_zscore['Actif'].values:
            z_val = df_zscore.loc[df_zscore['Actif'] == actif, 'Global_Score'].values[0]
            rec_val = df_zscore.loc[df_zscore['Actif'] == actif, 'Consensus'].values[0]
            z_score_display = f"{z_val:.2f}"
            rec_display = f"{rec_val:.1f}/5.0"
            
        donnees_tableau.append({
            "Instrument (Ticker)": instrument_str, 
            "Status": statut, 
            "Z-Score": z_score_display,
            "Analyst Rec": rec_display,
            "Max Drawdown": f"{max_dd[actif]*100:.1f}%", 
            "Volatility": f"{volatilite[actif]*100:.1f}%", 
            "Capital (EUR)": mnt
        })
        
    df_affichage = pd.DataFrame(donnees_tableau).sort_values(by="Capital (EUR)", ascending=False)
    st.dataframe(df_affichage.style.format({"Capital (EUR)": "{:.2f}"}).applymap(lambda x: 'background-color: #1a4222; color: #ffffff;' if x == 'ALLOCATED' else ('color: #8b0000;' if 'REJECT' in str(x) else ''), subset=['Status']), use_container_width=True, height=450)

with tab2:
    st.markdown("<h4 style='font-family: monospace;'>DYNAMIC BACKCAST SIMULATION</h4>", unsafe_allow_html=True)
    st.markdown("<p style='color: #888;'>Simulation projetée avec exclusion dynamique des actifs sans historique (Inception Bias Correction).</p>", unsafe_allow_html=True)
    
    if len(allocations) > 0:
        poids_test = (pd.Series(allocations) / budget).fillna(0)
        actifs_testes = poids_test[poids_test > 0].index.tolist()
        
        if len(actifs_testes) > 0:
            df_history = df_brut.copy()
            
            # Stress Test 1 : COVID CRASH (Fev-Avril 2020)
            df_covid_brut = df_history.loc['2020-02-15':'2020-04-30']
            
            # ANTI-NAN LOGIC (Exclusion des actifs trop récents)
            actifs_valides_covid = [a for a in actifs_testes if df_covid_brut[univers_etudie[a]["ticker"]].isna().sum() < 5]
            poids_covid = poids_test[actifs_valides_covid]
            
            if len(actifs_valides_covid) > 0 and poids_covid.sum() > 0:
                poids_covid = poids_covid / poids_covid.sum() # Renormalization to 100%
                ret_covid = df_covid_brut.pct_change().dropna()
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
                    dd_port_cov = (croissance_port_covid.min() - 100)
                    st.caption(f"Max Drawdown Engine: **{dd_port_cov:.1f}%**")
            else:
                col_st1, col_st2 = st.columns(2)
                with col_st1:
                    st.warning("No allocated assets have sufficient historical data for the 2020 period.")
                    
            # Stress Test 2 : INFLATION SHOCK (Jan-Oct 2022)
            df_inf_brut = df_history.loc['2022-01-01':'2022-10-31']
            actifs_valides_inf = [a for a in actifs_testes if df_inf_brut[univers_etudie[a]["ticker"]].isna().sum() < 5]
            poids_inf = poids_test[actifs_valides_inf]
            
            if len(actifs_valides_inf) > 0 and poids_inf.sum() > 0:
                poids_inf = poids_inf / poids_inf.sum()
                ret_inf = df_inf_brut.pct_change().dropna()
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
                    dd_port_inf = (croissance_port_inf.min() - 100)
                    st.caption(f"Max Drawdown Engine: **{dd_port_inf:.1f}%**")
            else:
                with col_st2:
                    st.warning("No allocated assets have sufficient historical data for the 2022 period.")
    else:
        st.warning("No allocation generated.")

with tab3:
    st.markdown("<h4 style='font-family: monospace;'>SYSTEMIC RISK FACTORS</h4>", unsafe_allow_html=True)
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("<span style='color: #888;'>Yield Curve (10Y - 3M Spread)</span>", unsafe_allow_html=True)
        df_yield = pd.DataFrame({"Spread (%)": (df_brut['^TNX'] - df_brut['^IRX']).tail(252)})
        st.line_chart(df_yield, color="#8b0000" if curve_inverted else "#4a90e2")
    with col_m2:
        st.markdown("<span style='color: #888;'>Interbank Credit Stress (HYG/IEF)</span>", unsafe_allow_html=True)
        df_credit = pd.DataFrame({"Ratio": df_brut['Credit_Spread'].tail(252)})
        st.line_chart(df_credit, color="#8b0000" if credit_stress else "#4a90e2")

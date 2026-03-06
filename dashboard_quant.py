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
st.set_page_config(page_title="Terminal Quantitatif V20", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# SYSTÈME D'AUTHENTIFICATION STRICTE
# ==========================================
if "authentifie" not in st.session_state:
    st.session_state.authentifie = False

if not st.session_state.authentifie:
    st.title("QUANTITATIVE ALLOCATION TERMINAL")
    st.markdown("AUTHENTIFICATION REQUISE. ACCÈS RESTREINT (V20 - Black-Litterman & CVaR).")
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
    # Ajout des capteurs Macro : ^IRX (3 Mois), HYG (High Yield Credit), IEF (Safe Bonds)
    tickers_complets = liste_tickers + ['^VIX', '^TNX', '^GSPC', '^IRX', 'HYG', 'IEF']
    # Passage à 5 ans pour les Stress-Tests
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

# --- TÉLÉCHARGEMENT MASSIF (5 ANS) ---
with st.spinner(f'Processing {len(univers_etudie)} instruments & running Black-Litterman Inference...'):
    liste_tickers_bruts = [v["ticker"] for k, v in univers_etudie.items()]
    df_brut = telecharger_donnees(liste_tickers_bruts)

# --- 1. DÉTECTION DE RÉGIME MULTI-FACTEURS ---
vix_actuel = float(df_brut['^VIX'].iloc[-1])
taux_10y = float(df_brut['^TNX'].iloc[-1])
taux_3m = float(df_brut['^IRX'].iloc[-1])

# A. Facteur Tendance Absolue (S&P 500 SMA 200)
sp500_close = float(df_brut['^GSPC'].iloc[-1])
sp500_sma200 = float(df_brut['^GSPC'].tail(200).mean())
trend_bear = sp500_close < sp500_sma200

# B. Facteur Courbe des Taux (Yield Curve Inversion)
yield_curve_spread = taux_10y - taux_3m
curve_inverted = yield_curve_spread < 0.0

# C. Facteur Stress de Crédit (High Yield vs Safe Bonds)
df_brut['Credit_Spread'] = df_brut['HYG'] / df_brut['IEF']
credit_spread_actuel = float(df_brut['Credit_Spread'].iloc[-1])
credit_spread_sma50 = float(df_brut['Credit_Spread'].tail(50).mean())
credit_stress = credit_spread_actuel < credit_spread_sma50

# Calcul du Systemic Risk Score (0 à 100)
risk_score = 0
if trend_bear: risk_score += 40
if curve_inverted: risk_score += 30
if credit_stress: risk_score += 30
if vix_actuel > seuil_vix: risk_score = min(100, risk_score + 20)

if risk_score >= 70:
    regime_marche = "CRITICAL BEAR MARKET"
    pourcentage_cash = 0.80 
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

# Extraction des prix des actifs
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

# --- PRE-FILTRAGE & SMART BETA ---
actifs_pre_eligibles = []
raisons = {}
for actif in univers_etudie.keys():
    if actif not in volatilite.index or pd.isna(volatilite[actif]): continue
    vol = volatilite[actif]
    dd = max_dd[actif]
    if vol > vol_max: raisons[actif] = f"REJETÉ (Volatilité > {vol_max*100:.0f}%)"
    elif dd < dd_max: raisons[actif] = f"REJETÉ (Max DD < {dd_max*100:.0f}%)"
    else: actifs_pre_eligibles.append(actif)

top_15_candidats = sortino_ajuste[actifs_pre_eligibles].sort_values(ascending=False).head(15).index.tolist()
actifs_eligibles_finaux = []

with st.spinner('Running Fundamental Due Diligence...'):
    for candidat in top_15_candidats:
        ticker_str = univers_etudie[candidat]["ticker"]
        if "Crypto" in univers_etudie[candidat]["nom"] or "ETF" in univers_etudie[candidat]["nom"] or "UCITS" in univers_etudie[candidat]["nom"] or "ETC" in univers_etudie[candidat]["nom"] or "Fund" in univers_etudie[candidat]["nom"]:
            actifs_eligibles_finaux.append(candidat)
            continue
        try:
            info = yf.Ticker(ticker_str).info
            pe_ratio = info.get('trailingPE', 15)
            if pe_ratio is None or pe_ratio < 0: raisons[candidat] = "FILTRÉ FONDAMENTAL (P/E Négatif)"
            elif pe_ratio > 100: raisons[candidat] = f"FILTRÉ FONDAMENTAL (Bulle P/E > 100)"
            else: actifs_eligibles_finaux.append(candidat)
        except:
            actifs_eligibles_finaux.append(candidat) 

# --- SÉLECTION TOP 5 ---
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

# --- 2. MODÈLE DE BLACK-LITTERMAN (MOTEUR BAYÉSIEN) ---
if len(top_5_actifs) > 0:
    try:
        tau = 0.05
        cov_matrix = rendements_hebdo[top_5_actifs].cov().values * 52
        
        # Poids d'équilibre (On utilise la Parité des Risques comme Neutral Market)
        inv_vol_bl = 1 / volatilite[top_5_actifs].values
        w_eq = inv_vol_bl / inv_vol_bl.sum()
        
        # Rendements Implicites d'Équilibre (Pi)
        Pi = 2.5 * np.dot(cov_matrix, w_eq) # Lambda estimé à 2.5
        
        # Vues Quantitatives (Momentum 3 mois)
        rendements_3m = np.log(df_actifs[top_5_actifs].iloc[-1] / df_actifs[top_5_actifs].iloc[-13]).values
        P = np.eye(len(top_5_actifs)) # Matrice de lien
        Q = rendements_3m * 4 # Annualisation du Momentum
        
        # Matrice de Confiance (Omega) - Plus la variance est haute, moins on a confiance
        Omega = np.diag(np.diag(cov_matrix)) * tau
        
        # Inférence Bayésienne (Posterior Expected Returns)
        inv_tau_cov = inv(tau * cov_matrix)
        inv_Omega = inv(Omega)
        
        term1 = inv(inv_tau_cov + np.dot(np.dot(P.T, inv_Omega), P))
        term2 = np.dot(inv_tau_cov, Pi) + np.dot(np.dot(P.T, inv_Omega), Q)
        BL_returns = np.dot(term1, term2)
        
        # Optimisation Maximize Expected Return / Variance
        inv_cov = inv(cov_matrix)
        poids_optimaux = np.dot(inv_cov, BL_returns)
        
        # Nettoyage et Normalisation des poids (Pas de shorting)
        poids_optimaux = np.clip(poids_optimaux, 0, None)
        if poids_optimaux.sum() == 0: poids_optimaux = w_eq
        else: poids_optimaux = poids_optimaux / poids_optimaux.sum()

    except Exception as e:
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
    st.sidebar.download_button(label="Export Execution Order (CSV)", data=csv_data, file_name=f"Ordre_DCA_V20.csv", mime="text/csv")
if st.sidebar.button("Terminate Session"):
    st.session_state.authentifie = False
    st.rerun()

# --- INTERFACE PRINCIPALE ---
st.title("QUANTITATIVE ALLOCATION TERMINAL")

col1, col2, col3, col4 = st.columns(4)
couleur_regime = "normal" if risk_score < 40 else "inverse"
col1.metric("Systemic Risk Score", f"{risk_score}/100", delta=regime_marche, delta_color=couleur_regime)
col2.metric("Yield Curve (10Y-3M)", f"{yield_curve_spread:.2f}%", delta="Inverted" if curve_inverted else "Normal", delta_color="inverse" if curve_inverted else "normal")
col3.metric("Black-Litterman Engine", "ACTIVE", delta="Bayesian Shrinkage", delta_color="normal")
col4.metric("Defensive Cash Reserve", f"{reserve_cash:.2f} EUR", delta=f"{pourcentage_cash*100}% exposure")

tab1, tab2, tab3 = st.tabs(["ALLOCATION MATRIX (BLACK-LITTERMAN)", "RISK & STRESS TESTS (CVaR)", "MACRO DASHBOARD"])

with tab1:
    st.markdown("*L'allocation finale est le résultat de l'inférence bayésienne du modèle Black-Litterman combinant la Minimum Variance et les Vues Quantitatives de Momentum.*")
    donnees_tableau = []
    actifs_a_afficher = list(dict.fromkeys(top_5_actifs + actifs_pre_eligibles[:20])) 
    
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
            "Max Drawdown (5y)": max_dd[actif]*100, 
            "Volatilité": volatilite[actif]*100, 
            "Allocation (EUR)": mnt
        })
        
    df_affichage = pd.DataFrame(donnees_tableau).sort_values(by="Allocation (EUR)", ascending=False)
    st.dataframe(df_affichage.style.format({"Sortino (Ajusté)": "{:.2f}", "Max Drawdown (5y)": "{:.1f}%", "Volatilité": "{:.1f}%", "Allocation (EUR)": "{:.2f}"}).applymap(lambda x: 'background-color: #1a4222; color: #ffffff;' if x == 'ALLOUÉ' else ('color: #ff4b4b;' if 'FONDAMENTAL' in str(x) else ''), subset=['Statut']), use_container_width=True, height=450)

with tab2:
    st.markdown("### 3. CONDITIONAL VALUE AT RISK (CVaR 95%) & MONTE CARLO")
    st.write("Évaluation de la perte moyenne attendue (Expected Shortfall) lors des 5% des pires scénarios de marché stochastiques.")
    
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
        
        # NOUVEAU : Calcul de la CVaR (Expected Shortfall)
        cvar_95 = np.mean(valeur_finale[valeur_finale <= var_95])
        
        fig_mc = go.Figure()
        for i in range(100): fig_mc.add_trace(go.Scatter(y=simulated_paths[:, i], mode='lines', line=dict(color='rgba(100, 100, 100, 0.1)'), showlegend=False))
        fig_mc.add_hline(y=var_95, line_dash="dash", line_color="#ff4b4b", annotation_text=f"VaR 95% : {var_95:.2f} EUR")
        fig_mc.add_hline(y=cvar_95, line_dash="solid", line_color="#8b0000", annotation_text=f"CVaR 95% (Extreme Crash) : {cvar_95:.2f} EUR")
        fig_mc.update_layout(title="Stochastic Future Equity Distribution (CVaR Engine)", xaxis_title="Jours (T+252)", yaxis_title="Capital (EUR)", template="plotly_dark", height=400)
        st.plotly_chart(fig_mc, use_container_width=True)
        
        st.error(f"**EXPECTED SHORTFALL (CVaR) :** En cas d'événement cygne noir (les 5% de pires cas absolus), la modélisation estime que la valeur résiduelle moyenne de votre allocation chutera à **{cvar_95:.2f} €**.")
    else:
        st.warning("Simulation impossible : 100% Cash.")

with tab3:
    st.markdown("### MULTI-FACTOR REGIME DETECTION")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("**1. Stress de Crédit Interbancaire (HYG/IEF)**")
        st.write("Ce ratio compare les obligations risquées (Junk Bonds) aux obligations d'État sécurisées. Une chute de ce ratio indique que les institutions financières paniquent et retirent leurs liquidités des actifs risqués.")
        df_credit = pd.DataFrame({"Ratio HYG/IEF": df_brut['Credit_Spread'].tail(252), "SMA 50": df_brut['Credit_Spread'].tail(252).rolling(50).mean()})
        st.line_chart(df_credit)
    with col_m2:
        st.markdown("**2. Tendance Absolue S&P 500 (Prix vs SMA 200)**")
        st.write("Le filtre de confirmation majeur. Si le prix passe sous la ligne, la tendance mondiale à long terme est statistiquement brisée.")
        df_sp = pd.DataFrame({"S&P 500": df_brut['^GSPC'].tail(252), "SMA 200": df_brut['^GSPC'].rolling(200).mean().tail(252)})
        st.line_chart(df_sp)

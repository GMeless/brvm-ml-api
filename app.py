import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# Configuration de la page
st.set_page_config(page_title="BRVM Predictor", layout="wide")

# ==============================
# BARRE LATÉRALE (SIDEBAR)
# ==============================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100) # Icône de profil
    st.title("👨‍💻 À propos de l'auteur")
    st.markdown("""
    **Nom :** BeniMeless  
    **Expertise :** Data Science & Finance (BRVM)
    
    Ce tableau de bord vous donne et prédit les tendances boursières basées sur les indicateurs techniques (Lags, Volatilité, Momentum).
    """)
    
    st.divider()
    
    st.markdown("### 📩 Me contacter")
    st.info("Disponible pour des projets d'analyse de données ou de déploiement d'IA.")
    
    # Liens réseaux sociaux (Corrigés avec https://)
    st.link_button("Mon LinkedIn", "https://www.linkedin.com/in/meless-m-gnagne-21261a196")
    st.link_button("Mon GitHub", "https://github.com/GMeless")

st.title("📊 BRVM Investisseurs Dashboard")

# ==============================
# CHARGEMENT DONNÉES
# ==============================
@st.cache_data
def load_data():
    df = pd.read_csv("brvm_with_indicators.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values("Date")
    return df

df = load_data()

# ==============================
# FILTRES
# ==============================
col1, col2 = st.columns(2)

with col1:
    code = st.selectbox("Choisir entreprise", sorted(df['Code'].unique()))

with col2:
    start_date = st.date_input("Date début", df['Date'].min())
    end_date = st.date_input("Date fin", df['Date'].max())

# Filtrage pour l'affichage des graphiques et indicateurs
df_period = df[
    (df['Date'] >= pd.to_datetime(start_date)) & 
    (df['Date'] <= pd.to_datetime(end_date))
]

df_company = df_period[df_period['Code'] == code]

st.markdown("---")

# ==============================
# INDICATEURS ENTREPRISE
# ==============================
st.subheader(f"📌 Indicateurs détaillés : {code}")

if len(df_company) > 30:
    total_return = df_company['Cours_jour'].iloc[-1] / df_company['Cours_jour'].iloc[0] - 1
    annual_return = (1 + total_return)**(252/len(df_company)) - 1 if len(df_company) > 0 else 0
    volatility = df_company['Return'].std() * np.sqrt(252)
    max_drawdown = df_company['Drawdown'].min()
    
    sharpe = (
        df_company['Return'].mean() / df_company['Return'].std()
        if df_company['Return'].std() != 0 else 0
    )
    
    avg_volume = df_company['Volume_echange'].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rendement période", f"{total_return*100:.2f}%")
    col2.metric("Rendement annualisé", f"{annual_return*100:.2f}%")
    col3.metric("Volatilité annualisée", f"{volatility*100:.2f}%")
    col4.metric("Sharpe Ratio", f"{sharpe:.2f}")

    col5, col6 = st.columns(2)
    col5.metric("Drawdown max", f"{max_drawdown*100:.2f}%")
    col6.metric("Volume moyen", f"{avg_volume:,.0f}")
else:
    st.warning("Pas assez de données sur cette période pour calculer les indicateurs.")

st.markdown("---")

# ==============================
# GRAPHIQUE PRIX
# ==============================
fig = px.line(
    df_company,
    x='Date',
    y='Cours_jour',
    title=f"📈 Evolution du Prix - {code}",
    template="plotly_dark"
)
st.plotly_chart(fig, use_container_width=True)

# ==============================
# CLASSEMENT TOP & BOTTOM
# ==============================
st.subheader("🔥 Classement sur la période sélectionnée")

performance = (
    df_period
    .groupby('Code')
    .apply(lambda x: x['Cours_jour'].iloc[-1] / x['Cours_jour'].iloc[0] - 1)
    .reset_index(name='Performance')
)

performance = performance.sort_values(by='Performance', ascending=False)
top10 = performance.head(10)
bottom10 = performance.tail(10).sort_values(by='Performance')

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🟢 Top 10 Performances")
    fig_top = px.bar(
        top10, x='Performance', y='Code', orientation='h',
        text=top10['Performance'].apply(lambda x: f"{x*100:.2f}%"),
        template="plotly_dark"
    )
    fig_top.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_top, use_container_width=True)

with col2:
    st.markdown("### 🔴 Flop 10 Performances")
    fig_bottom = px.bar(
        bottom10, x='Performance', y='Code', orientation='h',
        text=bottom10['Performance'].apply(lambda x: f"{x*100:.2f}%"),
        template="plotly_dark"
    )
    fig_bottom.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_bottom, use_container_width=True)

# ==============================
# ZONE PRÉDICTION ML
# ==============================
st.markdown("## 🤖 Prédiction Machine Learning")

horizon_dict = {"1 Jour": 1, "5 Jours": 5, "20 Jours (~1 mois)": 20}
horizon_label = st.selectbox("Choisir horizon de prédiction", list(horizon_dict.keys()))
horizon = horizon_dict[horizon_label]

# Utilisation de l'historique complet pour un entraînement robuste
df_full_history = df[df['Code'] == code].copy()

if len(df_full_history) > 60:
    with st.spinner(f"🔮 L'IA analyse l'historique complet de {code}..."):
        df_ml = df_full_history.copy()
        df_ml['Lag_1'] = df_ml['Return'].shift(1)
        df_ml['Lag_5'] = df_ml['Return'].shift(5)
        df_ml['Lag_20'] = df_ml['Return'].shift(20)
        df_ml['Target'] = df_ml['Return'].shift(-horizon)

        df_ml = df_ml.dropna()
        features = ['Lag_1','Lag_5','Lag_20','Volatility_30','Momentum_6M']

        if not df_ml.empty:
            X = df_ml[features]
            y = df_ml['Target']

            model = RandomForestRegressor(n_estimators=200, random_state=42)
            model.fit(X, y)

            # Inférence sur la dernière ligne connue
            latest_features = df_ml[features].iloc[-1:].values
            predicted_return = model.predict(latest_features)[0]

            # Prix actuel basé sur la dernière date disponible (pas forcément le filtre)
            current_price = df_full_history['Cours_jour'].iloc[-1]
            predicted_price = current_price * (1 + predicted_return)

            col1, col2 = st.columns(2)
            col1.metric("Prix actuel", f"{current_price:,.0f} FCFA")
            # Ajout d'un delta pour le style visuel
            col2.metric("Prix estimé futur", f"{predicted_price:,.0f} FCFA", delta=f"{predicted_return*100:.2f}%")

            st.success(f"Analyse terminée : Rendement estimé de {predicted_return*100:.2f}% sur {horizon_label}")
        else:
            st.error("Données insuffisantes pour l'entraînement.")
else:
    st.warning("Historique insuffisant sur cette valeur pour générer une prédiction.")
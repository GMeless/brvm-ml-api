import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(layout="wide")

st.title("📊 BRVM Investisseurs Dashboard")

# ==============================
# CHARGEMENT DONNÉES
# ==============================
df = pd.read_csv("brvm_with_indicators.csv")
df['Date'] = pd.to_datetime(df['Date'])

# ==============================
# FILTRES
# ==============================
col1, col2 = st.columns(2)

with col1:
    code = st.selectbox("Choisir entreprise", sorted(df['Code'].unique()))

with col2:
    start_date = st.date_input("Date début", df['Date'].min())
    end_date = st.date_input("Date fin", df['Date'].max())

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
        top10,
        x='Performance',
        y='Code',
        orientation='h',
        text=top10['Performance'].apply(lambda x: f"{x*100:.2f}%"),
        template="plotly_dark"
    )
    fig_top.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_top, use_container_width=True)

with col2:
    st.markdown("### 🔴 Flop 10 Performances")
    fig_bottom = px.bar(
        bottom10,
        x='Performance',
        y='Code',
        orientation='h',
        text=bottom10['Performance'].apply(lambda x: f"{x*100:.2f}%"),
        template="plotly_dark"
    )
    fig_bottom.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_bottom, use_container_width=True)

# ==============================
# ZONE PRÉDICTION ML
# ==============================
st.markdown("## 🤖 Prédiction Machine Learning")

horizon_dict = {
    "1 Jour": 1,
    "5 Jours": 5,
    "20 Jours (~1 mois)": 20
}

horizon_label = st.selectbox(
    "Choisir horizon de prédiction",
    list(horizon_dict.keys())
)

horizon = horizon_dict[horizon_label]

if len(df_company) > 200:

    df_ml = df_company.copy()

    df_ml['Lag_1'] = df_ml['Return'].shift(1)
    df_ml['Lag_5'] = df_ml['Return'].shift(5)
    df_ml['Lag_20'] = df_ml['Return'].shift(20)

    df_ml['Target'] = df_ml['Return'].shift(-horizon)

    df_ml = df_ml.dropna()

    features = ['Lag_1','Lag_5','Lag_20','Volatility_30','Momentum_6M']

    X = df_ml[features]
    y = df_ml['Target']

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

    latest_features = df_ml[features].iloc[-1:].values

    predicted_return = model.predict(latest_features)[0]

    current_price = df_company['Cours_jour'].iloc[-1]

    predicted_price = current_price * (1 + predicted_return)

    col1, col2 = st.columns(2)

    col1.metric("Prix actuel", f"{current_price:,.0f} FCFA")
    col2.metric("Prix estimé futur", f"{predicted_price:,.0f} FCFA")

    st.success(f"Rendement estimé : {predicted_return*100:.2f}% sur {horizon_label}")
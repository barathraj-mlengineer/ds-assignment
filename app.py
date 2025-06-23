import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bitcoin Sentiment vs Trader Performance", layout="wide")
st.title("📊 Bitcoin Market Sentiment vs Trader Performance")
st.markdown("Analyze how market sentiment (Fear/Greed) influences trading outcomes on Hyperliquid")

#Load Sentiment Data directly
sentiment_df = pd.read_csv("fear_greed_index.csv")
sentiment_df.columns = sentiment_df.columns.str.strip().str.lower()
sentiment_df = sentiment_df[['date', 'classification']]
sentiment_df.columns = ['Date', 'Sentiment']
sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
sentiment_df['Sentiment'] = sentiment_df['Sentiment'].str.upper()

#Load Trader Data directly
trader_df = pd.read_csv("historical_data_reduced.csv")
trader_df.columns = trader_df.columns.str.strip().str.lower().str.replace(" ", "_")
trader_df.rename(columns={'coin': 'symbol'}, inplace=True)  # Adjust for naming consistency
trader_df['timestamp'] = pd.to_datetime(trader_df['timestamp_ist'], format="%d-%m-%Y %H:%M")
trader_df['Date'] = trader_df['timestamp'].dt.normalize()

#Merge datasets
merged_df = pd.merge(trader_df, sentiment_df, on='Date', how='inner')

st.subheader("📅 Merged Data Sample")
st.dataframe(merged_df.head())

#Profit or Loss Classification
merged_df['ProfitOrLoss'] = merged_df['closed_pnl'].apply(lambda x: 'Profit' if x > 0 else 'Loss')

#Average PnL
avg_pnl = merged_df.groupby('Sentiment')['closed_pnl'].mean().reset_index()
st.subheader("💰 Average PnL by Sentiment")
fig1, ax1 = plt.subplots()
sns.barplot(data=avg_pnl, x='Sentiment', y='closed_pnl', palette='coolwarm', ax=ax1)
ax1.set_title("Average Trader Profit & Loss by Market Sentiment")
ax1.set_ylabel("Average PnL")
ax1.grid(True)
fig1.tight_layout(pad=0.3)
st.pyplot(fig1)

#Outcome Distribution
st.subheader("📉 Trade Outcome Count by Sentiment")
fig2, ax2 = plt.subplots()
sns.countplot(data=merged_df, x='Sentiment', hue='ProfitOrLoss', palette='Set2', ax=ax2)
ax2.set_title("Trade Outcomes (Profit vs Loss) by Sentiment")
ax2.set_ylabel("Number of Trades")
ax2.grid(True)
fig2.tight_layout(pad=0.3)
st.pyplot(fig2)

#Average Leverage
if 'leverage' in merged_df.columns:
    st.subheader("📈 Average Leverage Used by Sentiment")
    avg_leverage = merged_df.groupby('Sentiment')['leverage'].mean().reset_index()
    fig3, ax3 = plt.subplots()
    sns.barplot(data=avg_leverage, x='Sentiment', y='leverage', palette='flare', ax=ax3)
    ax3.set_title("Average Leverage Used in Different Market Sentiments")
    ax3.set_ylabel("Average Leverage")
    ax3.grid(True)
    fig3.tight_layout(pad=0.3)
    st.pyplot(fig3)

#Symbol-wise Performance
if 'symbol' in merged_df.columns:
    st.subheader("🔍 Symbol-wise PnL per Sentiment")
    symbol_sentiment = merged_df.groupby(['Sentiment', 'symbol'])['closed_pnl'].mean().reset_index()
    pivot_df = symbol_sentiment.pivot(index='symbol', columns='Sentiment', values='closed_pnl').fillna(0)
    st.dataframe(pivot_df)

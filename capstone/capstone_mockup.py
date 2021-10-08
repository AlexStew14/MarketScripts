from plotly.tools import make_subplots
import plotly.figure_factory as ff
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from arch import arch_model
import numpy as np

def plot_price_history(price_df, ticker):
    fig = make_subplots(specs=[[{'secondary_y': True}]])

    # Price candlestick
    fig.add_trace(go.Candlestick(x=price_df.index,
                    open=price_df['Open'],
                    high=price_df['High'],
                    low=price_df['Low'],
                    close=price_df['Close'],
                    name=f'{ticker} Price'), secondary_y=True)

    # Volume chart
    fig.add_trace(go.Bar(x=price_df.index, y=price_df.Volume, opacity=.1, name=f'{ticker} Volume'), secondary_y=False)
    
    fig.update_layout(xaxis_rangeslider_visible=False, height=800)
    fig.layout.yaxis2.showgrid=False
    st.plotly_chart(fig, use_container_width=True)


st.set_page_config(layout='wide',page_title='Options Frontier', initial_sidebar_state='auto')
st.title('Options Frontier')


with st.form(key='tickers_form'):
    with st.sidebar:
        tickers = pd.read_csv('tickers.csv')
        tickers_input = st.multiselect('Tickers', tickers, key='selected_tickers')        
        number_columns = st.number_input('Columns',key='ticker_columns', min_value=1, max_value=3, step=1)
        horizon = st.slider('Forecast Horizon',key='forecast_horizon', min_value=1, max_value=15, step=1)
        sim_count = st.slider('Simulation Count',key='sim_count', min_value=100, max_value=500, step=25)
        submit = st.form_submit_button(label='Submit')



if len(st.session_state.selected_tickers) > 0:    
    show_prices = st.checkbox("Show price history", value=True)
    if show_prices:        
        num_columns = int(st.session_state.ticker_columns)
        columns = st.columns(num_columns)
        for idx,ticker in enumerate(st.session_state.selected_tickers):
            with columns[idx % num_columns]:
                # Fetching and plotting prices
                stock = yf.Ticker(ticker)
                hist = stock.history(period='1y', interval='1d')
                plot_price_history(hist, ticker)

                # Garch Fitting
                returns = 100 * hist.Close.pct_change().dropna()
                model = arch_model(returns, p=1, q=1, mean='constant')
                results = model.fit()

                # Run simulations
                sim_count = int(st.session_state.sim_count)
                horizon = st.session_state.forecast_horizon
                forecasts = results.forecast(horizon=horizon, method='bootstrap', simulations=sim_count, reindex=True)
                sims = forecasts.simulations

                # Use Simulations to generate final price distribution
                sim_vals = sims.values[-1]
                sim_vals = (sim_vals / 100) + 1
                starting_price = np.full(shape=(sim_count, 1), fill_value=hist.Close.iloc[-1])
                sim_vals = np.column_stack((starting_price, sim_vals))
                sim_vals = np.cumprod(sim_vals, axis=1)
                end_vals = sim_vals[:, -1]

                #Plot distribution
                fig = ff.create_distplot([end_vals], group_labels=[ticker], histnorm='probability', show_rug=False)
                fig.update_layout(height=800)
                st.plotly_chart(fig, use_container_width=True)






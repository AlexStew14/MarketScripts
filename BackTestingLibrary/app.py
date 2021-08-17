from re import sub
import backtesting_engine as be
import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

def get_state(state, default=None):
    if state in st.session_state:
        return st.session_state[state]

    return default


ticker = get_state('ticker')
period_type = get_state('period_type')
period_count = get_state('period_count')
stock_data = get_state('stock_data')
frequency_type = get_state('frequency_type')
input_sub = False

# Get stock info
with st.form("input_form"):
    st.write("Choose a stock to backtest")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        ticker = st.text_input("ticker")
        ticker = ticker.upper()
    with col2:        
        period_type = st.selectbox("period type", ['year'])
    with col3:
        period_count = st.number_input("period count", min_value=1, max_value=10)
    with col4:
        frequency_type = st.selectbox("frequency type", ['daily', 'weekly'])

    input_sub = st.form_submit_button("Submit")
    
    if input_sub:
        if ticker == '':
            input_sub = False
            st.write("Please enter a valid ticker!")
        else:
            st.session_state['ticker'] = ticker
            st.session_state['period_type'] = period_type
            st.session_state['period_count'] = period_count
            st.session_state['frequency_type'] = frequency_type
            st.session_state['input_sub'] = True

            stock_data = be.get_daily_stock_data(
                ticker, period_type, period_count, frequency_type)
            st.session_state['stock_data'] = stock_data
            st.session_state['stock_state'] = (ticker, period_type, period_count, frequency_type)


input2_sub = False
indicator, buy_value, sell_value, starting_money = None, None, None, None


input_sub = get_state('input_sub', False)
    # Get indicator info
if input_sub:
    with st.expander("Backtesting"):
        with st.form("backtest_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                indicator = st.selectbox("Choose Indicator", ['RSI'])
            with col2:
                buy_value = st.slider("Buy Low Value", min_value=10, max_value=40, value=30)
            with col3:
                sell_value = st.slider("Sell Low Value", min_value=45, max_value=90, value=70)

            starting_money = st.number_input("Starting Capital", min_value=10000, max_value = 1000000 )

            input2_sub = st.form_submit_button("Submit")

    if input2_sub:
        indicator_data = be.run_indicators(stock_data, buy_value, sell_value)

        capital, owned_shares, trade_counts, trades = be.backtest(indicator_data, starting_money)

        if trades is not None:
            st.dataframe(trades)

        report = be.backtest_report(stock_data.shape[0],trades, {'rsi_low': buy_value, 'rsi_up': sell_value}, starting_money)

        if report is not None:
            st.write(report)
        else:
            st.write("No trades taken")

        be.plot_trades(indicator_data, pd.DataFrame(trades), "Test")

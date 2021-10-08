from api_info import client_id, redirect_uri
from td.client import TDClient
import numpy as np
import pandas as pd
import pandas_ta as pta
import matplotlib.pyplot as plt
import streamlit as st


def get_daily_stock_data(ticker, period_type, num_period, frequency_type):

    TDSession = TDClient(
        client_id=client_id,
        redirect_uri=redirect_uri,
        credentials_path='./td_state.json'
    )

    # Login to the session
    TDSession.login()

    stock_history = TDSession.get_price_history(
        ticker, extended_hours=False, period=num_period, period_type=period_type, frequency=1, frequency_type=frequency_type)

    stock_df = pd.DataFrame(stock_history).drop(columns=['empty', 'symbol'])

    # Replace candles column with corresponding open, high, low, etc columns
    stock_flat_df = pd.concat([stock_df.drop(columns=['candles']), pd.DataFrame(
        stock_df.candles.values.tolist())], axis=1)
    stock_flat_df['datetime_formatted'] = pd.to_datetime(
        stock_flat_df['datetime'], unit='ms')

    stock_flat_df['returns'] = stock_flat_df.close.pct_change()

    stock_flat_df.dropna(inplace=True)

    return stock_flat_df


def run_indicators(data, rsi_lower, rsi_upper):
    data['rsi'] = pta.rsi(data['close'], length=14)

    pd.options.mode.chained_assignment = None  # default='warn'
    data['buy'] = 0
    data.buy[data.rsi <= rsi_lower] = 1
    data.buy[data.rsi >= rsi_upper] = -1
    return data


def show_indicators(data):
    plt.figure(figsize=(12, 8))
    plt.plot(data.datetime_formatted, data.rsi)
    plt.show()


def backtest(buy_stock_data, starting_capital, long_only=True):
    trades = []
    capital = starting_capital
    owned_shares = 0
    trade_count = 0
    buy_price = buy_stock_data.close.iloc[0]
    buy_date = None
    trade_drawdown = 0
    sell_price = 0

    for idx in buy_stock_data.index:
        sell_price = buy_stock_data.close[idx]
        trade_drawdown = max(
            trade_drawdown, (1 - sell_price / buy_price) * 100)

        if buy_stock_data.buy[idx] == 1 and capital >= buy_stock_data.close[idx]:
            buy_price = sell_price
            shares_bought = int(capital / sell_price)
            capital -= shares_bought * sell_price
            owned_shares += shares_bought
            buy_date = buy_stock_data.datetime_formatted[idx]
            trade_drawdown = 0
        elif buy_stock_data.buy[idx] == -1 and owned_shares > 0:
            capital += owned_shares * sell_price
            trades.append({'buy_price': buy_price, 'sell_price': sell_price, 'buy_date': buy_date.date(), 'sell_date': buy_stock_data.datetime_formatted[idx].date(),
                           'shares_sold': owned_shares, 'pl': (sell_price - buy_price) * owned_shares, 'max_drawdown': trade_drawdown})
            owned_shares = 0
            trade_count += 1

    if owned_shares > 0:
        capital += owned_shares * sell_price
        trades.append({'buy_price': buy_price, 'sell_price': sell_price, 'buy_date': buy_date.date(), 'sell_date': buy_stock_data.datetime_formatted[idx].date(),
                       'shares_sold': owned_shares, 'pl': (sell_price - buy_price) * owned_shares, 'max_drawdown': trade_drawdown})
        owned_shares = 0
        trade_count += 1

    return capital, owned_shares, trade_count, trades


def backtest_report(total_days, input_trades, params, starting_capital):
    if len(input_trades) > 0:
        trades = pd.DataFrame(input_trades)
        winning_trades = trades.loc[trades.pl > 0]
        losing_trades = trades.loc[trades.pl < 0]

        if (trades.shape[0] > 0):
            avg_pl = np.average(
                trades.pl / (trades.buy_price * trades.shares_sold) * 100)
        else:
            avg_pl = 0

        if (winning_trades.shape[0] > 0):
            avg_profit = np.average(
                winning_trades.pl / (winning_trades.buy_price * winning_trades.shares_sold) * 100)
        else:
            avg_profit = 0

        if (losing_trades.shape[0] > 0):
            avg_loss = np.average(
                losing_trades.pl / (losing_trades.buy_price * losing_trades.shares_sold) * 100)
        else:
            avg_loss = 0

        pct_win = winning_trades.shape[0] / trades.shape[0] * 100
        total_profit = trades.pl.sum()
        total_days_holding = (trades.sell_date - trades.buy_date).dt.days.sum()

        report = {"average_pl_pct": avg_pl, "Average_Profit_pct": avg_profit, "Average_Loss_pct": avg_loss,
                "Winning_pct": pct_win, "trade_count": trades.shape[0], "total_profit_pct": total_profit / starting_capital * 100,
                "Market_Exposure_pct": total_days_holding / total_days * 100, 'max_trade_drawdowns': trades.max_drawdown.values}

        for k, v in params.items():
            report[k] = v

        return report

    return None


def determine_best_indicator_values(data, rsi_lower_range=range(15, 40, 5),
                                    rsi_upper_range=range(40, 95, 5), starting_capital=100000):
    best_params = ()
    worst_params = ()

    best_capital = 0
    worst_capital = 100000000000000
    best_trade_count = 0
    reports = []
    all_trades = {}

    for rsi_low in rsi_lower_range:
        for rsi_up in rsi_upper_range:
            run_indicators(data, rsi_low, rsi_up)
            capital, owned_shares, trade_count, trades = backtest(
                starting_capital)

            all_trades[(rsi_low, rsi_up)] = trades

            if capital > best_capital:
                best_capital = capital
                best_params = (rsi_low, rsi_up)
            elif capital < worst_capital:
                capital = worst_capital
                worst_params = (rsi_low, rsi_up)

            if (len(trades) > 0):
                reports.append(backtest_report(
                    trades, {'rsi_low': rsi_low, 'rsi_up': rsi_up}, starting_capital))

    return best_params, worst_params, reports


def plot_trades(data, trades, label):
    fig = plt.figure(figsize=(12, 6))
    plt.scatter(data[data.buy == 1].datetime_formatted,
                data[data.buy == 1].close, label="Indicator Buy", s=35)
    plt.scatter(data[data.buy == -1].datetime_formatted,
                data[data.buy == -1].close, label="Indicator Sell", s=35)

    plt.scatter(trades.buy_date, trades.buy_price, marker='P', c='black', s=100, label="Buy Trade")
    plt.scatter(trades.sell_date, trades.sell_price, marker='X', c='red', s=100, label='Sell Trade')


    plt.plot(data.datetime_formatted, data.close, c='green')
    plt.title(label)
    fig.legend()
    plt.show()
    st.pyplot(fig)


def plot_strategy_trades(all_trades, params, ticker, num_period, period_type):
    trades_df = pd.DataFrame(all_trades[params])
    plot_trades(trades_df,
                f"Trades From {params[0]},{params[1]} RSI Strategy for {ticker} {num_period} {period_type}")


def process_strategy_reports(reports, rsi_lower_range, rsi_upper_range):
    reports_df = pd.DataFrame(reports)

    reports_df['max_drawdown'] = reports_df.max_trade_drawdowns.map(np.max)
    reports_df['avg_drawdown'] = reports_df.max_trade_drawdowns.map(np.average)

    return reports_df


def plot_reports_summary(reports_df, rsi_lower_range, rsi_upper_range, ticker, num_period, period_type):
    plt.figure(figsize=(20, 12))
    for rs in rsi_lower_range:
        subset = reports_df[reports_df.rsi_low == rs]
        if (subset.shape[0] == 0):
            continue

        plt.plot(subset.rsi_up, subset.average_pl_pct,
                 label=f'Buy at RSI: {rs}')

    plt.title(
        f"Average PL of Different RSI Strats for {ticker} {num_period} {period_type}")
    plt.ylabel("Average PL")
    plt.xlabel("RSI Sell Level")
    plt.xticks(rsi_upper_range)
    plt.legend()
    plt.axhline(0, c='black')
    plt.show()

    plt.figure(figsize=(20, 12))
    for rs in rsi_lower_range:
        subset = reports_df[reports_df.rsi_low == rs]
        if (subset.shape[0] == 0):
            continue

        plt.plot(subset.rsi_up, subset.trade_count,
                 label=f'Buy at RSI: {rs}')

    plt.title(
        f"Number of Trades for Different RSI Strats for {ticker} {num_period} {period_type}")
    plt.ylabel("Number of Trades")
    plt.xlabel("RSI Sell Level")
    plt.xticks(rsi_upper_range)
    plt.legend()
    plt.show()

    plt.figure(figsize=(20, 12))
    for rs in rsi_lower_range:
        subset = reports_df[reports_df.rsi_low == rs]
        if (subset.shape[0] == 0):
            continue

        plt.plot(subset.rsi_up, subset.max_drawdown,
                 label=f'Buy at RSI: {rs}')

    plt.title(
        f"Max Drawdown % on a Trade for Different RSI Strats for {ticker} {num_period} {period_type}")
    plt.ylabel("Max Drawdown")
    plt.xlabel("RSI Sell Level")
    plt.xticks(rsi_upper_range)
    plt.legend()
    plt.show()

    plt.figure(figsize=(20, 12))
    for rs in rsi_lower_range:
        subset = reports_df[reports_df.rsi_low == rs]
        if (subset.shape[0] == 0):
            continue

        plt.plot(subset.rsi_up, subset.avg_drawdown,
                 label=f'Buy at RSI: {rs}')

    plt.title(
        f"Average Drawdown % on a Trade for Different RSI Strats for {ticker} {num_period} {period_type}")
    plt.ylabel("Average Drawdown")
    plt.xlabel("RSI Sell Level")
    plt.xticks(rsi_upper_range)
    plt.legend()
    plt.show()

    plt.figure(figsize=(20, 12))
    for rs in rsi_lower_range:
        subset = reports_df[reports_df.rsi_low == rs]
        if (subset.shape[0] == 0):
            continue

        plt.plot(subset.rsi_up, subset.total_profit_pct,
                 label=f'Buy at RSI: {rs}')

    plt.title(f"Total Profit % for Different RSI Strats for {ticker} {num_period} {period_type}")
    plt.ylabel("Total Profit %")
    plt.xlabel("RSI Sell Level")
    plt.xticks(rsi_upper_range)
    plt.legend()
    plt.show()

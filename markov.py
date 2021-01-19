import yfinance as yf
import pandas as pd
import numpy as np
import random


def daily_change(ticker, start_date, end_date):
    """
    Produces a list of the daily change in stock price of the references stock ticker
    :param ticker: a string of the ticker of the stock
    :param start_date: a string of the start date (ex. '2020-01-01')
    :param end_date: a string of the end date (ex. '2020-12-01')
    :return: 3 lists: 1st being the frac-change, frac-high, frac-low
    """
    frac_change = []
    frac_high = []
    frac_low = []
    stock = yf.Ticker(ticker)
    stock_history = stock.history(start=start_date, end=end_date)
    for row in range(0, len(stock_history.index)):
        close = stock_history.iloc[row, 3]
        open = stock_history.iloc[row, 0]
        high = stock_history.iloc[row, 1]
        low = stock_history.iloc[row, 2]
        if row != 0 and row != len(stock_history.index):
            frac_change.append((close - open) / open)
            frac_high.append((high - open) / open)
            frac_low.append((open - low) / open)
    return frac_change, frac_high, frac_low


def simplify(data):
    """
    Categorizes the percent daily change in a list of data into 6 bins:
        0 <-- (-infinity, -.1), 1 <-- (-.1, -.01), 2 <-- (-.001, 0),
        3 <-- (0, .01), 4 <-- (.01, .1), 5 <-- (.1, infinity)
    :param data: 3 list of integers representing percentage change in stock data
    :return: a list of representing the percentage change in stock data based on the allocated bins
    """
    new_daybyday = []
    for daily in data:
        if daily < -.1:
            new_daybyday.append(0)
        if daily < -.01:
            new_daybyday.append(1)
        if daily < 0:
            new_daybyday.append(2)
        if daily < .01:
            new_daybyday.append(3)
        if daily < .1:
            new_daybyday.append(4)
        else:
            new_daybyday.append(5)
    return new_daybyday


def markov(data, n_degree):
    """
    Uses a Markov Chain to determine the likelihood of a future possibility
    :param data: a list of ints representing previously collected data
    :param n_degree: an integer representing the desired order of the Markov Chain
    :return: a dictionary that represents the Markov Chain
    """
    chain = {}
    dates = len(data)
    for idx in range(dates - n_degree - 1):
        current = tuple(data[idx: idx + n_degree - 1])
        if current not in chain.keys():
            chain[current] = {}
        next = chain[current]
        if data[idx + n_degree] in next.keys():
            next[data[idx + n_degree]] += 1
        else:
            next[data[idx + n_degree]] = 1
        chain[current] = next
    # Normalizes everything
    for i, j in chain.items():
        total = 0
        for ii, jj in j.items():
            total += jj
        for ii, jj in j.items():
            chain[i][ii] = jj / total
    return chain


def predict(chain, last, num):
    """
    Predicts the next number given the model and the last value
    :param chain: a dictionary representing a Markov Chain
    :param last: a list (with length of the order of the Markov chain)
                representing the previous states
    :param num: an integer representing the number of desired future states
    :return: a list of integers that are the next num states
    """
    choices = []
    for trial in range(num):
        if tuple(last) in chain.keys():
            rndm = random.randint(0, 100)
            total = 0
            for possibility, chance in chain[tuple(last)].items():
                total += chance * 100
                if rndm < total:
                    choices.append(possibility)
                    break
        else:
            choices.append(random.randint(1, 4))
        last.pop(0)
        last.append(choices[-1])
    return choices


def mse(prediction, actual):
    """
    Finds the mean squared error between the predicted outcome and the actual outcome
    :param prediction: a list representing the predicted states
    :param actual: a list representing the actual states
    :return: a value of the mean squared error, the lower the better
    """
    total = 0
    for i in range(len(prediction)):
        total += (prediction[i] - actual[i]) ** 2
    total = total / len(prediction)
    return total


def experiment(ticker, start_date, end_date, start_date2, end_date2, degree, last, num):
    """
    Higher order function that just encapsulates the above functions
    :param ticker: string of stock ticker
    :param start_date: string
    :param end_date: string
    :param start_date2: string
    :param end_date2: string
    :param degree: degree of markov chain
    :param last: list of last states
    :param num: number of future state
    :return: a prediction
    """
    back_test = daily_change(ticker, start_date, end_date)

    fchange = simplify(back_test[0])
    fhigh = simplify(back_test[1])
    flow = simplify(back_test[2])

    m_fchange = markov(fchange, degree)
    m_fhigh = markov(fhigh, degree)
    m_flow = markov(flow, degree)

    change_prediction = predict(m_fchange, last, num)
    high_prediction = predict(m_fhigh, last, num)
    low_prediction = predict(m_flow, last, num)

    front_test = daily_change(ticker, start_date2, end_date2)
    change_mse = mse(change_prediction, simplify(front_test[0]))
    high_mse = mse(high_prediction, simplify(front_test[1]))
    low_mse = mse(low_prediction, simplify(front_test[1]))

    return change_mse, high_mse, low_mse


def minimize_mse(ticker, start_date, end_date, start_date2, end_date2, last, num):
    lowest_mse = {}
    for i in range(1, 6):
        total = 0
        for j in range(10):
            total += experiment(ticker, start_date, end_date, start_date2, end_date2, i, last, num)[0]
        total = total/100
        lowest_mse[i] = total
    return lowest_mse


print(minimize_mse('aapl', '2019-01-01', '2020-12-01', '2020-12-01', '2020-12-4', [2, 3, 2], 3))

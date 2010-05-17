def retrieve_preceding_prices(symbol, date, time, window, concurrent=True):
    with h5py.File(filename) as root:
        names = root[date]["names"]
        symbol_index = list(names).index(symbol)
        prices = root[date]["prices"][:, symbol_index]
        times = root[date]["dates"].value
        last_index = times.tolist().index(time)
        if concurrent == True:
            last_index += 1
        first_index = last_index - window
        if first_index >= 0:
            return np.vstack((prices[first_index:last_index],
                              times[first_index:last_index]
                            ))
        else:
            remaining_window = -first_index
            trade_dates = list(root)
            trade_date_index = trade_dates.index(date)
            previous_date = trade_dates[trade_date_index-1]
            previous_date_last_time = root[previous_date]['dates'][-1]
            return np.hstack(( retrieve_preceding_prices(symbol, previous_date,
                                                         previous_date_last_time,
                                                         remaining_window,
                                                         concurrent=True),
                               np.vstack((prices[:last_index],
                                          times[:last_index]))
                             ))

def beta(stock_prices, benchmark_prices, timestamps, step=1):
    stock_prices = stock_prices[::step]
    benchmark_prices = benchmark_prices[::step]
    timestamps = timestamps[::step]
    stock_returns = np.diff(np.log(stock_prices))
    benchmark_returns = np.diff(np.log(benchmark_prices))
    covariance = np.cov(stock_returns, benchmark_returns)[0, 1]
    benchmark_variance = close_close_vol(benchmark_prices, timestamps)
    print benchmark_variance, np.sqrt(benchmark_variance)
    benchmark_variance = np.var(benchmark_returns)
    return covariance / benchmark_variance



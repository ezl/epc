import h5py
import numpy as np
from matplotlib import pyplot
from instrument import FinancialInstrument
from IPython.Shell import IPShellEmbed
import datetime

ipshell = IPShellEmbed("Dropping to IPython shell")
filename = "SPY-VXX-20090507-20100427.hdf5"

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

def close_close_vol(prices, timestamps, step=1):
    '''timestamps in epoch times, ascending'''
    prices = prices[::step]
    timestamps = timestamps[::step]
    secs_per_year = 60 * 60 * 24 * 365
    raw_variances = np.diff(np.log(prices)) ** 2
    total_time = (timestamps[-1] - timestamps[0]) / secs_per_year
    annualized_variance = raw_variances.sum() / total_time
    return np.sqrt(annualized_variance)

def test_cointegration():
    '''+ SPY.qty * SPY.last * SPY.beta * SPY.vol =
       - VXX.qty * VXX.last * VXX.beta * VXX.vol'''
    ''' don't need beta and vol ratio'''
    ''' a pair can be cointegrated with zero beta, so this is a deceptive metric'''
    ''' d(SPY portfolio value * spy correlation * spy vol) is = to the same for VXX
        assume correlation of -1 for the trade

        SPYqty * SPYprice * SPYvol = -1 * VXXqty * VXXprice * VXXvol
        SPYqty/VXXqty = VXXvol/SPYvol * VXXprice/SPYprice
    '''
    """
    stable value portfolio = noptional * beta - notional * beta
    using vol ratio instead: beta discounts the offsetting stock amount by the
    correlation. since we don't know which stock to discount (i.e. which leads)
    we'll pretend the correlation is 100%
    """

    pass

def backtest():
    # trade parameters
    vol_window = 1000 # how many timesteps to look back when computing vol
    portfolio_window = 1000 # ... and for the stable value moving average
    vol_nans = np.array([np.nan for i in range(vol_window)])
    portfolio_nans = np.array([np.nan for i in range(portfolio_window)])
    step = 1

    root = h5py.File(filename)
    trade_dates = list(root)

    spy_prices = np.hstack([root[t]["prices"].value[:, 0] for t in trade_dates])
    vxx_prices = np.hstack([root[t]["prices"].value[:, 1] for t in trade_dates])
    spy_returns = np.hstack((0, np.diff(np.log(spy_prices))))
    vxx_returns = np.hstack((0, np.diff(np.log(vxx_prices))))

    timestamps = np.hstack([root[t]["dates"].value for t in trade_dates])
    spy_positions = np.zeros_like(spy_prices)
    vxx_positions = np.zeros_like(spy_prices)
    closes = np.hstack([root[t]["dates"].value[-1] for t in trade_dates])

    start_time = datetime.datetime.now()
    print start_time
    spy_vol = np.hstack((vol_nans[1:],
                          [close_close_vol(spy_prices[t-vol_window:t],
                                           timestamps[t-vol_window:t],
                                           step=step)
                           for t in range(vol_window, len(spy_prices)+1)]
                        ))
    vxx_vol = np.hstack((vol_nans[1:],
                          [close_close_vol(vxx_prices[t-vol_window:t],
                                           timestamps[t-vol_window:t],
                                           step=step)
                           for t in range(vol_window, len(spy_prices)+1)]
                        ))
    spy_qty = np.ones_like(spy_prices)
    vxx_qty = spy_vol / vxx_vol * spy_prices / vxx_prices
    portfolio = spy_qty * spy_prices + vxx_qty * vxx_prices
    portfolio_std = np.hstack((portfolio_nans[1:],
                               [portfolio[t-portfolio_window:t].std()
                                for t in range(portfolio_window, len(spy_prices)+1)]
                             ))
    portfolio_moving_average = np.hstack((portfolio_nans[1:],
                                          np.convolve(np.ones(portfolio_window) / portfolio_window,
                                                      portfolio, mode= "valid")
                                        ))
    signal = portfolio - portfolio_moving_average
    signal_crosses_zero = np.hstack((np.nan,
                                     [cmp(s, 0) for s in signal[1:] * signal[:-1]]
                                   )) <= 0
    signal_moving_average = np.hstack((portfolio_nans[1:],
                                       np.convolve(np.ones(portfolio_window) / portfolio_window,
                                                   signal, mode= "valid")
                                     ))
    signal_std = np.hstack((portfolio_nans[1:],
                            [signal[t-portfolio_window:t].std()
                             for t in range(portfolio_window, len(spy_prices)+1)]
                          ))
    big_signal = abs(signal) > signal_std
    big2_signal = abs(signal) > 2 * signal_std

    # determine position
    for i in range(len(spy_prices)):
# carry position over. will be rewritten below on changes
        spy_positions[i] = spy_positions[i-1]
        vxx_positions[i] = vxx_positions[i-1]
# entry condition
        # 2 stddevs away
        if abs(signal[i]) > 2 * signal_std[i]:
            # and no current position
            if spy_positions[i] == 0:
                if signal[i] > 0:
                    # sell both
                    spy_positions[i] = -spy_qty[i]
                    vxx_positions[i] = -vxx_qty[i]
                else:
                    # buy both
                    spy_positions[i] = spy_qty[i]
                    vxx_positions[i] = vxx_qty[i]
# exit condition
        # close on signal crosses zero
        if signal_crosses_zero[i]:
            spy_positions[i] = 0
            vxx_positions[i] = 0
    # close position at end
    spy_positions[-1] = 0

    total_trades = sum(abs(np.diff(spy_positions != 0)))
    #TODO: does the zero go athe beginning or end?
    spy_trades = np.hstack((0, np.diff(spy_positions)))
    vxx_trades = np.hstack((0, np.diff(vxx_positions)))
    cash = - vxx_trades * vxx_prices - spy_trades * spy_prices

    print datetime.datetime.now() - start_time
    pyplot.plot(1)
    pyplot.show()
    pyplot.plot(spy_prices)
    pyplot.plot(vxx_prices)
    pyplot.twinx()
    pyplot.plot(signal)
    ipshell()

if __name__ == "__main__":
    backtest()




import h5py
import numpy as np
from matplotlib import pyplot
from instrument import FinancialInstrument
from IPython.Shell import IPShellEmbed

ipshell = IPShellEmbed("Dropping to IPython shell")
filename = "SPY-VXX-20090507-20100427.hdf5"

def retrieve_preceding_prices(symbol, date, time, window, concurrent=True):
    '''Inputs:
           symbol -- string, "VXX"
           date -- string, "20100427"
           time -- epoch time, as a float
           window -- int, number of observations to return
           concurrent -- bool, do you want to return the price for the
                         concurrent period?
                            True to include the price at the "time" parameter
                            False to return only preceding

       Returns:
           numpy array(prices, times) of dimensions (2, $window)

       Example:
           # retrieve price vector in prices, timestammps in times
           prices, times = retrieve_preceding_prices("SPY", "20100427",
                                                     161216212.0, 50, False)
       Note:
           This process is slow as bananas.
           Retrieving 1000 tuples with window=1500 took 14 seconds.
           Retrieving 1000 tuples with window=150 took 4 seconds.
    '''

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

def time_this_bizatch(iterations):
    def timer_decorator(fxn):
        import datetime
        def wrapped():
            start = datetime.datetime.now()
            for i in range(iterations):
                fxn()
            end = datetime.datetime.now()
            print "%s iterations, time_elapsed: %s" % (iterations, end - start)
        return wrapped
    return timer_decorator

def beta(stock_prices, benchmark_prices):
    stock_returns = np.diff(np.log(stock_prices))
    benchmark_returns = np.diff(np.log(benchmark_prices))
    covariance = np.corrcoef(stock_returns, benchmark_returns)[0, 1]
    benchmark_variance = np.var(benchmark_returns)
    return covariance / benchmark_variance

def close_close_vol(prices, timestamps):
    '''timestamps in epoch times, ascending'''
    secs_per_year = 60 * 60 * 24 * 365
    raw_variances = np.diff(np.log(prices)) ** 2
    total_time = (timestamps[-1] - timestamps[0]) / secs_per_year
    annualized_variance = raw_variances.sum() / total_time
    return np.sqrt(annualized_variance)

@time_this_bizatch(1)
def plot_SPY():
    window = 1
    i = 0
    with h5py.File(filename) as root:
        trade_dates = list(root)
        for trade_date in trade_dates:
            spy_prices = root[trade_date]["prices"].value[:, 0]
            vxx_prices = root[trade_date]["prices"].value[:, 1]
            timestamps = root[trade_date]["dates"].value
            for price, timestamp in zip(spy_prices, timestamps):
                if i < window:
                    pass
                else:
                    foo = retrieve_preceding_prices("SPY", trade_date, timestamp, window, True)
                    # print foo[1, 1]
                    # print timestamp
                    pass
                # print repr(i).rjust(9), repr(price).ljust(20), repr(timestamp).ljust(20)
                i += 1

def test_cointegration():
    '''+ SPY.qty * SPY.last * SPY.beta * SPY.vol =
       - VXX.qty * VXX.last * VXX.beta * VXX.vol'''
    ''' don't need beta and vol ratio'''
    ''' a pair can be cointegrated with zero beta, so this is a deceptive metric'''
    pass

@time_this_bizatch(1000)
def time_price_retrieval():
    spy, timespy = retrieve_preceding_prices("SPY", "20100427", 1272395115.0, 15, True)

if __name__ == "__main__":
    spy, timespy = retrieve_preceding_prices("SPY", "20100427", 1272395115.0, 1500, True)
    vxx, timevxx = retrieve_preceding_prices("VXX", "20100427", 1272395115.0, 1500, True)
    time_price_retrieval()
    # plot_SPY()

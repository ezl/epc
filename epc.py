import h5py
import numpy as np
from matplotlib import pyplot
from instrument import FinancialInstrument
from IPython.Shell import IPShellEmbed
import datetime

ipshell = IPShellEmbed("Dropping to IPython shell")
filename = "SPY-VXX-20090507-20100427.hdf5"

def close_close_vol(prices, timestamps, step=1):
    '''timestamps in epoch times, ascending'''
    prices = prices[::step]
    timestamps = timestamps[::step]
    secs_per_year = 60 * 60 * 24 * 365
    raw_variances = np.diff(np.log(prices)) ** 2
    total_time = (timestamps[-1] - timestamps[0]) / secs_per_year
    annualized_variance = raw_variances.sum() / total_time
    return np.sqrt(annualized_variance)

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
    spy_price_changes = np.hstack((0, np.diff(spy_prices)))
    vxx_price_changes = np.hstack((0, np.diff(vxx_prices)))
    spy_returns = np.hstack((0, np.diff(np.log(spy_prices))))
    vxx_returns = np.hstack((0, np.diff(np.log(vxx_prices))))

    timestamps = np.hstack([root[t]["dates"].value for t in trade_dates])
    spy_positions = np.zeros_like(spy_prices)
    vxx_positions = np.zeros_like(spy_prices)
    signal_positions = np.zeros_like(spy_prices)
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

    signal_price_changes = np.hstack((0, np.diff(signal)))
    # determine position
    for i in range(len(spy_prices)):
# carry position over. will be rewritten below on changes
        spy_positions[i] = spy_positions[i-1]
        vxx_positions[i] = vxx_positions[i-1]
        signal_positions[i] = signal_positions[i-1]
# entry condition
        # 2 stddevs away
        if abs(signal[i]) > 2 * signal_std[i]:
            multiplier = abs(signal[i]) / signal_std[i]
            # and no current position
            spy_pos_raw_calc = multiplier * spy_qty[i]
            vxx_pos_raw_calc = multiplier * vxx_qty[i]
            if spy_pos_raw_calc > abs(spy_positions[i]):
                if signal[i] > 0:
                    # sell both
                    spy_positions[i] = -spy_pos_raw_calc
                    vxx_positions[i] = -vxx_pos_raw_calc
                    signal_positions[i] = -multiplier
                else:
                    # buy both
                    spy_positions[i] = spy_pos_raw_calc
                    vxx_positions[i] = vxx_pos_raw_calc
                    signal_positions[i] = multiplier
# exit condition
        # close on signal crosses zero
        # if signal_crosses_zero[i]:
        # close on within one standard deviation
        if abs(signal[i]) < signal_std[i]:
            if spy_positions[i] != 0:
                signal_positions[i] = 0
                spy_positions[i] = 0
                vxx_positions[i] = 0
    # close position at end
    spy_positions[-1] = 0

    total_trades = sum(abs(np.diff(spy_positions != 0)))
    spy_trades = np.hstack((0, np.diff(spy_positions)))
    vxx_trades = np.hstack((0, np.diff(vxx_positions)))
    cash = - vxx_trades * vxx_prices - spy_trades * spy_prices
    spy_pnls = spy_positions * spy_price_changes
    vxx_pnls = vxx_positions * vxx_price_changes
    total_pnls = spy_pnls + vxx_pnls
    signal_pnls = signal_positions * signal_price_changes

    cost_per_share = .02
    spy_costs = abs(spy_trades) * cost_per_share
    vxx_costs = abs(vxx_trades) * cost_per_share
    total_costs = spy_costs + vxx_costs

    print datetime.datetime.now() - start_time
    pyplot.plot(1)
    pyplot.show()

    pyplot.plot(spy_prices, "k")
    pyplot.twinx()
    pyplot.plot(vxx_prices, "r")

    pyplot.figure()
    pyplot.plot(signal)
    pyplot.plot(signal_std *2, 'r--')
    pyplot.plot(-signal_std *2, 'r--')

    pyplot.figure()
    pyplot.plot(np.cumsum(total_pnls))
    pyplot.plot(np.cumsum(total_costs))
    ipshell()

if __name__ == "__main__":
    backtest()




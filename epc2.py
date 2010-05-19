import h5py
import numpy as np
from matplotlib import pyplot
from instrument import FinancialInstrument
from IPython.Shell import IPShellEmbed
import datetime

ipshell = IPShellEmbed("Dropping to IPython shell")
filename = "SPY-VXX-20090507-20100427.hdf5"

def nans_like(a):
    return np.nan * np.zeros_like(a)

def stack(instrlist, attr, start_index=0, end_index=None):
    if end_index is None:
        end_index = len(instrlist)
    return np.hstack([getattr(instrlist[t], attr) for t in range(start_index, end_index)])

def backtest():
    # trade parameters
    vol_window = 1000 # how many timesteps to look back when computing vol
    portfolio_window = 1000 # ... and for the stable value moving average
    vol_nans = np.array([np.nan for i in range(vol_window)])
    portfolio_nans = np.array([np.nan for i in range(portfolio_window)])
    step = 1

    root = h5py.File(filename)
    trade_dates = list(root)


    spy = []; vxx = []; stable_value_portfolio = []
    vol_ratio_window = 1
    for i in range(len(trade_dates)):
        trade_date = trade_dates[i]
        print i, trade_date
        spy.append(FinancialInstrument(root[trade_date]["prices"].value[:, 0],
                                       root[trade_date]["dates"].value))
        vxx.append(FinancialInstrument(root[trade_date]["prices"].value[:, 1],
                                       root[trade_date]["dates"].value))
        stable_value_portfolio.append(FinancialInstrument(nans_like(spy[i].prices),
                                                          root[trade_date]["dates"].value))
        spy[i].trade_date = trade_date
        vxx[i].trade_date = trade_date
        stable_value_portfolio[i].trade_date = trade_date
        if i < vol_ratio_window:
            spy[i].vol = np.nan
            vxx[i].vol = np.nan
            spy[i].portfolio_weight = np.nan
            vxx[i].portfolio_weight = np.nan
        else:
            previous_dates = range(i - vol_ratio_window, i)
            spy[i].vol = sum([spy[p].volatility for p in previous_dates]) / len(previous_dates)
            vxx[i].vol = sum([vxx[p].volatility for p in previous_dates]) / len(previous_dates)
            vol_ratio = spy[i].vol / vxx[i].vol
            last_closing_price_ratio = spy[i - 1].prices[-1] / vxx[i - 1].prices[-1]
            spy[i].portfolio_weight = 1
            vxx[i].portfolio_weight = vol_ratio * last_closing_price_ratio
            stable_value_portfolio[i].prices = spy[i].portfolio_weight * spy[i].prices + vxx[i].portfolio_weight * vxx[i].prices

    pyplot.plot(1)
    pyplot.show()

    pyplot.plot(stack(spy, "prices"), "k")
    pyplot.twinx()
    pyplot.plot(stack(vxx, "prices"), "r")
    pyplot.title("vxx/spy prices")

    pyplot.figure()
    pyplot.plot(stack(stable_value_portfolio, "prices"))
    pyplot.title("stable value portfolio")

    pyplot.figure()
    pyplot.plot(stack(spy, "vol"), "k")
    pyplot.twinx()
    pyplot.plot(stack(vxx, "vol"), "r")
    pyplot.title("vxx/spy vol")

    pyplot.figure()
    pyplot.plot(stack(spy, "portfolio_weight"), "k")
    pyplot.twinx()
    pyplot.plot(stack(vxx, "portfolio_weight"), "r")
    pyplot.title("vxx/spy portfolio weights")
    ipshell()

#     spy_vol = np.hstack((vol_nans[1:],
#                           [close_close_vol(spy_prices[t-vol_window:t],
#                                            timestamps[t-vol_window:t],
#                                            step=step)
#                            for t in range(vol_window, len(spy_prices)+1)]
#                         ))
#     vxx_vol = np.hstack((vol_nans[1:],
#                           [close_close_vol(vxx_prices[t-vol_window:t],
#                                            timestamps[t-vol_window:t],
#                                            step=step)
#                            for t in range(vol_window, len(spy_prices)+1)]
#                         ))
#     spy_qty = np.ones_like(spy_prices)
#     vxx_qty = spy_vol / vxx_vol * spy_prices / vxx_prices
#     portfolio = spy_qty * spy_prices + vxx_qty * vxx_prices
#     portfolio_std = np.hstack((portfolio_nans[1:],
#                                [portfolio[t-portfolio_window:t].std()
#                                 for t in range(portfolio_window, len(spy_prices)+1)]
#                              ))
#     portfolio_moving_average = np.hstack((portfolio_nans[1:],
#                                           np.convolve(np.ones(portfolio_window) / portfolio_window,
#                                                       portfolio, mode= "valid")
#                                         ))
#     signal = portfolio - portfolio_moving_average
#     signal_crosses_zero = np.hstack((np.nan,
#                                      [cmp(s, 0) for s in signal[1:] * signal[:-1]]
#                                    )) <= 0
#     signal_moving_average = np.hstack((portfolio_nans[1:],
#                                        np.convolve(np.ones(portfolio_window) / portfolio_window,
#                                                    signal, mode= "valid")
#                                      ))
#     signal_std = np.hstack((portfolio_nans[1:],
#                             [signal[t-portfolio_window:t].std()
#                              for t in range(portfolio_window, len(spy_prices)+1)]
#                           ))
#     big_signal = abs(signal) > signal_std
#     big2_signal = abs(signal) > 2 * signal_std
# 
#     signal_price_changes = np.hstack((0, np.diff(signal)))
#     # determine position
#     for i in range(len(spy_prices)):
# # carry position over. will be rewritten below on changes
#         spy_positions[i] = spy_positions[i-1]
#         vxx_positions[i] = vxx_positions[i-1]
#         signal_positions[i] = signal_positions[i-1]
# # entry condition
#         # 2 stddevs away
#         if abs(signal[i]) > 2 * signal_std[i]:
#             multiplier = abs(signal[i]) / signal_std[i]
#             # and no current position
#             spy_pos_raw_calc = multiplier * spy_qty[i]
#             vxx_pos_raw_calc = multiplier * vxx_qty[i]
#             if spy_pos_raw_calc > abs(spy_positions[i]):
#                 if signal[i] > 0:
#                     # sell both
#                     spy_positions[i] = -spy_pos_raw_calc
#                     vxx_positions[i] = -vxx_pos_raw_calc
#                     signal_positions[i] = -multiplier
#                 else:
#                     # buy both
#                     spy_positions[i] = spy_pos_raw_calc
#                     vxx_positions[i] = vxx_pos_raw_calc
#                     signal_positions[i] = multiplier
# # exit condition
#         # close on signal crosses zero
#         # if signal_crosses_zero[i]:
#         # close on within one standard deviation
#         if abs(signal[i]) < signal_std[i]:
#             if spy_positions[i] != 0:
#                 signal_positions[i] = 0
#                 spy_positions[i] = 0
#                 vxx_positions[i] = 0
#     # close position at end
#     spy_positions[-1] = 0
# 
#     total_trades = sum(abs(np.diff(spy_positions != 0)))
#     spy_trades = np.hstack((0, np.diff(spy_positions)))
#     vxx_trades = np.hstack((0, np.diff(vxx_positions)))
#     cash = - vxx_trades * vxx_prices - spy_trades * spy_prices
#     spy_pnls = spy_positions * spy_price_changes
#     vxx_pnls = vxx_positions * vxx_price_changes
#     total_pnls = spy_pnls + vxx_pnls
#     signal_pnls = signal_positions * signal_price_changes
# 
#     cost_per_share = .02
#     spy_costs = abs(spy_trades) * cost_per_share
#     vxx_costs = abs(vxx_trades) * cost_per_share
#     total_costs = spy_costs + vxx_costs
# 
#     print datetime.datetime.now() - start_time
#     pyplot.plot(1)
#     pyplot.show()
# 
#     pyplot.plot(spy_prices, "k")
#     pyplot.twinx()
#     pyplot.plot(vxx_prices, "r")
# 
#     pyplot.figure()
#     pyplot.plot(signal)
#     pyplot.plot(signal_std *2, 'r--')
#     pyplot.plot(-signal_std *2, 'r--')
# 
#     pyplot.figure()
#     pyplot.plot(np.cumsum(total_pnls))
#     pyplot.plot(np.cumsum(total_costs))
#     ipshell()

if __name__ == "__main__":
    backtest()




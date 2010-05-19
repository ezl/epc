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
    '''Stack attribute from FinancialInstrument object from specified days into one numpy array
       usage: stack(spy, "prices", start_index=10, end_index=20)
              will return attribute "prices" from object spy for list index range(10,20)
    '''

    if end_index is None:
        end_index = len(instrlist)
    return np.hstack([getattr(instrlist[t], attr) for t in range(start_index, end_index)])

def retrieve_preceding(instrlist, day, attr, last_index, window, concurrent=True):
    if concurrent == True:
        last_index += 1
    first_index = last_index - window
    if first_index >= 0:
        return getattr(instrlist[day], attr)[first_index:last_index]
    else:
        remaining_window = -first_index
        if day == 0: # thats the end of the chain
            return None
        else:
            previous_last_index = instrlist[day-1].index[-1]
        preceding = retrieve_preceding(instrlist, day-1, attr, previous_last_index, remaining_window, concurrent=True)
        if not preceding is None:
            return np.hstack((preceding, getattr(instrlist[day], attr)[:last_index]))
        else:
            return None

def backtest():
    # trade parameters
    root = h5py.File(filename)
    trade_dates = list(root)
    plot =False
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

    if plot == True:
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

        pyplot.figure()
        pyplot.plot(stack(vxx, "vol")/stack(spy, "vol"))
        pyplot.title("spy/vxx vol ratio")

    ipshell()

if __name__ == "__main__":
    backtest()




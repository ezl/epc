import h5py
import numpy as np
from matplotlib import pyplot
from instrument import FinancialInstrument
from IPython.Shell import IPShellEmbed
import datetime
import pdb

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

def get_preceding(instrlist, day, attr, last_index, window, day_index, concurrent=True):
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
            try:
               previous_last_index = day_index[day-1][-1]
            except Exception, e:
               ipshell(e)
        preceding = get_preceding(instrlist, day-1, attr, previous_last_index, remaining_window, day_index=day_index, concurrent=True)
        if not preceding is None:
            return np.hstack((preceding, getattr(instrlist[day], attr)[:last_index]))
        else:
            return None

def get_vol(instrlist, day, last_index, window, day_index, concurrent=True):
    # '''Assume: all timedeltas are 15 seconds. replace overnight yields with 0.'''
    returns = get_preceding(instrlist=instrlist, day=day, attr="log_returns", last_index=last_index, window=window, day_index=day_index, concurrent=True)
    if returns == None:
        return np.nan
    else:
        years = 15.0 * window / (60. * 60. * 24. * 365.)
        return np.sqrt((returns ** 2).sum() / years)

def get_moving_average(instrlist, day, last_index, window, day_index, concurrent=True):
    prices = get_preceding(instrlist=instrlist, day=day, attr="prices", last_index=last_index, window=window, day_index=day_index, concurrent=True)
    if prices == None:
        return np.nan
    else:
        # np.convolve(np.ones(window) / len(window), prices, "valid")
        return prices.mean()

def get_std(instrlist, day, last_index, window, day_index, concurrent=True):
    prices = get_preceding(instrlist=instrlist, day=day, attr="prices", last_index=last_index, window=window, day_index=day_index, concurrent=True)
    if prices == None:
        return np.nan
    else:
        return prices.std()

def get_stat(instrlist, day, last_index, window, attr, fn, concurrent=True):
    data = get_preceding(instrlist=instrlist, day=day, attr=attr, last_index=last_index, window=window, concurrent=True)
    if data == None:
        return np.nan
    else:
        return fn(data)

def plot_day_dividers(closes):
    [pyplot.axvline(x=c, color="#B0E0E6") for c in closes]

def backtest():
    # trade parameters
    plot = True
    daily = True
    cost_per_share = .02
    vol_ratio_window = 1500
    portfolio_window = 1000
    start_day = 0
    end_day = 245

    root = h5py.File(filename)
    trade_dates = list(root)
    spy = []; vxx = []; stable_value_portfolio = []; signal = []
    timestamps = []; timesteps = []; closes = []; day_index = []
    for day in range(start_day, end_day):
        trade_date = trade_dates[day]
        print day, trade_date
        timestamps.append(root[trade_date]["dates"].value)
        spy.append(FinancialInstrument(root[trade_date]["prices"].value[:, 0],
                                       timestamps[day]))
        vxx.append(FinancialInstrument(root[trade_date]["prices"].value[:, 1],
                                       timestamps[day]))
        assert vxx[day].prices.shape == spy[day].prices.shape
        day_index.append(np.cumsum(np.ones_like(spy[day].prices)) - 1)
        try:
            timesteps.append(day_index[day] + timesteps[day-1][-1] + 1)
        except IndexError:
            timesteps.append(day_index[day])
        closes.append(timesteps[day][-1])

        # recompute vols and portfolio weights ratios only every day
        if daily == True:
            if day == 0:
                spy[day].vol = nans_like(spy[day].prices)
                vxx[day].vol = nans_like(spy[day].prices)
                spy[day].portfolio_weight = nans_like(spy[day].prices)
                vxx[day].portfolio_weight = nans_like(spy[day].prices)
                # all nans
            else:
                spy[day].vol = np.ones_like(spy[day].prices) * spy[day-1].volatility
                vxx[day].vol = np.ones_like(vxx[day].prices) * vxx[day-1].volatility
                vol_ratio = spy[day].vol / vxx[day].vol
                last_ratio = np.ones_like(spy[day].prices) * (spy[day-1].prices[-1] / vxx[day-1].prices[-1])
                spy[day].portfolio_weight = np.ones_like(spy[day].prices)
                vxx[day].portfolio_weight = vol_ratio * last_ratio

        # recompute everything at every timestep
        else:
            spy[day].vol = np.array([get_vol(instrlist=spy, day=day, last_index=i, window=vol_ratio_window, day_index=day_index, concurrent=True) for i in day_index[day]])
            vxx[day].vol = np.array([get_vol(instrlist=vxx, day=day, last_index=i, window=vol_ratio_window, day_index=day_index, concurrent=True) for i in day_index[day]])
            vol_ratio = spy[day].vol / vxx[day].vol
            last_ratio = spy[day].prices / vxx[day].prices
            spy[day].portfolio_weight = np.ones_like(spy[day].prices)
            vxx[day].portfolio_weight = vol_ratio * last_ratio

        stable_value_portfolio.append(FinancialInstrument(spy[day].portfolio_weight * spy[day].prices + vxx[day].portfolio_weight * vxx[day].prices,
                                                          timestamps[day]))
        stable_value_portfolio[day].moving_average = np.array([get_moving_average(instrlist=stable_value_portfolio, day=day, last_index=i, window=portfolio_window, day_index=day_index, concurrent=True) for i in day_index[day]])
        signal.append(FinancialInstrument(stable_value_portfolio[day].prices - stable_value_portfolio[day].moving_average,
                                          root[trade_date]["dates"].value))
        signal[day].std = np.array([get_std(instrlist=signal, day=day, last_index=i, window=portfolio_window, day_index=day_index, concurrent=True) for i in day_index[day]])
        signal[day].strength = signal[day].prices / signal[day].std # hugo's signal? how many standard deviations away is it

        # define the trade
        spy[day].pos = np.zeros_like(spy[day].prices)
        vxx[day].pos = np.zeros_like(vxx[day].prices)
        # start the day with zero position, but first timestep ends in 15 seconds and you can put on a position then.
        # pos[0] can be nonzero, still started the day with zero. pos[0] is the pos put on at end of time[0], into time[1]

        for i in day_index[day]:
            # default to carrying over previous position unless trade signal is received
            spy[day].pos[i] = spy[day].pos[i-1]
            vxx[day].pos[i] = vxx[day].pos[i-1]
            # entry condition
            if abs(signal[day].strength[i]) > 2:
                # TODO: rebalance the whole portfolio together, not one name at a time.
                spy[day].pos[i] = -cmp(signal[day].prices[i], 0) * max(spy[day].portfolio_weight[i] * abs(signal[day].strength[i]),
                                                                       abs(spy[day].pos[i-1]))
                vxx[day].pos[i] = -cmp(signal[day].prices[i], 0) * max(vxx[day].portfolio_weight[i] * abs(signal[day].strength[i]),
                                                                       abs(vxx[day].pos[i-1]))
                if spy[day].pos[i-1] == 0 and not spy[day].pos[i] ==0:
                    print "                  ENTER a trade"
            # exit condition
            elif abs(signal[day].strength[i]) < 1:
                spy[day].pos[i] = 0
                vxx[day].pos[i] = 0
                if not spy[day].pos[i-1] == 0 and spy[day].pos[i] ==0:
                    print "                                       EXIT a trade"

        # close position at end of day
        spy[day].pos[-1] = 0
        vxx[day].pos[-1] = 0

        # compute p&l
        spy[day].pnl = spy[day].pos * spy[day].price_changes
        vxx[day].pnl = vxx[day].pos * vxx[day].price_changes
        spy[day].daypnl = np.cumsum(spy[day].pnl)
        vxx[day].daypnl = np.cumsum(vxx[day].pnl)
        spy[day].trades = np.diff(np.hstack((0, spy[day].pos)))
        vxx[day].trades = np.diff(np.hstack((0, vxx[day].pos)))
        spy[day].cost = abs(spy[day].trades * cost_per_share)
        vxx[day].cost = abs(vxx[day].trades * cost_per_share)
        spy[day].daycost = np.cumsum(spy[day].cost)
        vxx[day].daycost = np.cumsum(vxx[day].cost)
        if day != 0:
            spy[day].totalpnl = spy[day].daypnl + spy[day-1].totalpnl[-1]
            vxx[day].totalpnl = vxx[day].daypnl + vxx[day-1].totalpnl[-1]
            spy[day].totalcost = spy[day].daycost + spy[day-1].totalcost[-1]
            vxx[day].totalcost = vxx[day].daycost + vxx[day-1].totalcost[-1]
        else:
            spy[day].totalpnl = spy[day].daypnl
            vxx[day].totalpnl = vxx[day].daypnl
            spy[day].totalcost = spy[day].daycost
            vxx[day].totalcost = vxx[day].daycost
        # compute costs





    # plot them
    if plot == True:
        pyplot.plot(1)
        pyplot.show()

#         pyplot.plot(np.hstack(timesteps), stack(spy, "prices"), "k")
#         pyplot.twinx()
#         pyplot.plot(np.hstack(timesteps), stack(vxx, "prices"), "r")
#         plot_day_dividers(closes)
#         pyplot.title("vxx/spy prices")
#
#         pyplot.figure()
#         pyplot.plot(np.hstack(timesteps), stack(spy, "vol"), "k")
#         pyplot.twinx()
#         pyplot.plot(np.hstack(timesteps), stack(vxx, "vol"), "r")
#         plot_day_dividers(closes)
#         pyplot.title("vxx/spy vol")
#
#         pyplot.figure()
#         pyplot.plot(np.hstack(timesteps), stack(spy, "portfolio_weight"), "k")
#         pyplot.twinx()
#         pyplot.plot(np.hstack(timesteps), stack(vxx, "portfolio_weight"), "r")
#         plot_day_dividers(closes)
#         pyplot.title("vxx/spy portfolio weights")
#
#         pyplot.figure()
#         pyplot.plot(np.hstack(timesteps), stack(vxx, "vol")/stack(spy, "vol"))
#         plot_day_dividers(closes)
#         pyplot.title("spy/vxx vol ratio")

#         pyplot.figure()
#         pyplot.plot(np.hstack(timesteps), stack(stable_value_portfolio, "prices"))
#         plot_day_dividers(closes)
#         pyplot.title("stable value portfolio")

        pyplot.figure()
        pyplot.plot(np.hstack(timesteps), stack(signal, "prices"))
        pyplot.plot(np.hstack(timesteps), stack(signal, "std"), "r--")
        pyplot.plot(np.hstack(timesteps), -stack(signal, "std"), "r--")
        pyplot.plot(np.hstack(timesteps), 2 * stack(signal, "std"), "g--")
        pyplot.plot(np.hstack(timesteps), -2 * stack(signal, "std"), "g--")
        plot_day_dividers(closes)
        pyplot.title("signal")

        pyplot.figure()
        pyplot.plot(np.hstack(timesteps), stack(spy, "pos"), "k")
        pyplot.plot(np.hstack(timesteps), stack(vxx, "pos"), "r")
        plot_day_dividers(closes)
        pyplot.title("positions")

        pyplot.figure()
        pyplot.plot(np.hstack(timesteps), stack(spy, "totalpnl"), "k")
        pyplot.plot(np.hstack(timesteps), stack(vxx, "totalpnl"), "r")
        pyplot.plot(np.hstack(timesteps), stack(spy, "totalpnl") + stack(vxx, "totalpnl"), "g")
        pyplot.plot(np.hstack(timesteps), stack(spy, "totalcost") + stack(vxx, "totalcost"), "g--")
        plot_day_dividers(closes)
        pyplot.title("pnl")
    ipshell()

if __name__ == "__main__":
    backtest()




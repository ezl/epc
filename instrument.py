import numpy as np

class FinancialInstrument(object):
    def __init__(self, prices, timestamps):
        '''timestamps as epoch time'''
        assert prices.shape == timestamps.shape
        self.prices = prices
        self.timestamps = timestamps

    def __repr__(self):
        info = """
        o, h, l, c: %s, %s, %s, %s
        shape: %s """ % (self.prices[0],
                         self.prices.max(),
                         self.prices.min(),
                         self.prices[-1],
                         self.prices.shape)
        return info

    @property
    def log_returns(self):
        return np.hstack((0, np.diff(np.log(self.prices))))

    @property
    def price_changes(self):
        return np.hstack((0, np.diff(self.prices)))

    @property
    def volatility(self):
        '''close-close volatility'''
        secs_per_year = 60 * 60 * 24 * 365
        raw_variance = (self.log_returns ** 2).sum()
        total_time = (self.timestamps[-1] - self.timestamps[0]) / secs_per_year
        annualized_variance = raw_variance / total_time
        return np.sqrt(annualized_variance)

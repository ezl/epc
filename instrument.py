import numpy as np

class FinancialInstrument(object):
    def __init__(self, prices):
        self.prices = prices

    def __repr__(self):
        repr = """
               prices: %s
               shape: %s

               """ % (self.prices, self.prices.shape)
        return repr

    @property
    def log_yields(self):
        return np.hstack((0, np.diff(np.log(self.prices))))

    @property
    def price_changes(self):
        return np.hstack((0, np.diff(self.prices)))

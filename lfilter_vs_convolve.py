import numpy as np
from scipy.signal import lfilter, fftconvolve
from IPython.Shell import IPShellEmbed
ipshell = IPShellEmbed("Dropping to IPython shell")

def time_function(iterations):
    import datetime
    def timer_decorator(fxn):
        def wrapped(*args, **kwargs):
            start = datetime.datetime.now()
            for i in range(iterations):
                fxn()
            end = datetime.datetime.now()
            print "%s iterations, time_elapsed: %s" % (iterations, end - start)
        return wrapped
    return timer_decorator

window = 200
iterations = 20
signal = range(100000)

@time_function(iterations)
def scipylfilter():
    lfilter(np.ones(window), window, signal)

@time_function(iterations)
def scipyfftconvolve():
    fftconvolve(np.ones(window), np.ones(window) / window, signal)

@time_function(iterations)
def numpyconvolve():
    np.convolve(np.ones(window)/window, signal, mode="valid")

print "lfilter:"
scipylfilter()
print "fftconvolve:"
scipyfftconvolve()
print "convolve:"
numpyconvolve()

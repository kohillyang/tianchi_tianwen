import pywt
import numpy as np
import scipy.signal as signal
from matplotlib import pyplot as plt

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a,shape=shape, strides=strides)

spcts = np.load('all_train_normalized.npy')[0:10,:]
print spcts.shape

spct = spcts[1,:]


from twaugment import SpctFeatureExtract,waveletSmooth



spct, spct_sub,T0, T1, spct_line = SpctFeatureExtract(spct)

T0 = spct*0 + T0

wavelet = waveletSmooth(spct)


# plt.plot(spct_sub,'o-',markersize=2)
# plt.plot(T0)
# plt.plot(T1)
# plt.figure()
# plt.plot(T0)
# plt.plot(T1)
# plt.plot(spct_line,'o-',markersize=2)
# plt.figure()
# plt.plot(spct,alpha=0.5)
# plt.plot(wavelet,'o-',markersize=2)
# plt.show()

# down sample
plt.plot(spct)
plt.show()



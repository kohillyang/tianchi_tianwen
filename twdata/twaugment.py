from __future__ import division
import torch
import numpy as np
from numpy import random
import scipy.signal as signal
import pywt
from statsmodels.robust import mad
from scipy import interpolate

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a,shape=shape, strides=strides)

def windowRMS(spct, w ):
    p = int((w - 1) / 2)
    a = np.pad(spct, (p, p), 'constant', constant_values=(0, 0))
    window_spct = rolling_window(a, w)
    return np.sqrt(1.0*((window_spct)**2).sum(axis=1)/window_spct.shape[0])

def SpctFeatureExtract(spct):
    contin = signal.medfilt(spct, 301)
    spct_sub = spct - contin
    T1 = 2.5 * windowRMS(spct_sub, 251 )
    T0 = np.sqrt(1.0*((spct_sub)**2).sum()/spct_sub.shape[0])
    spct_line = spct_sub.copy()
    spct_line[((spct_line >= -T0) & (spct_line <= T0))] = 0
    spct_line[((spct_line >= -T1) & (spct_line <= T1))] = 0

    return spct, contin, spct_line





def waveletSmooth(x, wavelet="db4", level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")

    sigma = mad(coeff[-level])

    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="soft") for i in coeff[1:])

    y = pywt.waverec(coeff, wavelet, mode="per")
    return y



class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, spct):
        for t in self.transforms:
            spct = t(spct)
        return spct


class WTDenoise():
    def __call__(self, spct):
        return waveletSmooth(spct)


class AddNoise():
    def __init__(self, A=0.1):
        self.A = A

    def __call__(self, spct):
        # add noise
        mid = np.median(np.abs(np.diff(spct, n=1)))
        spct = spct + np.random.normal(0, self.A*mid, spct.shape)
        return spct


class RandomAmplitude():
    def __init__(self, l=0.8,h=1.2):
        self.l = l
        self.h = h

    def __call__(self, spct):
        spct = spct * np.random.uniform(self.l, self.h)
        return spct


class RandomShiftCrop():
    def __init__(self, ori_len=2600, new_len=2600*0.8):
        self.ori_len = ori_len
        self.new_len = int(new_len)

    def __call__(self, spct):
        start = np.random.randint(self.ori_len - self.new_len)
        end = start + self.new_len
        spct = spct[start: end]

        return spct

class ShiftCrop():
    def __init__(self,start_point, end_point):
        self.start_point = int(start_point)
        self.end_point = int(end_point)

    def __call__(self, spct):
        spct = spct[self.start_point: self.end_point]

        return spct



class CenterCrop():
    def __init__(self, ori_len=2600, new_len=2600*0.8):
        self.ori_len = ori_len
        self.new_len = int(new_len)

    def __call__(self, spct):
        start = int((self.ori_len - self.new_len)/2)
        end = start + self.new_len

        spct = spct[start: end]
        return spct


class RedShift():
    def __init__(self,ori_len=2600, beta=0.8):
        self.zmin = (1-beta) * (-0.5945)
        self.zmax = (1-beta) * 1.4661
        self.x, self.step = np.linspace(3690, 9100, 2600, retstep=True)
        self.new_len = int(beta*ori_len)
        self.ori_len = int(ori_len)

    def __call__(self, spct):
        z = np.random.uniform(self.zmin, self.zmax)

        if z == 0:
            # fall back to random crop
            start = np.random.randint(self.ori_len - self.new_len)
            end = start + self.new_len
            spct = spct[start: end]

        else:
            x_sft = self.x + z * self.x
            f = interpolate.interp1d(x_sft,spct,kind='linear')

            if z>0:
                xnew = np.arange(x_sft[0], self.x[-1], step=self.step)
            else:
                xnew = np.arange(self.x[0], x_sft[-1], step=self.step)

            if len(xnew)<self.new_len:
                print '%.3f %d/%d [%.2f, %.2f]'%(z,len(xnew),self.new_len,xnew[0],xnew[-1])
                start = np.random.randint(self.ori_len - self.new_len)
                end = start + self.new_len
                spct = spct[start: end]
            elif len(xnew)==self.new_len:
                spct = f(xnew)
            else:
                spct = f(xnew)
                start = np.random.randint(len(spct) - self.new_len)
                end = start + self.new_len
                spct = spct[start: end]


        return spct



class Normalize():
    def __init__(self, mean=5390.4822, std=1917880.56):
        self.mean = mean
        self.std = std

    def __call__(self, spct):
        spct = (spct - self.mean) / self.std
        return spct


class FlowNormalize():
    def __init__(self, mode='standard'):
        self.mode = mode

    def __call__(self, spct):
        if self.mode == 'standard':
            spct = spct / np.sqrt((spct**2).sum())
        elif self.mode == 'max':
            spct = spct / spct.max()
        elif self.mode == 'min':
            spct = spct / spct.min()
        return spct

class SpctLine():
    def __call__(self, spct):
        spct, contin, spct_line = SpctFeatureExtract(spct)

        spct = np.vstack((spct[np.newaxis,:],
                   contin[np.newaxis, :],
                   spct_line[np.newaxis, :]))


        return spct


class DownSample():
    def __init__(self, size):
        self.size = size

    def __call__(self, spct):
        x = np.linspace(1,spct.shape[0],spct.shape[0])
        f = interpolate.interp1d(x,spct,kind='linear')
        xnew = np.linspace(1,self.size,self.size)
        spct = f(xnew)

        return spct

class AddAxis():
    def __call__(self, spct):
        return spct[np.newaxis,:]



class PCAreduce():
    def __init__(self, pca):
        self.pca = pca

    def __call__(self, spct):
        spct = self.pca.transform(spct)
        return spct




class TWAug(object):
    def __init__(self):
        self.augment = Compose([
            AddNoise(A=0.1),
            RandomAmplitude(l=0.9,h=1.1),
            # RandomShiftCrop(ori_len=2600, new_len=2600*0.8),
            FlowNormalize(),
            AddAxis()
        ])

    def __call__(self, spct):
        return self.augment(spct)

class TWAugVal(object):
    def __init__(self):
        self.augment = Compose([
            # CenterCrop(ori_len=2600, new_len=2600*0.8),
            FlowNormalize(),
            AddAxis()
        ])

    def __call__(self, spct):
        return self.augment(spct)



class TWAug3C():
    def __init__(self):
        self.augment = Compose([
            AddNoise(A=0.1),
            RandomAmplitude(l=0.9,h=1.1),
            # RandomShiftCrop(ori_len=2600, new_len=2600*0.8),
            FlowNormalize(),
            SpctLine()
        ])

    def __call__(self, spct):
        return self.augment(spct)

class TWAug3CVal():
    def __init__(self):
        self.augment = Compose([
            # CenterCrop(ori_len=2600, new_len=2600*0.8),
            FlowNormalize(),
            SpctLine()
        ])

    def __call__(self, spct):
        return self.augment(spct)



class TWAug3Cnpy():
    def __init__(self):
        self.augment = Compose([
            AddNoise(A=0.1),
            RandomAmplitude(l=0.9,h=1.1),
        ])
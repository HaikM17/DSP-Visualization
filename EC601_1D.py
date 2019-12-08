# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 18:37:08 2019

@author: haikm
"""

import os, json, io
from urllib.request import urlopen

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

def one_d_fft(y):
    
    # Perform the FFT
    F = np.fft.fft(y)

    # The full spectrum F consists of complex numbers ordered by frequency
    # Extract frequency, amplitude, phase information 
    fft_freq = np.fft.fftfreq(len(F))*n_samples
    fft_amp = np.abs(F)/n_samples
    fft_ph = np.angle(F)

    # Plot the result
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(fft_amp,'r.')
    xt = np.linspace(0,len(fft_freq), 5).astype(int)
    plt.xticks(xt, 1+fft_freq[xt-1].astype(int))
    plt.xlabel("Frequency",fontsize=20), plt.ylabel("Amplitude",fontsize=20)

    plt.subplot(1,2,2)
    plt.plot(fft_ph,'b.')
    plt.xticks(xt, 1+fft_freq[xt-1].astype(int))
    plt.xlabel("Frequency",fontsize=20), plt.ylabel("Phase",fontsize=20)
    plt.show()
    
    return fft_amp, fft_ph


# Set frequency and sampling rate
ph = np.pi/2
freq = 20
amp = 1
n_samples = 200

# Define and 1-D signal
x = np.linspace(0,1,n_samples)
y = np.sin(ph + (2*np.pi*freq*x))*amp

plt.figure(figsize=(10,1))
plt.plot(x,y,'b-')
plt.title("One Dimensional Signal",fontsize=20)
plt.show()


# FFT 
fft_amp, fft_ph = one_d_fft(y)

#Inverse Fourier transform
i_F = fft_amp*n_samples * np.exp(1j*fft_ph)
i_y = np.fft.ifft(i_F).real

# Plot reconstructed 1-D signal
plt.figure(figsize=(10,1))
plt.ylim([-1,1])
plt.plot(x,y,'b-')
plt.plot(x,i_y,'r-')
plt.title("Inverse Fourier 1-D Signal",fontsize=20)
plt.show()


'''
Complex Signal
More complex signals will show a more complex 
pattern of amplitudes and phases
'''
ph1 = np.pi/2; ph2 = 0; ph3 = 5*np.pi/4
freq1 = 9; freq2 = 20; freq3 = 35
amp1 = 1; amp2 = 1.5; amp3= 0.75
n_samples = 200

x = np.linspace(0,1,n_samples)
y1 = np.sin(ph1 + (2*np.pi*freq1*x))*amp1
y2 = np.sin(ph2 + (2*np.pi*freq2*x))*amp2
y3 = np.sin(ph3 + (2*np.pi*freq3*x))*amp3
ys = y1+y2+y3

plt.figure(figsize=(10,1))
plt.plot(x,y1,'b-'); plt.plot(x,y2,'r-'); plt.plot(x,y3,'g-')
plt.plot(x,ys,'k-', linewidth=2)
plt.title("Fourier Spectrum of a Sum of Three Sine Waves",fontsize=20)
plt.show()

'''
Inverse FFT
This is how MP3 compression works: components that are 
less relevant (e.g. low amplitude, or low sensitivity to its frequency) 
are removed from the Fourier transform to reduce the total amount of 
data that needs to be saved.
'''
fft_amp_ys, fft_ph_ys = one_d_fft(ys)



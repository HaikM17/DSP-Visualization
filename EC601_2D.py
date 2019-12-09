# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 19:01:54 2019

@author: haikm
"""
import os, json, io
from urllib.request import urlopen

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def ft(img):
        
    F = np.fft.fft2(img)
    
    # Center the spectrum on the lowest frequency
    F_centered = np.fft.fftshift(F)
    
    # Extract amplitude and phase
    amp = np.abs(F_centered).real
    phase = np.angle(F_centered).real
    
    return amp, phase, F

def inv_ft(amp, phase):
    #Reconstruct the image from amplitude and phase spectrum#
    F_centered = amp * np.exp(1j*phase)
    F = np.fft.ifftshift(F_centered)
    img = np.fft.ifft2(F).real
    return img

def render_gabor(pars):
    #Render a single Gabor patch
    vals = np.linspace(-pars['hsize'], pars['hsize'], (2*pars['hsize'])+1)
    xgr, ygr = np.meshgrid(vals, vals)
    gaussian = np.exp(-(xgr**2+ygr**2)/(2*pars['sigma']**2))
    slant = (ygr*(2*np.pi*pars['freq']*np.cos(pars['or'])) +
             xgr*(2*np.pi*pars['freq']*np.sin(pars['or']))) 
    grating = pars['amp']*np.cos(slant+pars['phase'])
    
    return gaussian*grating

def filter_image(flt,img):
    #Filter an image with one filter
    
    hfltr, vfltr = flt.shape
    himg, vimg = img.shape
    hres = hfltr+himg
    vres = vfltr+vimg
    hres2 = 2**int(np.log(hres)/np.log(2.0) + 1.0 )
    vres2 = 2**int(np.log(vres)/np.log(2.0) + 1.0 )
    img = img[::-1,::-1]
     
    fftimage = (np.fft.fft2(flt, s=(hres2, vres2))*
                np.fft.fft2(img, s=(hres2, vres2)))
    res = np.fft.ifft2(fftimage).real
    
    # Cut the actual filtered image from the result
    res = res[(hfltr//2):hres-(hfltr//2)-1,(vfltr//2):vres-(vfltr//2)-1][::-1, ::-1]
 
    return res  


'''
Fourier Transform in 2 Dimensions
'''
plt.figure(figsize=(15,5))

img=mpimg.imread('lake.jpg')
plt.imshow(img)
plt.title("Original Image", fontsize=20)
plt.show()

img = np.mean(img,axis=2)
plt.imshow(img, cmap='gray')
plt.title("Low Pass Filtered", fontsize=20)
plt.show()

amp, phase, F = ft(img)

plt.imshow(np.log(amp), cmap='gray')
plt.title("FFT Reconstruction from Amplitude", fontsize=20)
plt.show()
plt.imshow(phase, cmap='gray')
plt.title("FFT Reconstruction from Phase", fontsize=20)

plt.show()



'''
Fourier analysis can be extended to the low-pass, 
high-pass or bandpass filtering of images
Artifacts familiar from JPEG compressed images, exaggerated here 

This is how the compression algorithm works: 
higher frequency components are removed.
'''

amp, phase, F = ft(img)
Af = amp.copy()

# Compute one frequency at each pixel
fx = np.fft.fftshift(np.fft.fftfreq(amp.shape[0]))
fy = np.fft.fftshift(np.fft.fftfreq(amp.shape[1]))
fx,fy = np.meshgrid(fy,fx)
freq = np.hypot(fx,fy)
    
# Filter and reconstruct
bandpass = (freq>0) * (freq<0.05)
Af[~bandpass] = 0
f_img = inv_ft(Af, phase)

plt.figure(figsize=(10,10))
plt.imshow(f_img, cmap='gray')
plt.axis('off')
plt.title("Compressed with Artifacts", fontsize=20)
plt.show()


'''
Generate and apply a uniform amplitude spectrum
Amplitude and frequency can be decorrelated by a process called whitening
High frequencies will "dominate"
'''

w = np.ones(amp.shape)
w_img = inv_ft(w, phase)

plt.figure(figsize=(10,10))
plt.imshow(w_img, cmap='gray'), plt.axis('off')
plt.title("'Whitened' Image", fontsize=20)
plt.show()

'''
Edge Filtering
'''
# Render the Gabor edge filter
gab_par = {}
gab_par['freq'] = 0.1
gab_par['sigma'] = 2.5
gab_par['amp'] = 1.
gab_par['phase'] = np.pi/2
gab_par['hsize'] = 15.
gab_par['or'] = 0.
gab = render_gabor(gab_par)

img1=mpimg.imread('cat.jpg')
img1 = np.mean(img1, 2)
plt.figure(figsize=(10,10))
plt.imshow(img1, cmap='gray', interpolation='nearest')
plt.title("Original", fontsize=20)
plt.show()

#Filtered using edge effects
filtered = filter_image(gab,img1)
plt.figure(figsize=(10,10))
plt.imshow(filtered, cmap='gray', interpolation='nearest')
plt.title("Edge Filtered", fontsize=20)
plt.show()


'''
Filterbank
'''

n = 10
ors = np.linspace(0,2*np.pi,n+1)[:-1]
res = np.zeros(img1.shape+(n,))

for idx,this_or in enumerate(ors):
    
    # Render the filter
    gab_par['or'] = this_or
    gab = render_gabor(gab_par)
    
    # Filter the image
    filtered = filter_image(gab,img1)
    
    # Save the result in a numpy array
    res[:,:,idx] = filtered
    
    

#Find out the maximal response strength
str_resp = np.amax(res, axis=2)

# Find out to which filter it corresponds
str_or = np.argmax(res, axis=2)

plt.figure(figsize=(10,10))
plt.imshow(str_resp, cmap='gray', interpolation='nearest')
plt.title("Filter Bank Kernel 1", fontsize=20)
plt.show()

plt.figure(figsize=(10,10))
plt.imshow(str_or, cmap='gray', interpolation='nearest')
plt.title("Filter Bank Kernel 2", fontsize=20)
plt.show()    

# Combine these into one image using an HSV colorspace 
#(hue-saturation-value)
H = str_or.astype(float) / (len(ors)-1)
S = np.ones_like(H)
V = (str_resp-np.min(str_resp)) / np.max(str_resp)
hsv = np.dstack((H,S,V))
rgb = hsv_to_rgb(hsv)

# Render a hue circle as legend
s = 100
x,y = np.meshgrid(range(s),range(s))
rad = np.hypot(x-s/2,y-s/2)
ang = 0.5+(np.arctan2(y-s/2,x-s/2)/(2*np.pi))
mask = (rad<s/2)&(rad>s/4)

hsv_legend = np.dstack((ang, np.ones_like(ang, dtype='float'), mask.astype('float')))
rgb_legend = hsv_to_rgb(hsv_legend)
rgb[:s,:s,:] = rgb_legend[::-1,::]

plt.figure(figsize=(10,10))
plt.imshow(rgb, interpolation='nearest')
plt.title("Combined Kernels w/HSV", fontsize=20)
plt.show()
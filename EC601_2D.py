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

def do_ft(img):
        
    # Do the fft
    F = np.fft.fft2(img)
    
    # Center the spectrum on the lowest frequency
    F_centered = np.fft.fftshift(F)
    
    # Extract amplitude and phase
    A = np.abs(F_centered).real
    P = np.angle(F_centered).real
    
    # Return amplitude, phase, and the full spectrum
    return A, P, F

def do_inv_ft(A, P):
    #Reconstruct the image from amplitude and phase spectrum#
    F_centered = A * np.exp(1j*P)
    F = np.fft.ifftshift(F_centered)
    img = np.fft.ifft2(F).real
    return img

def render_gabor(pars):
    """Render a single Gabor patch"""
    vals = np.linspace(-pars['hsize'], pars['hsize'], (2*pars['hsize'])+1)
    xgr, ygr = np.meshgrid(vals, vals)
    gaussian = np.exp(-(xgr**2+ygr**2)/(2*pars['sigma']**2))
    slant = (ygr*(2*np.pi*pars['freq']*np.cos(pars['or'])) +
             xgr*(2*np.pi*pars['freq']*np.sin(pars['or']))) 
    grating = pars['amp']*np.cos(slant+pars['phase'])
    
    return gaussian*grating

def filter_image(flt,img):
    """Filter an image with one filter"""
    
    # Preparation: pad the arrays
    hflt, vflt = flt.shape
    himg, vimg = img.shape
    hres = hflt+himg
    vres = vflt+vimg
    hres2 = 2**int(np.log(hres)/np.log(2.0) + 1.0 )
    vres2 = 2**int(np.log(vres)/np.log(2.0) + 1.0 )
    img = img[::-1,::-1]
      
    # !!!THE FILTERING!!!
    fftimage = (np.fft.fft2(flt, s=(hres2, vres2))*
                np.fft.fft2(img, s=(hres2, vres2)))
    res = np.fft.ifft2(fftimage).real
    
    # Cut the actual filtered image from the result
    res = res[(hflt//2):hres-(hflt//2)-1,(vflt//2):vres-(vflt//2)-1][::-1, ::-1]
    
    # Return it
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

A, P, F = do_ft(img)

plt.imshow(np.log(A), cmap='gray')
plt.title("FFT Reconstruction from Amplitude", fontsize=20)
plt.show()
plt.imshow(P, cmap='gray')
plt.title("FFT Reconstruction from Phase", fontsize=20)

plt.show()



'''
Fourier analysis can be extended to the low-pass, 
high-pass or bandpass filtering of images
Artifacts familiar from JPEG compressed images, exaggerated here 
This is how the compression algorithm works: 
higher frequency components are removed.
'''

A, P, F = do_ft(img)
Af = A.copy()

# Compute one frequency at each pixel
fx = np.fft.fftshift(np.fft.fftfreq(A.shape[0]))
fy = np.fft.fftshift(np.fft.fftfreq(A.shape[1]))
fx,fy = np.meshgrid(fy,fx)
freq = np.hypot(fx,fy)
    
# Filter and reconstruct
bandpass = (freq>0) * (freq<0.05)
Af[~bandpass] = 0
f_img = do_inv_ft(Af, P)

# Show result
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

wA = np.ones(A.shape)
w_img = do_inv_ft(wA, P)

plt.figure(figsize=(10,10))
plt.imshow(w_img, cmap='gray'), plt.axis('off')
plt.title("'Whitened' Image", fontsize=20)
plt.show()

'''
Edge Filtering
'''
# Render the Gabor edge filter
gabpars = {}
gabpars['freq'] = 0.1
gabpars['sigma'] = 2.5
gabpars['amp'] = 1.
gabpars['phase'] = np.pi/2
gabpars['hsize'] = 15.
gabpars['or'] = 0.
gab = render_gabor(gabpars)

#Read Image
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

n = 50
ors = np.linspace(0,2*np.pi,n+1)[:-1]
res = np.zeros(img1.shape+(n,))

for idx,this_or in enumerate(ors):
    
    # Render the filter
    gabpars['or'] = this_or
    gab = render_gabor(gabpars)
    
    # Filter the image
    filtered = filter_image(gab,img1)
    
    # Save the result in a numpy array
    res[:,:,idx] = filtered
    
    

#Find out the maximal response strength
strongest_resp = np.amax(res, axis=2)

# Find out to which filter it corresponds
strongest_or = np.argmax(res, axis=2)

# Show each array
plt.figure(figsize=(10,10))
plt.imshow(strongest_resp, cmap='gray', interpolation='nearest')
plt.title("Filter Bank Kernel 1", fontsize=20)
plt.show()

plt.figure(figsize=(10,10))
plt.imshow(strongest_or, cmap='gray', interpolation='nearest')
plt.title("Filter Bank Kernel 2", fontsize=20)
plt.show()    

# Now let's combine these into one image
# Using an HSV colorspace (hue-saturation-value)
H = strongest_or.astype(float) / (len(ors)-1)
S = np.ones_like(H)
V = (strongest_resp-np.min(strongest_resp)) / np.max(strongest_resp)
HSV = np.dstack((H,S,V))
RGB = hsv_to_rgb(HSV)

# Render a hue circle as legend
sz = 100
x,y = np.meshgrid(range(sz),range(sz))
rad = np.hypot(x-sz/2,y-sz/2)
ang = 0.5+(np.arctan2(y-sz/2,x-sz/2)/(2*np.pi))
mask = (rad<sz/2)&(rad>sz/4)

hsv_legend = np.dstack((ang, np.ones_like(ang, dtype='float'), mask.astype('float')))
rgb_legend = hsv_to_rgb(hsv_legend)
RGB[:sz,:sz,:] = rgb_legend[::-1,::]

# Show result
plt.figure(figsize=(10,10))
plt.imshow(RGB, interpolation='nearest')
plt.title("Combined Kernels w/HSV", fontsize=20)
plt.show()
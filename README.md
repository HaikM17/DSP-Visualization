# DSP-Visualization

The following shows examples of ways that python can be utilized to visually display digital signal processing (dsp) in one dimension (sound) and two dimensions (images).

Code uploaded to repository can be run as a small demonstration of python's capabilities in dsp . 
The original images used are uploaded as well for reconstruction purposes

## One Dimension 

#### Background Info:
The **Fourier transform** is a mathematical function that decomposes a waveform, a function of time, into the frequencies that make it up. The result produced by the transform is a complex valued function of frequency. The **Inverse Fourier transform** is used to convert the signal back from the frequency domain to the time domain.
A **Fast Fourier transform** (FFT) is an algorithm that computes the discrete Fourier transform (DFT) of a sequence, or its inverse (IDFT). The FFT is significant in that it has made working in the frequency domain as computationally feasible as working in the time domain. 


![](Results/1-D%20signal.png)

   Figure shows a generated one dimensional signal
   
![](Results/FFT.png)

  Figure shows the FFT of the signal, displaying both the amplitude and phase against frequency


![](Results/inv_ft.png)

  Figure shows the inverse fourier transform of the signal


![](Results/complex_sig.png)

Figure shows a complex signal; more complex signals will show a more complex 
pattern of amplitudes and phasessignal

![](Results/inv_fft.png)

Inverse FFT of the signal.
This is similar to how MP3 compression works: components that are 
less relevant (e.g. low amplitude, or low sensitivity to its frequency) 
are removed to reduce the total amount of data that needs to be saved.



## Two Dimensions

#### Background Info:

Digital image processing, which is a subset of digital signal processing, uses computer algorithms to perform image processing on digital images.
Taking the fourier-transform of a 2D image provides a **magnitude** and a **phase** distribution. The magnitude values details how much a spacial frequency contributes to the image. The phase shows the shifts of the periodic pattern along the image. Hence, the phase is relevant for how the periodic patterns of all wave functions add up, and the phase information carries the majority of the spatial information of an image. The phase contains more important information concerning an image.

JPEG **compression** reduces the image quality and effectively loses image information. JPEG image compression is not suitable for images with sharp edges and lines. A **compression artifact** is a noticeable distortion of media caused by the application of lossy compression. Lossy data compression involves discarding some of the media's data so that it becomes small enough to be stored within the desired disk space or transmitted within the available bandwidth (known as the data rate or bit rate). If the compressor cannot store enough data in the compressed version, the result is a loss of quality, or introduction of artifacts. 

**Whitening** involves transforming the image in a way that amplitude and frequency are decorrelated. A uniform amplitude spectrum is applied while retaining the phase spectrum. It is called whitening in reference to white noise. This is an important step in most pre-processing algorithms.

**Edge detection** includes a variety of mathematical methods that aim at identifying points in a digital image at which the image brightness changes sharply or has discontinuities. The points at which image brightness changes sharply are typically organized into a set of curved line segments termed edges. Edge detection is a fundamental tool in image processing, machine vision and computer vision, particularly in the areas of feature detection and feature extraction.

A **Gabor filter** is a linear filter used for texture analysis; it analyzes if there are any specific frequency content in the image in specific directions in a localized region around the point or region of analysis. Frequency and orientation representations of Gabor filters are claimed to be similar to the human visual system. In the spatial domain, a 2D Gabor filter is a Gaussian kernel function modulated by a sinusoidal plane wave.



![](Images/lake.jpg)
Original Image

![](Results/lake_lp_filter.png)

Image passed through a filter (and grey scaled)

![](Results/lake_amp.png)

Image reconstructed using only extracted amplitudes of image

![](Results/lake_phase.png)

Image reconstructed using only extracted phases of image

![](Results/lake_compress.png)

Image after fourier analysis. Present are artifacts that are common with jpeg compressed images. This shows how in general compression algorithms work: **higher frequency components are removed.**

![](Results/lake_whitened.png)

The image shows how high frequencies dominate the perception, despite all frequencies are equally strong.

The same phenomenon is present in white noise. Such noise has a uniform amplitude spectrum, but perceptually the lower frequencies are not strongly detected.


![](Images/cat.jpg)

Original Image

![](Results/snow_edge.png)

A Gabor filter is applied to display edge effects.


### Filterbank filter
![](Results/snow_fb1.png)

Extracted kernel from filterbank that displays the maximal response strength from the varying orientations

![](Results/snow_fb2.png)

Extracted kernel from filterbank that is used to find out which filter it corresponds to

![](Results/snow_combo_fb.png)

Combined image using a hue saturation value 







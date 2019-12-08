# DSP-Visualization

The following shows examples of ways that python can be utilized to visually display digital signal processing in one dimension (sound)
and two dimensions (images).


## One Dimension 

![](Results/1-D%20signal.png)

   Figure shows a generated one dimensional signal
   
.   
.   
.

![](Results/FFT.png)

  Figure shows the FFT of the signal, displaying both the amplitude and phase against frequency

.
.
.

![](Results/inv_ft.png)

  **Figure shows the inverse fourier transform of the signal**
  .
  .
  .
  

![](Results/complex_sig.png)

 Figure shows a complex signal; more complex signals will show a more complex 
pattern of amplitudes and phasessignal

![](Results/inv_fft.png)

Inverse FFT of the signal.
This is how MP3 compression works: components that are 
less relevant (e.g. low amplitude, or low sensitivity to its frequency) 
are removed from the Fourier transform to reduce the total amount of 
data that needs to be saved.
.
.
.
## Two Dimensions

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

Amplitude and frequency can be decorrelated by a process called **whitening**. A uniform amplitude spectrum is applied while retaining the phase spectrum. The image shows how high frequencies dominate the perception, despite all frequencies are equally strong.

The same phenomenon is present in white noise. Such noise has a uniform amplitude spectrum, but perceptually the lower frequencies are not strongly detected.


![](Images/cat.jpg)

Original Image

![](Results/snow_edge.png)

A Gabor filter is applied to display edge effects.
A Gabor filter is a linear filter used for texture analysis, which means that it basically analyzes whether there are any specific frequency content in the image in specific directions in a localized region around the point or region of analysis.


### Filterbank filter
![](Results/snow_fb1.png)

Extracted kernel from filterbank that displays the maximal response strength from the varying orientations

![](Results/snow_fb2.png)

Extracted kernel from filterbank that is used to find out which filter it corresponds to

![](Results/snow_combo_fb.png)

Combined image using a hue saturation value 







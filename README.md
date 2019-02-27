# wavelet_convolution
Implement wavelet transforms using convolution. The codes are easy to understand, especially for beginners who are insterested in wavelet transforms and also JPEG2000.

Since covolution is used, the codes should be easily transcripted to PyTorch and run on GPU. Maybe I will submit a GPU version in the future.

For now, I only implement Cohen–Daubechies–Feauveau 9-7 (CDF9-7) wavelet, which is applied in JPEG2000 for lossy compression. In addition, I plan to implement CDF5-3, also called the LeGall 5/3 wavelet, which is used for lossless compression in JPEG2000.

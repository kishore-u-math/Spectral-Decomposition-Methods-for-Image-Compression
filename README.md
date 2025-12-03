# Spectral-Decomposition-Methods-for-Image-Compression

To address the critical storage demands of the digital era, this thesis investigates
Singular Value Decomposition (SVD) as a powerful linear algebraic tool for image compression. 
Bridging theoretical linear algebra with practical computation, the study rigorously benchmarks 
the Power Method, QR Algorithm, and Krylov subspace methods. %(Arnoldi and Lanczos). 
A MATLAB custom-based SVD Algorithm is developed to perform dimensionality reduction on both 
grayscale and full-color images. Comprehensive error analysis validates that isolating dominant 
top \(k\) singular values achieves significant data compression with minimal visual loss. 
For any given image, optimal compression is achieved by identifying the minimum \(k\) value 
at which Peak Signal-to-Noise Ratio (PSNR) reaches \(\geq 40\) dB, balancing maximum compression 
with excellent visual quality. The study effectively demonstrates the power of spectral methods 
in real-world data science applications.

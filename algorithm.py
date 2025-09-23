
## the plan
import numpy as np
import cv2
from scipy.fft import fft
from scipy.fft import ifft
from scipy.ndimage import gaussian_filter
from skimage import measure
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.fft import fftshift, ifftshift

def EdgeDetection(data):
    blur = cv2.GaussianBlur(data, (9, 9), 2)
    return cv2.Canny(data.astype(np.uint8), 100, 200)



import numpy as np
from scipy.fft import fft
from scipy.interpolate import interp1d
import cv2

"""def findcoeffs(image, n_coeffs=50):

    # 1. Edge detection
    filtered_image = EdgeDetection(image)

    # 2. Find contours
    contours, _ = cv2.findContours(filtered_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return np.zeros(n_coeffs, dtype=complex)

    # 3. Sort contours (optional: left-to-right)
    contours_sorted = sorted(contours, key=lambda c: c[0][0][0])

    # 4. Merge all contours into one sequence
    merged_contour = []
    for cont in contours_sorted:
        merged_contour.extend(cont)

    # 5. Convert points to complex numbers
    z = np.array([complex(pt[0][0], height - pt[0][1]) for pt in merged_contour])

    # 6. Resample along arc length for even spacing
    arc_len = np.cumsum(np.abs(np.diff(z, prepend=z[0])))
    arc_len /= arc_len[-1]  # normalize 0..1

    N_samples = max(len(z), n_coeffs * 2)  # enough samples for FFT stability
    t_new = np.linspace(0, 1, N_samples, endpoint=False)
    interp_real = interp1d(arc_len, z.real, kind='cubic')
    interp_imag = interp1d(arc_len, z.imag, kind='cubic')
    z_resampled = interp_real(t_new) + 1j*interp_imag(t_new)

    # 7. Center contour (DC = 0)
    z_centered = z_resampled - np.mean(z_resampled)

    # 8. Compute FFT and normalize
    C = (fft(z_centered) / len(z_centered))

    C_shifted = fftshift(C)
    mid = len(C_shifted)//2
    k = n_coeffs//2
    C_trunc = C_shifted[mid-k : mid+k+1]  # symmetric around DC
    C_trunc = ifftshift(C_trunc)

    I_C = ifft(C_trunc)

    plt.plot(np.real(I_C), np.imag(I_C))
    plt.show()
    # 9. Keep first n_coeffs coefficients (DC first)
    coeffs = C_trunc

    return coeffs"""

def findcoeffs(image, n_coeffs=50):
    """
    Compute Fourier coefficients from an image, merging all contours.

    Parameters:
    - image: 2D binary/edge image
    - n_coeffs: number of Fourier coefficients to keep (including DC)
    
    Returns:
    - coeffs: complex array of Fourier coefficients (length n_coeffs)
    """
    height, width = image.shape

    # 1. Edge detection
    filtered_image = EdgeDetection(image)

    # 2. Find contours
    contours, _ = cv2.findContours(filtered_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return np.zeros(n_coeffs, dtype=complex)

    # 3. Sort contours (optional: left-to-right)
    contours_sorted = sorted(contours, key=lambda c: c[0][0][0])

    # 4. Merge all contours into one sequence
    merged_contour = []
    for cont in contours_sorted:
        merged_contour.extend(cont)

    # 5. Convert points to complex numbers
    z = np.array([complex(pt[0][0], height - pt[0][1]) for pt in merged_contour])

    # 7. Center contour (DC = 0)
    z_centered = z - np.mean(z)

    # 8. Compute FFT and normalize
    C = (fft(z_centered) / len(z_centered))

    C_shifted = fftshift(C)
    mid = len(C_shifted)//2
    k = n_coeffs//2
    C_trunc = C_shifted[mid-k : mid+k+1]  # symmetric around DC
    C_trunc = ifftshift(C_trunc)

    I_C = ifft(C)

    plt.plot(np.real(z), np.imag(z))
    plt.show()
    # 9. Keep first n_coeffs coefficients (DC first)
    coeffs = C_trunc

    return coeffs






    





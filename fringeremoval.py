import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy.polynomial import Polynomial
import pywt


def defringeflat():

    pass


def compute_enhanced_rows(img: np.ndarray) -> np.ndarray:

    # Replace each pixel with the median of the n adjacent row pixels
    n = 30
    for row in range(img.shape[0]):

        lower_idx = np.max((0, row - n //2))
        upper_idx = np.min((img.shape[0] - 1, row + n // 2))

        adjacent_rows = img[lower_idx:upper_idx, :]
        img[row, :] = np.median(adjacent_rows, axis=0)
    
    return img

def subtract_rows_polyfit(img: np.ndarray) -> np.ndarray:

    n = 4
    row_x = np.arange(img.shape[1])
    img = np.float64(img)

    for row in range(img.shape[0]):
        p = Polynomial.fit(row_x, img[row, :], n)
        fit = p.convert()(row_x)
        img[row, :] -= fit

    return img

def wavelet_transform(img: np.ndarray, rows) -> np.ndarray:

    img = np.float64(img)
    # row = img[1000, :]
    x = np.arange(0, img.shape[1])

    wavelet = "cmor3.0-1.0" # bandwidth - freq
    widths = np.geomspace(1, 512, num=100)
    sampling_period = np.diff(x).mean()
    data = []
    row_freqs = []
    for row in rows:
    # widths = np.arange(1, 30)

        cwtmatr, row_freqs = pywt.dwt(img[row, :], wavelet) # sampling_period=1)
        data.append(cwtmatr)
    # print(cwtmatr)
    return data, row_freqs

def inverse_wavelet_transform(coeffs, widths):

    widths = np.geomspace(1, 512, num=100)
    img = pywt.icwt(coeffs, widths, "cmor3.0-1.0")
    return img

if __name__ == "__main__":

    img = cv.imread(r"C:\Users\Weld Lab\Desktop\DMD\DMDProject\Feed forward\test_images\solid_t.png", cv.IMREAD_GRAYSCALE)
    img2 = compute_enhanced_rows(img)
    img3 = subtract_rows_polyfit(img2)

    rows = [800, 1200, 1500]
    rows = [1200]
    data, freqs = wavelet_transform(img3, rows)
    # print(wavelet_transform(img3))

    coeffs = data[0][:-1,:-1]
    coeffs[np.abs(coeffs) > 17] = 10
    amplitudes = np.abs(coeffs)

    imgr = inverse_wavelet_transform(coeffs, 0)

    fig, ax = plt.subplots(1,2, figsize=(8, 4), sharey=True)
    pcm = ax[0].pcolormesh(np.arange(0, img3.shape[1]), freqs, amplitudes)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("col x (px)")
    ax[0].set_ylabel("Frequency (Hz)")
    ax[0].set_title(f"Row {rows[0]}")
    ax[1].imshow(np.real(imgr))
    fig.colorbar(pcm, ax=ax[0], orientation="horizontal", location="bottom", label="amplitude")
    # ax[0].imshow(img2, cmap="gray")
    # ax[0].set_title("Median replacement")
    # ax[1].imshow(img3, cmap="gray")
    fig.suptitle("Morlet wavelet transform")
    # ax[1].set_title("Polyfit subtracted")
    
    plt.show()
    # cv.imshow("test",img3)
    # cv.moveWindow("test",0, 0)
    # cv.namedWindow("test", cv.WND_PROP_FULLSCREEN)
    # cv.waitKey(0)
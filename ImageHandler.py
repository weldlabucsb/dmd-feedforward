import cv2 as cv
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class ImageHandler:

    def __init__(self, xdim, ydim):

        self.xdim = xdim
        self.ydim = ydim

    @staticmethod
    def dither(img: np.ndarray):
        """
        Returns the Floyd-Steinberg binary unit8 image conversion of img
        img: (np.ndarray) The image to be dithered
        """
        img = np.uint8(img / np.max(img) * 255)
        img = Image.fromarray(img, mode="L")
        dithered = np.asarray(img.convert(mode="1", dither=1), np.uint8)
        dithered = cv.normalize(dithered, None, 0, 255, cv.NORM_MINMAX)
        
        return dithered


    def linear_gradient(self, angle: float, xmin: int=None, xmax: int=None, ymin: int=None, 
                        ymax: int=None, bg_invert: bool=False, no_dither: bool=False):
        """
        angle: angle to rotate linear gradient at (degrees)
        xmin, xmax, ymin, ymax: bounding box coordinates to draw the gradient in. Defaults to
            full area resolution (xdim, ydim)
        returns:
            Dithered np.array, dtype=np.uint8, shape=(xdim, ydim) of linear gradient at angle
        """
        # Set default params
        xmin = 0 if xmin is None else xmin
        xmax = self.xdim if xmax is None else xmax
        ymin = 0 if ymin is None else ymin
        ymax = self.ydim if ymax is None else ymax

        img = np.zeros((self.ydim, self.xdim), dtype=np.uint8)
        # bg_invert inverts the background to solid field (on) instead of off
        if bg_invert:
            img += 1

        dx = np.cos(angle * np.pi / 180)
        dy = np.sin(angle * np.pi / 180)

        x = np.linspace(xmin, xmax-1, xmax - xmin)
        y = np.linspace(ymin, ymax-1, ymax - ymin)
        xx, yy = np.meshgrid(x, y)

        gradient = dx * xx + dy * yy
        gradient_normalized = cv.normalize(gradient, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        gradient_normalized = np.uint8(gradient_normalized)
        gradient_img = Image.fromarray(gradient_normalized, mode="L")

        # Dither using Floyd-Steinberg algorithm
        converted = np.asarray(gradient_img.convert(mode="1", dither=1), np.uint8)
        print(np.max(converted))
        print(np.min(converted))

        if no_dither:
            converted = np.asarray(gradient_img.convert(mode="L"), np.uint8)

        img[ymin:ymax, xmin:xmax] = converted
        img = cv.normalize(img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

        return img
    
    def solid_field(self, invert=False, xmin: int=None, xmax: int=None, ymin: int=None, 
                        ymax: int=None):
        """
        invert: sets all 1 pixels to 0. Default false
        xmin, xmax, ymin, ymax: bounding box coordinates to draw the gradient in. Defaults to
            full area resolution (xdim, ydim)
        
        returns:
            np.array, dtype=np.uint8, shape=(xdim, ydim)
        """
        # Set default params
        xmin = 0 if xmin is None else xmin
        xmax = self.xdim if xmax is None else xmax
        ymin = 0 if ymin is None else ymin
        ymax = self.ydim if ymax is None else ymax

        img = np.zeros((self.ydim, self.xdim), dtype=np.uint8)
        solid = np.ones((ymax-ymin, xmax-xmin), dtype=np.uint8) * 255

        if invert:
            img += 255
            solid -= 255
        
        img[ymin:ymax, xmin:xmax] = solid
        return img
        # return np.zeros((1600, 2560), dtype=np.uint8)

    def invert_image(self, img: np.ndarray):
        """
        img: the image to be binary inverted (255 <-> 0)
        returns: inverted image as np.ndarray
        """
        return np.abs(img - np.ones((self.ydim, self.xdim), dtype=np.uint8)*255)
    
    def border(self, width: int, invert=False):

        img = np.zeros((self.ydim, self.xdim), dtype=np.uint8)

        img[:width, :] = 255  # Top border
        img[-width:, :] = 255  # Bottom border
        img[:, :width] = 255  # Left border
        img[:, -width:] = 255  # Right border

        if invert:
            img = 255 - img
        return img

if __name__ == "__main__":
    gen = ImageHandler(2560, 1600)
    field = gen.linear_gradient(0, no_dither=True)
    # field = gen.border(10)
    # inverted = gen.invert_image(field)
    plt.imshow(field)
    plt.show()
    # cv.imshow("test",field)
    # cv.waitKey(0)
    # plt.imshow(field)
    # plt.colorbar()
    # plt.show()
"""
Implements Feedforward class for DMD operation and optimization

author: Daniel Harrington
8/2024
"""


import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from DMDController import DMDController
from FemtoEasyBP import FemtoEasyBP
from ImageHandler import ImageHandler
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
import time
from scipy.optimize import curve_fit
from scipy.ndimage import convolve
mpl.rcParams.update({"mathtext.fontset": "cm", "font.family": "serif", "figure.constrained_layout.use": True})


class Feedforward:

    def __init__(self, cam_ip: str, cam_port: int=8000, invert: bool=False):
        
        # self.dmd = DMDController(isImageLock=False)
        self.dmd = DMDController()
        self.dmd_xdim, self.dmd_ydim = self.dmd.get_size()
        self.img_gen = ImageHandler(self.dmd_xdim, self.dmd_ydim)
        #
        self.camera = FemtoEasyBP(cam_ip, cam_port)
        self.result = None
        self.invert = invert

        self.M = np.eye(3)
        self.flat_field = np.ones((self.dmd.ydim, self.dmd.xdim))

    @staticmethod
    def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta):
        x, y = xy
        xo = float(xo)
        yo = float(yo)    
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                                + c*((y-yo)**2)))
        return g.ravel()
    
    
    @staticmethod
    def percentile_normalize(array: np.ndarray, p: float) -> np.ndarray:
    
        pval = np.percentile(array, p)
        array = array / pval
        array[array > 1] = 1
        return array
    
    
    def get_flat_field(self, p_opt: list=None) -> np.ndarray:

        self.dmd.update_array(self.img_gen.solid_field(invert=self.invert))
        solid = self.camera.get_image()

        # Transform and normalze to [0, 1]
        solid_t = cv.normalize(np.float32(self.transform(solid)), None, norm_type=cv.NORM_MINMAX)
        solid_t = self.percentile_normalize(solid_t, 98)

        # initial_guess = (0.8, 1500, 800, 1000, 1000, 135,)
        initial_guess = [7.85867699e-01, 1.68624522e+03, 7.32967556e+02, 7.11864416e+02,
            5.80838498e+02, 1.34337083e+02]
        
        x = np.arange(self.dmd.xdim)
        y = np.arange(self.dmd.ydim)
        a_fit = (self.dmd.xdim - self.dmd.ydim) // 2
        b_fit = self.dmd.xdim - a_fit

        # Crop to a square for fitting
        ax = 800
        bx = 2400
        ay = 0
        by = 1600
        solid_cropped = solid_t[ay:by, :][:, ax:bx]

        N = 10
        kernel = np.ones((N, N)) / N**2
        solid_cropped_smoothed = convolve(solid_cropped, kernel)

        xf = x[(x >= ax) & (x < bx)]
        yf = y[(y >= ay) & (y < by)]
        xx, yy = np.meshgrid(xf, yf)

        if p_opt is None:
            print("Optimizing fit...")
            p_opt, p_cov = curve_fit(self.gaussian_2d, (xx, yy), solid_cropped_smoothed.ravel(), p0=initial_guess)
            print("Fit found.")

        print(p_opt)
        amplitude_fit, xo_fit, yo_fit, sigma_x_fit, sigma_y_fit, theta_fit = p_opt
        
        xxbig, yybig = np.meshgrid(x, x)
        yo_translated = yo_fit + a_fit  # Account for the difference in size
        fit = self.gaussian_2d((xxbig, yybig), amplitude_fit, xo_fit, yo_translated, sigma_x_fit, sigma_y_fit, theta_fit)
        fit = np.float32(fit).reshape(self.dmd.xdim, self.dmd.xdim)

        # flat_field = solid_t - fit[a_fit:b_fit, :] * alpha
        flat_field = np.array(solid_t)
        flat_field[fit[a_fit:b_fit, :] > 0.01] /= fit[a_fit:b_fit, :][fit[a_fit:b_fit, :] > 0.01]
        flat_field[flat_field <= 0.01] = 1

        flat_field = np.float32(cv.normalize(flat_field, None, 0, 255, cv.NORM_MINMAX))
        # flat_field[flat_field < np.min(flat_field) + 10] = np.median(flat_field)
        
        self.flat_field = flat_field
        return solid_t, fit[a_fit:b_fit], self.flat_field



    def show_test_image(self, target):

        # target = self.img_gen.linear_gradient(0, xmin=200, bg_invert=True)
        # target = self.img_gen.solid_field(xmin=200, invert=self.invert)
        # target = cv.imread(r"C:\Users\Weld Lab\PycharmProjects\DMDProject\oldfiles\RescaledImages\amongus_adjusted.bmp", cv.IMREAD_UNCHANGED)
        self.dmd.update_array(target)
        time.sleep(3)
        img = self.camera.get_image()
        # plt.imshow(img)
        self.result = img
        # plt.imshow(target)
        # plt.colorbar()
        # plt.show()
        self.dmd.close()
        return img
    

    def transform(self, img: np.ndarray):

        transformed = cv.warpAffine(img, self.M, (self.dmd_xdim, self.dmd_ydim))
        return np.float32(transformed)


    def set_transformation(self, dmd_pts: np.ndarray, show_pts=False):
        """
        dmd_pts: array-like of shape (3,)  
            Three points defining the top left, top right, and bottom right of the DMD image in camera coordinates
        defines the transformation matrix between camera and target images
        """

        if show_pts:
            for pt in dmd_pts:
                cv.circle(self.result, np.uint16(pt), 30, (100), -1)

        # top left, top right, bottom right
        target_img_pts = np.float32([(0, 0), (self.dmd.xdim-1, 0), (self.dmd.xdim-1, self.dmd.ydim-1)]) 
        self.M = cv.getAffineTransform(dmd_pts, target_img_pts)

        return self.M

    def compute_rms(self, measured: np.ndarray, target: np.ndarray) -> float:
        
        t = 0.1
        integrand = np.float32(measured[target > t]) / np.float32(target[target > t]) - 1
        integrand = integrand.flatten()**2
        rms = np.sqrt(np.sum(integrand))
        return rms


    def compute_error(self, measured_image: np.ndarray, target_image: np.array, alpha: float=1):
        
        # measured_image /= np.mean(measured_image)
        # target_image /= np.mean(target_image)
        # target_image *= 2
        error = alpha * (measured_image - target_image) / np.max(measured_image)
        return error
    
    def create_update_map(self, error: np.ndarray, target: np.ndarray, eta=0.2) -> np.ndarray:

        update = target - eta * error

        # update = self.percentile_normalize(update, 95)
        # update = cv.normalize(update, None, 0, 1, cv.NORM_MINMAX)
        update[update < 0] = 0
        update[update > 1] = 1
        # update[:, -500:] = 0
        print(f"max update: {np.max(update)}")
        print(f"min update: {np.min(update)}")
        return update

    def run(self, target: np.ndarray, solid: np.ndarray, n_iter: int=1, t_slope: float=1) -> float:
        """
        target: (np.ndarray) Target image
        n_iter: (int) Number of iterations to loop through
        Runs n_iter number of update steps
        """
        delay = 2
        

        # Make target match image orientation
        target_continuous = self.img_gen.linear_gradient(180, bg_invert=False, no_dither=True)
        target_continuous = np.float32(target_continuous[:, ::-1])
        target_continuous = t_slope * target_continuous / np.max(target_continuous)

        # Initialize target
        target_0_cts = np.array(target_continuous).astype(np.float32)
        solid /= np.max(solid)
        target_0_cts[solid > 0.001] /= solid[solid > 0.001]
        # target_0_cts -= 50
        # target_0_cts[target_0_cts < 0] = 0
        target_0_cts = np.sqrt(target_0_cts)
        target_0_cts = ff.percentile_normalize(target_0_cts, 99)

        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(target_continuous)
        ax[1].imshow(solid)
        ax[2].imshow(target_0_cts)
        plt.show()

        # Make target match DMD orientation and dither
        target_0_dithered = (1 - target_0_cts)[:, ::-1]
        target_0_dithered = self.img_gen.dither(target_0_dithered)


        errs = []
        target_i_dithered = target_0_dithered
        target_i_cts = target_0_cts

        captures = []
        for i in range(n_iter):

            self.dmd.update_array(target_i_dithered)
            time.sleep(delay)
            measured = self.transform(self.camera.get_image())
            measured = self.percentile_normalize(measured, 96) 
            # measured = cv.normalize(measured, None, 0, 1, cv.NORM_MINMAX)
            captures.append(measured)

            err = self.compute_error(measured, target_continuous/t_slope)
            errs.append(err)
            target_i_cts = self.create_update_map(err, target_i_cts)

            target_i_dithered = (1 - target_i_cts)[:, ::-1]
            target_i_dithered = self.img_gen.dither(target_i_dithered)



        return target_0_cts, 255 - target_i_dithered[:, ::-1], captures, errs

if __name__ == "__main__":


    ff = Feedforward("128.111.8.162", 8000, invert=True)

    # img = ff.show_test_image(ff.img_gen.linear_gradient(0))
    

    tl = (305, 315)
    tr = (2955, 111)
    # bl = (352, 1400)
    br = (3085, 1785)
    ff.set_transformation(np.float32([tl, tr, br]), show_pts=False)
    # t = ff.transform(img)

    # p = [7.95607253e-01, 1.70821661e+03, 7.10791491e+02, 7.10309487e+02,
    #     5.81035417e+02, 1.34338774e+0]
    _, solid, flat = ff.get_flat_field()

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(_)
    ax[1].imshow(solid)
    ax[2].imshow(flat)
    plt.show()
   
    
    # flat = ff.percentile_normalize(flat, 98)
    # # solid = ff.transform(ff.show_test_image(ff.img_gen.solid_field(xmin=500, invert=True)))
    # # solid = ff.percentile_normalize(solid, 95)
    # t_slope = 1
    # target = np.float32(ff.img_gen.linear_gradient(0, bg_invert=True, no_dither=True))
    # target = t_slope * target / np.max(target)


    # target_0, target_i, caps, errs = ff.run([], solid, 1, t_slope=t_slope)

    # fig, ax = plt.subplots(2, 4, figsize=(10, 5), constrained_layout=True)

    # avg = np.sum(caps[0], axis=0)

    # filtered = np.array(caps[0]) 
    # filtered[flat > 0.1] /= flat[flat > 0.1]
    # filtered = ff.percentile_normalize(filtered, 98)

    # cax00 = ax[0].imshow(caps[0])
    # cax01 = ax[1].imshow(errs[0], cmap="RdBu", vmin=-1, vmax=1)
    # ax[2].plot(avg / np.max(avg))
    # # # ax[1].plot(caps[0][800, :])
    # # ax[2].plot(target[800, :] * 2)
    # cv.imwrite("optimized_grad_dithered_0.5_iter=8.png", target_i)

    # rms_0 = ff.compute_rms(ff.percentile_normalize(caps[0], 99) * t_slope, target)
    # rms_3 = ff.compute_rms(ff.percentile_normalize(caps[-1], 99) * t_slope, target)
    # print(f"RMS_0 = {rms_0:.2f}. RMS_7 = {rms_3:.2f}")

    # cax00 = ax[0, 0].imshow(target_0)
    # ax[0, 0].set_title("$\mathrm{DMD}_0(x,y)$")
    # cax01 = ax[0, 1].imshow(target_i)
    # ax[0, 1].set_title("$\mathrm{DMD}_7(x,y) \; \eta = 0.1$")
    # cax02 = ax[0, 2].imshow(caps[0])
    # ax[0, 2].set_title("$T_0 (x, y)$")
    # cax03 = ax[0, 3].imshow(caps[-1])
    # ax[0, 3].set_title("$T_7 (x, y)$")
    # # cax04 = ax[0, 4].imshow(solid_fit)
    # cax10 = ax[1, 0].imshow(errs[0], cmap="RdBu")
    # ax[1, 0].set_title("$e_0(x, y)$")
    # cax11 = ax[1, 1].imshow(errs[-1], cmap="RdBu")
    # ax[1, 1].set_title("$e_7(x, y)$")

    # ax[1, 2].imshow(t)

    # ax[1, 2].plot(np.sum(caps[0], axis=0) / 1600)
    # ax[1, 2].plot(target[800, :]/t_slope)
    # ax[1, 2].set_title("$\int \; \mathrm{DMD}_0(x, y) \, dy$")
    # ax[1, 3].plot(np.sum(caps[-1], axis=0) / 1600)
    # ax[1, 3].plot(target[800, :]/t_slope)
    # ax[1, 3].set_title("$\int \; \mathrm{DMD}_7(x, y) \, dy$")
    # # ax[1, 4].plot(solid_fit[800, :])
    # # cax3 = ax[3].imshow((fixed - new_target) / np.max(fixed), cmap="RdBu")

    # # print(np.min(emap))
    # # print(np.max(emap))

    # fig.colorbar(cax00, ax=ax[0, 0], orientation="vertical", location="left")
    # fig.colorbar(cax01, ax=ax[0, 1], orientation="horizontal", location="bottom")
    # fig.colorbar(cax02, ax=ax[0, 2], orientation="horizontal", location="bottom")
    # fig.colorbar(cax03, ax=ax[0, 3], orientation="horizontal", location="bottom")
    # # fig.colorbar(cax04, ax=ax[0, 4], orientation="horizontal", location="bottom")
    # fig.colorbar(cax10, ax=ax[1, 0], orientation="vertical", location="right")
    # fig.colorbar(cax11, ax=ax[1, 1], orientation="horizontal", location="bottom")

    # plt.show()

    

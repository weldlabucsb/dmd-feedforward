"""
FemtoEasyBP.py

07/11/24 - Daniel Harrington

Class for interfacing with the Femto Easy beam profiler over HTTP. Beam profiler 
must be plugged in and software running for the server to be open. Image files 
are saved in .tiff format or returned as numpy arrays
"""


import requests as rq
import numpy as np
import cv2 as cv
import time
import matplotlib.pyplot as plt


class FemtoEasyBP:

    def __init__(self, host_ip: str, port: int=8000):
        self.url = f"http://{host_ip}:{port}"
        self.camera_ready = False

        print("Connecting...")
        data = self.make_get_request("/about").json()
        print(f"Successfully connected. Server version {data['version']}.")

        data = self.make_get_request("/status").json()
        
        if data["cameraInitialized"]:
            print("Camera online.")
        else:
            raise Exception("Camera offline.")
        
        params = self.get_params()
        self.min_exp = params["exposureTime"]["min_value"]
        self.max_exp = params["exposureTime"]["max_value"]
        self.min_gain = params["gain"]["min_value"]
        self.max_gain = params["gain"]["max_value"]
        
        self.min_roi_width = params["roiWidth"]["min_value"]
        self.max_roi_width = params["roiWidth"]["max_value"]
        self.min_roi_height = params["roiHeight"]["min_value"]
        self.max_roi_height = params["roiHeight"]["max_value"]

        self.start_acquisition()

        
    def path(self, path: str):
        """
        path: str, e.g. "/path/after/url"
        returns:
            url with path added
        """
        return self.url + path
    
    def make_get_request(self, path: str):
        """
        path: e.g. "/path/after/url"
        returns: 
            request json response. raises errors
        """
        try:
            res = rq.get(self.path(path))
            res.raise_for_status()
            return res
        except rq.exceptions.HTTPError as err:
            print(f"HTTP error when making request to {path}:")
            raise(err)
        except Exception as err:
            print(f"Error when making request to {path}:")
            raise(err)
        
    def make_put_request(self, path: str, data: dict={}):
        """
        path: e.g. "/path/after/url"
        data: put request json
        returns: 
            request json response. raises errors
        """
        try:
            res = rq.put(self.path(path), json=data)
            res.raise_for_status()
            return res
        except rq.exceptions.HTTPError as err:
            print(f"HTTP error when making request to {path}:")
            raise(err)
        except Exception as err:
            print(f"Error when making request to {path}:")
            raise(err)
    
    def set_params(self, params: dict):
        """
        params: parameter dict of settings to send to camera
        sets parameters
        """
        res = self.make_put_request("/camera/parameters", params)
        print(res.text)
    
    def get_params(self):
        """
        returns current camera parameters as JSON
        """
        return self.make_get_request("/camera/parameters").json()

    def set_exposure(self, exp_time: float):
        """
        exp_time: exposure time (microseconds)
        sets camera exposure time
        """
        if (exp_time >= self.min_exp) and (exp_time < self.max_exp):
            print("Setting Exposure...")
            self.set_params({"exposureTime": exp_time})
            print("Exposure set.")
        else:
            print(f"Exposure time {exp_time} not in valid range.")

    def get_exposure(self):
        """
        returns current camera exposure
        """
        return self.get_params()["exposureTime"]["value"]
    

    def get_gain(self):
        """
        returns current camera gain
        """
        return self.get_params()["gain"]["value"]

    def set_gain(self, gain: float):
        """
        gain: gain for image
        sets camera gain
        """
        if (gain >= self.min_gain) and (gain <= self.max_gain):
            print("Setting gain...")
            self.set_params({"gain": gain})
            print("Gain set.")
        else:
            print(f"Gain {gain} not in valid range.")


    def get_roi(self):
        """
        returns current camera ROI as (width, height)
        """
        return (self.get_params()["roiWidth"]["value"], self.get_params()["roiHeight"]["value"])

    def set_roi(self, width: int, height: int):
        """
        width, height: dimensions of ROI
        sets camera region of interest (crops image to given area)
        """
        if (width >= self.min_roi_width) and (width <= self.max_roi_width):
            if (height >= self.min_roi_height) and (height <= self.max_roi_height):
                print("Setting ROI...")
                self.set_params({"roiWidth": width, "roiHeight": height })
                print("ROI set.")
            else:
                print(f"ROI height {height} not in valid range.")
        else:
            print(f"ROI width {width} not in valid range.")


    def get_image(self):
        """
        returns: 8bit single channel np.ndarray of image
        """
        print(f"Getting image...")
        self.make_put_request("/runner/start")
        time.sleep(1)
        fe_result = self.make_get_request("/runner/lastResult/sourceImage")

        img_bytes = np.frombuffer(fe_result.content, np.uint8)
        img = cv.imdecode(img_bytes, cv.IMREAD_GRAYSCALE)
        print("Image received.")
        return img

    def start_acquisition(self):
        """
        stops the camera recording
        """
        self.make_put_request("/runner/start")
        time.sleep(3)

   
    def save_image(self, filepath: str):
        """
        filepath: path and file name for image (must include extension .tiff)
        writes image file in .tiff format and returns path
        """
        
        print(f"Getting image file...")
        self.make_put_request("/runner/start")
        fe_result = self.make_get_request("/runner/lastResult/sourceImage")
        self.make_put_request("/runner/stop")

        with open(filepath, "wb") as file:
            content = fe_result.content
            file.write(content)

        print(f"Image written to {filepath}.")
        return f"{filepath}"



if __name__ == "__main__":
    fe = FemtoEasyBP("localhost", 8000)
    # print(fe.get_roi())
    plt.imshow(fe.get_image())
    plt.show()
import numpy as np
import cv2 as cv
import time
from screeninfo import get_monitors
import threading, faulthandler

class DMDController:

    def __init__(self, monitor_idx: int=1, ):
        """
        Args:
            monitor_idx: The index of the monitor that the DMD acts as (1 = second monitor, 2 = third, etc.)
        """

        self.xdim = get_monitors()[monitor_idx].width
        self.ydim = get_monitors()[monitor_idx].height
        self.window_name = "array"

        # Get the main display width, which is how much to move the DMD image by
        # to display it on the second monitor (the DMD)
        self.monitor_offset = get_monitors()[0].width

    def get_size(self):
        """
        Returns the size in px of the DMD array
        """
        
        return self.xdim, self.ydim

    def update_array(self, image_array: np.ndarray, delay: int=1000) -> None:
        """
        Displays the image on the DMD array. Scales image to fit DMD if image does not
            match DMD size
        Args:
            image_array (np.ndarray): single channel 8-bit image pixel array
            delay (int): Delay in ms after displaying image.
        Returns: None
        """
        
        # Rescale if necessary
        if (image_array.shape != (self.ydim, self.xdim)):
            image_array = cv.resize(image_array, (self.xdim, self.ydim))

        cv.namedWindow(self.window_name, cv.WND_PROP_FULLSCREEN)
        cv.moveWindow(self.window_name, self.monitor_offset, 0) # Move to DMD screen
        cv.setWindowProperty(self.window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

        cv.imshow(self.window_name, image_array)
        cv.waitKey(100) # 10ms delay needed for proper display
        cv.waitKey(delay)



    def close(self) -> None:
        """
        Closes the DMD array display window
        """
        cv.destroyAllWindows()
        # # If the window is still open
        # if (cv.getWindowProperty(self.window_name, cv.WND_PROP_VISIBLE) >= 1):
        #     cv.destroyAllWindows()
        


if __name__ == "__main__":
    
    # image = cv.imread(r"C:\Users\Weld Lab\PycharmProjects\DMDProject\oldfiles\RescaledImages\amongus_adjusted.bmp",
    #              cv.IMREAD_UNCHANGED)
    image = cv.imread(r"C:\Users\Weld Lab\PycharmProjects\DMDProject\DMDImageFolder\darkfield.bmp",
                 cv.IMREAD_UNCHANGED)
    
    dmd = DMDController()
    dmd.get_size()
    dmd.update_array(image)
    time.sleep(10)
    dmd.close()

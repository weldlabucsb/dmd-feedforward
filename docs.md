# Code Documentation

---
## `class FemtoEasyBP:`

Interfaces with the Femto Easy beam profiler (LP6.3). Getters and setters for all acquisition parameters are also implemented.

### `__init__(host_ip: str, port: int=8000)`

> **Parameters:**
- **host_ip** *string* Local IP address of the computer that the Femto easy is connected to.
- **port** *int* Port of the API endpoint. Default 8000.

### `get_image()`

Gets the current image from the beam profiler in array form.

> **Returns:**
- **image** *array-like* Latest image


### `save_image(filepath: str)`

Gets and saves the current image to disk in .tiff format.

> **Parameters:**
- **filepath** *string* Path to the file to write to. Must include .tiff extension

> **Returns:** None

---
## `class ImageHandler:`

Create images to be displayed on the DMD.

### `__init__(xdim, ydim)`

> **Parameters:**
- **xdim** *int* $x$ (width) dimension of the images to be created
- **ydim** *int* $y$ (height) dimension of the images to be created

> **Returns:** *ImageHandler* object

### `dither(img: np.ndarray)`

Dithers an image using Floyd-Steinberg error diffusion algorithm.
> **Parameters:**  
- **img** *array-like* Continuous image to be dithered
> **Returns:** None

### `linear_gradient(angle, xmin=None, xmax=None, ymin=None, ymax=None, bg_invert=False, no_dither=False)`

> **Parameters:** 
- **angle:** *float* angle of the gradient with respect to the $x$ axis in degrees. 
- **xmin, xmax, ymin, ymax:** Defines the domain over which the gradient should be created. Defaults to the full array. 
- **bg_invert:** *bool* Inverts the background, switching off pixels to on. Applies only when domain does not cover full array. Default false.
- **no_dither** *bool* Skip dithering the image if true. Default false.

> **Returns:** 
- **gradient** *array-like* Gradient as an 8 bit image array.

### `solid_field(invert=False,  xmin=None, xmax=None, ymin=None, ymax=None)`

Creates a solid image over the specified domain.

> **Parameters:**  
- **invert** *bool* Invert on and off values if true. Default false.  
- **xmin, xmax, ymin, ymax:** Defines the domain over which the soild field should be created. Defaults to the full array. 
> **Returns:** 
- **solid** *array-like* Soild field as an 8 bit image array.

### `invert_image(img)`

Inverts the intensity of each pixel between 0 <--> 255.
> **Parameters:**  
- **img** *np.ndarray* Image (8 bit) to be inverted  
> **Returns:** 
- **inverted** *np.ndarray* Inverted image

### `border(width: int, invert=False)`

Creates a border frame image, helpful in testing.

> **Parameters:**  
- **width** *int* Width of the border to create.  
- **invert** *bool* Inverts the image if true. Default false.  
> **Returns:** 
- **border** *array-like* Image of the border

---
## `class DMDController:`

Displays images on the DMD over HDMI.

### `__init__(monitor_idx: int=1):`

> **Parameters:**
- **monitor_idx** *int* Index of the monitor that corresponds to the DMD, where 0 is primary, 1 is secondary, etc. Default 1.

> **Returns:**

- **DMDController** object

### `get_size()`

> **Returns:**

- **(width, height)** *tuple[int]* The dimensions of the detected DMD.

### `update_array(image_array: np.ndarray, delay: int=1000)`

Updates the DMD array.
> **Parameters:**
- **image_array** *array-like* of size equal to DMD dimensions. The binary (0 or 255) image to display on the DMD.
- **delay** *float* Delay in ms after calling creating the window on the DMD display. Required for proper operation. Default 1000.

### `close()`

Closes the current DMD display window. DMD mirrors set to off state.

---
## `class Feedforward:`

Executes noise removal and feed forward optimization routines.

### `__init__(cam_ip: str, cam_port: int=8000, invert: bool=False)`

Constructs a Feedforward object
> **Parameters:**
- **cam_ip:** *str* IP address of the Femto easy camera, e.g. "localhost" or "127.0.0.1"
- **cam_port:** *int* port number of the Femto easy API endpoint. Default 8000.

> **Returns:**
- **Feedforward** object

### `gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta)`

> **Parameters:**
- **xy:** *array-like* of shape (N, N)  
- **amplitude, xo, yo, sigma_x, sigma_y, theta:** *int*s parameters for fit

> **Returns:**
- **gaussian:** *array-like* Gaussian function with rotation angle theta, offset *(xo, yo)*, standard deviations *(sigma_x, sigma_y)* and rotation angle *theta*.

### `remove_border(img: np.ndarray, border_width: int)`

> **Parameters:**
- **img:** *array-like* of shape (M, N) 
- **border_width:** *int* width of border to zero out

> **Returns**:  
- **zeroed**: *np.ndarray* of shape (M, N). Pixels on the outer border are set to zero

    
### `percentile_normalize(array: np.ndarray, p: float)`

Normalizes an array to the value of the $p^{th}$ percentile such that those and all values above are set to 1.

> **Parameters**
- **img:** *array-like* of shape (M, N) 

> **Returns**:  P
- **normalized**: *np.ndarray* of shape (M, N), normalized



### `transform(img)`

> **Parameters:**

- **img:** *array-like* The image to transform  

Transforms the most recent result using the set transformation matrix $M$.

### `set_transformation(dmd_pts, show_pts=False)`

Sets the camera and DMD coordinate points to define the transformation matrix $M$.

> **Parameters:**

- **dmd_pts:** *array-like* of shape (3,)  
Three points defining the top left, top right, and bottom right of the DMD image in camera coordinates
- **show_pts:** *bool*   
Draws circles at *dmd_pts* if set to True.

> **Returns**:  
- **M**: *np.ndarray* of shape (2, 3). The computed transformation matrix.

### `get_flat_field(p_opt: list=None)`

Gets a flat field by taking a solid field image, smoothing, fitting to a 2D Gaussian, then dividing the fit from the solid field image. Sets `self.flat_field`. Must set transformation using `set_transformation()` first.

> **Parameters**
- **p_opt**: *list* list of the parameters for Gaussian fitting in order of `gaussian_2d()` parameters. Overrides automatic fitting if set. Default None.

> **Returns**:  
- **(solid, fit, flat_field)**: *tuple, np.ndarray* the transformed solid field, Gaussian fit, and computed flat fields.

### `compute_rms(measured: np.ndarray, target: np.ndarray)`

Computes the normalized RMS error as

$$
\varepsilon_k = \sqrt{\sum_{i,j} \left( \frac{n_M(i,j)}{n_T(i,j)} - 1 \right)^2}
$$

> **Parameters**
- **measured** *array-like* measured DMD image array $n_M$
- **target** *array-like* continous (non-dithered) target array $n_T$

> **Returns**:  
- **rms**: *float* the rms error

### `create_update_map(self, error: np.ndarray, target: np.ndarray, eta: float=0.2)`

Takes in the error and targe and computes the updated continuous pixel map as
$$
D_{k+1}(i, j) = D_k (i, j) - \eta E_k (i, j)
$$
for step size $\eta$. Values outside of [0, 1] are clipped to the boundary.

> **Parameters:**
- **error:** *array-like* Error map as computed from `compute_error()`
- **target:** *array-like* Continuous (non-dithered) normalized target image

> **Returns**:  
- **update** *array-like* The updated pixel map. Must be dithered before display on DMD


### `compute_error(measured_image, target_image, alpha=1)`

Computes the error map as

$$
E(i, j) = \alpha \frac{n_M (i, j) - n_T (i, j)}{\max \{n_M\}}
$$
> **Parameters:**
- **measured_image:** *array-like* of DMD dimension shape. ($n_M$)
- **target_image:** *array-like* of DMD dimension shape (not dithered). ($n_T$)
- **alpha:** *float* Step size to scale error by

> **Returns**:  *np.ndarray* $E(i, j)$ as above 

**Note:**  
Should be updated in the future to include term accounting for potential depth when using feed forward with atom density, see eq. (4.11) on p. 104 of Gauthier Guillaume's thesis.


### `run(n_iter)`

Runs the main step loop *n_iter* number of times.

Main loop:  
1. Capture current DMD display image
2. Divide by flat field
3. Take difference between target and image
4. Update pixels

Note: currently setup to reverse the captured image for comparison due to the image inversion from the
odd number of mirrors after the DMD.

> **Parameters:**
- **n_iter:** *int*  
Number of iterations to step

> **Returns:**
Returns tuple of the following, in order:
- **target_0_cts:** *array-like* Initial target image
- **target_last_dithered:** *array-like* Last iteration dithered image displayed on the DMD
- **captures:** *list* List of images after each iteration
- **errs:** *list* List of error maps after each iteration

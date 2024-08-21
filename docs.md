# Code Plans


## `class Feedforward:`


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

### `get_flat_field()`

Gets a flat field by taking a solid field image, smoothing, fitting to a 2D Gaussian, then subtracting the fit from the solid field image. Sets `self.flat_field`.

> **Returns**:  
- **flat_field**: *np.ndarray* of shape equal to DMD dimensions


### `compute_error(measured_image, target_image, alpha=1)`

Computes the error map as

$$
E(i, j) = \alpha \frac{n_M (i, j) - n_T (i, j)}{\max \{n_M\}}
$$
> **Parameters:**
- **measured_image:** *array-like* of DMD dimension shape. ($n_M$)
- **target_image:** *array-like* of DMD dimension shape (not dithered). ($n_T$)
- **alpa:** *float* Step size to scale error by

> **Returns**:  *np.ndarray* $E(i, j)$ as above 

**Note:**  
Should be updated in the future to include term accounting for potential depth when using feed forward with atom density, see eq. (4.11) on p. 104 of Gauthier Guillaume's thesis.


### `run(n_iter)`

Runs the main step loop *n_iter* number of times.

> **Parameters:**
- **n_iter:** *int*  
Number of iterations to step


Main loop:  
1. Capture current DMD display image
2. Divide by flat field
3. Take difference between target and image
4. Update pixels
# Code Documentation


## `class Feedforward:`

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

> **Parameters:**
- **n_iter:** *int*  
Number of iterations to step


Main loop:  
1. Capture current DMD display image
2. Divide by flat field
3. Take difference between target and image
4. Update pixels
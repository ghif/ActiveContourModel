import numpy as np

import scipy
from skimage.util import img_as_float
from skimage.filters import sobel, gaussian
from scipy.interpolate import RectBivariateSpline

import image_processor as iproc

class Snakes(object):
    def __init__(self, name='snakes'):
        self.name = name
        self.hist = {}

    def fit_kaas(self, Im, init_snake, alpha=0.01, beta=0.1,
            w_line=0, w_edge=1, tau=100,
            bc='periodic', max_px_move=1.0,
            max_iter=2500, convergence=0.1, convergence_order=10):
        """
        Snake fitting based on the original method 
            Kaas, M., Witkin, A., Terzopoulos, D. "Snakes: Active Contour Models". [IJCV 1998]

        Adopted from skimage active_contour
            https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/active_contour_model.py

        Args: 
            Im (ndarray)            : (H x W) grayscaled image
            init_snake (ndarray)    : (n, 2) initial snake contour
                For periodic snakes, it should not include duplicate endpoints
            alpha (float)           : Snake length shape parameter
            beta (float)            : Snake smoothness shape parameter
            w_line (float)          : Controls attraction to brightness. Use negative values to attract to dark regions
            w_edge (float)          : Controls attraction to edges. Use negative values to repel snake from edges
            tau (float)             : time step
            bc {'periodic', 'free', 'fixed'}    : Boundary conditions for snake. 'periodic' attaches the two ends of the snake
            max_px_move             : maximum pixel distance to move per iteration
            max_iter                : maximum iterations to optimize snake shape
            convergence             : convergence criteria

        Returns:
            snake (ndarray)         : (n, 2) final snake contour
            Edge (ndarray)          : (H x W) sobel edge image 
            Map (ndarray)           : (H x W) external energy

        """
        # Gaussian smoothing
        Img = img_as_float(Im)
        Img = gaussian(Img, 2)
                
        # Compute edge
        Edge = sobel(Img)

        # Superimpose intensity and edge images
        Img = iproc.normalizeRange(Img, minVal=0, maxVal=1) # normalize to 0-1 values, IMPORTANT!
        Edge = iproc.normalizeRange(Edge, minVal=0, maxVal=1)
        Map = w_line * Img + w_edge * Edge
        
        # Interpolate for smoothness
        intp_fn = RectBivariateSpline(
                np.arange(Map.shape[1]),
                np.arange(Map.shape[0]),
                Map.T, kx=2, ky=2, s=0
            )

        # Get snake contour axes
        x0, y0 = init_snake[:, 0].astype(np.float), init_snake[:, 1].astype(np.float)
        
        # store snake progress
        sn = np.array([x0, y0]).T
        self.hist['snakes'] = []
        self.hist['snakes'].append(sn) 

        # for convergence computation
        xsave = np.empty((convergence_order, len(x0)))
        ysave = np.empty((convergence_order, len(y0)))

        # Build finite difference matrices
        n = len(x0)
	A = np.roll(np.eye(n), -1, axis=0) + \
		np.roll(np.eye(n), -1, axis=1) - \
		2*np.eye(n)  # second order derivative, central difference
	B = np.roll(np.eye(n), -2, axis=0) + \
		np.roll(np.eye(n), -2, axis=1) - \
		4*np.roll(np.eye(n), -1, axis=0) - \
		4*np.roll(np.eye(n), -1, axis=1) + \
		6*np.eye(n)  # fourth order derivative, central difference
	Z = -alpha*A + beta*B
 
        
        # Impose boundary conditions different from periodic:
	sfixed = False
	if bc.startswith('fixed'):
            Z[0, :] = 0
            Z[1, :] = 0
            Z[1, :3] = [1, -2, 1]
            sfixed = True

	efixed = False
	if bc.endswith('fixed'):
            Z[-1, :] = 0
            Z[-2, :] = 0
            Z[-2, -3:] = [1, -2, 1]
            efixed = True

	sfree = False
	if bc.startswith('free'):
            Z[0, :] = 0
            Z[0, :3] = [1, -2, 1]
            Z[1, :] = 0
            Z[1, :4] = [-1, 3, -3, 1]
            sfree = True

	efree = False
	if bc.endswith('free'):
            Z[-1, :] = 0
            Z[-1, -3:] = [1, -2, 1]
            Z[-2, :] = 0
            Z[-2, -4:] = [-1, 3, -3, 1]
            efree = True

        # Calculate inverse
        # Zinv = scipy.linalg.inv(Z + gamma * np.eye(n))
        Zinv = scipy.linalg.inv(np.eye(n) + tau * Z)

        # Snake energy minimization
        x = np.copy(x0)
        y = np.copy(y0)

        

        for i in range(max_iter):
            fx = intp_fn(x, y, dx=1, grid=False)
	    fy = intp_fn(x, y, dy=1, grid=False)

            # # Normalize external forces
            # fx = fx / (np.linalg.norm(fx) + 1e-6)
            # fy = fy / (np.linalg.norm(fy) + 1e-6)
        
            if sfixed:
                fx[0] = 0
                fx[0] = 0
            if efixed:
                fx[-1] = 0
                fy[-1] = 0
            if sfree:
                fx[0] *= 2
                fy[0] *= 2
            if efree:
                fx[-1] *= 2
                fy[-1] *= 2

            # xn = np.dot(Zinv, gamma *x + fx)
            # yn = np.dot(Zinv, gamma *y + fy)
            xn = np.dot(Zinv, x + tau * fx)
            yn = np.dot(Zinv, y + tau * fy)
            
            # Movements are capped to max_px_move per iter
            dx = max_px_move * np.tanh(xn-x)
            dy = max_px_move * np.tanh(yn-y)

            if sfixed:
                dx[0] = 0
                dy[0] = 0

            if efixed:
                dx[-1] = 0
                dy[-1] = 0

            x += dx
            y += dy

            # Convergence criteria 
            # - x(t) - x(t-i} < convergence 
            # - i: the last k (convergence_order) of previous snakes
	    
	    j = i % (convergence_order+1)
            dist = 10000000
	    if j < convergence_order:
                xsave[j, :] = x
                ysave[j, :] = y
	    else:
                dist = np.min(np.max(np.abs(xsave-x[None, :]) +
                           np.abs(ysave-y[None, :]), 1))
                if dist < convergence:
                    break
                print('Iter-%d/%d: convergence_order: %d, j: %d, dist: %f' % (i, max_iter, convergence_order, j, dist))
                # store snake progress
                sn = np.array([x, y]).T
                self.hist['snakes'].append(sn)
        # end for

        snake = np.array([x, y]).T

        return snake, Edge, Map, Img

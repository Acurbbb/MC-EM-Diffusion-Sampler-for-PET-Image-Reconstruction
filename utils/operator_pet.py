import os
import time
import torch
import numpy as np
import scipy.sparse as sp
from .scanner import PETScanner
from .map import penalized_ot_lange_step

from typing import Optional, Tuple, Union

ArrayLike = np.ndarray

class PETOperator():
    def __init__(self, **kwargs):
        """
            G:  geometric system matrix in scipy array
            P:  iPSF sparse matrix
        """
        for key, value in kwargs.items():
            setattr(self, key, self._initialize_data(value))

        self.sens = (self.ni @ self.G) @ self.P
        self.device = 'cpu'

    def _initialize_data(self, data):
        if isinstance(data, np.ndarray):
            return data.astype(np.float32)
        elif sp.issparse(data):
            return data.astype(np.float32)
        else:
            raise ValueError("Initialization only supports numpy array or scipy sparse matrix.")
        
    def _get_backend(self):
        """
        Return torch or numpy module based on current device.
        """
        return torch if self.device != 'cpu' else np

    def _get_device(self, data):
        if isinstance(data, np.ndarray) or sp.issparse(data):
            return 'cpu'
        elif isinstance(data, torch.Tensor):
            return 'cuda' if data.is_cuda else 'cpu'
        else:
            raise ValueError("Unknown data type.")

    def _get_data_vars(self):
        data_vars = []
        for name, value in self.__dict__.items():
            if isinstance(value, (np.ndarray, torch.Tensor, sp.spmatrix)):
                data_vars.append(name)
        return data_vars
    
    def _scipy_sparse_to_torch_sparse(self, mat):
        mat = mat.tocoo()
        indices = np.vstack((mat.row, mat.col))
        indices = torch.from_numpy(indices).long()
        values = torch.from_numpy(mat.data).float()
        shape = mat.shape
        return torch.sparse_coo_tensor(indices, values, shape)

    def _torch_sparse_to_scipy_sparse(self, tensor):
        tensor = tensor.coalesce()
        indices = tensor.indices().cpu().numpy()
        values = tensor.values().cpu().numpy()
        shape = tensor.shape
        return sp.coo_matrix((values, (indices[0], indices[1])), shape=shape)

    def to(self, device):
        if isinstance(device, torch.device):
            device = str(device)

        for var in self._get_data_vars():
            value = getattr(self, var)
            if device == 'cpu':
                if isinstance(value, torch.Tensor):
                    if value.is_sparse:
                        setattr(self, var, self._torch_sparse_to_scipy_sparse(value))
                    else:
                        setattr(self, var, value.cpu().numpy())
                # already numpy or scipy.sparse — no change needed
            elif device.startswith('cuda'):
                if isinstance(value, np.ndarray):
                    setattr(self, var, torch.from_numpy(value).to(device))
                elif sp.issparse(value):
                    setattr(self, var, self._scipy_sparse_to_torch_sparse(value).to(device))
                elif isinstance(value, torch.Tensor):
                    setattr(self, var, value.to(device))
            else:
                raise ValueError(f"Unknown device {device}")
        
        self.device = device
        return self
    
    def ensure_device(self, x):
        if self.device == 'cpu':
            if isinstance(x, torch.Tensor):
                if x.is_sparse:
                    x = self._torch_sparse_to_scipy_sparse(x)
                else:
                    x = x.cpu().numpy()
        else:
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).to(self.device)
            elif sp.issparse(x):
                x = self._scipy_sparse_to_torch_sparse(x).to(self.device)
            else:
                x = x.to(self.device)
        return x
           

    def forward(self, x):
        return self.ni * (self.G @ (self.P @ x.flatten())) + self.ri
    
    def backproj(self, y):
        return ((y * self.ni) @ self.G) @ self.P
    
    def transpose(self, mat):
        if isinstance(mat, torch.Tensor):
            return mat.transpose(0, 1)
        elif sp.issparse(mat) or isinstance(mat, np.ndarray):
            return mat.transpose((0,1))
        else:
            raise ValueError('Unknown data type!')
    
    def get_log_likelihood(self, x, eps=1e-5):
        bd = self._get_backend()
        fp = self.forward(x) + eps
        return bd.sum(self.yi * bd.log(fp) - fp)
    
    def get_KL_distance(self, x, eps=1e-5):
        """KL(y,Hx)=y*log(y/(Hx))+Hx-y"""
        bd = self._get_backend()
        fp = self.forward(x)
        idx_b = fp > 0
        return bd.sum(self.yi[idx_b] * bd.log(self.yi[idx_b] / fp[idx_b] + eps) + fp[idx_b] - self.yi[idx_b])
    
    def get_grad_log_likelihood(self, x, eps=1e-5):
        fp = self.forward(x) + eps
        return (((self.ni * self.yi / fp) @ self.G) @ self.P - self.sens).reshape(x.shape)
    
    def do_mlem(self, x=None, maxit=50, savestep=1):
        yi = self.yi

        if x is None:
            x = np.ones((self.G.shape[1])) * 1e-5

        # Prepare
        wx = self.sens
        yeps = np.mean(yi) * 1e-9

        # Output
        xs = []
        likelihoods = []

        # Iterate
        for it in range(maxit):
            # Save data
            # print(f'Iteration {it + 1}')
            if savestep > 0 and it % savestep == 0:
                xs.append(np.copy(x).flatten())

            # EM update
            yb = self.forward(x)
            yy = yi / (yb + yeps)
            yy[(yb == 0) & (yi == 0)] = 1
            xb = self.backproj(yy)
            x = x / wx * xb

            # Objective function value
            likelihoods.append(self.get_log_likelihood(x))
            if it > 0 and likelihoods[it] < likelihoods[it - 1]:
                print('Warning: Objective function is not increasing')

        xs = np.stack(xs + [np.copy(x).flatten()], axis=0)
        return x, xs, likelihoods

    def do_pml(self, 
                beta: float,
                delta: float,
                imgsize: Tuple,
                x=None,
                patch_size: Union[int, Tuple[int, ...]] = 3,
                neigh_size: Union[int, Tuple[int, ...]] = 3,
                maxit: int = 50,
                savestep = 1):
        """Penalized ML, Wang and Qi, TMI, 2012"""
        yi = self.yi

        if x is None:
            x = np.ones((self.G.shape[1])) * 1e-5

        if torch.cuda.is_available():
            backend_conv = "torch" 
            print("CUDA is available, will perform convolution on GPU")
        else:
            backend_conv = "numpy"
            print("CUDA not avialable, will perform convolution on CPU")

        # Prepare
        wx = self.sens
        yeps = np.mean(yi) * 1e-9

        # Output
        xs = []
        likelihoods = []

        # Iterate
        for it in range(maxit):
            # Save data
            # print(f'Iteration {it + 1}')
            if savestep > 0 and it % savestep == 0:
                xs.append(np.copy(x).flatten())

            # EM update
            yb = self.forward(x)
            yy = yi / (yb + yeps)
            yy[(yb == 0) & (yi == 0)] = 1
            xb = self.backproj(yy)
            x_em = x / wx * xb

            if beta > 0:
                x, x_em = x.reshape(imgsize), x_em.reshape(imgsize)
                _, x = penalized_ot_lange_step(x, x_em, beta, delta, wx.reshape(imgsize), patch_size, neigh_size, backend_conv)
                x = x.ravel()
            else:
                x = x_em

        xs = np.stack(xs + [np.copy(x).flatten()], axis=0)
        return x, xs, likelihoods
    
    def em_update(self, x, eps=1e-5):
        image_shape = x.shape
        x = x.ravel()
        # forward projection + additive + small floor
        ybar = self.forward(x) + eps

        # OSEM update: x <- x * [P^T(1/ybar)] / (sens/num_subsets)
        x_em = (x / 
               self.sens *   
               self.backproj(self.yi / ybar))

        return x_em.reshape(image_shape)
    
    def em_update_prox(self, x, x0, lamda, eps=1e-5):
        image_shape = x.shape
        x = x.ravel()
        x0 = x0.ravel()
        # forward projection + additive + small floor
        ybar = self.forward(x) + eps

        # OSEM update: x <- x * [P^T(1/ybar)] / (sens/num_subsets)
        x_em = (x / 
               self.sens *   
               self.backproj(self.yi / ybar))
        
        x_em += 2 * lamda * x / self.sens * (x0 - x)
        x_em = x_em.clip(min=0.)
        
        return x_em.reshape(image_shape)
    
    def grad_fn(self, x, step_size=1., eps=1e-5):
        image_shape = x.shape
        x = x.ravel()
        grad = self.get_grad_log_likelihood(x, eps)

        return step_size * grad.reshape(image_shape)


class PETOperatorLM():
    def __init__(
        self,
        scanner: PETScanner,
        num_subsets: int,
        imgsize: Tuple[float,float,float],
        voxsize: Tuple[float,float,float],
        lmdata: np.ndarray | str,
        add_fac: np.ndarray | str | None = None,
        # mul_fac: np.ndarray | None = None,
        sens: np.ndarray | None = None,
        blur_sens = True,
    ):
        """
        proj      : sinogram‐based projector (has .in_shape, .voxel_size, .lor_descriptor, .tof, .tof_parameters)
        res_model : resolution model operator (post‐projection blur)
        num_subsets: number of subsets for OSEM
        u_map     : attenuation map (same shape as proj.in_shape)
        lmdata    : N×5 int16 array [t1, z1, t2, z2, tof_bin]
        add_fac   : optional additive corrections per event
        mul_fac   : optional multiplicative corrections per event
        sens      : optional precomputed sensitivity image
        """
        self.scanner = scanner
        self.lmdata = lmdata
        try:
            self.num_events = lmdata.shape[0]
        except AttributeError:
            self.num_events = os.path.getsize(lmdata) // 10 # hard-coded 10 bytes per events

        self.imgsize = imgsize
        self.voxsize = voxsize

        # per-event corrections
        self.add_fac = None if add_fac is None else add_fac
        # self.mul_fac = np.ones(self.num_events, dtype=np.float32) \
        #                if mul_fac is None else mul_fac

        self.num_subsets = num_subsets

        # blur sensitivity image with scanner.psf_sigma_mm
        if blur_sens:
            self.sens = self._apply_psf(sens.reshape(imgsize), scanner.psf_sigma_mm, voxsize)
        else:
            self.sens = sens.reshape(imgsize)


    def grad_fn(self, x, subset_id, step_size=1., eps=1e-5):
        sl = slice(subset_id, None, self.num_subsets)
        lm_subset = self.lmdata[sl]
        af_subset = self.add_fac[sl]
        # forward projection + additive + small floor
        ybar = self.scanner.fproj_tof_mt(x, 
                                         self.imgsize, 
                                         self.voxsize, 
                                         lm_subset, 
                                         apply_psf=True) + af_subset + eps

        # OSEM update: x <- x * [P^T(1/ybar)] / (sens/num_subsets)
        grad =  self.scanner.bproj_tof_mt(1. / ybar,
                                         self.imgsize,
                                         self.voxsize,
                                         lm_subset,
                                         apply_psf=True,
                                         return_dtype=np.float32) * self.num_subsets - self.sens

        return step_size * grad
    

    def em_update(self, x, subset_id, eps=1e-5):
        sl = slice(subset_id, None, self.num_subsets)
        lm_subset = self.lmdata[sl]
        af_subset = self.add_fac[sl]
        # forward projection + additive + small floor
        ybar = self.scanner.fproj_tof_mt(x, 
                                         self.imgsize, 
                                         self.voxsize, 
                                         lm_subset, 
                                         apply_psf=True) + af_subset + eps

        # OSEM update: x <- x * [P^T(1/ybar)] / (sens/num_subsets)
        x_em = (x / 
               self.sens *   
               self.scanner.bproj_tof_mt(1. / ybar,
                                         self.imgsize,
                                         self.voxsize,
                                         lm_subset,
                                         apply_psf=True,
                                         return_dtype=np.float32) * 
               self.num_subsets)

        return x_em
    

    def em_update_prox(self, x, subset_id, x0, lamda, eps=1e-5):
        sl = slice(subset_id, None, self.num_subsets)
        lm_subset = self.lmdata[sl]
        af_subset = self.add_fac[sl]
        # forward projection + additive + small floor
        ybar = self.scanner.fproj_tof_mt(x, 
                                         self.imgsize, 
                                         self.voxsize, 
                                         lm_subset, 
                                         apply_psf=True) + af_subset + eps

        # OSEM update: x <- x * [P^T(1/ybar)] / (sens/num_subsets)
        x_em = (x / 
               self.sens *   
               self.scanner.bproj_tof_mt(1. / ybar,
                                         self.imgsize,
                                         self.voxsize,
                                         lm_subset,
                                         apply_psf=True,
                                         return_dtype=np.float32) * 
               self.num_subsets)
        
        x_em += 2 * lamda * x / self.sens * (x0 - x)
        x_em = x_em.clip(min=0.)
        
        return x_em


    def em_update_quad(self, x, subset_id, beta, eps=1e-5):
        (dimz, dimy, dimx) = x.shape

        # prepare for regularization
        # 0.5*beta*sum_ijk(|x_{ijk}-x_{ijk+1}|+|x_{ijk}-x_{ijk-1}|)
        # quadratic surrogate:
        # 0.5*beta*sum_ijk(1/(2d)(x_{ijk}-x_{ijk+1})^2+1/(2d)*(x_{ijk}-x_{ijk-1})^2)
        diff_up = np.concatenate((np.abs(x[1:] - x[:-1]), 
                                  np.zeros((1,dimy,dimx),dtype=np.float32)), axis=0)
        diff_dn = np.concatenate((np.zeros((1,dimy,dimx),dtype=np.float32), 
                                  np.abs(x[:-1] - x[1:])), axis=0)
        kappa_up = 0.5 / (diff_up + eps)
        kappa_dn = 0.5 / (diff_dn + eps)
        x_up = np.concatenate((np.zeros((1,dimy,dimx),dtype=np.float32), 
                               x[1:]), axis=0)
        x_dn = np.concatenate((x[:-1], 
                               np.zeros((1,dimy,dimx),dtype=np.float32)), axis=0)

        sl = slice(subset_id, None, self.num_subsets)
        lm_subset = self.lmdata[sl]
        af_subset = self.add_fac[sl]
        # forward projection + additive + small floor
        ybar = self.scanner.fproj_tof_mt(x, 
                                         self.imgsize, 
                                         self.voxsize, 
                                         lm_subset, 
                                         apply_psf=True) + af_subset + eps

        # OSEM update: x <- x * [P^T(1/ybar)] / (sens/num_subsets)
        xem = (x *  
               self.scanner.bproj_tof_mt(1. / ybar,
                                         self.imgsize,
                                         self.voxsize,
                                         lm_subset,
                                         apply_psf=True))

        # regularization
        A = 2 * beta * (kappa_up + kappa_dn)
        B = self.sens - \
            beta * (
                kappa_up * (x + x_up) +
                kappa_dn * (x + x_dn)
            )
        C = xem * self.num_subsets

        return 2 * C / (B + np.sqrt(np.clip(B**2 + 4*A*C, a_min=0., a_max=None)) + eps)

    def em_update_rdpz(self, x, subset_id, beta, xi=1., eps=1e-5):
        (dimz, dimy, dimx) = x.shape

        # prepare for regularization
        # 0.5*beta*sum_ijk(|x_{ijk}-x_{ijk+1}|+|x_{ijk}-x_{ijk-1}|)
        # quadratic surrogate:
        # 0.5*beta*sum_ijk(1/(2d)(x_{ijk}-x_{ijk+1})^2+1/(2d)*(x_{ijk}-x_{ijk-1})^2)
        up = np.concatenate((x[0][np.newaxis,...],
                             x[:-1]), axis=0)
        dn = np.concatenate((x[1:],
                             x[-1][np.newaxis,...]), axis=0)
      
        sl = slice(subset_id, None, self.num_subsets)
        lm_subset = self.lmdata[sl]
        af_subset = self.add_fac[sl]
        # forward projection + additive + small floor
        ybar = self.scanner.fproj_tof_mt(x, 
                                         self.imgsize, 
                                         self.voxsize, 
                                         lm_subset, 
                                         apply_psf=True) + af_subset + eps

        # OSEM update: x <- x * [P^T(1/ybar)] / (sens/num_subsets)
        x_em = (x / 
               self.sens *   
               self.scanner.bproj_tof_mt(1. / ybar,
                                         self.imgsize,
                                         self.voxsize,
                                         lm_subset,
                                         apply_psf=True,
                                         return_dtype=np.float32) * 
               self.num_subsets)

        # RDP
        r_up = x / (up + eps)
        partial_r  = - (r_up - 1) * (xi * np.abs(r_up - 1) + r_up + 3) / (r_up + 1 + xi * np.abs(r_up - 1))**2
        r_dn = x / (dn + eps)
        partial_r += - (r_dn - 1) * (xi * np.abs(r_dn - 1) + r_dn + 3) / (r_dn + 1 + xi * np.abs(r_dn - 1))**2

        return x_em + beta * x / self.sens * partial_r 
    
    def osem(
        self,
        n_iter: int,
        x: np.ndarray | None = None,
        eps: float = 1e-5,
        save_steps=0,
    ) -> np.ndarray:
        """
        Run OSEM for n_iter iterations.
        Returns the reconstructed image.
        """
        if x is None:
            x = 1e-5 * np.ones(self.imgsize, dtype=np.float64)
        elif x.shape == self.imgsize:
            x = x
        else:
            raise ValueError("Initial x must agree with the size stored!")
            
        subset_slices = [slice(i, None, self.num_subsets)
                         for i in range(self.num_subsets)]

        xs = []
        for it in range(n_iter):
            for ss, sl in enumerate(subset_slices):
                # endpoints + tof for this subset
                lm_subset = self.lmdata[sl]

                ybar = self.scanner.fproj_tof_mt(x, 
                                                 self.imgsize, 
                                                 self.voxsize, 
                                                 lm_subset, 
                                                 apply_psf=True) + self.add_fac[sl] + eps
                
                bp = self.scanner.bproj_tof_mt(1. / ybar,
                                               self.imgsize,
                                               self.voxsize,
                                               lm_subset,
                                               apply_psf=True)

                x = x * bp / self.sens * self.num_subsets               

                print(f'[{it}/{n_iter}][{ss}/{self.num_subsets}] finished...')
            
            if save_steps > 0 and it % save_steps == 0:
                xs.append(x)
        
        if save_steps > 0:
            return x, np.stack(xs, axis=0)
        else:
            return x      
        

    def pml(
        self,
        n_iter: int,
        beta: float,
        delta: float,
        x=None, 
        patch_size: Union[int, Tuple[int, ...]] = 3,
        neigh_size: Union[int, Tuple[int, ...]] = 3,
        eps=1e-5,
        save_steps = 0,
    ) -> np.ndarray:
        """
            PML reconstruction using FAIR potential function, Wang and Qi, TMI, 2012
        """
        if x is None:
            x = 1e-5 * np.ones(self.imgsize, dtype=np.float64)
        elif x.shape == self.imgsize:
            x = x
        else:
            raise ValueError("Initial x must agree with the size stored!")

        if torch.cuda.is_available():
            backend_conv = "torch" 
            print("CUDA is available, will perform convolution on GPU")
        else:
            backend_conv = "numpy"
            print("CUDA not avialable, will perform convolution on CPU")

        subset_slices = [slice(i, None, self.num_subsets)
                         for i in range(self.num_subsets)]

        xs = []
        for it in range(n_iter):
            for ss, sl in enumerate(subset_slices):
                # endpoints + tof for this subset
                lm_subset = self.lmdata[sl]

                ybar = self.scanner.fproj_tof_mt(x, 
                                                 self.imgsize, 
                                                 self.voxsize, 
                                                 lm_subset, 
                                                 apply_psf=True) + self.add_fac[sl] + eps
                
                bp = self.scanner.bproj_tof_mt(1. / ybar,
                                               self.imgsize,
                                               self.voxsize,
                                               lm_subset,
                                               apply_psf=True)

                x_em = x * bp / self.sens * self.num_subsets    

                if beta > 0:
                    _, x = penalized_ot_lange_step(x, x_em, beta, delta, self.sens, patch_size, neigh_size, backend_conv)    
                else:
                    x = x_em 

                print(f'[{it}/{n_iter}][{ss}/{self.num_subsets}] finished...')
            
            if save_steps > 0 and it % save_steps == 0:
                xs.append(x)
        
        if save_steps > 0:
            return x, np.stack(xs, axis=0)
        else:
            return x      
                


    def osem_no_memmap(
        self,
        n_iter: int,
        x: np.ndarray | None = None,
        eps: float = 1e-5,
        save_steps=0,
    ) -> np.ndarray:
        """
        OSEM compatible with np.memmap
        """
        if x is None:
            x = 1e-5 * np.ones(self.imgsize, dtype=np.float64)
        elif x.shape == self.imgsize:
            x = x
        else:
            raise ValueError("Initial x must agree with the size stored!")

        xs = []
        for it in range(n_iter):
            for ss in range(self.num_subsets):
                # endpoints + tof for this subset
                lm_subset = _read_every_kth_no_memmap(self.lmdata, self.num_subsets, fields_per_record=5, dtype=np.int16, start_index=ss)
                if self.add_fac is not None:
                    af_subset = _read_every_kth_no_memmap(self.add_fac, self.num_subsets, fields_per_record=1, dtype=np.float32, start_index=ss) 
                else:
                    af_subset = np.zeros(lm_subset.shape[0], dtype=np.float32)

                ybar = self.scanner.fproj_tof_mt(x, 
                                                 self.imgsize, 
                                                 self.voxsize, 
                                                 lm_subset, 
                                                 apply_psf=True) + af_subset + eps
                
                bp = self.scanner.bproj_tof_mt(1. / ybar,
                                               self.imgsize,
                                               self.voxsize,
                                               lm_subset,
                                               apply_psf=True)

                x = x * bp / self.sens * self.num_subsets               

                print(f'[{it}/{n_iter}][{ss}/{self.num_subsets}] finished...')

            if save_steps > 0 and it % save_steps == 0:
                xs.append(x)
        
        if save_steps > 0:
            return x, np.stack(xs, axis=0)
        else:
            return x      
    
    def _apply_psf(self, img: ArrayLike, psf_sigma_mm: Tuple, voxel_size_mm: Tuple) -> ArrayLike:
        """Apply separable Gaussian blur with σ specified in mm (converted to voxels)."""
        if all(s == 0.0 for s in psf_sigma_mm):
            return img

        sigma_vox = tuple(max(0.0, s_mm / vs) for s_mm, vs in zip(psf_sigma_mm, voxel_size_mm))

        # NumPy branch (SciPy preferred)
        try:
            from scipy.ndimage import gaussian_filter
        except Exception as e:
            raise RuntimeError(
                "NumPy PSF blur requires SciPy (scipy.ndimage.gaussian_filter). "
                "Either install SciPy or pass/keep tensors as torch.Tensor."
            ) from e
        # Accept (D,H,W) or (C,D,H,W) or (N,C,D,H,W)
        arr = np.asarray(img)
        if arr.ndim == 3:
            return gaussian_filter(arr, sigma=sigma_vox, mode="reflect")
        elif arr.ndim == 4:
            # Treat channel as independent; filter per-channel
            out = np.empty_like(arr)
            for c in range(arr.shape[0]):
                out[c] = gaussian_filter(arr[c], sigma=sigma_vox, mode="reflect")
            return out
        elif arr.ndim == 5:
            out = np.empty_like(arr)
            for n in range(arr.shape[0]):
                for c in range(arr.shape[1]):
                    out[n, c] = gaussian_filter(arr[n, c], sigma=sigma_vox, mode="reflect")
            return out
        else:
            raise ValueError("Expected image dims in {3,4,5}, got {}".format(arr.ndim))

    def gaussian_blur(self, img: ArrayLike, psf_sigma_mm: float | Tuple, voxel_size_mm: Tuple) -> ArrayLike:
        """Blur the image using Gaussian"""
        if isinstance(psf_sigma_mm, (float, int)):
            psf_sigma_mm = (psf_sigma_mm,) * 3
        return self._apply_psf(img, psf_sigma_mm, voxel_size_mm)


def get_projection(G, P, x, u, count, random=0.2):

    # n_tofbin = int(G.shape[0] / G_nontof.shape[0])
    G_nontof = G
    x = np.atleast_2d(x)
    num_frame = x.shape[0]
    
    # noise-free geometric projection
    proj = []
    for m in range(num_frame):
        proj.append(G @ (P @ x[m]))
    proj = np.stack(proj, axis=0)

    # attenuation
    ai = np.atleast_2d(np.exp(-G_nontof @ (P @ u)))
    # ai = np.tile(ai.reshape(-1,1), (1, n_tofbin))
    # ai = ai.flatten()

    # Background (randoms and scatters)
    mean_ai_proj = np.mean(ai * proj, axis=1)
    ri = random * np.diag(mean_ai_proj) @ np.ones_like(proj)  # uniform background

    # Total noiseless projection
    y0 = ai * proj + ri

    # Normalized sinograms
    cs = count / y0.sum()
    y0 *= cs
    ri *= cs

    # Generate noisy projection
    yi = np.random.poisson(y0)

    # Multiplicative factor
    ni = ai * cs

    return yi.squeeze(), ni.squeeze(), ri.squeeze()


from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple, Union
import numpy as np
from rayprj import fproj_mt, bproj_mt, fproj_tof_mt, bproj_tof_mt

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

ArrayLike = np.ndarray


class PETScanner:
    """
    Minimal PET scanner wrapper that:
      - stores crystal locations and LOR pairing
      - stores TOF meta (bin width, sigma, etc.)
      - applies image-domain PSF (Gaussian) for forward/backward ops
      - calls user-supplied forward/backward projector functions

    Parameters
    ----------
    crystal_xyz : (N_det, 3) array
        3D positions (mm) of each detector crystal center.
    lor_pairs : (N_LOR, 2) int array
        Each row is [idx1, idx2] indexing into `crystal_xyz` to form an LOR.
    image_shape : (D, H, W)
        Voxel grid size for images.
    voxel_size_mm : (dz, dy, dx)
        Physical voxel size in mm.
    projector_fwd : callable
        Signature: proj = projector_fwd(image, lor_endpoints, tof)
        where:
          - image is a (D,H,W) ndarray or torch tensor (or batched CxDxHxW)
          - lor_endpoints is (N_LOR, 2, 3) float array (mm)
          - tof is a dict with optional keys
    projector_bwd : callable
        Signature: img = projector_bwd(proj, lor_endpoints, tof)
        Should return an image in the same layout as `image`.
    psf_sigma_mm : (sz, sy, sx)
        Gaussian sigma (mm) for image-domain PSF. Set zeros to disable.
    tof : dict, optional
        Keys you might use in your projector: 
        {
          "bin_width_ps": float,
          "num_bins": int,
          "sigma_t_ps": float,         # timing resolution (1σ) in ps
          "sigma_mm_along_lor": float  # computed as c*σ_t/2 in mm (filled if not provided)
        }
    """
    def __init__(
        self,
        scanner: str,
    ):
        if scanner == "nx-3d":
            _here = Path(__file__).resolve().parent
            self.xtal_xy = np.fromfile(_here / "xtal_pos_xy_uih", dtype=np.float32).reshape((-1,2)).astype(np.float64)
            self.xtal_z  = np.fromfile(_here / "xtal_pos_z", dtype=np.float32).astype(np.float64)
            self.tof_info = np.array([240, 12.2])
            psf_fwhm_mm = (1.6, 0.8, 0.8)
            self.psf_sigma_mm = tuple(fwhm / 2.355 for fwhm in psf_fwhm_mm)

    def fproj_mt(
            self, 
            image: ArrayLike, 
            imgsize: Tuple, 
            voxsize: Tuple, 
            lmdata: ArrayLike,  
            apply_psf: bool = True,
            ) -> ArrayLike:
        """
        Forward projection. If apply_psf=True, blurs the image by Gaussian PSF
        before projection (common in image-domain PSF modeling).
        """
        x = self._apply_psf(image, voxsize) if apply_psf else image
        x = np.ascontiguousarray(x, dtype=np.float64)
        lmdata = np.ascontiguousarray(lmdata)
        assert lmdata.dtype == np.int16
        return fproj_mt(
            x.flatten(),      
            np.asarray(imgsize, dtype=np.int32),    
            np.asarray(voxsize),    
            self.xtal_xy,    
            self.xtal_z,     
            lmdata,     
        )

    def bproj_mt(
            self,
            proj: ArrayLike,
            imgsize: Tuple,
            voxsize: Tuple,
            lmdata: ArrayLike,
            apply_psf: bool = True,
            return_dtype = np.float64,
            ) -> ArrayLike:
        """
        Backprojection. If apply_psf=True, applies Gaussian PSF after backprojection.
        For symmetric Gaussian, this equals the transpose operation for the blur.
        """
        proj = np.ascontiguousarray(proj, dtype=np.float64)
        lmdata = np.ascontiguousarray(lmdata)
        assert lmdata.dtype == np.int16
        img = bproj_mt(
            proj,      
            np.asarray(imgsize, dtype=np.int32),    
            np.asarray(voxsize),   
            self.xtal_xy,   
            self.xtal_z,    
            lmdata,    
        ).reshape(imgsize)

        return np.asarray(self._apply_psf(img, voxsize) if apply_psf else img, dtype=return_dtype)

    def fproj_tof_mt(
            self, 
            image: ArrayLike, 
            imgsize: Tuple, 
            voxsize: Tuple, 
            lmdata: ArrayLike,  
            apply_psf: bool = True,
            ) -> ArrayLike:
        """
        Forward projection. If apply_psf=True, blurs the image by Gaussian PSF
        before projection (common in image-domain PSF modeling).
        """
        x = self._apply_psf(image, voxsize) if apply_psf else image
        x = np.ascontiguousarray(x, dtype=np.float64)
        lmdata = np.ascontiguousarray(lmdata)
        assert lmdata.dtype == np.int16
        return fproj_tof_mt(
            x.flatten(),      
            np.asarray(imgsize, dtype=np.int32),    
            np.asarray(voxsize),    
            self.xtal_xy,    
            self.xtal_z,     
            lmdata,     
            np.asarray(self.tof_info, dtype=np.float64),
        )

    def bproj_tof_mt(
            self,
            proj: ArrayLike,
            imgsize: Tuple,
            voxsize: Tuple,
            lmdata: ArrayLike,
            apply_psf: bool = True,
            return_dtype = np.float64,
            ) -> ArrayLike:
        """
        Backprojection. If apply_psf=True, applies Gaussian PSF after backprojection.
        For symmetric Gaussian, this equals the transpose operation for the blur.
        """
        proj = np.ascontiguousarray(proj, dtype=np.float64)
        lmdata = np.ascontiguousarray(lmdata)
        assert lmdata.dtype == np.int16
        img = bproj_tof_mt(
            proj,      
            np.asarray(imgsize, dtype=np.int32),    
            np.asarray(voxsize),   
            self.xtal_xy,   
            self.xtal_z,    
            lmdata,    
            np.asarray(self.tof_info, dtype=np.float64),
        ).reshape(imgsize)
        
        return np.asarray(self._apply_psf(img, voxsize) if apply_psf else img, dtype=return_dtype)


    def set_psf_sigma_mm(self, sigma_mm: Tuple[float, float, float]) -> None:
        """Update image-domain PSF sigma (mm)."""
        self.psf_sigma_mm = tuple(float(s) for s in sigma_mm)
        self._torch_kernel_cache.clear()

    def set_tof(self, tof: Dict) -> None:
        """Update TOF metadata; fills derived fields if possible."""
        self.tof = dict(tof)
        if "sigma_mm_along_lor" not in self.tof and "sigma_t_ps" in self.tof:
            self.tof["sigma_mm_along_lor"] = (
                self._C_MM_PER_PS * float(self.tof["sigma_t_ps"]) / 2.0
            )

    # ----------------------------- Helper methods -----------------------------
    def _apply_psf(self, img: ArrayLike, voxel_size_mm: Tuple) -> ArrayLike:
        """Apply separable Gaussian blur with σ specified in mm (converted to voxels)."""
        if all(s == 0.0 for s in self.psf_sigma_mm):
            return img

        sigma_vox = tuple(max(0.0, s_mm / vs) for s_mm, vs in zip(self.psf_sigma_mm, voxel_size_mm))

        if _HAS_TORCH and isinstance(img, torch.Tensor):
            return self._gaussian_blur_torch(img, sigma_vox)
        else:
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

    # ---------------------------- Torch blur kernel ---------------------------

    def _gaussian_blur_torch(self, x: "torch.Tensor", sigma_vox: Tuple[float, float, float]) -> "torch.Tensor":
        """
        Torch 3D Gaussian via conv3d with a separable 3D kernel (built from 1D gaussians).
        Supports (D,H,W), (C,D,H,W), or (N,C,D,H,W). Reflect-like padding via 'replicate'.
        """
        if x.ndim == 3:
            x_in = x[None, None]  # 1x1xDxHxW
        elif x.ndim == 4:
            # (C,D,H,W) -> (1,C,D,H,W)
            x_in = x[None, ...]
        elif x.ndim == 5:
            x_in = x
        else:
            raise ValueError("Expected torch image dims in {3,4,5}, got {}".format(x.ndim))

        N, C, D, H, W = x_in.shape
        device = x_in.device
        dtype = x_in.dtype

        # Build/cached kernel
        key = (dtype, device, tuple(round(s, 6) for s in sigma_vox))
        weight = self._torch_kernel_cache.get(key, None)
        if weight is None:
            kz, ky, kx = [self._gauss1d_torch(s, dtype, device) for s in sigma_vox]
            # Outer product -> (Kz, Ky, Kx)
            K3 = kz[:, None, None] * ky[None, :, None] * kx[None, None, :]
            K3 = K3 / K3.sum()
            weight = K3[None, None]  # (out_channels=1, in_channels=1, Kz,Ky,Kx)
            self._torch_kernel_cache[key] = weight

        kz, ky, kx = weight.shape[-3:]
        pad = (kx // 2, ky // 2, kz // 2)  # (padW, padH, padD)
        # depthwise conv
        w = weight.to(device=device, dtype=dtype).repeat(C, 1, 1, 1, 1)  # (C,1,Kz,Ky,Kx)
        x_pad = F.pad(x_in, (pad[0], pad[0], pad[1], pad[1], pad[2], pad[2]), mode="replicate")
        y = F.conv3d(x_pad, w, bias=None, stride=1, padding=0, groups=C)

        if x.ndim == 3:
            return y[0, 0]
        elif x.ndim == 4:
            return y[0]
        else:
            return y

    @staticmethod
    def _gauss1d_torch(sigma: float, dtype: "torch.dtype", device: "torch.device") -> "torch.Tensor":
        """
        Make a 1D Gaussian kernel for torch. If sigma==0 -> delta kernel.
        Kernel length = 2*ceil(3*sigma)+1 (covers ~99.7% support).
        """
        if sigma <= 0.0:
            # 1-tap identity
            k = torch.zeros(1, dtype=dtype, device=device)
            k[0] = 1.0
            return k
        radius = int(max(1, np.ceil(3.0 * sigma)))
        xs = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
        k = torch.exp(-0.5 * (xs / sigma) ** 2)
        k /= k.sum()
        return k

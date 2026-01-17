import copy
import os, glob, random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset


DTYPE_MAP = {
    "float32": np.float32, "single": np.float32, "f32": np.float32, "float": np.float32,
    "float64": np.float64, "double": np.float64, "f64": np.float64,
    "float16": np.float16, "half": np.float16, "f16": np.float16,
    "int16": np.int16, "short": np.int16, "i16": np.int16,
    "int32": np.int32, "i32": np.int32,
    "uint8": np.uint8, "u8": np.uint8,
}


def zero_one_to_mone_one(img):
    return (img - 0.5) * 2


def mone_one_to_zero_one(img):
    return (img + 1.) * 0.5
 

def get_pad_margin(N, p):
    """N: unpadded image dimension, p: patch size"""
    return (N // p + 1) * p - N


TRANSFORM_MAP = {
    "zero_one": {
        "normalize": zero_one_to_mone_one,
        "unnormalize": mone_one_to_zero_one,
    },    
}


class RandomCropper():
    def __init__(
        self,
        patch_sizes,
        patch_probs,
    ):
        self.patch_sizes = patch_sizes
        patch_probs = np.asarray(patch_probs, dtype=np.float64) 
        self.patch_probs = patch_probs / np.sum(patch_probs)
        self.idx = 0

    def choose(self, batch_shape):
        self.idx = np.random.choice(len(self.patch_sizes), p=self.patch_probs)
        image_size = list(batch_shape[-3:])
        assert list(self.patch_sizes[0]) == list(image_size) # largest patch size should match
        # lattice offsets
        patch_size = self.patch_sizes[self.idx]
        self.k = random.randrange(0, max(image_size[0]-patch_size[0], 1))
        self.i = random.randrange(0, max(image_size[1]-patch_size[1], 1))
        self.j = random.randrange(0, max(image_size[2]-patch_size[2], 1))
        self.patch_size = patch_size

    def crop(self, batch):
        return batch[..., self.k : self.k + self.patch_size[0], self.i : self.i + self.patch_size[1], self.j : self.j + self.patch_size[2]]


class RandomPaddedPatchDataset3D(Dataset):
    """
    Random 3D patch sampler from float32 .bin volumes using a 'dithered' grid:
      1) Conceptually zero-pad each volume by (pad_d, pad_h, pad_w).
      2) Draw offsets k∈[0,pad_d-1], i∈[0,pad_h-1], j∈[0,pad_w-1] to set the grid.
      3) Enumerate all patch start positions on that grid; sample a random index r.

    - Volumes are memmapped as (D, H, W), dtype float32. Only the needed slice is read.
    - Returns: image patch (1, pD, pH, pW) and position map (3, pD, pH, pW) in [-1, 1].
    """

    def __init__(
        self,
        data_dir: str,
        patch_size: tuple[int, int, int],
        height: int,
        width: int,
        pad_margin: int | tuple[int, int, int] = None, 
        patches_per_volume: int = 1,
        dtype: str = "float32",
        data_normalization=None,
        recursive: bool = False,
        empty_frac_threshold: float = 0.95,       # if >= this fraction of voxels are near background -> “empty”
        empty_thresh: float = 0.05,                # greater than threshold to survive
        max_resample: int = 0,                    # max tries before accepting whatever we got
        augment_shift: tuple[int, int, int] = None,
    ):
        self.data_dir = data_dir
        self.pD, self.pH, self.pW = patch_size
        self.H, self.W = height, width
        self.patches_per_volume = int(patches_per_volume)
        try:
            self.normalize = TRANSFORM_MAP[data_normalization]["normalize"]
            self.unnormalize = TRANSFORM_MAP[data_normalization]["unnormalize"]
        except KeyError:
            raise ValueError(f"Unsupported normalization style '{data_normalization}',. Choose from: {sorted(TRANSFORM_MAP.keys())}")
        try:
            self.dtype = DTYPE_MAP[dtype]
        except KeyError:
            raise ValueError(f"Unsupported dtype '{dtype}'. Choose from: {sorted(DTYPE_MAP.keys())}")

        # Determine pad margins
        if pad_margin is None:
            pad_h, pad_w = map(get_pad_margin, 
                               [self.H,  self.W],
                               [self.pH, self.pW])
        elif isinstance(pad_margin, int):
            pad_d = pad_h = pad_w = int(pad_margin)
        else:
            assert len(pad_margin) == 3
            pad_d, pad_h, pad_w = map(int, pad_margin)

        # Collect *.bin files
        pattern = "**/*.bin" if recursive else "*.bin"
        self.files = sorted(glob.glob(os.path.join(data_dir, pattern), recursive=recursive))
        if not self.files:
            raise FileNotFoundError(f"No .bin files found in: {data_dir}")

        # Infer D per file using filesize; memmaps are created lazily
        self._shapes = []
        self._pad_margins = []
        for f in self.files:
            nbytes = os.path.getsize(f)
            denom = self.H * self.W * np.dtype(self.dtype).itemsize
            if nbytes % denom != 0:
                raise ValueError(f"File size not divisible by H*W*4: {f}")
            D = nbytes // denom
            self._shapes.append((int(D), self.H, self.W))
            if pad_margin is None:
                self._pad_margins.append((int(get_pad_margin(D, self.pD)),
                                        int(pad_h),
                                        int(pad_w)))
            else:
                self._pad_margins.append((pad_d, pad_h, pad_w))


        # rejection controls
        self.empty_frac_threshold = float(empty_frac_threshold)
        self.empty_thresh = float(empty_thresh)
        self.max_resample = int(max_resample)

        self._maps = None  # lazy init per worker

        # shift augmentation
        self.aug_shift = augment_shift

    def _init_memmaps(self):
        self._maps = []
        for f, (D, H, W) in zip(self.files, self._shapes):
            self._maps.append(np.memmap(f, mode='r', dtype=self.dtype, shape=(D, H, W)))

    def __len__(self):
        return len(self.files) * self.patches_per_volume

    # ----------------------
    # Cropping helper method
    # ----------------------
    def _is_empty_patch(self, patch: np.ndarray) -> bool:
        # fraction of voxels near the background_value within tolerance
        frac_near_bg = np.mean(patch <= self.empty_thresh)
        return frac_near_bg >= self.empty_frac_threshold


    def _crop_from_padded(
        self,
        mm: np.memmap,
        shape: tuple[int, int, int],
        start_pad: tuple[int, int, int],
        pad_margin: tuple[int, int, int],
    ) -> np.ndarray:
        """
        Given a memmap `mm` with shape (D,H,W), a padded start coord (d0_pad,h0_pad,w0_pad),
        and patch size (pD,pH,pW) & pad margins (pad_d,pad_h,pad_w), return
        a (pD,pH,pW) patch where out-of-bounds regions are implicitly zero-padded.
        """
        D, H, W = shape
        pD, pH, pW = self.pD, self.pH, self.pW
        pad_d, pad_h, pad_w = pad_margin
        d0_pad, h0_pad, w0_pad = start_pad

        # Map padded to original coordinates
        d0_orig = d0_pad - pad_d
        h0_orig = h0_pad - pad_h
        w0_orig = w0_pad - pad_w

        # Intersection with original volume
        d_src_lo = max(d0_orig, 0)
        h_src_lo = max(h0_orig, 0)
        w_src_lo = max(w0_orig, 0)
        d_src_hi = min(d0_orig + pD, D)
        h_src_hi = min(h0_orig + pH, H)
        w_src_hi = min(w0_orig + pW, W)

        # Allocate output patch; fill zeros by default
        patch = np.zeros((pD, pH, pW), dtype=np.float32)

        # If there is overlap, copy the minimal subvolume
        if (d_src_lo < d_src_hi) and (h_src_lo < h_src_hi) and (w_src_lo < w_src_hi):
            td_lo = d_src_lo - d0_orig
            th_lo = h_src_lo - h0_orig
            tw_lo = w_src_lo - w0_orig
            td_hi = td_lo + (d_src_hi - d_src_lo)
            th_hi = th_lo + (h_src_hi - h_src_lo)
            tw_hi = tw_lo + (w_src_hi - w_src_lo)

            subvol = mm[d_src_lo:d_src_hi, h_src_lo:h_src_hi, w_src_lo:w_src_hi]
            patch[td_lo:td_hi, th_lo:th_hi, tw_lo:tw_hi] = np.ascontiguousarray(subvol)

        return patch

    def _make_pos_map_3d(
        self,
        shape: tuple[int, int, int],
        start_pad: tuple[int, int, int],
        pad_margin: tuple[int, int, int],
    ) -> np.ndarray:
        """
        Build a (3, pD, pH, pW) map for (z, y, x) positions following the paper's 2D logic:
        - Indices are taken in the **padded** coordinate frame (start_pad + local offsets).
        - Normalization uses the **original** (D,H,W), i.e., (idx / (dim-1) - 0.5) * 2.
          This yields values outside [-1,1] for padded regions, matching the 2D code.
        """
        D, H, W = shape
        pD, pH, pW = self.pD, self.pH, self.pW
        d0_pad, h0_pad, w0_pad = start_pad
        pad_d, pad_h, pad_w = pad_margin

        # Avoid division by zero if a dimension equals 1
        den_z = float(max((D + 2*pad_d) - 1, 1))
        den_y = float(max((H + 2*pad_h) - 1, 1))
        den_x = float(max((W + 2*pad_w) - 1, 1))

        # Local indices in the patch, then shift by padded start
        z_idx = (np.arange(pD, dtype=np.int64) + d0_pad).reshape(pD, 1, 1)
        y_idx = (np.arange(pH, dtype=np.int64) + h0_pad).reshape(1, pH, 1)
        x_idx = (np.arange(pW, dtype=np.int64) + w0_pad).reshape(1, 1, pW)

        # Broadcast to full grids
        Z = np.broadcast_to(z_idx, (pD, pH, pW)).astype(np.float32)
        Y = np.broadcast_to(y_idx, (pD, pH, pW)).astype(np.float32)
        X = np.broadcast_to(x_idx, (pD, pH, pW)).astype(np.float32)

        # Normalize to [-1, 1] using original dims (same as author's 2D code)
        Z = (Z / den_z - 0.5) * 2.0
        Y = (Y / den_y - 0.5) * 2.0
        X = (X / den_x - 0.5) * 2.0

        # Concatenate along "channel" axis: (3, pD, pH, pW)
        pos_map = np.stack([Z, Y, X], axis=0)
        return pos_map

    def __getitem__(self, idx):
        if self._maps is None:
            self._init_memmaps()

        # Choose volume for this sample
        vidx = idx // self.patches_per_volume
        mm = self._maps[vidx]
        D, H, W = self._shapes[vidx]

        pD, pH, pW = self.pD, self.pH, self.pW
        pad_d, pad_h, pad_w = self._pad_margins[vidx]

        # Draw random lattice offsets
        k = random.randrange(0, max(pad_d, 1))
        i = random.randrange(0, max(pad_h, 1))
        j = random.randrange(0, max(pad_w, 1))

        # Compute valid starts in the padded frame
        PD, PH, PW = D + 2 * pad_d, H + 2 * pad_h, W + 2 * pad_w
        d_starts = np.arange(k, max(PD - pD + 1, 1), pD, dtype=int)
        h_starts = np.arange(i, max(PH - pH + 1, 1), pH, dtype=int)
        w_starts = np.arange(j, max(PW - pW + 1, 1), pW, dtype=int)

        # Choose random patch index on this grid
        nd, nh, nw = len(d_starts), len(h_starts), len(w_starts)

        for attempt in range(self.max_resample + 1):
            r_d = random.randrange(0, nd)
            r_h = random.randrange(0, nh)
            r_w = random.randrange(0, nw)
            start_pad = (int(d_starts[r_d]), int(h_starts[r_h]), int(w_starts[r_w]))

            # Actual cropping/zero-padding
            patch = self._crop_from_padded(mm, (D, H, W), start_pad, (pad_d, pad_h, pad_w))

            if attempt < self.max_resample and self._is_empty_patch(patch):
                continue
            else:
                break

        # Position map
        if self.aug_shift is not None:
            start_pad_pos_map = [max(min(start_pad[i] + random.randrange(-shift,shift+1), siz-1), 0) 
                                 for i, (shift, siz) in enumerate(zip(self.aug_shift, (PD,PH,PW)))]
        else:
            start_pad_pos_map = start_pad
        pos_map = self._make_pos_map_3d((D, H, W), start_pad_pos_map, (pad_d, pad_h, pad_w))   # (3, pD, pH, pW)

        if self.normalize is not None:
            patch = self.normalize(patch)

        x = torch.from_numpy(patch).unsqueeze(0)                # (1, pD, pH, pW), float32

        pos = torch.from_numpy(pos_map)                         # (3, pD, pH, pW), float32
        return x, pos


class ListPaddedPatchDataset3D(Dataset):
    """
    Random 3D patch sampler from float32 .bin volumes using a 'dithered' grid:
      1) Conceptually zero-pad each volume by (pad_d, pad_h, pad_w).
      2) Draw offsets k∈[0,pad_d-1], i∈[0,pad_h-1], j∈[0,pad_w-1] to set the grid.
      3) Enumerate all patch start positions on that grid; sample a random index r.

    - Volumes are memmapped as (D, H, W), dtype float32. Only the needed slice is read.
    - Returns: image patch (1, pD, pH, pW) and position map (3, pD, pH, pW) in [-1, 1].
    """

    def __init__(
        self,
        padded: torch.Tensor,
        patch_size: tuple[int, int, int],
        pad_margin: int | tuple[int, int, int] = None,
    ):
        self.pD, self.pH, self.pW = map(int, patch_size)
        B, C, self.PD, self.PH, self.PW = map(int, padded.shape)
        assert B == 1 and C == 1
        self.pad_d, self.pad_h, self.pad_w = map(int, pad_margin)
        self.D, self.H, self.W = self.PD-2*self.pad_d, self.PH-2*self.pad_h, self.PW-2*self.pad_w
        # Draw random lattice offsets
        self.k = random.randrange(0, max(self.pad_d, 1))
        self.i = random.randrange(0, max(self.pad_h, 1))
        self.j = random.randrange(0, max(self.pad_w, 1))

        # number of patches
        self.np_d, self.np_h, self.np_w = map(self._get_num_patches, 
                                              [self.D, self.H, self.W],
                                              [self.pD, self.pH, self.pW])
        
        # Store patch start coordinates
        self.coords = []
        for dd in range(self.np_d):
            for hh in range(self.np_h):
                for ww in range(self.np_w):
                    self.coords.append((self.k + dd*self.pD,
                                        self.i + hh*self.pH,
                                        self.j + ww*self.pW,))

        self.device = padded.device
        self.padded = padded.squeeze()

    def __len__(self):
        return self.np_d * self.np_h * self.np_w

    # ----------------------
    # Cropping helper method
    # ----------------------
    def _get_num_patches(self, N, p):
        """N: unpadded image size"""
        return int(N) // int(p) + 1


    def _make_pos_map_3d(
        self,
        start_pad: tuple[int, int, int],
    ) -> np.ndarray:
        """
        Build a (3, pD, pH, pW) map for (z, y, x) positions following the paper's 2D logic:
        - Indices are taken in the **padded** coordinate frame (start_pad + local offsets).
        - Normalization uses the **original** (D,H,W), i.e., (idx / (dim-1) - 0.5) * 2.
          This yields values outside [-1,1] for padded regions, matching the 2D code.
        """
        D, H, W = self.D, self.H, self.W
        pD, pH, pW = self.pD, self.pH, self.pW
        d0_pad, h0_pad, w0_pad = start_pad

        # Avoid division by zero if a dimension equals 1
        den_z = float(max((self.PD) - 1, 1))
        den_y = float(max((self.PH) - 1, 1))
        den_x = float(max((self.PW) - 1, 1))

        # Local indices in the patch, then shift by padded start
        z_idx = (torch.arange(pD, dtype=torch.float32, device=self.device) + d0_pad).reshape(pD, 1, 1)
        y_idx = (torch.arange(pH, dtype=torch.float32, device=self.device) + h0_pad).reshape(1, pH, 1)
        x_idx = (torch.arange(pW, dtype=torch.float32, device=self.device) + w0_pad).reshape(1, 1, pW)

        # Broadcast to full grids
        Z = torch.broadcast_to(z_idx, (pD, pH, pW))
        Y = torch.broadcast_to(y_idx, (pD, pH, pW))
        X = torch.broadcast_to(x_idx, (pD, pH, pW))

        # Normalize to [-1, 1] using original dims (same as author's 2D code)
        Z = (Z / den_z - 0.5) * 2.0
        Y = (Y / den_y - 0.5) * 2.0
        X = (X / den_x - 0.5) * 2.0

        # Concatenate along "channel" axis: (3, pD, pH, pW)
        pos_map = torch.stack([Z, Y, X], axis=0)
        return pos_map

    def __getitem__(self, idx):  
        start_pad = self.coords[idx] # start index in padded image

        patch =  self.padded[start_pad[0] : (start_pad[0] + self.pD),
                             start_pad[1] : (start_pad[1] + self.pH),
                             start_pad[2] : (start_pad[2] + self.pW)].clone().unsqueeze(0)
        
        pos = self._make_pos_map_3d(start_pad)

        return patch, pos
    
    def reconstruct_volume(self, patch, margin_val=None, margin=None):
        """
            patch:  (B,C,D,H,W), patches concatenated along batch dimension
            margin_val: margin value
            margin: same shape padded, with margin
        """
        device = patch.device
        volume = margin_val * torch.ones((1,1,*self.padded.shape), dtype=torch.float32, device=device) if margin is None else margin

        for idx, (dd, hh, ww) in enumerate(self.coords):
            volume[..., dd:dd+self.pD, hh:hh+self.pH, ww:ww+self.pW] = patch[idx].squeeze()

        return volume
    
    def clone_with_image(self, new_image):
        """Clone the dataset with a new image """
        new_ds = object.__new__(type(self))  # avoid __init__
        has_set = False
        for k, v in self.__dict__.items():
            if k == "padded":
                setattr(new_ds, "padded", new_image.squeeze())  # swap, no copy
                has_set = True
            else:
                setattr(new_ds, k, copy.deepcopy(v)) # deep-copy others
        assert has_set, "New image has not been set!"
        return new_ds


class RandomPaddedPatchDataset2D(Dataset):
    """
    Random 3D patch sampler from float32 .bin volumes using a 'dithered' grid:
      1) Conceptually zero-pad each volume by (pad_d, pad_h, pad_w).
      2) Draw offsets k∈[0,pad_d-1], i∈[0,pad_h-1], j∈[0,pad_w-1] to set the grid.
      3) Enumerate all patch start positions on that grid; sample a random index r.

    - Volumes are memmapped as (D, H, W), dtype float32. Only the needed slice is read.
    - Returns: image patch (1, pD, pH, pW) and position map (3, pD, pH, pW) in [-1, 1].
    """

    def __init__(
        self,
        data_dir: str,
        patch_size: tuple[int, int],
        height: int,
        width: int,
        pad_margin: int | tuple[int, int, int] = None, 
        dtype: str = "float32",
        data_normalization=None,
        recursive: bool = False,
        empty_frac_threshold: float = 0.95,       # if >= this fraction of voxels are near background -> “empty”
        empty_thresh: float = 0.05,                # greater than threshold to survive
        max_resample: int = 0,                    # max tries before accepting whatever we got
    ):
        self.data_dir = data_dir
        self.pH, self.pW = patch_size
        self.H, self.W = height, width
        self.patches_per_image = (self.H // self.pH + 1) * (self.W // self.pW + 1)
        try:
            self.normalize = TRANSFORM_MAP[data_normalization]["normalize"]
            self.unnormalize = TRANSFORM_MAP[data_normalization]["unnormalize"]
        except KeyError:
            raise ValueError(f"Unsupported normalization style '{data_normalization}',. Choose from: {sorted(TRANSFORM_MAP.keys())}")
        try:
            self.dtype = DTYPE_MAP[dtype]
        except KeyError:
            raise ValueError(f"Unsupported dtype '{dtype}'. Choose from: {sorted(DTYPE_MAP.keys())}")

        # Determine pad margins
        if pad_margin is None:
            pad_h, pad_w = map(get_pad_margin, 
                              [self.H,  self.W],
                              [self.pH, self.pW])
        elif isinstance(pad_margin, int):
            pad_h = pad_w = int(pad_margin)
        else:
            assert len(pad_margin) == 3
            pad_h, pad_w = map(int, pad_margin)
        self.pad_h = pad_h
        self.pad_w = pad_w

        # Collect *.bin files
        pattern = "**/*.bin" if recursive else "*.bin"
        self.files = sorted(glob.glob(os.path.join(data_dir, pattern), recursive=recursive))
        if not self.files:
            raise FileNotFoundError(f"No .bin files found in: {data_dir}")

        # rejection controls
        self.empty_frac_threshold = float(empty_frac_threshold)
        self.empty_thresh = float(empty_thresh)
        self.max_resample = int(max_resample)

        self._maps = None  # lazy init per worker

    # def _init_memmaps(self):
    #     self._maps = []
    #     for f in self.files:
    #         self._maps.append(np.memmap(f, mode='r', dtype=self.dtype, shape=(self.H, self.W)))

    def __len__(self):
        return len(self.files) * self.patches_per_image

    # ----------------------
    # Cropping helper method
    # ----------------------
    def _is_empty_patch(self, patch: np.ndarray) -> bool:
        # fraction of voxels near the background_value within tolerance
        frac_near_bg = np.mean(patch <= self.empty_thresh)
        return frac_near_bg >= self.empty_frac_threshold


    def _crop_from_padded(
        self,
        mm: np.memmap,
        shape: tuple[int, int, int],
        start_pad: tuple[int, int, int],
        pad_margin: tuple[int, int, int],
    ) -> np.ndarray:
        """
        Given a memmap `mm` with shape (D,H,W), a padded start coord (d0_pad,h0_pad,w0_pad),
        and patch size (pD,pH,pW) & pad margins (pad_d,pad_h,pad_w), return
        a (pD,pH,pW) patch where out-of-bounds regions are implicitly zero-padded.
        """
        H, W = shape
        pH, pW = self.pH, self.pW
        pad_h, pad_w = pad_margin
        h0_pad, w0_pad = start_pad

        # Map padded to original coordinates
        h0_orig = h0_pad - pad_h
        w0_orig = w0_pad - pad_w

        # Intersection with original volume
        h_src_lo = max(h0_orig, 0)
        w_src_lo = max(w0_orig, 0)
        h_src_hi = min(h0_orig + pH, H)
        w_src_hi = min(w0_orig + pW, W)

        # Allocate output patch; fill zeros by default
        patch = np.zeros((pH, pW), dtype=np.float32)

        # If there is overlap, copy the minimal subvolume
        if (h_src_lo < h_src_hi) and (w_src_lo < w_src_hi):
            th_lo = h_src_lo - h0_orig
            tw_lo = w_src_lo - w0_orig
            th_hi = th_lo + (h_src_hi - h_src_lo)
            tw_hi = tw_lo + (w_src_hi - w_src_lo)

            subvol = mm[h_src_lo:h_src_hi, w_src_lo:w_src_hi]
            patch[th_lo:th_hi, tw_lo:tw_hi] = np.ascontiguousarray(subvol)

        return patch

    def _make_pos_map_2d(
        self,
        shape: tuple[int, int],
        start_pad: tuple[int, int],
        pad_margin: tuple[int, int],
    ) -> np.ndarray:
        """
        Build a (3, pD, pH, pW) map for (z, y, x) positions following the paper's 2D logic:
        - Indices are taken in the **padded** coordinate frame (start_pad + local offsets).
        - Normalization uses the **original** (D,H,W), i.e., (idx / (dim-1) - 0.5) * 2.
          This yields values outside [-1,1] for padded regions, matching the 2D code.
        """
        H, W = shape
        pH, pW = self.pH, self.pW
        h0_pad, w0_pad = start_pad
        pad_h, pad_w = pad_margin

        # Avoid division by zero if a dimension equals 1
        den_y = float(max((H + 2*pad_h) - 1, 1))
        den_x = float(max((W + 2*pad_w) - 1, 1))

        # Local indices in the patch, then shift by padded start
        y_idx = (np.arange(pH, dtype=np.int64) + h0_pad).reshape(pH, 1)
        x_idx = (np.arange(pW, dtype=np.int64) + w0_pad).reshape(1, pW)

        # Broadcast to full grids
        Y = np.broadcast_to(y_idx, (pH, pW)).astype(np.float32)
        X = np.broadcast_to(x_idx, (pH, pW)).astype(np.float32)

        # Normalize to [-1, 1] using original dims (same as author's 2D code)
        Y = (Y / den_y - 0.5) * 2.0
        X = (X / den_x - 0.5) * 2.0

        # Concatenate along "channel" axis: (3, pD, pH, pW)
        pos_map = np.stack([Y, X], axis=0)
        return pos_map

    def __getitem__(self, idx):
        # Choose volume for this sample
        vidx = idx // self.patches_per_image
        mm = np.fromfile(self.files[vidx], dtype=self.dtype).reshape((self.H, self.W))
        H, W = self.H, self.W

        pH, pW = self.pH, self.pW
        pad_h, pad_w = self.pad_h, self.pad_w

        # Draw random lattice offsets
        i = random.randrange(0, max(pad_h, 1))
        j = random.randrange(0, max(pad_w, 1))

        # Compute valid starts in the padded frame
        PH, PW = H + 2 * pad_h, W + 2 * pad_w
        h_starts = np.arange(i, max(PH - pH + 1, 1), pH, dtype=int)
        w_starts = np.arange(j, max(PW - pW + 1, 1), pW, dtype=int)

        # Choose random patch index on this grid
        nh, nw = len(h_starts), len(w_starts)

        for attempt in range(self.max_resample + 1):
            r_h = random.randrange(0, nh)
            r_w = random.randrange(0, nw)
            start_pad = (int(h_starts[r_h]), int(w_starts[r_w]))

            # Actual cropping/zero-padding
            patch = self._crop_from_padded(mm, (H, W), start_pad, (pad_h, pad_w))

            if attempt < self.max_resample and self._is_empty_patch(patch):
                continue
            else:
                break

        # Position map
        pos_map = self._make_pos_map_2d((H, W), start_pad, (pad_h, pad_w))   # (2, pH, pW)

        if self.normalize is not None:
            patch = self.normalize(patch)

        x = torch.from_numpy(patch).unsqueeze(0)                # (1, pH, pW), float32

        pos = torch.from_numpy(pos_map)                         # (2, pH, pW), float32
        return x, pos
    

class ListPaddedPatchDataset2D(Dataset):
    """
    Random 3D patch sampler from float32 .bin volumes using a 'dithered' grid:
      1) Conceptually zero-pad each volume by (pad_d, pad_h, pad_w).
      2) Draw offsets k∈[0,pad_d-1], i∈[0,pad_h-1], j∈[0,pad_w-1] to set the grid.
      3) Enumerate all patch start positions on that grid; sample a random index r.

    - Volumes are memmapped as (D, H, W), dtype float32. Only the needed slice is read.
    - Returns: image patch (1, pD, pH, pW) and position map (3, pD, pH, pW) in [-1, 1].
    """

    def __init__(
        self,
        padded_batch: torch.Tensor,
        patch_size: tuple[int, int, int],
        pad_margin: int | tuple[int, int, int] = None,
    ):
        self.pH, self.pW = map(int, patch_size)
        self.B, C, self.PH, self.PW = map(int, padded_batch.shape)
        assert C == 1
        self.pad_h, self.pad_w = map(int, pad_margin)
        self.H, self.W = self.PH-2*self.pad_h, self.PW-2*self.pad_w
        # Draw random lattice offsets
        self.i = random.randrange(0, max(self.pad_h, 1))
        self.j = random.randrange(0, max(self.pad_w, 1))

        # number of patches
        self.np_h, self.np_w = map(self._get_num_patches, 
                                  [self.H, self.W],
                                  [self.pH, self.pW])
        
        
        # Store patch start coordinates
        self.coords = []
        for hh in range(self.np_h):
            for ww in range(self.np_w):
                self.coords.append((self.i + hh*self.pH,
                                    self.j + ww*self.pW,))

        self.device = padded_batch.device
        self.padded = padded_batch.squeeze()

    def __len__(self):
        return self.B * self.np_h * self.np_w

    # ----------------------
    # Cropping helper method
    # ----------------------
    def _get_num_patches(self, N, p):
        """N: unpadded image size"""
        return int(N) // int(p) + 1


    def _make_pos_map_2d(
        self,
        start_pad: tuple[int, int],
    ) -> np.ndarray:
        """
        Build a (3, pD, pH, pW) map for (z, y, x) positions following the paper's 2D logic:
        - Indices are taken in the **padded** coordinate frame (start_pad + local offsets).
        - Normalization uses the **original** (D,H,W), i.e., (idx / (dim-1) - 0.5) * 2.
          This yields values outside [-1,1] for padded regions, matching the 2D code.
        """
        H, W = self.H, self.W
        pH, pW = self.pH, self.pW
        h0_pad, w0_pad = start_pad

        # Avoid division by zero if a dimension equals 1
        den_y = float(max((self.PH) - 1, 1))
        den_x = float(max((self.PW) - 1, 1))

        # Local indices in the patch, then shift by padded start
        y_idx = (torch.arange(pH, dtype=torch.float32, device=self.device) + h0_pad).reshape(pH, 1)
        x_idx = (torch.arange(pW, dtype=torch.float32, device=self.device) + w0_pad).reshape(1, pW)

        # Broadcast to full grids
        Y = torch.broadcast_to(y_idx, (pH, pW))
        X = torch.broadcast_to(x_idx, (pH, pW))

        # Normalize to [-1, 1] using original dims (same as author's 2D code)
        Y = (Y / den_y - 0.5) * 2.0
        X = (X / den_x - 0.5) * 2.0

        # Concatenate along "channel" axis: (3, pD, pH, pW)
        pos_map = torch.stack([Y, X], axis=0)
        return pos_map

    def __getitem__(self, idx):  
        idx_image = idx // (self.np_h * self.np_w)
        idx_patch = idx % (self.np_h * self.np_w)
        start_pad = self.coords[idx_patch] # start index in padded image

        patch =  self.padded[idx_image,
                             start_pad[0] : (start_pad[0] + self.pH),
                             start_pad[1] : (start_pad[1] + self.pW)].clone().unsqueeze(0)
        
        pos = self._make_pos_map_2d(start_pad)

        return patch, pos
    
    def reconstruct_image(self, patch, margin_val=None, margin=None):
        """
            patch:  (B,C,D,H,W), patches concatenated along batch dimension
            margin_val: margin value
            margin: same shape padded, with margin
        """
        device = patch.device
        batched_image = margin_val * torch.ones((self.B,1,*self.padded.shape[-2:]), dtype=torch.float32, device=device) if margin is None else margin

        for bb in range(self.B):
            for idx, (hh, ww) in enumerate(self.coords):
                batched_image[bb, :, hh:hh+self.pH, ww:ww+self.pW] = patch[bb * self.np_h * self.np_w + idx].squeeze()

        return batched_image
    

class NumpyImageDataset(Dataset):
    def __init__(self, folder_path, data_normalization="zero_one"):
        """
        Args:
            folder_path (str): Path to the folder containing .npy files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.folder_path = folder_path
        self.file_list = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
        try:
            self.normalize = TRANSFORM_MAP[data_normalization]["normalize"]
            self.unnormalize = TRANSFORM_MAP[data_normalization]["unnormalize"]
        except KeyError:
            raise ValueError(f"Unsupported normalization style '{data_normalization}',. Choose from: {sorted(TRANSFORM_MAP.keys())}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx, percentile=99.999):
        file_path = os.path.join(self.folder_path, self.file_list[idx])
        image = np.load(file_path).astype(np.float32)  # Ensure float32 for PyTorch
        
        # Normalize to [0, 1]
        image = (image - np.min(image)) / (np.percentile(image, percentile) - np.min(image))

        # Convert to tensor
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        if self.normalize:
            image = self.normalize(image)

        return image

def plot_image_grid(images,
                    n,
                    cmap='gray',
                    vmin=None,
                    vmax=None,
                    figsize=None,
                    dpi=300,
                    save_path=None,
                    save_kwargs=None):
    """
    Splice a list of grayscale images into a single n x n grid and plot with one colorbar.
    Optionally save the result to a PNG file.

    Parameters:
    - images: list or numpy array of shape (n*n, H, W)
    - n: grid dimension (number of rows and columns)
    - cmap: colormap for display (default 'gray')
    - figsize: tuple for figure size (default: (4*n, 4*n))
    - dpi: resolution in dots per inch (default 300)
    - save_path: str or None; if given, save figure to this path (should end in '.png')
    - save_kwargs: dict of additional keyword args to pass to fig.savefig()

    Returns:
    - fig: matplotlib Figure
    - ax: matplotlib Axes
    """
    images = np.array(images)
    if images.ndim != 3 or images.shape[0] != n*n:
        raise ValueError(f"Expected images of shape (n*n, H, W), got {images.shape}")

    # Build the grid
    rows = []
    for i in range(n):
        row_imgs = [images[i*n + j] for j in range(n)]
        rows.append(np.hstack(row_imgs))
    grid = np.vstack(rows)

    # Determine vmin/vmax
    vmin = grid.min() if vmin is None else vmin
    vmax = grid.max() if vmax is None else vmax

    # Plot
    fig, ax = plt.subplots(figsize=figsize or (4*n, 4*n), dpi=dpi)
    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=16)

    # Save if requested
    if save_path:
        save_kwargs = save_kwargs or {}
        # force PNG extension if not provided
        if not save_path.lower().endswith('.png'):
            save_path += '.png'
        fig.savefig(save_path, dpi=dpi, **save_kwargs)

    return fig, ax

def save_gray_with_colorbar(image, filename, vmin=None, vmax=None, cmap='gray', dpi=300):
    """
    Save a grayscale image with a colorbar and optional intensity window.

    Parameters:
        image (np.ndarray): 2D numpy array representing the grayscale image.
        filename (str): Output filename (e.g., 'output.png').
        vmin (int or float): Minimum intensity value for display. If None, uses image min.
        vmax (int or float): Maximum intensity value for display. If None, uses image max.
        cmap (str): Colormap to use (default is 'gray').
        dpi (int): Resolution of saved image.
    """
    if vmin is None:
        vmin = np.min(image)
    if vmax is None:
        vmax = np.max(image)

    plt.figure(figsize=(6, 5))
    img_plot = plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(img_plot)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()

def show_orthogonal_views(volume, slice_indices=None, cmap='gray', vmin=None, vmax=None, dpi=300, save_path=None):
    """
    Display three orthogonal slices (axial, coronal, sagittal) from a 3D volume,
    each in its own figure with an individual colorbar.

    Parameters:
        volume (np.ndarray): 3D numpy array with shape (depth, height, width).
        slice_indices (tuple of ints, optional): (z, y, x) indices for slices.
            Defaults to the center slices if None.
        cmap (str, optional): Colormap for display. Defaults to 'gray'.
    """
    # Determine slice indices
    if slice_indices is None:
        z, y, x = [dim // 2 for dim in volume.shape]
    else:
        z, y, x = slice_indices

    # Extract slices
    axial    = volume[z,    :, :]
    coronal  = volume[:,    y,    :]
    sagittal = volume[:,    :,    x]

    # Prepare views
    views = [
        ('Axial',    axial,    'Z', z),
        ('Coronal',  coronal,  'Y', y),
        ('Sagittal', sagittal, 'X', x)
    ]
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=dpi)
    for ax, (name, img, axis_label, idx) in zip(axes, views):
        lower = img.min() if vmin is None else vmin
        upper = img.max() if vmax is None else vmax
        im = ax.imshow(img, cmap=cmap, vmin=lower, vmax=upper)
        ax.set_title(f'{name} ({axis_label}={idx})')
        ax.axis('off')
        # Add an individual colorbar to each subplot
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if save_path:
        # force PNG extension if not provided
        if not save_path.lower().endswith('.png'):
            save_path += '.png'
        fig.savefig(save_path, dpi=dpi)


def get_cond_schedule(num_steps, **kwargs):
    """
        Get a logical array specifying at which step conditional projection is taken
    """
    if kwargs['name'] == 'linear':
        t_start = kwargs.get('t_start', num_steps-1) # start from t = T-1
        t_end = kwargs.get('t_end', 0) # end with t = 0
        intv_start = kwargs['interval_start']
        intv_end = kwargs['interval_end']

        schedule = [] 
        t_last = num_steps
        for t in reversed(range(num_steps)):
            intv_moving = round(intv_end + (t - t_end) / (t_start-t_end) * (intv_start - intv_end)) 
            
            if t <= t_start and t >= t_end:
                if t_last - t >= intv_moving or t == t_end or t == t_start:
                    schedule.append(1)
                    t_last = t
                    continue
            
            schedule.append(0)

    elif kwargs['name'] == 'subiter':
        schedule = []
        
        for t in reversed(range(num_steps)):
            schedule.append(kwargs['subiter'])

    return list(reversed(schedule))


def psnr(pred: torch.Tensor,
         target: torch.Tensor,
         data_range: float,
         reduction: str = 'mean',
         eps: float = 1e-8) -> torch.Tensor:
    """
    Compute PSNR (Peak Signal-to-Noise Ratio) between two batches of images.

    Args:
        pred (torch.Tensor): Predicted images, shape (B, C, ...).
        target (torch.Tensor): Reference images, same shape as `pred`.
        data_range (float): Maximum possible pixel value range (default=1.0 for images in [0,1]).
        reduction (str): 'mean' to return a single scalar (average over batch),
                         'none' to return a tensor of shape (B,) with per-image PSNR.
        eps (float): Small value to avoid log(0).

    Returns:
        torch.Tensor: PSNR in decibels. Shape is () if `reduction='mean'`, or (B,) if `reduction='none'`.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape, got {pred.shape} vs {target.shape}")

    # Compute per-image MSE over all but the batch dimension
    mse = torch.mean((pred - target) ** 2, dim=tuple(range(1, pred.ndim)))

    # PSNR = 10 * log10( (data_range^2) / mse )
    psnr_vals = 10.0 * torch.log10((data_range ** 2) / (mse + eps))

    if reduction == 'mean':
        return psnr_vals.mean()
    elif reduction == 'none':
        return psnr_vals
    else:
        raise ValueError(f"reduction must be 'mean' or 'none', got {reduction}")


def _gaussian_window(window_size: int, sigma: float, channel: int, dims: int, device, dtype):
    """
    Create a 2D or 3D Gaussian window for SSIM.
    """
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g1d = torch.exp(-(coords**2) / (2 * sigma**2))
    g1d /= g1d.sum()

    if dims == 2:
        # outer product for 2D
        kernel = g1d[:, None] * g1d[None, :]
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1,1,WS,WS]
    elif dims == 3:
        # outer product for 3D
        k2 = g1d[:, None] * g1d[None, :]
        kernel = (g1d.view(-1,1,1) * k2.view(1,*k2.shape)).unsqueeze(0).unsqueeze(0)  # [1,1,WS,WS,WS]
    else:
        raise ValueError(f"Unsupported dims={dims}")

    # expand to [C,1,...,WS,...]
    kernel = kernel.expand(channel, 1, *kernel.shape[-dims:])
    return kernel

import torch.nn.functional as F

def ssim(pred: torch.Tensor,
         target: torch.Tensor,
         data_range: float,
         window_size: int = 11,
         sigma: float = 1.5,
         reduction: str = 'mean',
         eps: float = 1e-8) -> torch.Tensor:
    """
    Compute SSIM (Structural Similarity) between two batches of images/volumes.

    Args:
        pred (torch.Tensor): Predicted images, shape (B, C, H, W) or (B, C, D, H, W).
        target (torch.Tensor): Reference images, same shape as `pred`.
        data_range (float): Value range of input (e.g., 1.0, or HU window width).
        window_size (int): Size of the Gaussian kernel (default=11).
        sigma (float): Standard deviation of the Gaussian kernel (default=1.5).
        reduction (str): 'mean' returns a single scalar (batch-mean SSIM),
                         'none' returns a tensor of shape (B,) with per-sample SSIM.
        eps (float): Small constant to avoid division by zero.

    Returns:
        torch.Tensor: SSIM values in [–1,1]. Shape is () if `reduction='mean'`, or (B,) if `reduction='none'`.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape, got {pred.shape} vs {target.shape}")
    if pred.ndim not in (4, 5):
        raise ValueError("pred/target must be 4D (B,C,H,W) or 5D (B,C,D,H,W) tensors")

    # choose between conv2d/conv3d
    dims = pred.ndim - 2
    conv = F.conv2d if dims == 2 else F.conv3d
    padding = window_size // 2

    B, C = pred.shape[:2]
    device, dtype = pred.device, pred.dtype

    # make Gaussian window
    window = _gaussian_window(window_size, sigma, C, dims, device, dtype)

    # compute local means
    mu1 = conv(pred,   window, padding=padding, groups=C)
    mu2 = conv(target, window, padding=padding, groups=C)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # compute local variances / covariances
    sigma1_sq = conv(pred * pred,     window, padding=padding, groups=C) - mu1_sq
    sigma2_sq = conv(target * target, window, padding=padding, groups=C) - mu2_sq
    sigma12   = conv(pred * target,   window, padding=padding, groups=C) - mu1_mu2

    # constants for stability (per original SSIM paper)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) \
               / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + eps)

    # average over spatial (and channel) dims to get per-sample SSIM
    ssim_per_sample = ssim_map.view(B, -1).mean(dim=1)

    if reduction == 'mean':
        return ssim_per_sample.mean()
    elif reduction == 'none':
        return ssim_per_sample
    else:
        raise ValueError(f"reduction must be 'mean' or 'none', got {reduction}")


def get_nrmse(pred: np.ndarray, target: np.ndarray, norm_type: str = 'minmax') -> float:
    """
    Compute the Normalized Root Mean Squared Error (NRMSE) between prediction and target.

    Parameters:
    - pred: np.ndarray, predicted image or data
    - target: np.ndarray, ground truth image or data
    - norm_type: str, normalization type, 'minmax' or 'mean'
    
    Returns:
    - nrmse: float, normalized RMSE
    """
    pred = np.asarray(pred).astype(np.float32)
    target = np.asarray(target).astype(np.float32)
    
    mse = np.mean((pred - target) ** 2)
    rmse = np.sqrt(mse)

    if norm_type == 'minmax':
        norm = target.max() - target.min()
    elif norm_type == 'mean':
        norm = np.mean(target)
    else:
        raise ValueError("norm_type must be 'minmax' or 'mean'")
    
    if norm == 0:
        raise ValueError("Normalization factor is zero. Cannot compute NRMSE.")

    return rmse / norm


if __name__ == "__main__":
    # Test dataset
    dataset = RandomPaddedPatchDataset3D(
        "/home/raid7/bybhuang/UDPET/Bern-Inselspital-2022_3d/data/",
        patch_size=(128,144,144),
        height=144,
        width=144,
        patches_per_volume=1,
        dtype="float32",
        transform="zero_one_to_mone_one",
    )

    patch = dataset[10]
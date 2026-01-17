import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from typing import Optional, Tuple, Union, Literal

Array = np.ndarray


def _as_tuple(v: Union[int, Tuple[int, ...]], ndim: int) -> Tuple[int, ...]:
    if isinstance(v, int):
        return (v,) * ndim
    assert len(v) == ndim
    return tuple(int(x) for x in v)


def _make_inv_distance_kernel(patch_size: Tuple[int, ...], eps: float = 1e-12) -> Array:
    """
    Build h_l in Eq. (10): normalized inverse spatial distance to the patch center. :contentReference[oaicite:6]{index=6}
    """
    ndim = len(patch_size)
    assert all(s % 2 == 1 for s in patch_size), "patch_size must be odd in every dimension"
    radii = [s // 2 for s in patch_size]

    grids = np.meshgrid(*[np.arange(-r, r + 1) for r in radii], indexing="ij")
    dist2 = np.zeros_like(grids[0], dtype=np.float64)
    for g in grids:
        dist2 += g.astype(np.float64) ** 2
    dist = np.sqrt(dist2)

    h = 1.0 / (dist + 1.0)  # center gets weight 1
    h = h / (h.sum() + eps)
    return h.astype(np.float64)


# def _conv_same_nd(x: Array, k: Array) -> Array:
#     """
#     N-D zero-padded "same" convolution for small kernels, via sliding_window_view.
#     x: shape (...), k: shape (k1,k2[,k3])
#     """
#     ndim = x.ndim
#     assert k.ndim == ndim
#     pad = [(s // 2, s // 2) for s in k.shape]
#     xp = np.pad(x, pad_width=pad, mode="constant", constant_values=0.0)

#     win = sliding_window_view(xp, k.shape)  # shape = x.shape + k.shape
#     # sum_{u} win[..., u] * k[u]
#     axes_win = tuple(range(ndim, 2 * ndim))
#     axes_k = tuple(range(ndim))
#     return np.tensordot(win, k, axes=(axes_win, axes_k))


def _conv_same_nd(
    x: Array,
    k: Array,
    backend: Literal["numpy", "torch"] = "numpy",
    device: str = "cuda",
    dtype: Optional[str] = None,
) -> Array:
    """
    N-D zero-padded "same" convolution for small kernels.

    Parameters
    ----------
    x : np.ndarray
        Input array of shape (H,W) or (D,H,W).
    k : np.ndarray
        Kernel of shape (kh,kw) or (kd,kh,kw). Should be small and odd-sized.
    backend : {"numpy","torch"}
        - "numpy": sliding_window_view + tensordot (original).
        - "torch": uses torch conv2d/conv3d (can run on GPU).
    device : str
        "cuda" or "cpu" (only used when backend="torch").
    dtype : Optional[str]
        If given, cast x and k to this dtype before convolution (e.g., "float32").

    Returns
    -------
    y : np.ndarray
        Convolution result with the same shape as x.
    """
    ndim = x.ndim
    assert ndim in (2, 3), "x must be 2D (H,W) or 3D (D,H,W)"
    assert k.ndim == ndim, "kernel ndim must match x ndim"
    assert all(s % 2 == 1 for s in k.shape), "kernel sizes must be odd for 'same' padding"

    if dtype is not None:
        x = x.astype(dtype, copy=False)
        k = k.astype(dtype, copy=False)

    if backend == "numpy":
        pad = [(s // 2, s // 2) for s in k.shape]
        xp = np.pad(x, pad_width=pad, mode="constant", constant_values=0.0)
        win = sliding_window_view(xp, k.shape)  # shape = x.shape + k.shape
        axes_win = tuple(range(ndim, 2 * ndim))
        axes_k = tuple(range(ndim))
        return np.tensordot(win, k, axes=(axes_win, axes_k))

    if backend == "torch":
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as e:
            raise ImportError("backend='torch' requires PyTorch installed.") from e

        # Choose a conv-friendly dtype (float32 is usually fastest on GPU)
        # If x/k are float64, you *can* keep float64, but it'll be slower on many GPUs.
        x_t = torch.from_numpy(np.ascontiguousarray(x))
        k_t = torch.from_numpy(np.ascontiguousarray(k))

        # Ensure floating type
        if not torch.is_floating_point(x_t):
            x_t = x_t.float()
        if not torch.is_floating_point(k_t):
            k_t = k_t.float()

        x_t = x_t.to(device)
        k_t = k_t.to(device)

        # Build weight in the shape conv expects:
        # conv2d: (out_channels=1, in_channels=1, kh, kw)
        # conv3d: (out_channels=1, in_channels=1, kd, kh, kw)
        if ndim == 2:
            # x: (H,W) -> (N=1,C=1,H,W)
            x_in = x_t[None, None, :, :]
            w = k_t[None, None, :, :]
            padding = (k.shape[0] // 2, k.shape[1] // 2)
            y = F.conv2d(x_in, w, bias=None, stride=1, padding=padding)
            y = y[0, 0, :, :]

        else:  # ndim == 3
            # x: (D,H,W) -> (N=1,C=1,D,H,W)
            x_in = x_t[None, None, :, :, :]
            w = k_t[None, None, :, :, :]
            padding = (k.shape[0] // 2, k.shape[1] // 2, k.shape[2] // 2)
            y = F.conv3d(x_in, w, bias=None, stride=1, padding=padding)
            y = y[0, 0, :, :, :]

        return y.detach().cpu().numpy()

    raise ValueError(f"Unknown backend: {backend}")


def _shift_zero(x: Array, offset: Tuple[int, ...]) -> Array:
    """
    Shift x by offset with zero fill (no wrap).
    offset: (dy,dx) for 2D or (dz,dy,dx) for 3D
    """
    out = np.zeros_like(x)
    src_slices = []
    dst_slices = []
    for dim, s in enumerate(offset):
        n = x.shape[dim]
        if s >= 0:
            src = slice(0, n - s)
            dst = slice(s, n)
        else:
            src = slice(-s, n)
            dst = slice(0, n + s)
        src_slices.append(src)
        dst_slices.append(dst)
    out[tuple(dst_slices)] = x[tuple(src_slices)]
    return out


def _neighbor_offsets(neigh_size: Tuple[int, ...]) -> Tuple[Tuple[int, ...], ...]:
    """
    All offsets in a neigh_size window excluding the zero offset.
    neigh_size: (W,W) or (W,W,W), each odd.
    """
    ndim = len(neigh_size)
    assert all(s % 2 == 1 for s in neigh_size), "neigh_size must be odd in every dimension"
    radii = [s // 2 for s in neigh_size]

    # Cartesian product of ranges
    grids = np.meshgrid(*[np.arange(-r, r + 1) for r in radii], indexing="ij")
    offs = np.stack([g.reshape(-1) for g in grids], axis=1)  # (N, ndim)
    out = []
    for o in offs:
        t = tuple(int(v) for v in o)
        if any(v != 0 for v in t):
            out.append(t)
    return tuple(out)


def penalized_ot_lange_step(
    x_n: Array,
    x_em: Array,
    beta: float,
    delta: float,
    p: Optional[Array] = None,
    patch_size: Union[int, Tuple[int, ...]] = 3,
    neigh_size: Union[int, Tuple[int, ...]] = 3,
    backend: Literal["numpy", "torch"] = "numpy",
    eps: float = 1e-12,
    renormalize_edges: bool = True,
) -> Tuple[Array, Array]:
    """
    Algorithm 1 (line 5 & 6):
      - Image smoothing: compute x_reg via Eq. (38) using w_jk(x^n) from Eq. (34). :contentReference[oaicite:7]{index=7}
      - Pixel-by-pixel fusion: compute x^{n+1} via Eq. (45) with beta_j^n in Eq. (44). :contentReference[oaicite:8]{index=8}

    Inputs:
      x_n   : current image x^n, shape (H,W) or (D,H,W)
      x_em  : unregularized update x_em^{n+1} (you already computed this)
      beta  : regularization strength (paper's beta)
      delta : Lange hyper-parameter in psi (Eq. (7)); only w^psi(t)=1/(delta+t) is needed. :contentReference[oaicite:9]{index=9} :contentReference[oaicite:10]{index=10}
      p     : sensitivity image p_j (Eq. (23)); if None, use ones
    """
    assert x_n.shape == x_em.shape
    assert x_n.ndim in (2, 3), "x_n must be 2D (H,W) or 3D (D,H,W)"
    ndim = x_n.ndim

    patch_size = _as_tuple(patch_size, ndim)
    neigh_size = _as_tuple(neigh_size, ndim)

    x_n = x_n.astype(np.float64, copy=False)
    x_em = x_em.astype(np.float64, copy=False)
    p = np.ones_like(x_n, dtype=np.float64) if p is None else p.astype(np.float64, copy=False)

    # h_l in Eq. (10) and used as the spatial averaging weights in Eq. (34). :contentReference[oaicite:11]{index=11} :contentReference[oaicite:12]{index=12}
    h = _make_inv_distance_kernel(patch_size, eps=eps)

    ones = np.ones_like(x_n, dtype=np.float64)

    w_sum = np.zeros_like(x_n, dtype=np.float64)  # w_j^n = sum_k w_jk (Eq. (37)) :contentReference[oaicite:13]{index=13}
    num = np.zeros_like(x_n, dtype=np.float64)    # numerator in Eq. (38) :contentReference[oaicite:14]{index=14}

    for off in _neighbor_offsets(neigh_size):
        # x_k aligned at j: x_k[j] = x_n[j+off]
        x_k = _shift_zero(x_n, off)

        # valid center positions where k=j+off exists
        valid_center = _shift_zero(ones, off)

        # ---- Patch distance d(j,k) (Eq. (9)) computed efficiently:
        # For fixed off, (x_{j+u} - x_{k+u}) = (x - shift(x, off)) evaluated at (j+u).
        # So d^2(j,k) = sum_u h(u) * (diff_off(j+u))^2 = conv_same(diff_off^2, h) at j. :contentReference[oaicite:15]{index=15}
        diff = x_n - x_k
        diff2 = diff * diff

        if renormalize_edges:
            denom_d = _conv_same_nd(valid_center, h, backend=backend)
            d2 = _conv_same_nd(diff2 * valid_center, h, backend=backend) / (denom_d + eps)
        else:
            d2 = _conv_same_nd(diff2, h, backend=backend)

        d = np.sqrt(d2 + eps)

        # Lange: psi(t)=delta(|t|/delta - log(1+|t|/delta)) (Eq. (7)),
        # and w^psi(t)=psi'(t)/t = 1/(delta + t) for t>=0 (Eq. (12)). :contentReference[oaicite:16]{index=16} :contentReference[oaicite:17]{index=17}
        wpsi = 1.0 / (delta + d)

        # ---- Patch OT weight w_jk(x^n) in Eq. (34): spatial average of wpsi over patch positions. :contentReference[oaicite:18]{index=18}
        if renormalize_edges:
            denom_w = _conv_same_nd(valid_center, h, backend=backend)
            wjk = _conv_same_nd(wpsi * valid_center, h, backend=backend) / (denom_w + eps)
        else:
            wjk = _conv_same_nd(wpsi, h, backend=backend) * valid_center  # still zero out invalid centers

        w_sum += wjk
        num += wjk * (x_k + x_n)

    # Eq. (38): x_reg = (1/(2 w_j^n)) * sum_k w_jk(x^n) * (x_k^n + x_j^n). :contentReference[oaicite:19]{index=19}
    x_reg = num / (2.0 * w_sum + eps)

    # Eq. (44): beta_j^n = beta * w_j^n / p_j. :contentReference[oaicite:20]{index=20}
    beta_j = (beta * w_sum) / (p + eps)

    # Eq. (45): x^{n+1} = 2*x_em / ( sqrt((1 - beta_j*x_reg)^2 + 4*beta_j*x_em) + (1 - beta_j*x_reg) ). :contentReference[oaicite:21]{index=21}
    t = 1.0 - beta_j * x_reg
    rad = np.sqrt(t * t + 4.0 * beta_j * x_em + eps)
    x_new = (2.0 * x_em) / (rad + t + eps)

    # Nonnegativity (consistent with PET activity)
    x_new = np.maximum(x_new, 0.0)

    return x_reg, x_new

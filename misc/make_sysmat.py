import time
import numpy as np

from scipy.sparse import coo_matrix, save_npz
import scipy.io as sio

import parallelproj
from parallelproj import Array
import array_api_compat.numpy as xp
from scipy.spatial import cKDTree

def get_geo_projector(scanner, img_shape, voxel_size, fine_tofbin=True):

    # choose a device (CPU or CUDA GPU)
    if "numpy" in xp.__name__:
        # using numpy, device must be cpu
        dev = "cpu"
    elif "cupy" in xp.__name__:
        # # using cupy, only cuda devices are possible
        # dev = xp.cuda.Device(0)
        raise ValueError('cupy not supported yet!')
    elif "torch" in xp.__name__:
        # # using torch valid choices are 'cpu' or 'cuda'
        # if parallelproj.cuda_present:
        #     dev = "cuda"
        # else:
        #     dev = "cpu"
        raise ValueError('torch not supported yet!')

    if scanner == 'nx-2d':
        radius=271.75
        num_sides=20
        num_lor_endpoints_per_side=48
        lor_spacing=1.6648
        ring_positions=xp.array([0])
        symmetry_axis=0
        phis = (
            2
            * xp.pi
            * xp.arange(num_sides, dtype=xp.float32, device=dev)
            / num_sides
            - 0.5 * xp.pi   # offset for orientation of image
        )
        
        radial_trim=240
        max_ring_difference=None

        if fine_tofbin:
            num_tofbins=201
            tofbin_width_ps=12.2
        else:
            num_tofbins=51
            tofbin_width_ps=50
        fwhm_ps=240

        ipsf_fwhm=(1.6, 0.8, 0.8)

    elif scanner == 'nx-3d':
        # scanner geometry
        radius=271.75
        num_sides=20
        num_lor_endpoints_per_side=48
        lor_spacing=1.6648 # from config file, assuming equal interval
        ring_positions=np.fromfile('/home/raid7/bybhuang/diffusion/diffusion-posterior-sampling/util/xtal_pos_z', dtype=np.float32)
        symmetry_axis=0
        phis = (
            2
            * xp.pi
            * xp.arange(num_sides, dtype=xp.float32, device=dev)
            / num_sides
            # - 0.5 * xp.pi   # offset for orientation of image
        )
        # LOR
        radial_trim=240
        max_ring_difference=None
        
        num_tofbins=201
        tofbin_width_ps=12.2
        fwhm_ps=240

        ipsf_fwhm=(1.6, 0.8, 0.8)

    else:
        raise NotImplementedError("Scanner not implemented!")
    
    scanner = parallelproj.RegularPolygonPETScannerGeometry(
        xp,
        dev,
        radius=radius,
        num_sides=num_sides,
        num_lor_endpoints_per_side=num_lor_endpoints_per_side,
        lor_spacing=lor_spacing,
        ring_positions=ring_positions,
        symmetry_axis=symmetry_axis,
        phis=phis
    )

    lor_desc = parallelproj.RegularPolygonPETLORDescriptor(
        scanner,
        radial_trim=radial_trim,
        max_ring_difference=max_ring_difference,
        sinogram_order=parallelproj.SinogramSpatialAxisOrder.RVP,
    )

    proj = parallelproj.RegularPolygonPETProjector(
        lor_desc, img_shape=img_shape, voxel_size=voxel_size
    )

    # tofbin_ps[ns] * (speed of light / 2) [mm/ns]
    tofbin_width = tofbin_width_ps * 0.001 * 299.792 / 2
    sigma_tof = (299.792 / 2) * (
        fwhm_ps / 1000 / 2.355
    )  # (speed_of_light [mm/ns] / 2) * TOF FWHM [ns] / 2.355

    proj.tof_parameters = parallelproj.TOFParameters(
        num_tofbins=num_tofbins, tofbin_width=tofbin_width, sigma_tof=sigma_tof
    )

    res_model = parallelproj.GaussianFilterOperator(
        proj.in_shape, sigma=xp.asarray(ipsf_fwhm) / (2.35 * proj.voxel_size), truncate=3.
    )
    return proj, res_model

def get_lm_projection(sino_proj, res_model, count, x_true, u_map, random=0.2):
    """
    Simulate list‐mode events from an image + attenuation map.

    Returns
    -------
    lmdata : np.ndarray, shape (N_events, 5), dtype int16
        Each row is [tx1, ax1, tx2, ax2, tof_bin].
    contamination : np.ndarray
        The contamination sinogram that was added before Poisson noise.
    """
    # 1) forward (sinogram) + attenuation
    sino_proj.tof = False
    att_sino = np.exp(-sino_proj(u_map.reshape(sino_proj.in_shape)))
    sino_proj.tof = True

    # setup the attenuation multiplication operator which is different
    # for TOF and non-TOF since the attenuation sinogram is always non-TOF
    if sino_proj.tof:
        att_op = parallelproj.TOFNonTOFElementwiseMultiplicationOperator(
            sino_proj.out_shape, att_sino
        )
    else:
        att_op = parallelproj.ElementwiseMultiplicationOperator(att_sino)

    # compose all 3 operators into a single linear operator
    pet_lin_op = parallelproj.CompositeLinearOperator((att_op, sino_proj, res_model))

    # simulated noise-free data
    noise_free_data = pet_lin_op(x_true)

    # generate a contant contamination sinogram
    contamination = np.full(
        noise_free_data.shape,
        random * float(np.mean(noise_free_data)),
        device=sino_proj.lor_descriptor.dev,
        dtype=np.float32,
    )

    noise_free_data += contamination
    scale = count / noise_free_data.sum()
    noise_free_data *= scale
    contamination *= scale

    # add Poisson noise
    y = np.asarray(
        np.random.poisson(parallelproj.to_numpy_array(noise_free_data)),
        device=sino_proj.lor_descriptor.dev,
        dtype=np.int32,
    )

    coords_s, coords_e, tof_bins = sino_proj.convert_sinogram_to_listmode(
        y
    )

    # ---------------------------------------------------------------
    # map world‐coords back to (module, index_in_module)
    scanner = sino_proj.lor_descriptor.scanner
    # all endpoints as (M,3) array
    all_pts = np.asarray(scanner.all_lor_endpoints)
    mods  = np.asarray(scanner.all_lor_endpoints_module_number)
    offs  = np.asarray(scanner.all_lor_endpoints_index_offset)

    # build KD‐tree once
    tree = cKDTree(all_pts)

    # find nearest endpoint for each event
    idx_s = tree.query(np.asarray(coords_s), k=1)[1]
    idx_e = tree.query(np.asarray(coords_e), k=1)[1]

    # module and in‐module index
    mod_s = mods[idx_s].astype(np.int16)
    im_s  = (idx_s - offs[mod_s]).astype(np.int16)
    mod_e = mods[idx_e].astype(np.int16)
    im_e  = (idx_e - offs[mod_e]).astype(np.int16)

    # tof (if None, set 0)
    if tof_bins is None:
        tof16 = np.zeros(len(idx_s), dtype=np.int16)
    else:
        tof16 = np.asarray(tof_bins, dtype=np.int16)

    # stack into Nx5 lmdata
    lmdata = np.stack([im_s, mod_s, im_e, mod_e, tof16], axis=1)

    return lmdata, contamination, scale


def cal_geo_mat(proj, imsize):
    sp_idx_i = []
    sp_idx_j = []
    sp_data = []
    num_j = np.array(imsize).prod().item()
    num_i = np.array(proj.out_shape).prod().item()

    time_start = time.time()
    for j in range(num_j):  
        test_img = np.zeros(imsize)
        idx3d = np.unravel_index(j, imsize)
        test_img[idx3d] = 1.

        fp = proj(test_img)

        sp_idx_i_list = np.flatnonzero(fp > 0).tolist()
        data_tmp = fp.flatten()[sp_idx_i_list].tolist()
        sp_idx_i += sp_idx_i_list
        sp_idx_j += len(sp_idx_i_list) * [j]
        sp_data += data_tmp

        if j % 100 == 0:
            time_elapsed = time.time()-time_start
            print(f'{j/num_j*100}%, time elapsed {time_elapsed/60} min, estimated remaining time {time_elapsed*(num_j-j)/(j+1)/60} min') 
        
    return coo_matrix((sp_data, (sp_idx_i, sp_idx_j)), shape=(num_i, num_j), dtype=np.float32)

imsize = (1, 128, 128)
voxsize = (1.65, 1.65, 1.65)
proj, res_model = get_geo_projector('nx-2d', imsize, voxsize, fine_tofbin=False)

### --- Non-TOF --- ###
proj.tof = False

G = cal_geo_mat(proj, imsize)

save_npz('G_nontof_nx-2d_481x480_128x128_1.65mm.npz', G)
print('G finished...')

# # MATLAB 
# G_csc = G.astype(np.float64).tocsc()
# sio.savemat('G_nontof_nx-2d_481x480_128x128_1.65mm.mat', {"G": G_csc}, do_compression=True)

# ### --- TOF --- ###
# proj.tof = True

# G = cal_geo_mat(proj, imsize)

# save_npz('G_tof_nx-2d_481x480x51_128x128_1.65mm.npz', G)
# print('G finished...')

### --- iPSF --- ###
from itertools import product

def gaussian_kernel1d(sigma, radius=None):
    if radius is None:
        radius = int(3 * sigma)
    x = np.arange(-radius, radius+1)
    kernel = np.exp(-(x**2)/(2*sigma**2))
    kernel /= kernel.sum()
    return kernel

def gaussian_kernel_nd(shape, sigma):
    if isinstance(sigma, (int, float)):
        sigma = [sigma] * len(shape)
    kernels = [gaussian_kernel1d(s, round(3*s)) for s in sigma]
    coords = list(product(*[range(-len(k)//2 + 1, len(k)//2 + 1) for k in kernels]))
    kernel_vals = np.array([np.prod([kernels[d][c[d] + len(kernels[d])//2] for d in range(len(shape))])
                            for c in coords])
    return coords, kernel_vals

from scipy.sparse import coo_matrix, diags

def make_sparse_gaussian_matrix(image_shape, sigma, dtype=np.float32):
    coords, values = gaussian_kernel_nd(image_shape, sigma)
    n_voxels = np.prod(image_shape)
    
    def index_nd_to_1d(index, shape):
        return np.ravel_multi_index(index, shape)

    rows, cols, data = [], [], []

    for idx in product(*[range(s) for s in image_shape]):
        idx1d = index_nd_to_1d(idx, image_shape)
        for offset, weight in zip(coords, values):
            neighbor = tuple(i + o for i, o in zip(idx, offset))
            if all(0 <= n < s for n, s in zip(neighbor, image_shape)):
                idx_neighbor = index_nd_to_1d(neighbor, image_shape)
                rows.append(idx1d)
                cols.append(idx_neighbor)
                data.append(weight)

    sparse_matrix = coo_matrix((data, (rows, cols)), shape=(n_voxels, n_voxels), dtype=dtype)
    return sparse_matrix

def normalize_sparse_rows(sparse_matrix):
    row_sums = np.array(sparse_matrix.sum(axis=1)).flatten()
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    inv_row_sums = 1.0 / row_sums
    D_inv = diags(inv_row_sums)
    return D_inv @ sparse_matrix


sigma = (np.array((1.6, 0.8, 0.8)) / (2.35 * np.array(voxsize))).tolist()
P = normalize_sparse_rows(make_sparse_gaussian_matrix(imsize, sigma, dtype=np.float32)).tocoo()

# save_npz('ipsf_nontof_nx-2d.npz', P)

# MATLAB 
P_csc = P.astype(np.float64).tocsc()
sio.savemat('ipsf_nontof_nx-2d.mat', {"P": P_csc}, do_compression=True)
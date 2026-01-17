/*
## Geometry
### Davis matlab convention
recon geometry    y ^      __                   
                    |   /  __2 \
                    |  |  |     | _> tx id = 0
                    |   \ 1__  /
                     _______________>  x
Matlab coordinate
### rayprj convention
recon geometry    y ^      __ 
                    |   /  __3 \
                    |  |  |     | _> tx id = 0
                    |   \ 2__  /
                     _______________>  x

3D image: <br>
    3rd dim (last in numpy) increases as x increases <br>
    2nd dim increases as y decreases for plt.imshow correct image <br>
    1st dim increases as z increases
*/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <cmath>
#include "rtracer.h"

namespace py = pybind11;

// Helpers to read inputs safely; we’ll accept C- or Fortran-ordered arrays and copy if needed.
static inline void check_len3(const py::array &a) {
    if (a.ndim() != 1 || a.shape(0) != 3)
        throw std::runtime_error("array must be length-3");
}

// Replace the old checkers:
static inline void check_xtal_xy(const py::array &a) {
    if (a.ndim() != 2 || a.shape(1) != 2)
        throw std::runtime_error("xtal_xy must be of shape (N, 2) with columns [x, y]");
}
static inline void check_lm(const py::array &a) {
    if (a.ndim() != 2 || a.shape(1) != 5)
        throw std::runtime_error("lmdata must be of shape (N, 5) with columns [xtal1, ring1, xtal2, ring2, tbin]");
}


// ---- Forward projector (non-TOF) ----
#ifndef USE_TOF 
py::array_t<double> fproj_mt_py(
    py::array_t<double, py::array::c_style> image,   // flattened numpy array, c style
    py::array_t<int, py::array::forcecast> imgsize, // [z, y, x], numpy convention
    py::array_t<double> voxsize, // [z, y, x], numpy convention
    py::array_t<double> xtal_xy, // [[x, y],...]
    py::array_t<double> ring_z,  // [z]
    py::array_t<int16_t> lmdata
) 
#else
py::array_t<double> fproj_tof_mt_py(
    py::array_t<double, py::array::c_style> image,   // flattened numpy array, c style
    py::array_t<int, py::array::forcecast> imgsize, // [z, y, x], numpy convention
    py::array_t<double> voxsize, // [z, y, x], numpy convention
    py::array_t<double> xtal_xy, // [[x, y],...]
    py::array_t<double> ring_z,  // [z]
    py::array_t<int16_t> lmdata,
    py::array_t<double> tof_info // length-2: [tw_resolution, tw_spacing]
) 
#endif
{
    check_len3(imgsize); check_len3(voxsize);
    check_xtal_xy(xtal_xy); check_lm(lmdata);

    const int ni = imgsize.at(1), nj = imgsize.at(2), nk = imgsize.at(0); // re-order to matlab order [y, x, z]
    const int64_t nsize = ni * nj * nk;
    const int64_t num_prompts = static_cast<int64_t>(lmdata.shape(0));
    if (static_cast<int64_t>(image.size()) != nsize)
        throw std::runtime_error("invalid image vector or image size");

    // Build tracer (non-TOF => tw_sigma/tw_spacing negative)
    // ImageRayTracer::NUMBER_OF_SIGMA = 3;
    // ImageRayTracer::GWSAMPLESIZE    = 10240;

#ifndef USE_TOF
    ImageRayTracer tracer(
        ni, nj, nk,
        voxsize.at(1), voxsize.at(2), voxsize.at(0), // re-order to matlab order [y,x,z]
        -1.0, -1.0
    );
#else
    const double tw_res = tof_info.at(0);
    const double tw_spa = tof_info.at(1);
    ImageRayTracer tracer(
        ni, nj, nk,
        voxsize.at(1), voxsize.at(2), voxsize.at(0),
        tw_res * 0.15 / (2*std::sqrt(2*std::log(2))),  // sigma
        tw_spa * 0.15                                   // spacing
    );
#endif

    auto proj = py::array_t<double>(num_prompts);
    double* proj_ptr = proj.mutable_data();    // OK once, before releasing the GIL
    // auto pproj = proj.mutable_unchecked<1>();

    // // Access raw buffers
    // auto img   = image.unchecked<1>();                       // length nsize
    // auto xtal  = xtal_xy.unchecked<2>();                     // num_xtals x 2
    // auto rz    = ring_z.unchecked<1>();                      // nring
    // auto lm    = lmdata.unchecked<2>();                      // num_prompts x 5
    // Prepare raw pointers & accessors (outside the loop). Avoid using python functions
    auto lm_buf   = lmdata.request();
    auto xtal_buf = xtal_xy.request();
    auto rz_buf   = ring_z.request();
    auto img_buf    = image.request();       // if this is bproj

    const int16_t* lm_ptr   = static_cast<const int16_t*>(lm_buf.ptr);
    const double*  xtal_ptr = static_cast<const double*>(xtal_buf.ptr); // shape (2, n_xtals)
    const double*  rz_ptr   = static_cast<const double*>(rz_buf.ptr);
    const double*  img_ptr    = static_cast<const double*>(img_buf.ptr);

    // Release the GIL during the heavy loop (important)
    py::gil_scoped_release release;

    int64_t n;
#if USE_OMP
    #pragma omp parallel for private(n)
#endif
    // Loop over prompts (OpenMP will still parallelize the C++ loop inside tracer)
    for (n = 0; n < num_prompts; ++n) {
        // const int16_t xtal_id_1 = lm(n, 0);
        // const int16_t ring_id_1 = lm(n, 1);
        // const int16_t xtal_id_2 = lm(n, 2);
        // const int16_t ring_id_2 = lm(n, 3);
        // const int16_t tbin_id   = lm(n, 4);

        // const double p0x = -xtal(xtal_id_1, 1);
        // const double p0y = -xtal(xtal_id_1, 0);
        // const double p0z = rz(ring_id_1);
        // const double p1x = -xtal(xtal_id_2, 1);
        // const double p1y = -xtal(xtal_id_2, 0);
        // const double p1z = rz(ring_id_2);

        const int16_t xtal_id_1 = lm_ptr[n * 5];
        const int16_t ring_id_1 = lm_ptr[n * 5 + 1];
        const int16_t xtal_id_2 = lm_ptr[n * 5 + 2];
        const int16_t ring_id_2 = lm_ptr[n * 5 + 3];
        const int16_t tbin_id   = lm_ptr[n * 5 + 4];

        const double p0x = -xtal_ptr[xtal_id_1 * 2 + 1];
        const double p0y = -xtal_ptr[xtal_id_1 * 2];
        const double p0z = rz_ptr[ring_id_1];
        const double p1x = -xtal_ptr[xtal_id_2 * 2 + 1];
        const double p1y = -xtal_ptr[xtal_id_2 * 2];
        const double p1z = rz_ptr[ring_id_2];

        proj_ptr[n] = tracer.fproj(p0x, p0y, p0z, p1x, p1y, p1z, img_ptr, tbin_id);
    }
    return proj;
}

// ---- Back projector ----
#ifndef USE_TOF
py::array_t<double> bproj_mt_py(
    py::array_t<double> proj,
    py::array_t<int, py::array::forcecast> imgsize,
    py::array_t<double> voxsize,
    py::array_t<double> xtal_xy,
    py::array_t<double> ring_z,
    py::array_t<int16_t> lmdata
)
#else
py::array_t<double> bproj_tof_mt_py(
    py::array_t<double> proj,
    py::array_t<int, py::array::forcecast> imgsize,
    py::array_t<double> voxsize,
    py::array_t<double> xtal_xy,
    py::array_t<double> ring_z,
    py::array_t<int16_t> lmdata,
    py::array_t<double> tof_info // length-2: [tw_resolution, tw_spacing]
)
#endif
{
    check_len3(imgsize); check_len3(voxsize);
    check_xtal_xy(xtal_xy); check_lm(lmdata);

    const int ni = imgsize.at(1), nj = imgsize.at(2), nk = imgsize.at(0);
    const int64_t nsize = ni * nj * nk;
    const int64_t num_prompts = static_cast<int64_t>(lmdata.shape(0));
    if (static_cast<int64_t>(proj.size()) != num_prompts)
        throw std::runtime_error("wrong number of projections or prompts");

    // ImageRayTracer::NUMBER_OF_SIGMA = 3;
    // ImageRayTracer::GWSAMPLESIZE    = 10240;

#ifndef USE_TOF
    ImageRayTracer tracer(
        ni, nj, nk,
        voxsize.at(1), voxsize.at(2), voxsize.at(0), // re-order to matlab order [y,x,z]
        -1.0, -1.0
    );
#else
    const double tw_res = tof_info.at(0);
    const double tw_spa = tof_info.at(1);
    ImageRayTracer tracer(
        ni, nj, nk,
        voxsize.at(1), voxsize.at(2), voxsize.at(0),
        tw_res * 0.15 / (2*std::sqrt(2*std::log(2))),  // sigma
        tw_spa * 0.15                                   // spacing
    );
#endif

    auto out = py::array_t<double>(nsize);
    double* out_ptr = out.mutable_data();    // OK once, before releasing the GIL

    // Prepare raw pointers & accessors (outside the loop). Avoid using python functions
    auto lm_buf   = lmdata.request();
    auto xtal_buf = xtal_xy.request();
    auto rz_buf   = ring_z.request();
    auto w_buf    = proj.request();       // if this is bproj

    const int16_t* lm_ptr   = static_cast<const int16_t*>(lm_buf.ptr);
    const double*  xtal_ptr = static_cast<const double*>(xtal_buf.ptr); // shape (2, n_xtals)
    const double*  rz_ptr   = static_cast<const double*>(rz_buf.ptr);
    const double*  w_ptr    = static_cast<const double*>(w_buf.ptr);

    // initialize to zeros
    for (int64_t i = 0; i < nsize; ++i) out_ptr[i] = 0.0;

    // Release the GIL during the heavy loop (important)
    py::gil_scoped_release release;

    int64_t n;
#if USE_OMP
    #pragma omp parallel for private(n)
#endif
    for (n = 0; n < num_prompts; ++n) {
        const int16_t xtal_id_1 = lm_ptr[n * 5];
        const int16_t ring_id_1 = lm_ptr[n * 5 + 1];
        const int16_t xtal_id_2 = lm_ptr[n * 5 + 2];
        const int16_t ring_id_2 = lm_ptr[n * 5 + 3];
        const int16_t tbin_id   = lm_ptr[n * 5 + 4];

        const double p0x = -xtal_ptr[xtal_id_1 * 2 + 1];
        const double p0y = -xtal_ptr[xtal_id_1 * 2];
        const double p0z = rz_ptr[ring_id_1];
        const double p1x = -xtal_ptr[xtal_id_2 * 2 + 1];
        const double p1y = -xtal_ptr[xtal_id_2 * 2];
        const double p1z = rz_ptr[ring_id_2];

        const double wt = w_ptr[n];
        if (wt > 0.0) {
            tracer.bproj(p0x, p0y, p0z, p1x, p1y, p1z, wt, out_ptr, tbin_id);
        }
    }
    return out;
}

// #ifdef USE_TOF
// // ---- TOF versions (same as above, but read tof_info and pass tw_sigma & tw_spacing) ----
// py::array_t<double> fproj_tof_mt_py(
//     py::array_t<double> image,
//     py::array_t<double> imgsize,
//     py::array_t<double> voxsize,
//     py::array_t<double> xtal_xy,
//     py::array_t<double> ring_z,
//     py::array_t<int16_t> lmdata,
//     py::array_t<double> tof_info // length-2: [tw_resolution, tw_spacing]
// ) {
//     check_len3(imgsize); check_len3(voxsize); check_xtal_xy(xtal_xy); check_lm(lmdata);
//     if (tof_info.ndim() != 1 || tof_info.shape(0) < 2)
//         throw std::runtime_error("invalid tof_info (need [tw_resolution, tw_spacing])");

//     const int ni = imgsize.at(1), nj = imgsize.at(2), nk = imgsize.at(0);
//     const int nsize = ni * nj * nk;
//     const int num_prompts = static_cast<int>(lmdata.shape(0));
//     if (static_cast<int64_t>(image.size()) != static_cast<int64_t>(nsize))
//         throw std::runtime_error("invalid image vector or image size");

//     const double tw_res = tof_info.at(0);
//     const double tw_spa = tof_info.at(1);

//     // ImageRayTracer::NUMBER_OF_SIGMA = 3;
//     // ImageRayTracer::GWSAMPLESIZE    = 10240;
//     // Note: your MEX scales by 0.15 and converts FWHM->sigma; we do the same here:
//     ImageRayTracer tracer(
//         ni, nj, nk,
//         voxsize.at(1), voxsize.at(2), voxsize.at(0),
//         tw_res * 0.15 / (2*std::sqrt(2*std::log(2))),  // sigma
//         tw_spa * 0.15                                   // spacing
//     );

//     auto proj = py::array_t<double>(num_prompts);
//     auto pproj = proj.mutable_unchecked<1>();

//     // Access raw buffers
//     auto img   = image.unchecked<1>();                       // length nsize
//     auto xtal  = xtal_xy.unchecked<2>();                     // 2 x num_xtals
//     auto rz    = ring_z.unchecked<1>();                      // nring
//     auto lm    = lmdata.unchecked<2>();                      // 5 x num_prompts

//     // Loop over prompts (OpenMP will still parallelize the C++ loop inside tracer)
//     for (int n = 0; n < num_prompts; ++n) {
//         const int16_t xtal_id_1 = lm(n, 0);
//         const int16_t ring_id_1 = lm(n, 1);
//         const int16_t xtal_id_2 = lm(n, 2);
//         const int16_t ring_id_2 = lm(n, 3);
//         const int16_t tbin_id   = lm(n, 4);

//         const double p0x = -xtal(xtal_id_1, 1);
//         const double p0y = -xtal(xtal_id_1, 0);
//         const double p0z = rz(ring_id_1);
//         const double p1x = -xtal(xtal_id_2, 1);
//         const double p1y = -xtal(xtal_id_2, 0);
//         const double p1z = rz(ring_id_2);

//         pproj(n) = tracer.fproj(p0x, p0y, p0z, p1x, p1y, p1z,
//                                 image.data(), /*tbin*/ tbin_id);
//     }
//     return proj;
// }

// py::array_t<double> bproj_tof_mt_py(
//     py::array_t<double> proj,
//     py::array_t<double> imgsize,
//     py::array_t<double> voxsize,
//     py::array_t<double> xtal_xy,
//     py::array_t<double> ring_z,
//     py::array_t<int16_t> lmdata,
//     py::array_t<double> tof_info // length-2: [tw_resolution, tw_spacing]
// ) {
//     check_len3(imgsize); check_len3(voxsize); check_xtal_xy(xtal_xy); check_lm(lmdata);
//     if (tof_info.ndim() != 1 || tof_info.shape(0) < 2)
//         throw std::runtime_error("invalid tof_info (need [tw_resolution, tw_spacing])");

//     const int ni = imgsize.at(1), nj = imgsize.at(2), nk = imgsize.at(0);
//     const int nsize = ni * nj * nk;
//     const int num_prompts = static_cast<int>(lmdata.shape(0));
//     if (static_cast<int>(proj.size()) != num_prompts)
//         throw std::runtime_error("wrong number of projections or prompts");

//     const double tw_res = tof_info.at(0);
//     const double tw_spa = tof_info.at(1);

//     // ImageRayTracer::NUMBER_OF_SIGMA = 3;
//     // ImageRayTracer::GWSAMPLESIZE    = 10240;
//     ImageRayTracer tracer(
//         ni, nj, nk,
//         voxsize.at(1), voxsize.at(2), voxsize.at(0),
//         tw_res * 0.15 / (2*std::sqrt(2*std::log(2))),  // sigma
//         tw_spa * 0.15                                   // spacing
//     );

//     auto out = py::array_t<double>(nsize);
//     auto bpimg = out.mutable_unchecked<1>();

//     auto xtal  = xtal_xy.unchecked<2>();
//     auto rz    = ring_z.unchecked<1>();
//     auto lm    = lmdata.unchecked<2>();
//     auto w     = proj.unchecked<1>();

//     // initialize to zeros
//     for (int i = 0; i < nsize; ++i) bpimg(i) = 0.0;

//     for (int n = 0; n < num_prompts; ++n) {
//         const int16_t xtal_id_1 = lm(n, 0);
//         const int16_t ring_id_1 = lm(n, 1);
//         const int16_t xtal_id_2 = lm(n, 2);
//         const int16_t ring_id_2 = lm(n, 3);
//         const int16_t tbin_id   = lm(n, 4);

//         const double p0x = -xtal(xtal_id_1, 1);
//         const double p0y = -xtal(xtal_id_1, 0);
//         const double p0z = rz(ring_id_1);
//         const double p1x = -xtal(xtal_id_2, 1);
//         const double p1y = -xtal(xtal_id_2, 0);
//         const double p1z = rz(ring_id_2);

//         const double wt = w(n);
//         if (wt > 0.0) {
//             tracer.bproj(p0x, p0y, p0z, p1x, p1y, p1z, wt, out.mutable_data(), tbin_id);
//         }
//     }
//     return out;
// }
// #endif

#ifdef USE_TOF
PYBIND11_MODULE(_core_tof, m)
#else
PYBIND11_MODULE(_core, m)
#endif
{
#ifdef USE_TOF
    // TOF exports
    m.def("fproj_tof_mt", &fproj_tof_mt_py);
    m.def("bproj_tof_mt", &bproj_tof_mt_py);
#else
    // non-TOF exports
    m.def("fproj_mt", &fproj_mt_py);
    m.def("bproj_mt", &bproj_mt_py);
#endif
}

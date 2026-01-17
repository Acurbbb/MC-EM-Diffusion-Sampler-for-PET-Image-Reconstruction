#include <cmath>
#include <vector>
#include <algorithm>

#ifndef RTRACER_H
#define RTRACER_H

// added for python support
#ifndef RAYPRJ_ERROR
  #ifdef BUILD_MEX
    #include "mex.h"
    #define RAYPRJ_ERROR(msg) mexErrMsgTxt(msg)
  #else
    #include <stdexcept>
    #define RAYPRJ_ERROR(msg) throw std::runtime_error(msg)
  #endif
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
typedef int     INT;
typedef double  REAL;
typedef double  IMAGE_DATA_TYPE;

typedef struct  tagXTAL_XY_POSITION {
    double x;
    double y;
}XTAL_XY_POSITION;

typedef struct tagXTAL_TRANSAXIAL_POSITION {
	double x;
	double y;
	double z;
}XTAL_TRANSAXIAL_POSITION;

typedef struct  tagLMEVENT {
    short xtal_id_1;
    short ring_id_1;
    short xtal_id_2;
    short ring_id_2;
    short tbin_id;
}LMEVENT;

typedef struct tagDOILMEVENT {
    unsigned char xtal_id_1;
    unsigned char ring_id_1;
    unsigned char doi_id_1;
    unsigned char xtal_id_2;
    unsigned char ring_id_2;
    unsigned char doi_id_2;
    unsigned char step_id;
}DOILMEVENT;

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// ImageRayTracer
//
////////////////////////////////////////////////////////////////////////////////////////////////////
#define DIST_NO_SQRT
//#define BP_ATOMIC 1

class ImageRayTracer
{
public:
    ImageRayTracer(const INT vdim_i = 0,
                   const INT vdim_j = 0,
                   const INT vdim_k = 0,
                   const REAL vox_size_i = 0,
                   const REAL vox_size_j = 0,
                   const REAL vox_size_k = 0,
                   const REAL tw_sigma = -1,
                   const REAL tw_spacing = -1);
    ~ImageRayTracer();

public:
    REAL fproj(REAL p0x, REAL p0y, REAL p0z,
               REAL p1x, REAL p1y, REAL p1z,
               const IMAGE_DATA_TYPE* img, const INT tbin_id = 0);
    
    void bproj(REAL p0x, REAL p0y, REAL p0z,
               REAL p1x, REAL p1y, REAL p1z,
               const REAL weight, IMAGE_DATA_TYPE* img, const INT tbin_id = 0);
    
    void bproj_sq(REAL p0x, REAL p0y, REAL p0z,
                  REAL p1x, REAL p1y, REAL p1z,
                  const REAL weight, IMAGE_DATA_TYPE* img, const INT tbin_id = 0);

    REAL fprojBresenham(REAL p0x, REAL p0y, REAL p0z,
               REAL p1x, REAL p1y, REAL p1z,
               const IMAGE_DATA_TYPE* img, const INT tbin_id = 0);

    void bprojBresenham(REAL p0x, REAL p0y, REAL p0z,
               REAL p1x, REAL p1y, REAL p1z,
               const REAL weight, IMAGE_DATA_TYPE* img, const INT tbin_id = 0);

    REAL fprojLinterp(REAL p0x, REAL p0y, REAL p0z,
               REAL p1x, REAL p1y, REAL p1z,
               const IMAGE_DATA_TYPE* img, const INT tbin_id = 0);
               
    void bprojLinterp(REAL p0x, REAL p0y, REAL p0z,
               REAL p1x, REAL p1y, REAL p1z,
               const REAL weight, IMAGE_DATA_TYPE* img, const INT tbin_id = 0);
        
private:
    bool hitCheck(const REAL p0x, const REAL p0y, const REAL p0z,
                  const REAL p1x, const REAL p1y, const REAL p1z,
                  REAL& t_min, REAL& t_max);
    void calcTOFLOREndPoints(const INT tbin_id,
                             REAL& p0x, REAL& p0y, REAL& p0z,
                             REAL& p1x, REAL& p1y, REAL& p1z,
                             REAL& nrm_dx, REAL& nrm_dy, REAL& nrm_dz);
    
private:
    INT m_vdim_i;
    INT m_vdim_j;
    INT m_vdim_k;
    INT m_vdim_ixj;
    REAL m_vox_size_i;
    REAL m_vox_size_j;
    REAL m_vox_size_k;
    REAL m_vbd_x0;
    REAL m_vbd_x1;
    REAL m_vbd_y0;
    REAL m_vbd_y1;
    REAL m_vbd_z0;
    REAL m_vbd_z1;
    REAL m_tw_sigma;
    REAL m_tw_spacing;
    std::vector<REAL> m_gw_lut;
    REAL m_gs_inv;
    std::vector<REAL> m_x_cuts;
    std::vector<REAL> m_y_cuts;
    std::vector<REAL> m_z_cuts;
    
public:
    static const REAL PI;
    static const REAL FLTMIN;
    
public:
    static INT GWSAMPLESIZE;
    static INT NUMBER_OF_SIGMA;
};

const REAL ImageRayTracer::FLTMIN = 1e-12;
const REAL ImageRayTracer::PI = 3.1415926535897932384626;
INT ImageRayTracer::GWSAMPLESIZE = 10240;
INT ImageRayTracer::NUMBER_OF_SIGMA = 3; // actual range is +/- NUMBER_OF_SIGMA

ImageRayTracer::ImageRayTracer(const INT vdim_i,
                               const INT vdim_j,
                               const INT vdim_k,
                               const REAL vox_size_i,
                               const REAL vox_size_j,
                               const REAL vox_size_k,
                               const REAL tw_sigma,
                               const REAL tw_spacing) :
    m_vdim_i(vdim_i),
    m_vdim_j(vdim_j),
    m_vdim_k(vdim_k),
    m_vox_size_i(vox_size_i),
    m_vox_size_j(vox_size_j),
    m_vox_size_k(vox_size_k),
    m_vdim_ixj(vdim_i * vdim_j),
    m_tw_sigma(tw_sigma),
    m_tw_spacing(tw_spacing)
{
    m_vbd_y0 = -(vdim_i * vox_size_i) * 0.5;
    m_vbd_y1 = +(vdim_i * vox_size_i) * 0.5;
    m_vbd_x0 = -(vdim_j * vox_size_j) * 0.5;
    m_vbd_x1 = +(vdim_j * vox_size_j) * 0.5;
    m_vbd_z0 = -(vdim_k * vox_size_k) * 0.5;
    m_vbd_z1 = +(vdim_k * vox_size_k) * 0.5;

    m_x_cuts.resize(m_vdim_j);
    for (INT n = 0; n < m_vdim_j; n ++) {
        m_x_cuts[n] = (-m_vdim_j * 0.5 + n + 0.5) * m_vox_size_j;
    }
    m_y_cuts.resize(m_vdim_i);
    for (INT n = 0; n < m_vdim_i; n ++) {
        m_y_cuts[n] = (-m_vdim_i * 0.5 + n + 0.5) * m_vox_size_i;
    }
    m_z_cuts.resize(m_vdim_k);
    for (INT n = 0; n < m_vdim_k; n ++) {
        m_z_cuts[n] = (-m_vdim_k * 0.5 + n + 0.5) * m_vox_size_k;
    }

#if USE_TOF    
    if (m_tw_sigma < 0 || m_tw_spacing < 0) {
        RAYPRJ_ERROR("invalid timing info!");
        abort();
    }

    // precompute gaussian window lookuptable
    // +/- 3-sigma truncation
#if 0  
    REAL gw_c0 = 1.0 / (2.0 * m_tw_sigma * m_tw_sigma);
    REAL gw_c1 = 1.0 / (sqrt(2.0 * PI) * m_tw_sigma);
    REAL gw_stepsize = 3.0 * m_tw_sigma / GWSAMPLESIZE;
    m_gs_inv = 1.0 / gw_stepsize;

    REAL s0 = 0;
    for (INT i = 0; i < GWSAMPLESIZE; i ++) {
        m_gw_lut.push_back(exp(-((gw_stepsize * i) *
                                 (gw_stepsize * i)) * gw_c0) * gw_c1);
        s0 += m_gw_lut.back();
    }
#else
    REAL c0 = m_tw_spacing * 0.5;
	REAL c1 = m_tw_sigma*sqrt(2.0);
	REAL gw_stepsize = NUMBER_OF_SIGMA * m_tw_sigma / GWSAMPLESIZE;
	m_gs_inv = 1.0 / gw_stepsize;

	REAL s0 = 0;
	for (int i = 0; i < GWSAMPLESIZE; i ++) {
		m_gw_lut.push_back((erf((i*gw_stepsize + c0)/c1) - erf((i*gw_stepsize - c0)/c1)) / 2.0);
	    s0 += m_gw_lut.back();
	}
#endif    
    
#if 0
    mexPrintf("Gaussian window: %d, %.10f, %.10f\n", GWSAMPLESIZE, s0, s0 / GWSAMPLESIZE);
#endif
#endif
    
}

ImageRayTracer::~ImageRayTracer()
{
}
    
inline bool ImageRayTracer::hitCheck(const REAL p0x, const REAL p0y, const REAL p0z,
                                     const REAL dx, const REAL dy, const REAL dz,
                                     REAL& t_min, REAL& t_max)
{
#if CFOV_ENABLED
    
//    REAL fov_radius = 58 * m_vox_size_i * 0.5;
    REAL fov_radius = (m_vdim_i - 1) * m_vox_size_i * 0.5;
    REAL dx2dy2 = dx * dx + dy * dy;
    REAL det = dx2dy2 * fov_radius * fov_radius -
                 (dx * p0y - dy * p0x) * (dx * p0y - dy * p0x);

    if (det > 0.0) {
        REAL sqr_det = sqrtf(det);
        REAL xdx_ydy = -(dx * p0x + dy * p0y);
        t_min = (xdx_ydy - sqr_det) / dx2dy2;
        t_max = (xdx_ydy + sqr_det) / dx2dy2;
        return true;
    } else {
        return false;
    }

#else    
    // these are tricks to avoid the `divide by zero' error
    REAL ddx = (dx == 0) ? 1e-20 : dx;
    REAL ddy = (dy == 0) ? 1e-20 : dy;
    REAL ddz = (dz == 0) ? 1e-20 : dz;


    // parameter (need change if ray goes from p1 to p0)
#if 0
    REAL tx0 = (ddx > 0) ? ((m_vbd_x0 - m_vox_size_j - p0x) / dx) : ((m_vbd_x1 + m_vox_size_j - p0x) / ddx);
    REAL tx1 = (ddx > 0) ? ((m_vbd_x1 + m_vox_size_j - p0x) / dx) : ((m_vbd_x0 - m_vox_size_j - p0x) / ddx);
    REAL ty0 = (ddy > 0) ? ((m_vbd_y0 + m_vox_size_i - p0y) / dy) : ((m_vbd_y1 - m_vox_size_i - p0y) / ddy);
    REAL ty1 = (ddy > 0) ? ((m_vbd_y1 - m_vox_size_i - p0y) / dy) : ((m_vbd_y0 + m_vox_size_i - p0y) / ddy);
    REAL tz0 = (ddz > 0) ? ((m_vbd_z0 - p0z) / dz) : ((m_vbd_z1 - p0z) / ddz);
    REAL tz1 = (ddz > 0) ? ((m_vbd_z1 - p0z) / dz) : ((m_vbd_z0 - p0z) / ddz);
#else    
    REAL tx0 = (ddx > 0) ? ((m_vbd_x0 - p0x) / dx) : ((m_vbd_x1 - p0x) / ddx);
    REAL tx1 = (ddx > 0) ? ((m_vbd_x1 - p0x) / dx) : ((m_vbd_x0 - p0x) / ddx);
    REAL ty0 = (ddy > 0) ? ((m_vbd_y0 - p0y) / dy) : ((m_vbd_y1 - p0y) / ddy);
    REAL ty1 = (ddy > 0) ? ((m_vbd_y1 - p0y) / dy) : ((m_vbd_y0 - p0y) / ddy);
    REAL tz0 = (ddz > 0) ? ((m_vbd_z0 - p0z) / dz) : ((m_vbd_z1 - p0z) / ddz);
    REAL tz1 = (ddz > 0) ? ((m_vbd_z1 - p0z) / dz) : ((m_vbd_z0 - p0z) / ddz);
#endif
    /*
    if (dx == 0) {
        tx0 = -99.9;
        tx1 = +99.9;
    }

    if (dy == 0) {
        ty0 = -99.9;
        ty1 = +99.9;
    }

    if (dz == 0) {
        tz0 = -99.9;
        tz1 = +99.9;
    }     
     */
    
    // determine min and max
    t_min = std::max(std::max(tx0, std::max(ty0, tz0)), 0.0); 
    t_max = std::min(std::min(tx1, std::min(ty1, tz1)), 1.0);
    return (t_min < t_max);
#endif
}

inline void ImageRayTracer::calcTOFLOREndPoints(const INT tbin_id,
                                                REAL& p0x, REAL& p0y, REAL& p0z,
                                                REAL& p1x, REAL& p1y, REAL& p1z,
                                                REAL& nrm_dx, REAL& nrm_dy, REAL& nrm_dz)
{
    // ray direction vector
    REAL dx = p1x - p0x;
    REAL dy = p1y - p0y;
    REAL dz = p1z - p0z;

    // lor center
    REAL pcx = (p0x + p1x) * 0.5;
    REAL pcy = (p0y + p1y) * 0.5;
    REAL pcz = (p0z + p1z) * 0.5;

    // (inverse) length
    REAL inv_ray_length = 1.0 / sqrt(dx * dx + dy * dy + dz * dz);

    // normalized dir
    nrm_dx = dx * inv_ray_length;
    nrm_dy = dy * inv_ray_length;
    nrm_dz = dz * inv_ray_length;

    REAL tbin_offset1 = tbin_id * m_tw_spacing - NUMBER_OF_SIGMA * m_tw_sigma;
    REAL tbin_offset2 = tbin_id * m_tw_spacing + NUMBER_OF_SIGMA * m_tw_sigma;

    // get new end-points
    p0x = pcx + tbin_offset1 * nrm_dx;
    p0y = pcy + tbin_offset1 * nrm_dy;
    p0z = pcz + tbin_offset1 * nrm_dz;
    p1x = pcx + tbin_offset2 * nrm_dx;
    p1y = pcy + tbin_offset2 * nrm_dy;
    p1z = pcz + tbin_offset2 * nrm_dz;
}

REAL ImageRayTracer::fproj(REAL p0x, REAL p0y, REAL p0z,
                           REAL p1x, REAL p1y, REAL p1z,
                           const IMAGE_DATA_TYPE* img, 
                           const INT tbin_id)
{
#ifdef USE_TOF
    // normalized dir
    REAL nrm_dx;
    REAL nrm_dy;
    REAL nrm_dz;
    calcTOFLOREndPoints(tbin_id, 
                        p0x, p0y, p0z, 
                        p1x, p1y, p1z, 
                        nrm_dx, nrm_dy, nrm_dz);
    // timing bin center coord
    REAL tbc_x = (p0x + p1x) * 0.5;
    REAL tbc_y = (p0y + p1y) * 0.5;
    REAL tbc_z = (p0z + p1z) * 0.5;
#endif
    
    REAL dx = p1x - p0x;
    REAL dy = p1y - p0y;
    REAL dz = p1z - p0z;
    REAL t_min, t_max;
    if (!hitCheck(p0x, p0y, p0z, dx, dy, dz, t_min, t_max)) {
        return 0;
    }
#ifdef USE_TOF

    t_min = std::max(t_min, 0.0);
    t_max = std::min(t_max, 1.0);

    // double check if the ray segment is still inside the FOV
    if (t_min > t_max) {
        return 0.0;
    }

#ifdef DIST_NO_SQRT // required by type-II distance (if enabled)
    REAL coef_d = -(nrm_dx * tbc_x + nrm_dy * tbc_y + nrm_dz * tbc_z);
#endif
#endif    
    
    // calc length of ray
    REAL w0 = sqrt(dx * dx + dy * dy + dz * dz);     
    REAL p1stx = p0x + dx * t_min;
    REAL p1sty = p0y + dy * t_min;
    REAL p1stz = p0z + dz * t_min;
    
    // see the definition of coordination system
    // index of the first voxel hit by ray
    INT j = INT((p1stx - m_vbd_x0) / m_vox_size_j); // 1: x
    j = std::min(j, m_vdim_j - 1); // because the max index value is m_vdim_*-1

    INT i = INT((m_vbd_y1 - p1sty) / m_vox_size_i); // 0: y
    i = std::min(i, m_vdim_i - 1);

    INT k = INT((p1stz - m_vbd_z0) / m_vox_size_k); // 2 : z
    k = std::min(k, m_vdim_k - 1);
    
    // initial boundary
    REAL bx0 = (dx > 0) ? m_vbd_x0 + j * m_vox_size_j : m_vbd_x0 + (j + 1) * m_vox_size_j;
    REAL by0 = (dy > 0) ? m_vbd_y0 + (m_vdim_i - i - 1) * m_vox_size_i : 
                          m_vbd_y0 + (m_vdim_i - i) * m_vox_size_i;
    REAL bz0 = (dz > 0) ? m_vbd_z0 + k * m_vox_size_k : m_vbd_z0 + (k + 1) * m_vox_size_k;

    // step for update index
    INT di = (dy > 0) ? 1 : -1;
    INT dj = (dx > 0) ? 1 : -1;
    INT dk = (dz > 0) ? 1 : -1;
    
    // step for forward ray
    REAL ddx = (dx > 0) ? m_vox_size_j : -m_vox_size_j;
    REAL ddy = (dy > 0) ? m_vox_size_i : -m_vox_size_i;
    REAL ddz = (dz > 0) ? m_vox_size_k : -m_vox_size_k;

    INT idx;
    INT m = 0; // counter for nonzero element number
    REAL tm0 = t_min, tm1;
    REAL tx = (fabs(dx) < FLTMIN) ? 99.9 : (bx0 + ddx - p0x) / dx;
    REAL ty = (fabs(dy) < FLTMIN) ? 99.9 : (by0 + ddy - p0y) / dy;
    REAL tz = (fabs(dz) < FLTMIN) ? 99.9 : (bz0 + ddz - p0z) / dz;
    REAL dtx = (fabs(dx) < FLTMIN) ? 99.9 : ddx / dx;
    REAL dty = (fabs(dy) < FLTMIN) ? 99.9 : ddy / dy;
    REAL dtz = (fabs(dz) < FLTMIN) ? 99.9 : ddz / dz;

    REAL out = 0.0;
    // start main loop
    do {

        // current voxel subscripts
        INT i0 = i;
        INT j0 = j;
        INT k0 = k;
        
        // check which direction should be updated
        if (tx < ty) {
            if (tx < tz) {
                tm1 = tx;
                j += dj;
                tx += dtx;
            } else {
                tm1 = tz;
                k += dk;
                tz += dtz;
            }
        } else {
            if (ty < tz) {
                tm1 = ty;
                i -= di;    // ! see definition of coordinate system
                ty += dty;
            } else {
                tm1 = tz;
                k += dk;
                tz += dtz;
            }
        }

        if (tm1 > t_max) {
            tm1 = t_max;
        }

        // calc weight
        REAL ww = (tm1 - tm0) * w0;
        
#ifdef USE_TOF // TOF

#ifndef DIST_NO_SQRT
        REAL tdx = m_vbd_x0 + (j0 + 0.5) * m_vox_size_j - tbc_x; //m_x_cuts[j0] - tbc_x;
        REAL tdy = m_vbd_y1 - (i0 + 0.5) * m_vox_size_i - tbc_y; //m_y_cuts[m_img_size_i - i0 - 1] - tbc_y;
        REAL tdz = m_vbd_z0 + (k0 + 0.5) * m_vox_size_k - tbc_z; //m_z_cuts[k0] - tbc_z;
        REAL vox_to_tbc_dist = sqrt(tdx * tdx + tdy * tdy + tdz * tdz);
#else
        REAL vx = m_vbd_x0 + (j0 + 0.5) * m_vox_size_j; //m_x_cuts[j0];
        REAL vy = m_vbd_y1 - (i0 + 0.5) * m_vox_size_i; //m_y_cuts[m_img_size_i - i0 - 1];
        REAL vz = m_vbd_z0 + (k0 + 0.5) * m_vox_size_k; //m_z_cuts[k0];
        REAL vox_to_tbc_dist = fabs(nrm_dx * vx + nrm_dy * vy + nrm_dz * vz + coef_d);
#endif

        INT gw_bin = INT(vox_to_tbc_dist * m_gs_inv);
        if (gw_bin < GWSAMPLESIZE) {
            out += img[k0 * m_vdim_ixj + j0 * m_vdim_i + i0] * ww * m_gw_lut[gw_bin];
        }

#else // nonTOF
        // put value to image domain
        // calc voxel index
        out += ww * img[k0 * m_vdim_ixj + j0 * m_vdim_i + i0]; //img(i0, j0, k0);
#endif
        tm0 = tm1; // save previous result
    } while (fabs(tm0 - t_max) > FLTMIN);

    return out;
}

void ImageRayTracer::bproj(REAL p0x, REAL p0y, REAL p0z,
                           REAL p1x, REAL p1y, REAL p1z,
                           const REAL weight, IMAGE_DATA_TYPE* img, 
                           const INT tbin_id)
{
    if (weight == 0) {
        return;
    }
#ifdef USE_TOF
    // normalized dir
    REAL nrm_dx;
    REAL nrm_dy;
    REAL nrm_dz;
    calcTOFLOREndPoints(tbin_id, p0x, p0y, p0z, p1x, p1y, p1z, nrm_dx, nrm_dy, nrm_dz);
    
    // timing bin center coord
    REAL tbc_x = (p0x + p1x) * 0.5f;
    REAL tbc_y = (p0y + p1y) * 0.5f;
    REAL tbc_z = (p0z + p1z) * 0.5f;
#endif
    
    REAL dx = p1x - p0x;
    REAL dy = p1y - p0y;
    REAL dz = p1z - p0z;
    REAL t_min, t_max;

    if (!hitCheck(p0x, p0y, p0z, dx, dy, dz, t_min, t_max)) {
        return;
    }
    
#ifdef USE_TOF

    t_min = std::max(t_min, 0.0);
    t_max = std::min(t_max, 1.0);
    // double check if the ray segment is still inside the FOV
    if (t_min > t_max) {
        return;
    }

#ifdef DIST_NO_SQRT // required by type-II distance (if enabled)
    REAL coef_d = -(nrm_dx * tbc_x + nrm_dy * tbc_y + nrm_dz * tbc_z);
#endif
#endif    
    
    // calc length of ray
    REAL w0 = sqrt(dx * dx + dy * dy + dz * dz) * weight;     
    REAL p1stx = p0x + dx * t_min;
    REAL p1sty = p0y + dy * t_min;
    REAL p1stz = p0z + dz * t_min;
    
    // see the definition of coordination system
    // index of the first voxel hit by ray
    INT j = INT((p1stx - m_vbd_x0) / m_vox_size_j); // 1: x
    j = std::min(j, m_vdim_j - 1); // because the index max value is m_vdim_*-1

    INT i = INT((m_vbd_y1 - p1sty) / m_vox_size_i); // 0: y
    i = std::min(i, m_vdim_i - 1);

    INT k = INT((p1stz - m_vbd_z0) / m_vox_size_k); // 2 : z
    k = std::min(k, m_vdim_k - 1);

    // initial boundary
    REAL bx0 = (dx > 0) ? m_vbd_x0 + j * m_vox_size_j : m_vbd_x0 + (j + 1) * m_vox_size_j;
    REAL by0 = (dy > 0) ? m_vbd_y0 + (m_vdim_i - i - 1) * m_vox_size_i : 
                          m_vbd_y0 + (m_vdim_i - i) * m_vox_size_i;
    REAL bz0 = (dz > 0) ? m_vbd_z0 + k * m_vox_size_k : m_vbd_z0 + (k + 1) * m_vox_size_k;

    // step for update index
    INT di = (dy > 0) ? 1 : -1;
    INT dj = (dx > 0) ? 1 : -1;
    INT dk = (dz > 0) ? 1 : -1;

    // step for forward ray
    REAL ddx = (dx > 0) ? m_vox_size_j : -m_vox_size_j;
    REAL ddy = (dy > 0) ? m_vox_size_i : -m_vox_size_i;
    REAL ddz = (dz > 0) ? m_vox_size_k : -m_vox_size_k;
    
    INT idx;
    INT m = 0; // counter for nonzero element number
    REAL tm0 = t_min, tm1;
    REAL tx = (fabs(dx) < FLTMIN) ? 99.9 : (bx0 + ddx - p0x) / dx;
    REAL ty = (fabs(dy) < FLTMIN) ? 99.9 : (by0 + ddy - p0y) / dy;
    REAL tz = (fabs(dz) < FLTMIN) ? 99.9 : (bz0 + ddz - p0z) / dz;
    REAL dtx = (fabs(dx) < FLTMIN) ? 99.9 : ddx / dx;
    REAL dty = (fabs(dy) < FLTMIN) ? 99.9 : ddy / dy;
    REAL dtz = (fabs(dz) < FLTMIN) ? 99.9 : ddz / dz;

    // start main loop
    do {

        // current voxel subscripts
        INT i0 = i;
        INT j0 = j;
        INT k0 = k;
        
//        mexPrintf("%d %d %d, %.20f, tx=%.20f, ty=%.20f, tz=%.20f\n", i, j, k, tm0, tx,ty,tz);
        
        // check which direction should be updated
        if (tx < ty) {
            if (tx < tz) {
                tm1 = tx;
                j += dj;
                tx += dtx;
            } else {
                tm1 = tz;
                k += dk;
                tz += dtz;
            }
        } else {
            if (ty < tz) {
                tm1 = ty;
                i -= di;    // ! see definition of coordinate system
                ty += dty;
            } else {
                tm1 = tz;
                k += dk;
                tz += dtz;
            }
        }

        if (tm1 > t_max) {
            tm1 = t_max;
        }

        // calc weight
        REAL ww = (tm1 - tm0) * w0; 

#ifdef USE_TOF // TOF
////<---
#ifndef DIST_NO_SQRT
        REAL tdx = m_vbd_x0 + (j0 + 0.5) * m_vox_size_j - tbc_x; //m_x_cuts[j0] - tbc_x;
        REAL tdy = m_vbd_y1 - (i0 + 0.5) * m_vox_size_i - tbc_y; //m_y_cuts[m_img_size_i - i0 - 1] - tbc_y;
        REAL tdz = m_vbd_z0 + (k0 + 0.5) * m_vox_size_k - tbc_z; //m_z_cuts[k0] - tbc_z;
        REAL vox_to_tbc_dist = sqrt(tdx * tdx + tdy * tdy + tdz * tdz);
#else
        REAL vx = m_vbd_x0 + (j0 + 0.5) * m_vox_size_j; //m_x_cuts[j0];
        REAL vy = m_vbd_y1 - (i0 + 0.5) * m_vox_size_i; //m_y_cuts[m_img_size_i - i0 - 1];
        REAL vz = m_vbd_z0 + (k0 + 0.5) * m_vox_size_k; //m_z_cuts[k0];
        REAL vox_to_tbc_dist = fabs(nrm_dx * vx + nrm_dy * vy + nrm_dz * vz + coef_d);
#endif

#if 1        
        INT gw_bin = INT(vox_to_tbc_dist * m_gs_inv); 
        if (gw_bin < GWSAMPLESIZE) {
#ifdef USE_OMP
            #pragma omp atomic
#endif        
            img[k0 * m_vdim_ixj + j0 * m_vdim_i + i0] += ww * m_gw_lut[gw_bin];
        }
#endif

#else // nonTOF
        
        // put value to image domain
        // calc voxel index
#ifdef USE_OMP
//#if BP_ATOMIC
        #pragma omp atomic
//#endif
#endif        
        img[k0 * m_vdim_ixj + j0 * m_vdim_i + i0] += ww; 
#endif
        tm0 = tm1; // save previous result
    } while (fabs(tm0 - t_max) > FLTMIN);

}

void ImageRayTracer::bproj_sq(REAL p0x, REAL p0y, REAL p0z,
                              REAL p1x, REAL p1y, REAL p1z,
                              const REAL weight, IMAGE_DATA_TYPE* img, 
                              const INT tbin_id)
{
    if (weight == 0) {
        return;
    }
#ifdef USE_TOF
    // normalized dir
    REAL nrm_dx;
    REAL nrm_dy;
    REAL nrm_dz;
    calcTOFLOREndPoints(tbin_id, p0x, p0y, p0z, p1x, p1y, p1z, nrm_dx, nrm_dy, nrm_dz);
    
    // timing bin center coord
    REAL tbc_x = (p0x + p1x) * 0.5f;
    REAL tbc_y = (p0y + p1y) * 0.5f;
    REAL tbc_z = (p0z + p1z) * 0.5f;
#endif
    
    REAL dx = p1x - p0x;
    REAL dy = p1y - p0y;
    REAL dz = p1z - p0z;
    REAL t_min, t_max;

    if (!hitCheck(p0x, p0y, p0z, dx, dy, dz, t_min, t_max)) {
        return;
    }
    
#ifdef USE_TOF

    t_min = std::max(t_min, 0.0);
    t_max = std::min(t_max, 1.0);
    // double check if the ray segment is still inside the FOV
    if (t_min > t_max) {
        return;
    }

#ifdef DIST_NO_SQRT // required by type-II distance (if enabled)
    REAL coef_d = -(nrm_dx * tbc_x + nrm_dy * tbc_y + nrm_dz * tbc_z);
#endif
#endif    
    
    // calc length of ray
    REAL w0 = sqrt(dx * dx + dy * dy + dz * dz);     
    REAL p1stx = p0x + dx * t_min;
    REAL p1sty = p0y + dy * t_min;
    REAL p1stz = p0z + dz * t_min;
    
    // see the definition of coordination system
    // index of the first voxel hit by ray
    INT j = INT((p1stx - m_vbd_x0) / m_vox_size_j); // 1: x
    j = std::min(j, m_vdim_j - 1); // because the index max value is m_vdim_*-1

    INT i = INT((m_vbd_y1 - p1sty) / m_vox_size_i); // 0: y
    i = std::min(i, m_vdim_i - 1);

    INT k = INT((p1stz - m_vbd_z0) / m_vox_size_k); // 2 : z
    k = std::min(k, m_vdim_k - 1);

    // initial boundary
    REAL bx0 = (dx > 0) ? m_vbd_x0 + j * m_vox_size_j : m_vbd_x0 + (j + 1) * m_vox_size_j;
    REAL by0 = (dy > 0) ? m_vbd_y0 + (m_vdim_i - i - 1) * m_vox_size_i : 
                          m_vbd_y0 + (m_vdim_i - i) * m_vox_size_i;
    REAL bz0 = (dz > 0) ? m_vbd_z0 + k * m_vox_size_k : m_vbd_z0 + (k + 1) * m_vox_size_k;

    // step for update index
    INT di = (dy > 0) ? 1 : -1;
    INT dj = (dx > 0) ? 1 : -1;
    INT dk = (dz > 0) ? 1 : -1;

    // step for forward ray
    REAL ddx = (dx > 0) ? m_vox_size_j : -m_vox_size_j;
    REAL ddy = (dy > 0) ? m_vox_size_i : -m_vox_size_i;
    REAL ddz = (dz > 0) ? m_vox_size_k : -m_vox_size_k;
    
    INT idx;
    INT m = 0; // counter for nonzero element number
    REAL tm0 = t_min, tm1;
    REAL tx = (fabs(dx) < FLTMIN) ? 99.9 : (bx0 + ddx - p0x) / dx;
    REAL ty = (fabs(dy) < FLTMIN) ? 99.9 : (by0 + ddy - p0y) / dy;
    REAL tz = (fabs(dz) < FLTMIN) ? 99.9 : (bz0 + ddz - p0z) / dz;
    REAL dtx = (fabs(dx) < FLTMIN) ? 99.9 : ddx / dx;
    REAL dty = (fabs(dy) < FLTMIN) ? 99.9 : ddy / dy;
    REAL dtz = (fabs(dz) < FLTMIN) ? 99.9 : ddz / dz;

    // start main loop
    do {

        // current voxel subscripts
        INT i0 = i;
        INT j0 = j;
        INT k0 = k;
        
//        mexPrintf("%d %d %d, %.20f, tx=%.20f, ty=%.20f, tz=%.20f\n", i, j, k, tm0, tx,ty,tz);
        
        // check which direction should be updated
        if (tx < ty) {
            if (tx < tz) {
                tm1 = tx;
                j += dj;
                tx += dtx;
            } else {
                tm1 = tz;
                k += dk;
                tz += dtz;
            }
        } else {
            if (ty < tz) {
                tm1 = ty;
                i -= di;    // ! see definition of coordinate system
                ty += dty;
            } else {
                tm1 = tz;
                k += dk;
                tz += dtz;
            }
        }

        if (tm1 > t_max) {
            tm1 = t_max;
        }

        // calc weight (intersection length)
        REAL ww = (tm1 - tm0) * w0; 

#ifdef USE_TOF // TOF
////<---
#ifndef DIST_NO_SQRT
        REAL tdx = m_vbd_x0 + (j0 + 0.5) * m_vox_size_j - tbc_x; //m_x_cuts[j0] - tbc_x;
        REAL tdy = m_vbd_y1 - (i0 + 0.5) * m_vox_size_i - tbc_y; //m_y_cuts[m_img_size_i - i0 - 1] - tbc_y;
        REAL tdz = m_vbd_z0 + (k0 + 0.5) * m_vox_size_k - tbc_z; //m_z_cuts[k0] - tbc_z;
        REAL vox_to_tbc_dist = sqrt(tdx * tdx + tdy * tdy + tdz * tdz);
#else
        REAL vx = m_vbd_x0 + (j0 + 0.5) * m_vox_size_j; //m_x_cuts[j0];
        REAL vy = m_vbd_y1 - (i0 + 0.5) * m_vox_size_i; //m_y_cuts[m_img_size_i - i0 - 1];
        REAL vz = m_vbd_z0 + (k0 + 0.5) * m_vox_size_k; //m_z_cuts[k0];
        REAL vox_to_tbc_dist = fabs(nrm_dx * vx + nrm_dy * vy + nrm_dz * vz + coef_d);
#endif

#if 1        
        INT gw_bin = INT(vox_to_tbc_dist * m_gs_inv); 
        if (gw_bin < GWSAMPLESIZE) {
#ifdef USE_OMP
            #pragma omp atomic
#endif        
			// only square the length
            img[k0 * m_vdim_ixj + j0 * m_vdim_i + i0] += (ww * m_gw_lut[gw_bin]) * (ww * m_gw_lut[gw_bin]) * weight;
        }
#endif

#else // nonTOF
        
        // put value to image domain
        // calc voxel index
#ifdef USE_OMP
//#if BP_ATOMIC
        #pragma omp atomic
//#endif
#endif  
		// only square the length      
        img[k0 * m_vdim_ixj + j0 * m_vdim_i + i0] += ww * ww * weight; 
#endif
        tm0 = tm1; // save previous result
    } while (fabs(tm0 - t_max) > FLTMIN);

}

REAL ImageRayTracer::fprojBresenham(REAL p0x, REAL p0y, REAL p0z,
                           			REAL p1x, REAL p1y, REAL p1z,
                           			const IMAGE_DATA_TYPE* img, 
                           			const INT tbin_id)
{
#ifdef USE_TOF
    // normalized dir
    REAL nrm_dx;
    REAL nrm_dy;
    REAL nrm_dz;
    calcTOFLOREndPoints(tbin_id, 
                        p0x, p0y, p0z, 
                        p1x, p1y, p1z, 
                        nrm_dx, nrm_dy, nrm_dz);
    // timing bin center coord
    REAL tbc_x = (p0x + p1x) * 0.5;
    REAL tbc_y = (p0y + p1y) * 0.5;
    REAL tbc_z = (p0z + p1z) * 0.5;
#endif
    
    REAL dx = p1x - p0x;
    REAL dy = p1y - p0y;
    REAL dz = p1z - p0z;
    REAL t_min, t_max;
    if (!hitCheck(p0x, p0y, p0z, dx, dy, dz, t_min, t_max)) {
        return 0;
    }
#ifdef USE_TOF

    t_min = std::max(t_min, 0.0);
    t_max = std::min(t_max, 1.0);

    // double check if the ray segment is still inside the FOV
    if (t_min > t_max) {
        return 0.0;
    }

#ifdef DIST_NO_SQRT // required by type-II distance (if enabled)
    REAL coef_d = -(nrm_dx * tbc_x + nrm_dy * tbc_y + nrm_dz * tbc_z);
#endif
#endif    
    
    REAL out = 0.0;
    REAL w0;
    REAL abs_dx = fabs(dx);
    REAL abs_dy = fabs(dy);
    REAL abs_dz = fabs(dz);
        
	if (abs_dx > abs_dy) { // x-cuts
		
        REAL d = abs_dy / abs_dx;
		w0 = sqrt(1.0 + d * d);

        // use t_min and t_max to reduce the size of for loop
        REAL x0 = p0x + t_min * dx;
        REAL x1 = p0x + t_max * dx;
        INT n0 = INT((x0 - m_vbd_x0) / m_vox_size_j);
        INT n1 = INT((x1 - m_vbd_x0) / m_vox_size_j);
        INT n_min = (dx < 0) ? ( n1 < 0 ? 0 : n1 ) : ( n0 < 0 ? 0 : n0 );
        INT n_max = (dx < 0) ? ( n0 >= m_vdim_j ? m_vdim_j-1 : n0 ) : 
                               ( n1>=m_vdim_j ? m_vdim_j-1 : n1 );
        
        REAL t = (m_x_cuts[n_min] - p0x) / dx;
        REAL yc = (m_vbd_y1 - (p0y + t * dy)) / m_vox_size_i;
        REAL zc = (p0z + t * dz - m_vbd_z0) / m_vox_size_k;
        REAL yc_step = -m_vox_size_j / dx * dy / m_vox_size_i;
        REAL zc_step = m_vox_size_j / dx * dz / m_vox_size_k;
                
        do {                        
            
            if (yc>0 && yc<m_vdim_i && zc>0 && zc<m_vdim_k) {
                INT i = INT(yc);
                INT k = INT(zc);

#ifdef USE_TOF // TOF *****************************************************************************

#ifndef DIST_NO_SQRT
        			REAL tdx = m_x_cuts[n_min] - tbc_x;
    			    REAL tdy = m_y_cuts[m_vdim_i - i - 1] - tbc_y;
    			    REAL tdz = m_z_cuts[k] - tbc_z;
			    REAL vox_to_tbc_dist = sqrt(tdx*tdx + tdy*tdy + tdz*tdz);
#else
    			    REAL vx = m_x_cuts[n_min];
			    REAL vy = m_y_cuts[m_vdim_i - i - 1];
			    REAL vz = m_z_cuts[k];
			    REAL vox_to_tbc_dist = fabs(nrm_dx * vx + nrm_dy * vy + nrm_dz * vz + coef_d);
#endif
			    INT gw_bin = INT(vox_to_tbc_dist * m_gs_inv);
			    if (gw_bin < GWSAMPLESIZE) {
				    out += img[k * m_vdim_ixj + n_min * m_vdim_i + i] * m_gw_lut[gw_bin];
			    }

#else // non-TOF	 ******************************************************************************		

                out += img[k * m_vdim_ixj + n_min * m_vdim_i + i];

#endif                
            }                       
        
            yc += yc_step;
            zc += zc_step;
            n_min ++;
        } while (n_min <= n_max);
		
	} else { // y-cuts
			
		REAL d = abs_dx / abs_dy;
		w0 = sqrt(1.0 + d * d);

        REAL y0 = p0y + t_min * dy;
        REAL y1 = p0y + t_max * dy;
        INT n0 = INT((y0 - m_vbd_y0) / m_vox_size_i);
        INT n1 = INT((y1 - m_vbd_y0) / m_vox_size_i);        
        INT n_min = (dy < 0) ? ( n1 < 0 ? 0 : n1 ) : ( n0 < 0 ? 0 : n0 );
        INT n_max = (dy < 0) ? ( n0 >= m_vdim_i ? m_vdim_i-1 : n0 ) : 
                               ( n1 >= m_vdim_i ? m_vdim_i-1 : n1 );

        REAL t = (m_y_cuts[n_min] - p0y) / dy;
        REAL xc = (p0x + t * dx - m_vbd_x0) / m_vox_size_j;
        REAL zc = (p0z + t * dz - m_vbd_z0) / m_vox_size_k;
        REAL xc_step = m_vox_size_i / dy * dx / m_vox_size_j;
        REAL zc_step = m_vox_size_i / dy * dz / m_vox_size_k;
        
        do {
                                
            if (xc>0 && xc<m_vdim_j && zc>0 && zc<m_vdim_k) {
                INT j = INT(xc);
                INT k = INT(zc);

#ifdef USE_TOF // TOF *****************************************************************************

#ifndef DIST_NO_SQRT
        			REAL tdx = m_x_cuts[j] - tbc_x;
        			REAL tdy = m_y_cuts[n_min] - tbc_y;
        			REAL tdz = m_z_cuts[k] - tbc_z;
        			REAL vox_to_tbc_dist = sqrtf(tdx*tdx + tdy*tdy + tdz*tdz);
#else
        			REAL vx = m_x_cuts[j];
        			REAL vy = m_y_cuts[n_min];
        			REAL vz = m_z_cuts[k];
        			REAL vox_to_tbc_dist = fabs(nrm_dx * vx + nrm_dy * vy + nrm_dz * vz + coef_d);			
#endif
		        	INT gw_bin = INT(vox_to_tbc_dist * m_gs_inv);
		        	if (gw_bin < GWSAMPLESIZE) {
		        		out += img[k * m_vdim_ixj + j * m_vdim_i + (m_vdim_i-n_min-1)] * m_gw_lut[gw_bin]; 
		        	}

#else // non-TOF **********************************************************************************

                out += img[k * m_vdim_ixj + j * m_vdim_i + (m_vdim_i-n_min-1)];
#endif                
            }
            
            xc += xc_step;
            zc += zc_step;
            n_min ++;
        } while (n_min <= n_max);
	}
	
    return (out * w0);
}

/*
    This version is best in terms of quality!
*/
void ImageRayTracer::bprojBresenham(REAL p0x, REAL p0y, REAL p0z,
                           REAL p1x, REAL p1y, REAL p1z,
                           const REAL weight, IMAGE_DATA_TYPE* img, 
                           const INT tbin_id)
{
#ifdef USE_TOF
    // normalized dir
    REAL nrm_dx;
    REAL nrm_dy;
    REAL nrm_dz;
    calcTOFLOREndPoints(tbin_id, 
                        p0x, p0y, p0z, 
                        p1x, p1y, p1z, 
                        nrm_dx, nrm_dy, nrm_dz);
    // timing bin center coord
    REAL tbc_x = (p0x + p1x) * 0.5;
    REAL tbc_y = (p0y + p1y) * 0.5;
    REAL tbc_z = (p0z + p1z) * 0.5;
#endif
    
    REAL dx = p1x - p0x;
    REAL dy = p1y - p0y;
    REAL dz = p1z - p0z;
    REAL t_min, t_max;
    if (!hitCheck(p0x, p0y, p0z, dx, dy, dz, t_min, t_max)) {
        return;
    }
#ifdef USE_TOF

    t_min = std::max(t_min, 0.0);
    t_max = std::min(t_max, 1.0);

    // double check if the ray segment is still inside the FOV
    if (t_min > t_max) {
        return;
    }

#ifdef DIST_NO_SQRT // required by type-II distance (if enabled)
    REAL coef_d = -(nrm_dx * tbc_x + nrm_dy * tbc_y + nrm_dz * tbc_z);
#endif
#endif    
       
    REAL w0;
    REAL abs_dx = fabs(dx);
    REAL abs_dy = fabs(dy);
    REAL abs_dz = fabs(dz);
        
	if (abs_dx > abs_dy) { // x-cuts
		
        REAL d = abs_dy / abs_dx;
		w0 = sqrt(1.0 + d * d) * weight;

        // use t_min and t_max to reduce the size of for loop
        REAL x0 = p0x + t_min * dx;
        REAL x1 = p0x + t_max * dx;
        INT n0 = INT((x0 - m_vbd_x0) / m_vox_size_j);
        INT n1 = INT((x1 - m_vbd_x0) / m_vox_size_j);
        INT n_min = (dx < 0) ? ( n1 < 0 ? 0 : n1 ) : ( n0 < 0 ? 0 : n0 );
        INT n_max = (dx < 0) ? ( n0 >= m_vdim_j ? m_vdim_j-1 : n0 ) : 
                               ( n1>=m_vdim_j ? m_vdim_j-1 : n1 );
        
        REAL t = (m_x_cuts[n_min] - p0x) / dx;
        REAL yc = (m_vbd_y1 - (p0y + t * dy)) / m_vox_size_i;
        REAL zc = (p0z + t * dz - m_vbd_z0) / m_vox_size_k;
        REAL yc_step = -m_vox_size_j / dx * dy / m_vox_size_i;
        REAL zc_step = m_vox_size_j / dx * dz / m_vox_size_k;
                
        do {                        

            if (yc>0 && yc<m_vdim_i && zc>0 && zc<m_vdim_k) {
                INT i = INT(yc);
                INT k = INT(zc);

#ifdef USE_TOF // TOF *****************************************************************************

#ifndef DIST_NO_SQRT
        			REAL tdx = m_x_cuts[n_min] - tbc_x;
    			    REAL tdy = m_y_cuts[m_vdim_i - i - 1] - tbc_y;
    			    REAL tdz = m_z_cuts[k] - tbc_z;
			    REAL vox_to_tbc_dist = sqrt(tdx*tdx + tdy*tdy + tdz*tdz);
#else
    			    REAL vx = m_x_cuts[n_min];
			    REAL vy = m_y_cuts[m_vdim_i - i - 1];
			    REAL vz = m_z_cuts[k];
			    REAL vox_to_tbc_dist = fabs(nrm_dx * vx + nrm_dy * vy + nrm_dz * vz + coef_d);
#endif
			    INT gw_bin = INT(vox_to_tbc_dist * m_gs_inv);
			    if (gw_bin < GWSAMPLESIZE) {
#ifdef USE_OMP
            		    #pragma omp atomic
#endif
				    img[k * m_vdim_ixj + n_min * m_vdim_i + i] += w0 * m_gw_lut[gw_bin];
			    }
			    
#else // non-TOF	 ******************************************************************************
		
#ifdef USE_OMP
            		#pragma omp atomic
#endif
                img[k * m_vdim_ixj + n_min * m_vdim_i + i] += w0;
                
#endif                
            }                       
        
            yc += yc_step;
            zc += zc_step;
            n_min ++;
        } while (n_min <= n_max);
		
	} else { // y-cuts
			
		REAL d = abs_dx / abs_dy;
		w0 = sqrt(1.0 + d * d) * weight;

        REAL y0 = p0y + t_min * dy;
        REAL y1 = p0y + t_max * dy;
        INT n0 = INT((y0 - m_vbd_y0) / m_vox_size_i);
        INT n1 = INT((y1 - m_vbd_y0) / m_vox_size_i);
        INT n_min = (dy < 0) ? ( n1 < 0 ? 0 : n1 ) : ( n0 < 0 ? 0 : n0 );
        INT n_max = (dy < 0) ? ( n0 >= m_vdim_i ? m_vdim_i-1 : n0 ) : 
                               ( n1 >= m_vdim_i ? m_vdim_i-1 : n1 );

        REAL t = (m_y_cuts[n_min] - p0y) / dy;
        REAL xc = (p0x + t * dx - m_vbd_x0) / m_vox_size_j;
        REAL zc = (p0z + t * dz - m_vbd_z0) / m_vox_size_k;
        REAL xc_step = m_vox_size_i / dy * dx / m_vox_size_j;
        REAL zc_step = m_vox_size_i / dy * dz / m_vox_size_k;
        
        do {
                                
            if (xc>0 && xc<m_vdim_j && zc>0 && zc<m_vdim_k) {
                INT j = INT(xc);
                INT k = INT(zc);

#ifdef USE_TOF // TOF *****************************************************************************


#ifndef DIST_NO_SQRT
        			REAL tdx = m_x_cuts[j] - tbc_x;
        			REAL tdy = m_y_cuts[n_min] - tbc_y;
        			REAL tdz = m_z_cuts[k] - tbc_z;
        			REAL vox_to_tbc_dist = sqrtf(tdx*tdx + tdy*tdy + tdz*tdz);
#else
        			REAL vx = m_x_cuts[j];
        			REAL vy = m_y_cuts[n_min];
        			REAL vz = m_z_cuts[k];
        			REAL vox_to_tbc_dist = fabs(nrm_dx * vx + nrm_dy * vy + nrm_dz * vz + coef_d);			
#endif
		        	INT gw_bin = INT(vox_to_tbc_dist * m_gs_inv);
		        	if (gw_bin < GWSAMPLESIZE) {
#ifdef USE_OMP
                		#pragma omp atomic
#endif			
		        		img[k * m_vdim_ixj + j * m_vdim_i + (m_vdim_i-n_min-1)] += w0 * m_gw_lut[gw_bin]; 
		        	}

#else // non-TOF **********************************************************************************

#ifdef USE_OMP
                #pragma omp atomic
#endif			                
                img[k * m_vdim_ixj + j * m_vdim_i + (m_vdim_i-n_min-1)] += w0;

#endif

            }
            
            xc += xc_step;
            zc += zc_step;
            n_min ++;
        } while (n_min <= n_max);
	}
	
}

REAL ImageRayTracer::fprojLinterp(REAL p0x, REAL p0y, REAL p0z,
                           		  REAL p1x, REAL p1y, REAL p1z,
                           		  const IMAGE_DATA_TYPE* img, 
                           		  const INT tbin_id)
{
#ifdef USE_TOF
    // normalized dir
    REAL nrm_dx;
    REAL nrm_dy;
    REAL nrm_dz;
    calcTOFLOREndPoints(tbin_id, 
                        p0x, p0y, p0z, 
                        p1x, p1y, p1z, 
                        nrm_dx, nrm_dy, nrm_dz);
    // timing bin center coord
    REAL tbc_x = (p0x + p1x) * 0.5;
    REAL tbc_y = (p0y + p1y) * 0.5;
    REAL tbc_z = (p0z + p1z) * 0.5;
#endif
    
    REAL dx = p1x - p0x;
    REAL dy = p1y - p0y;
    REAL dz = p1z - p0z;
    REAL t_min, t_max;
    if (!hitCheck(p0x, p0y, p0z, dx, dy, dz, t_min, t_max)) {
        return 0;
    }
#ifdef USE_TOF

    t_min = std::max(t_min, 0.0);
    t_max = std::min(t_max, 1.0);

    // double check if the ray segment is still inside the FOV
    if (t_min > t_max) {
        return 0.0;
    }

#ifdef DIST_NO_SQRT // required by type-II distance (if enabled)
    REAL coef_d = -(nrm_dx * tbc_x + nrm_dy * tbc_y + nrm_dz * tbc_z);
#endif
#endif    
    
    REAL out = 0.0;
    REAL w0;
    REAL abs_dx = fabs(dx);
    REAL abs_dy = fabs(dy);
    REAL abs_dz = fabs(dz);
        
	if (abs_dx > abs_dy) { // x-cuts
		
        REAL d = abs_dy / abs_dx;
		w0 = sqrt(1.0 + d * d);

        // use t_min and t_max to reduce the size of for loop
        REAL x0 = p0x + t_min * dx;
        REAL x1 = p0x + t_max * dx;
        INT n0 = INT((x0 - m_vbd_x0) / m_vox_size_j);
        INT n1 = INT((x1 - m_vbd_x0) / m_vox_size_j);
        INT n_min = (dx < 0) ? ( n1 < 0 ? 0 : n1 ) : ( n0 < 0 ? 0 : n0 );
        INT n_max = (dx < 0) ? ( n0 >= m_vdim_j ? m_vdim_j-1 : n0 ) : 
                               ( n1>=m_vdim_j ? m_vdim_j-1 : n1 );
        
        REAL t = (m_x_cuts[n_min] - p0x) / dx;
        REAL yc = (m_vbd_y1 - (p0y + t * dy)) / m_vox_size_i;
        REAL zc = (p0z + t * dz - m_vbd_z0) / m_vox_size_k;
        REAL yc_step = -m_vox_size_j / dx * dy / m_vox_size_i;
        REAL zc_step = m_vox_size_j / dx * dz / m_vox_size_k;
                
        do {                        
            
            if (yc >= 0.5 && 
                yc <= m_vdim_i-0.5 && 
                zc >= 0.5 && 
                zc <= m_vdim_k-0.5) {
            
                INT i = INT(yc - 0.5);
                INT k = INT(zc - 0.5);
		        INT addr0 = k * m_vdim_ixj + n_min * m_vdim_i + i;
		        REAL c0 = yc - 0.5 - i;
		        REAL c1 = zc - 0.5 - k;

#if USE_TOF // TOF ********************************************************************************
//#ifndef DIST_NO_SQRT
                // not support yet!
//#else
    			    REAL vx = m_x_cuts[n_min];
			    REAL vy = m_y_cuts[m_vdim_i - i - 1];
			    REAL vz = m_z_cuts[k];
			    REAL vox_to_tbc_dist = fabs(nrm_dx * vx + nrm_dy * vy + nrm_dz * vz + coef_d);
//#endif
			    INT gw_bin = INT(vox_to_tbc_dist * m_gs_inv);
			    if (gw_bin < GWSAMPLESIZE) {
				    out += img[addr0] * m_gw_lut[gw_bin] * (1 - c0) * (1 - c1);
			    }
			    
			    if (i + 1 < m_vdim_i) {
			        vy = m_y_cuts[m_vdim_i - (i+1) - 1];
			        vz = m_z_cuts[k];
			        vox_to_tbc_dist = fabs(nrm_dx * vx + nrm_dy * vy + nrm_dz * vz + coef_d);
			        gw_bin = INT(vox_to_tbc_dist * m_gs_inv);
			        if (gw_bin < GWSAMPLESIZE) {
				        out += img[addr0 + 1] * m_gw_lut[gw_bin] * c0 * (1 - c1);
			        }
			    }

                if (k + 1 < m_vdim_k) {
                    vy = m_y_cuts[m_vdim_i - i - 1];
			        vz = m_z_cuts[k + 1];
			        vox_to_tbc_dist = fabs(nrm_dx * vx + nrm_dy * vy + nrm_dz * vz + coef_d);
			        gw_bin = INT(vox_to_tbc_dist * m_gs_inv);
			        if (gw_bin < GWSAMPLESIZE) {
				        out += img[addr0 + m_vdim_ixj] * m_gw_lut[gw_bin] * (1 - c0) * c1;
			        }
			        
			        if (i + 1 < m_vdim_i) {
			            vy = m_y_cuts[m_vdim_i - (i+1) - 1];
			            vz = m_z_cuts[k + 1];
			            vox_to_tbc_dist = fabs(nrm_dx * vx + nrm_dy * vy + nrm_dz * vz + coef_d);
			            gw_bin = INT(vox_to_tbc_dist * m_gs_inv);
			            if (gw_bin < GWSAMPLESIZE) {
				            out += img[addr0 + m_vdim_ixj + 1] * m_gw_lut[gw_bin] * c0 * c1;
			            }
			        } 
                }

#else // non-TOF **********************************************************************************		        
                out += img[addr0] * (1 - c0) * (1 - c1);
                if (i + 1 < m_vdim_i) {
                    out += img[addr0 + 1] * c0 * (1 - c1);
                }            
                if (k + 1 < m_vdim_k) {
                    out += img[addr0 + m_vdim_ixj] * (1 - c0) * c1;
                    if (i + 1 < m_vdim_i) {
                        out += img[addr0 + m_vdim_ixj + 1] * c0 * c1;                    
                    }
                }
#endif                
            }                       
        
            yc += yc_step;
            zc += zc_step;
            n_min ++;
        } while (n_min <= n_max);
		
	} else { // y-cuts
			
		REAL d = abs_dx / abs_dy;
		w0 = sqrt(1.0 + d * d);

        REAL y0 = p0y + t_min * dy;
        REAL y1 = p0y + t_max * dy;
        INT n0 = INT((y0 - m_vbd_y0) / m_vox_size_i);
        INT n1 = INT((y1 - m_vbd_y0) / m_vox_size_i);        
        INT n_min = (dy < 0) ? ( n1 < 0 ? 0 : n1 ) : ( n0 < 0 ? 0 : n0 );
        INT n_max = (dy < 0) ? ( n0 >= m_vdim_i ? m_vdim_i-1 : n0 ) : 
                               ( n1 >= m_vdim_i ? m_vdim_i-1 : n1 );

        REAL t = (m_y_cuts[n_min] - p0y) / dy;
        REAL xc = (p0x + t * dx - m_vbd_x0) / m_vox_size_j;
        REAL zc = (p0z + t * dz - m_vbd_z0) / m_vox_size_k;
        REAL xc_step = m_vox_size_i / dy * dx / m_vox_size_j;
        REAL zc_step = m_vox_size_i / dy * dz / m_vox_size_k;
        
        do {
                                
            if (xc >= 0.5 && 
                xc <= m_vdim_j-0.5 && 
                zc>=0.5 && 
                zc<=m_vdim_k-0.5) {
                
                INT j = INT(xc - 0.5);
                INT k = INT(zc - 0.5);
                INT addr0 = k * m_vdim_ixj + j * m_vdim_i + (m_vdim_i-n_min-1);
                
                REAL c0 = xc - 0.5 - j;
                REAL c1 = zc - 0.5 - k;

#if USE_TOF // TOF ********************************************************************************

//#ifndef DIST_NO_SQRT
                // not support yet!
//#else
        			REAL vx = m_x_cuts[j];
        			REAL vy = m_y_cuts[n_min];
        			REAL vz = m_z_cuts[k];
        			REAL vox_to_tbc_dist = fabs(nrm_dx * vx + nrm_dy * vy + nrm_dz * vz + coef_d);			
//#endif
		        	INT gw_bin = INT(vox_to_tbc_dist * m_gs_inv);
		        	if (gw_bin < GWSAMPLESIZE) {
		        		out += img[addr0] * m_gw_lut[gw_bin] * (1 - c0) * (1 - c1); 
		        	}

                if (j+1 < m_vdim_j) {
                    vx = m_x_cuts[j+1];
                    vz = m_z_cuts[k];
                    vox_to_tbc_dist = fabs(nrm_dx * vx + nrm_dy * vy + nrm_dz * vz + coef_d);			
		            gw_bin = INT(vox_to_tbc_dist * m_gs_inv);
    		            	if (gw_bin < GWSAMPLESIZE) {
		        		    out += img[addr0 + m_vdim_i] * m_gw_lut[gw_bin] * c0 * (1 - c1); 
		        	    }
                }
                
                if (k + 1 < m_vdim_k) {
                    vx = m_x_cuts[j];
                    vz = m_z_cuts[k + 1];
                    vox_to_tbc_dist = fabs(nrm_dx * vx + nrm_dy * vy + nrm_dz * vz + coef_d);			
		            gw_bin = INT(vox_to_tbc_dist * m_gs_inv);
    		            	if (gw_bin < GWSAMPLESIZE) {
		        		    out += img[addr0 + m_vdim_ixj] * m_gw_lut[gw_bin] * (1 - c0) * c1; 
		        	    }
		        	    
		        	    if (j + 1 < m_vdim_j) {
                        vx = m_x_cuts[j + 1];
                        vz = m_z_cuts[k + 1];
                        vox_to_tbc_dist = fabs(nrm_dx * vx + nrm_dy * vy + nrm_dz * vz + coef_d);			
        		            gw_bin = INT(vox_to_tbc_dist * m_gs_inv);
        		            	if (gw_bin < GWSAMPLESIZE) {
        		        		    out += img[addr0 + m_vdim_ixj + m_vdim_i] * m_gw_lut[gw_bin] * c0 * c1; 
        		        	    }
		        	    }
		        	    
                }

#else // non-TOF **********************************************************************************                
                
                out += img[addr0] * (1-c0) * (1-c1);
                if (j+1 < m_vdim_j) {
                    out += img[addr0 + m_vdim_i] * c0 * (1-c1);
                }
                if (k + 1 < m_vdim_k) {
                    out += img[addr0 + m_vdim_ixj] * (1-c0) * c1;
                    if (j+1 < m_vdim_j) {
                        out += img[addr0 + m_vdim_ixj + m_vdim_i] * c0 * c1;
                    }
                }

#endif
                
            }
            
            xc += xc_step;
            zc += zc_step;
            n_min ++;
        } while (n_min <= n_max);
	}
	
    return (out * w0);
}

void ImageRayTracer::bprojLinterp(REAL p0x, REAL p0y, REAL p0z,
                           REAL p1x, REAL p1y, REAL p1z,
                           const REAL weight, IMAGE_DATA_TYPE* img, 
                           const INT tbin_id)
{
#ifdef USE_TOF
    // normalized dir
    REAL nrm_dx;
    REAL nrm_dy;
    REAL nrm_dz;
    calcTOFLOREndPoints(tbin_id, 
                        p0x, p0y, p0z, 
                        p1x, p1y, p1z, 
                        nrm_dx, nrm_dy, nrm_dz);
    // timing bin center coord
    REAL tbc_x = (p0x + p1x) * 0.5;
    REAL tbc_y = (p0y + p1y) * 0.5;
    REAL tbc_z = (p0z + p1z) * 0.5;
#endif
    
    REAL dx = p1x - p0x;
    REAL dy = p1y - p0y;
    REAL dz = p1z - p0z;
    REAL t_min, t_max;
    if (!hitCheck(p0x, p0y, p0z, dx, dy, dz, t_min, t_max)) {
        return;
    }
#ifdef USE_TOF

    t_min = std::max(t_min, 0.0);
    t_max = std::min(t_max, 1.0);

    // double check if the ray segment is still inside the FOV
    if (t_min > t_max) {
        return;
    }

#ifdef DIST_NO_SQRT // required by type-II distance (if enabled)
    REAL coef_d = -(nrm_dx * tbc_x + nrm_dy * tbc_y + nrm_dz * tbc_z);
#endif
#endif    
       
    REAL w0;
    REAL abs_dx = fabs(dx);
    REAL abs_dy = fabs(dy);
    REAL abs_dz = fabs(dz);
        
	if (abs_dx > abs_dy) { // x-cuts
		
        REAL d = abs_dy / abs_dx;
		w0 = sqrt(1.0 + d * d) * weight;
        // use t_min and t_max to reduce the size of for loop
        
        REAL x0 = p0x + t_min * dx;
        REAL x1 = p0x + t_max * dx;
        INT n0 = INT((x0 - m_vbd_x0) / m_vox_size_j);
        INT n1 = INT((x1 - m_vbd_x0) / m_vox_size_j);
        INT n_min = (dx < 0) ? ( n1 < 0 ? 0 : n1 ) : ( n0 < 0 ? 0 : n0 );
        INT n_max = (dx < 0) ? ( n0 >= m_vdim_j ? m_vdim_j-1 : n0 ) : 
                               ( n1>=m_vdim_j ? m_vdim_j-1 : n1 );
        
        REAL t = (m_x_cuts[n_min] - p0x) / dx;
        REAL yc = (m_vbd_y1 - (p0y + t * dy)) / m_vox_size_i;
        REAL zc = (p0z + t * dz - m_vbd_z0) / m_vox_size_k;
        REAL yc_step = -m_vox_size_j / dx * dy / m_vox_size_i;
        REAL zc_step = m_vox_size_j / dx * dz / m_vox_size_k;
                
        do {
                                
            if (yc >= 0.5 && 
                yc <= m_vdim_i-0.5 && 
                zc >= 0.5 && 
                zc <= m_vdim_k-0.5) {
            
                INT i = INT(yc - 0.5);
                INT k = INT(zc - 0.5);
		        INT addr0 = k * m_vdim_ixj + n_min * m_vdim_i + i;
		        REAL c0 = yc - 0.5 - i;
		        REAL c1 = zc - 0.5 - k;

#if USE_TOF // TOF ********************************************************************************

//#ifndef DIST_NO_SQRT
                // not support yet!
//#else
    			    REAL vx = m_x_cuts[n_min];
			    REAL vy = m_y_cuts[m_vdim_i - i - 1];
			    REAL vz = m_z_cuts[k];
			    REAL vox_to_tbc_dist = fabs(nrm_dx * vx + nrm_dy * vy + nrm_dz * vz + coef_d);
//#endif
			    INT gw_bin = INT(vox_to_tbc_dist * m_gs_inv);
			    if (gw_bin < GWSAMPLESIZE) {
#ifdef USE_OMP
            		    #pragma omp atomic
#endif
				    img[addr0] += w0 * m_gw_lut[gw_bin] * (1 - c0) * (1 - c1);
			    }
			    
			    if (i + 1 < m_vdim_i) {
			        vy = m_y_cuts[m_vdim_i - (i+1) - 1];
			        vz = m_z_cuts[k];
			        vox_to_tbc_dist = fabs(nrm_dx * vx + nrm_dy * vy + nrm_dz * vz + coef_d);
			        gw_bin = INT(vox_to_tbc_dist * m_gs_inv);
			        if (gw_bin < GWSAMPLESIZE) {
#ifdef USE_OMP
            		        #pragma omp atomic
#endif
				        img[addr0 + 1] += w0 * m_gw_lut[gw_bin] * c0 * (1 - c1);
			        }
			    }

                if (k + 1 < m_vdim_k) {
                    vy = m_y_cuts[m_vdim_i - i - 1];
			        vz = m_z_cuts[k + 1];
			        vox_to_tbc_dist = fabs(nrm_dx * vx + nrm_dy * vy + nrm_dz * vz + coef_d);
			        gw_bin = INT(vox_to_tbc_dist * m_gs_inv);
			        if (gw_bin < GWSAMPLESIZE) {
#ifdef USE_OMP
            		        #pragma omp atomic
#endif
				        img[addr0 + m_vdim_ixj] += w0 * m_gw_lut[gw_bin] * (1 - c0) * c1;
			        }
			        
			        if (i + 1 < m_vdim_i) {
			            vy = m_y_cuts[m_vdim_i - (i+1) - 1];
			            vz = m_z_cuts[k + 1];
			            vox_to_tbc_dist = fabs(nrm_dx * vx + nrm_dy * vy + nrm_dz * vz + coef_d);
			            gw_bin = INT(vox_to_tbc_dist * m_gs_inv);
			            if (gw_bin < GWSAMPLESIZE) {
#ifdef USE_OMP
            		            #pragma omp atomic
#endif
				            img[addr0 + m_vdim_ixj + 1] += w0 * m_gw_lut[gw_bin] * c0 * c1;
			            }
			        } 
                }
                
#else // nonTOF ***********************************************************************************

#if USE_OMP
                #pragma omp atomic
#endif                		        
                img[addr0] += w0 * (1 - c0) * (1 - c1);
                if (i + 1 < m_vdim_i) {
#if USE_OMP
                    #pragma omp atomic
#endif                		        
                    img[addr0 + 1] += w0 * c0 * (1 - c1);
                }            
                if (k + 1 < m_vdim_k) {
#if USE_OMP
                    #pragma omp atomic
#endif                		        
                    img[addr0 + m_vdim_ixj] += w0 * (1 - c0) * c1;
                    if (i + 1 < m_vdim_i) {
#if USE_OMP
                        #pragma omp atomic
#endif                		        
                        img[addr0 + m_vdim_ixj + 1] += w0 * c0 * c1;                    
                    }
                }

#endif
                
            }                       
        
            yc += yc_step;
            zc += zc_step;
            n_min ++;
        } while (n_min <= n_max);
        
		
	} else { // y-cuts
			
		REAL d = abs_dx / abs_dy;
		w0 = sqrt(1.0 + d * d) * weight;

        REAL y0 = p0y + t_min * dy;
        REAL y1 = p0y + t_max * dy;
        INT n0 = INT((y0 - m_vbd_y0) / m_vox_size_i);
        INT n1 = INT((y1 - m_vbd_y0) / m_vox_size_i);        
        INT n_min = (dy < 0) ? ( n1 < 0 ? 0 : n1 ) : ( n0 < 0 ? 0 : n0 );
        INT n_max = (dy < 0) ? ( n0 >= m_vdim_i ? m_vdim_i-1 : n0 ) : 
                               ( n1 >= m_vdim_i ? m_vdim_i-1 : n1 );

        REAL t = (m_y_cuts[n_min] - p0y) / dy;
        REAL xc = (p0x + t * dx - m_vbd_x0) / m_vox_size_j;
        REAL zc = (p0z + t * dz - m_vbd_z0) / m_vox_size_k;
        REAL xc_step = m_vox_size_i / dy * dx / m_vox_size_j;
        REAL zc_step = m_vox_size_i / dy * dz / m_vox_size_k;
        
        do {
                                
            if (xc >= 0.5 && 
                xc <= m_vdim_j-0.5 && 
                zc>=0.5 && 
                zc<=m_vdim_k-0.5) {
                
                INT j = INT(xc - 0.5);
                INT k = INT(zc - 0.5);
                INT addr0 = k * m_vdim_ixj + j * m_vdim_i + (m_vdim_i-n_min-1);                
                REAL c0 = xc - 0.5 - j;
                REAL c1 = zc - 0.5 - k;

#if USE_TOF // TOF ********************************************************************************

//#ifndef DIST_NO_SQRT
                // not support yet!
//#else
        			REAL vx = m_x_cuts[j];
        			REAL vy = m_y_cuts[n_min];
        			REAL vz = m_z_cuts[k];
        			REAL vox_to_tbc_dist = fabs(nrm_dx * vx + nrm_dy * vy + nrm_dz * vz + coef_d);			
//#endif
		        	INT gw_bin = INT(vox_to_tbc_dist * m_gs_inv);
		        	if (gw_bin < GWSAMPLESIZE) {
#ifdef USE_OMP
                		#pragma omp atomic
#endif			
		        		img[addr0] += w0 * m_gw_lut[gw_bin] * (1 - c0) * (1 - c1); 
		        	}

                if (j+1 < m_vdim_j) {
                    vx = m_x_cuts[j+1];
                    vz = m_z_cuts[k];
                    vox_to_tbc_dist = fabs(nrm_dx * vx + nrm_dy * vy + nrm_dz * vz + coef_d);			
		            gw_bin = INT(vox_to_tbc_dist * m_gs_inv);
    		            	if (gw_bin < GWSAMPLESIZE) {
#ifdef USE_OMP
                		    #pragma omp atomic
#endif			
		        		    img[addr0 + m_vdim_i] += w0 * m_gw_lut[gw_bin] * c0 * (1 - c1); 
		        	    }
                }
                
                if (k + 1 < m_vdim_k) {
                    vx = m_x_cuts[j];
                    vz = m_z_cuts[k + 1];
                    vox_to_tbc_dist = fabs(nrm_dx * vx + nrm_dy * vy + nrm_dz * vz + coef_d);			
		            gw_bin = INT(vox_to_tbc_dist * m_gs_inv);
    		            	if (gw_bin < GWSAMPLESIZE) {
#ifdef USE_OMP
                		    #pragma omp atomic
#endif			
		        		    img[addr0 + m_vdim_ixj] += w0 * m_gw_lut[gw_bin] * (1 - c0) * c1; 
		        	    }
		        	    
		        	    if (j + 1 < m_vdim_j) {
                        vx = m_x_cuts[j + 1];
                        vz = m_z_cuts[k + 1];
                        vox_to_tbc_dist = fabs(nrm_dx * vx + nrm_dy * vy + nrm_dz * vz + coef_d);			
        		            gw_bin = INT(vox_to_tbc_dist * m_gs_inv);
        		            	if (gw_bin < GWSAMPLESIZE) {
#ifdef USE_OMP
                    		    #pragma omp atomic
#endif			
        		        		    img[addr0 + m_vdim_ixj + m_vdim_i] += w0 * m_gw_lut[gw_bin] * c0 * c1; 
        		        	    }
		        	    }
		        	    
                }


#else // nonTOF ***********************************************************************************                
                
#if USE_OMP
                #pragma omp atomic
#endif                		        
                
                img[addr0] += w0 * (1-c0) * (1-c1);
                if (j+1 < m_vdim_j) {
#if USE_OMP
                    #pragma omp atomic
#endif                		        
                    img[addr0 + m_vdim_i] += w0 * c0 * (1-c1);
                }
                if (k + 1 < m_vdim_k) {
#if USE_OMP
                    #pragma omp atomic
#endif                		        
                    img[addr0 + m_vdim_ixj] += w0 * (1-c0) * c1;
                    if (j+1 < m_vdim_j) {
#if USE_OMP
                        #pragma omp atomic
#endif                		        
                        img[addr0 + m_vdim_ixj + m_vdim_i] += w0 * c0 * c1;
                    }
                }

#endif                
                
            }
            
            xc += xc_step;
            zc += zc_step;
            n_min ++;
        } while (n_min <= n_max);
	}
	
}

#endif

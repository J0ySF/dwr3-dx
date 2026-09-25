#include "../propagation_medium.cuh"

/// Lambda in the numerical scheme
#define L1 0.57735026918962576450914878050196f
/// The reciprocal of Lambda in the numerical scheme
#define L1_REC 1.7320508075688772935274463415059f
/// Lambda squared in the numerical scheme
#define L2 0.33333333333333333333333333333333f
/// d_1 in the numerical scheme
#define D1 0.33333333333333333333333333333333f

// Coordinate linearization used to access p buffers
#define P_INDEX(X, Y, Z) ((X) + (size_x + pad_x) * ((Y) + size_y * (Z)))
// Coordinate linearization used to access x boundary buffers
#define B_X_INDEX(Y, Z) ((Y) + size_y * (Z))
// Coordinate linearization used to access y boundary buffers
#define B_Y_INDEX(X, Z) ((X) + size_x * (Z))
// Coordinate linearization used to access z boundary buffers
#define B_Z_INDEX(X, Y) ((X) + size_x * (Y))

/// Kernel block sizes
#define SIM_BORDER_BLOCK_DIM_X 32
#define SIM_BORDER_BLOCK_DIM_Y 4
#define SIM_CENTER_BLOCK_DIM_X 32
#define SIM_CENTER_BLOCK_DIM_Y 4
/// The center kernel uses a loop tiling technique to handle multiple z axis coordinates with a loop, each block contains
/// only SIM_CENTER_BLOCK_DIM_X x SIM_CENTER_BLOCK_DIM_Y threads
#ifndef SIM_CENTER_BLOCK_DIM_Z
/// Good value for real-time iteration on small-to-medium instances on GeForce RTX 40 series GPUs
#define SIM_CENTER_BLOCK_DIM_Z 10
#endif
/// Each thread in each warp for the center section kernel computes two adjacent x coordinates (with the first and last threads computing only one)
#define SIM_CENTER_BLOCK_COMPUTE_DIM_X (SIM_CENTER_BLOCK_DIM_X * 2 - 2)

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Macros used to implement kernel boundary parameters passage,
/// This is used since the __restrict__ attribute is ignored for non-base types such as arrays of __restrict__
/// pointers and structs containing __restrict__ pointers

#if DWR3_BOUNDARY_FILTER_ORDER > 6
#define MACRO_LIST_EXPANSION_7(PRE_A, PRE_C, POST_C, POST_A) , PRE_A PRE_C##7##POST_C POST_A
#define MACRO_LIST_EXPANSION_6(PRE_A, PRE_C, POST_C, POST_A)                                                           \
    , PRE_A PRE_C##6##POST_C POST_A MACRO_LIST_EXPANSION_7(PRE_A, PRE_C, POST_C, POST_A)
#else
#define MACRO_LIST_EXPANSION_6(PRE_A, PRE_C, POST_C, POST_A)
#endif

#if DWR3_BOUNDARY_FILTER_ORDER > 4
#define MACRO_LIST_EXPANSION_5(PRE_A, PRE_C, POST_C, POST_A)                                                           \
    , PRE_A PRE_C##5##POST_C POST_A MACRO_LIST_EXPANSION_6(PRE_A, PRE_C, POST_C, POST_A)
#define MACRO_LIST_EXPANSION_4(PRE_A, PRE_C, POST_C, POST_A)                                                           \
    , PRE_A PRE_C##4##POST_C POST_A MACRO_LIST_EXPANSION_5(PRE_A, PRE_C, POST_C, POST_A)
#else
#define MACRO_LIST_EXPANSION_4(PRE_A, PRE_C, POST_C, POST_A)
#endif

#if DWR3_BOUNDARY_FILTER_ORDER > 2
#define MACRO_LIST_EXPANSION_3(PRE_A, PRE_C, POST_C, POST_A)                                                           \
    , PRE_A PRE_C##3##POST_C POST_A MACRO_LIST_EXPANSION_4(PRE_A, PRE_C, POST_C, POST_A)
#define MACRO_LIST_EXPANSION_2(PRE_A, PRE_C, POST_C, POST_A)                                                           \
    , PRE_A PRE_C##2##POST_C POST_A MACRO_LIST_EXPANSION_3(PRE_A, PRE_C, POST_C, POST_A)
#else
#define MACRO_LIST_EXPANSION_2(PRE_A, PRE_C, POST_C, POST_A)
#endif

#if DWR3_BOUNDARY_FILTER_ORDER > 1
#define MACRO_LIST_EXPANSION_1(PRE_A, PRE_C, POST_C, POST_A)                                                           \
    , PRE_A PRE_C##1##POST_C POST_A MACRO_LIST_EXPANSION_2(PRE_A, PRE_C, POST_C, POST_A)
#else
#define MACRO_LIST_EXPANSION_1(PRE_A, PRE_C, POST_C, POST_A)
#endif

#define MACRO_LIST_EXPANSION_0(PRE_A, PRE_C, POST_C, POST_A)                                                           \
    PRE_A PRE_C##0##POST_C POST_A MACRO_LIST_EXPANSION_1(PRE_A, PRE_C, POST_C, POST_A)

#define KERNEL_BOUNDARY_FORMAL_PARAMS(SUFFIX)                                                                          \
    float *__restrict__ b_##SUFFIX##_g,                                                                                \
            float *__restrict__ b_##SUFFIX##_x0 MACRO_LIST_EXPANSION_1(, const float *__restrict__ b_##SUFFIX##_x,     \
                                                                       , ),                                            \
            float *__restrict__ b_##SUFFIX##_y0 MACRO_LIST_EXPANSION_1(, const float *__restrict__ b_##SUFFIX##_y,     \
                                                                       , ),                                            \
            const __grid_constant__ dwr3::digital_impedance_filter_coefficients d_##SUFFIX

static_assert(DWR3_BOUNDARY_FILTER_ORDER <= 8,
              "DWR3_BOUNDARY_FILTER_ORDER greater than 8 is currently unsupported");

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Updates a boundary filter state
inline __device__ void
update_filter(const float o, const float n, float *__restrict__ const b__g, // NOLINT(*-reserved-identifier)
              float *__restrict__ const b__x0 MACRO_LIST_EXPANSION_1(, const float *__restrict__ const b__x, ,), // NOLINT(*-reserved-identifier)
              float *__restrict__ const b__y0 MACRO_LIST_EXPANSION_1(, const float *__restrict__ const b__y, ,), // NOLINT(*-reserved-identifier)
              const dwr3::digital_impedance_filter_coefficients &d_) {
    float g = *b__g;
    const float x0 = (L1_REC * (n - o) - g) * dif_b_0_rec(d_);
    const float y0 = dif_b(d_, 0) * x0 + g;

    g = dif_b(d_, 1) * x0 - dif_a(d_, 1) * y0;
#if DWR3_BOUNDARY_FILTER_ORDER > 1
    g += dif_b(d_, 2) * *b__x1 - dif_a(d_, 2) * *b__y1;
#endif
#if DWR3_BOUNDARY_FILTER_ORDER > 2
    g += dif_b(d_, 3) * *b__x2 - dif_a(d_, 3) * *b__y2;
    g += dif_b(d_, 4) * *b__x3 - dif_a(d_, 4) * *b__y3;
#endif
#if DWR3_BOUNDARY_FILTER_ORDER > 4
    g += dif_b(d_, 5) * *b__x4 - dif_a(d_, 5) * *b__y4;
    g += dif_b(d_, 6) * *b__x5 - dif_a(d_, 6) * *b__y5;
#endif
#if DWR3_BOUNDARY_FILTER_ORDER > 6
    g += dif_b(d_, 7) * *b__x6 - dif_a(d_, 7) * *b__y6;
    g += dif_b(d_, 8) * *b__x7 - dif_a(d_, 8) * *b__y7;
#endif

    *b__g = g;
    *b__x0 = x0;
    *b__y0 = y0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Set of macros used to parameterize the generic boundary kernel implementations

/// First step macros
/**/
/// Gather X- boundary parameters WITH a coordinate check
#define UPDATE_MACRO_X1N                                                                                               \
    int xnd = -1;                                                                                                      \
    if (x == 0) {                                                                                                      \
        xnd = 1;                                                                                                       \
        b += dif_b_0_rec(d_xn);                                                                                        \
        g += b_xn_g[B_X_INDEX(y, z)] * dif_b_0_rec(d_xn);                                                              \
    }
/// Gather X- boundary parameters WITHOUT a coordinate check
#define UPDATE_MACRO_X1NS                                                                                              \
    {                                                                                                                  \
        b += dif_b_0_rec(d_xn);                                                                                        \
        g += b_xn_g[B_X_INDEX(y, z)] * dif_b_0_rec(d_xn);                                                              \
    }
/// Gather X+ boundary parameters WITH a coordinate check
#define UPDATE_MACRO_X1P                                                                                               \
    int xpd = 1;                                                                                                       \
    if (x == size_x - 1) {                                                                                             \
        xpd = -1;                                                                                                      \
        b += dif_b_0_rec(d_xp);                                                                                        \
        g += b_xp_g[B_X_INDEX(y, z)] * dif_b_0_rec(d_xp);                                                              \
    }
/// Gather X+ boundary parameters WITHOUT a coordinate check
#define UPDATE_MACRO_X1PS                                                                                              \
    {                                                                                                                  \
        b += dif_b_0_rec(d_xp);                                                                                        \
        g += b_xp_g[B_X_INDEX(y, z)] * dif_b_0_rec(d_xp);                                                              \
    }
/// Gather Y- boundary parameters WITH a coordinate check
#define UPDATE_MACRO_Y1N                                                                                               \
    int ynd = -(size_x + pad_x);                                                                                       \
    if (y == 0) {                                                                                                      \
        ynd = (size_x + pad_x);                                                                                        \
        b += dif_b_0_rec(d_yn);                                                                                        \
        g += b_yn_g[B_Y_INDEX(x, z)] * dif_b_0_rec(d_yn);                                                              \
    }
/// Gather Y- boundary parameters WITHOUT a coordinate check
#define UPDATE_MACRO_Y1NS                                                                                              \
    {                                                                                                                  \
        b += dif_b_0_rec(d_yn);                                                                                        \
        g += b_yn_g[B_Y_INDEX(x, z)] * dif_b_0_rec(d_yn);                                                              \
    }
/// Gather Y+ boundary parameters WITH a coordinate check
#define UPDATE_MACRO_Y1P                                                                                               \
    int ypd = (size_x + pad_x);                                                                                        \
    if (y == size_y - 1) {                                                                                             \
        ypd = -(size_x + pad_x);                                                                                       \
        b += dif_b_0_rec(d_yp);                                                                                        \
        g += b_yp_g[B_Y_INDEX(x, z)] * dif_b_0_rec(d_yp);                                                              \
    }
/// Gather Y+ boundary parameters WITHOUT a coordinate check
#define UPDATE_MACRO_Y1PS                                                                                              \
    {                                                                                                                  \
        b += dif_b_0_rec(d_yp);                                                                                        \
        g += b_yp_g[B_Y_INDEX(x, z)] * dif_b_0_rec(d_yp);                                                              \
    }
/// Gather Z- boundary parameters WITH a coordinate check
#define UPDATE_MACRO_Z1N                                                                                               \
    int znd = -(size_x + pad_x) * size_y;                                                                              \
    if (z == 0) {                                                                                                      \
        znd = (size_x + pad_x) * size_y;                                                                               \
        b += dif_b_0_rec(d_zn);                                                                                        \
        g += b_zn_g[B_Z_INDEX(x, y)] * dif_b_0_rec(d_zn);                                                              \
    }
/// Gather Z- boundary parameters WITHOUT a coordinate check
#define UPDATE_MACRO_Z1NS                                                                                              \
    {                                                                                                                  \
        b += dif_b_0_rec(d_zn);                                                                                        \
        g += b_zn_g[B_Z_INDEX(x, y)] * dif_b_0_rec(d_zn);                                                              \
    }
/// Gather Z+ boundary parameters WITH a coordinate check
#define UPDATE_MACRO_Z1P                                                                                               \
    int zpd = (size_x + pad_x) * size_y;                                                                               \
    if (z == size_z - 1) {                                                                                             \
        zpd = -(size_x + pad_x) * size_y;                                                                              \
        b += dif_b_0_rec(d_zp);                                                                                        \
        g += b_zp_g[B_Z_INDEX(x, y)] * dif_b_0_rec(d_zp);                                                              \
    }
/// Gather Z+ boundary parameters WITHOUT a coordinate check
#define UPDATE_MACRO_Z1PS                                                                                              \
    {                                                                                                                  \
        b += dif_b_0_rec(d_zp);                                                                                        \
        g += b_zp_g[B_Z_INDEX(x, y)] * dif_b_0_rec(d_zp);                                                              \
    }

/// Actual parameters used by macros for update_filter in the second step
#define BOUNDARY_ACTUAL_PARAMETERS_KERNEL_FILTER(SUF, I)                                                               \
    b_##SUF##_g + (I), MACRO_LIST_EXPANSION_0(, b_##SUF##_x, , +(I)), MACRO_LIST_EXPANSION_0(, b_##SUF##_y, , +(I)),   \
            d_##SUF

/// Second step macros
/**/
/// Update X- boundary state WITH a coordinate check
#define UPDATE_MACRO_X2N                                                                                               \
    if (x == 0) { update_filter(o, n, BOUNDARY_ACTUAL_PARAMETERS_KERNEL_FILTER(xn, B_X_INDEX(y, z))); }
/// Update X- boundary state WITHOUT a coordinate check
#define UPDATE_MACRO_X2NS                                                                                              \
    { update_filter(o, n, BOUNDARY_ACTUAL_PARAMETERS_KERNEL_FILTER(xn, B_X_INDEX(y, z))); }
/// Update X+ boundary state WITH a coordinate check
#define UPDATE_MACRO_X2P                                                                                               \
    if (x == size_x - 1) { update_filter(o, n, BOUNDARY_ACTUAL_PARAMETERS_KERNEL_FILTER(xp, B_X_INDEX(y, z))); }
/// Update X+ boundary state WITHOUT a coordinate check
#define UPDATE_MACRO_X2PS                                                                                              \
    { update_filter(o, n, BOUNDARY_ACTUAL_PARAMETERS_KERNEL_FILTER(xp, B_X_INDEX(y, z))); }
/// Update Y- boundary state WITH a coordinate check
#define UPDATE_MACRO_Y2N                                                                                               \
    if (y == 0) { update_filter(o, n, BOUNDARY_ACTUAL_PARAMETERS_KERNEL_FILTER(yn, B_Y_INDEX(x, z))); }
/// Update Y- boundary state WITHOUT a coordinate check
#define UPDATE_MACRO_Y2NS                                                                                              \
    { update_filter(o, n, BOUNDARY_ACTUAL_PARAMETERS_KERNEL_FILTER(yn, B_Y_INDEX(x, z))); }
/// Update Y+ boundary state WITH a coordinate check
#define UPDATE_MACRO_Y2P                                                                                               \
    if (y == size_y - 1) { update_filter(o, n, BOUNDARY_ACTUAL_PARAMETERS_KERNEL_FILTER(yp, B_Y_INDEX(x, z))); }
/// Update Y+ boundary state WITHOUT a coordinate check
#define UPDATE_MACRO_Y2PS                                                                                              \
    { update_filter(o, n, BOUNDARY_ACTUAL_PARAMETERS_KERNEL_FILTER(yp, B_Y_INDEX(x, z))); }
/// Update Z- boundary state WITH a coordinate check
#define UPDATE_MACRO_Z2N                                                                                               \
    if (z == 0) { update_filter(o, n, BOUNDARY_ACTUAL_PARAMETERS_KERNEL_FILTER(zn, B_Z_INDEX(x, y))); }
/// Update Z- boundary state WITHOUT a coordinate check
#define UPDATE_MACRO_Z2NS                                                                                              \
    { update_filter(o, n, BOUNDARY_ACTUAL_PARAMETERS_KERNEL_FILTER(zn, B_Z_INDEX(x, y))); }
/// Update Z+ boundary state WITH a coordinate check
#define UPDATE_MACRO_Z2P                                                                                               \
    if (z == size_z - 1) { update_filter(o, n, BOUNDARY_ACTUAL_PARAMETERS_KERNEL_FILTER(zp, B_Z_INDEX(x, y))); }
/// Update Z+ boundary state WITHOUT a coordinate check
#define UPDATE_MACRO_Z2PS                                                                                              \
    { update_filter(o, n, BOUNDARY_ACTUAL_PARAMETERS_KERNEL_FILTER(zp, B_Z_INDEX(x, y))); }

/// Generic macro to update a single boundary side
#define UPDATE_MACRO(VARS, S1, S2)                                                                                     \
    {                                                                                                                  \
        VARS;                                                                                                          \
        const int i = P_INDEX(x, y, z);                                                                                \
        const float *p_r = p + i;                                                                                      \
        float *p_aux_r = p_aux + i;                                                                                    \
        float g = 0;                                                                                                   \
        float b = 0;                                                                                                   \
        S1;                                                                                                            \
        const float o = *p_aux_r;                                                                                      \
        float n = p_r[znd];                                                                                            \
        n += p_r[ynd];                                                                                                 \
        n += p_r[xnd];                                                                                                 \
        n += p_r[xpd];                                                                                                 \
        n += p_r[ypd];                                                                                                 \
        n += p_r[zpd];                                                                                                 \
        n = (D1 * n + L2 * g + (L1 * b - 1.0f) * o) / (L1 * b + 1.0f);                                                 \
        *p_aux_r = n;                                                                                                  \
        S2;                                                                                                            \
    }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Each of the boundary kernels maps the coordinates to the respective boundaries and performs the UPDATE_MACRO
/// The kernels use reverse_z to reverse the mapping of threads on the z axis, increasing cached access across iterations
/// THe kernels use pad_x to correctly compute the p arrays coordinates (padding of 1 is added if size_x is odd)
/**/
/// X- and X+ boundaries single-time-step update handling kernel (no edges)
template<bool reverse_z, int pad_x>
__global__ void __launch_bounds__(SIM_BORDER_BLOCK_DIM_X * SIM_BORDER_BLOCK_DIM_Y)
sim_iter_x(const int size_x, const int size_y, const int size_z, const float *__restrict__ p,
           float *__restrict__ p_aux, KERNEL_BOUNDARY_FORMAL_PARAMS(xn), KERNEL_BOUNDARY_FORMAL_PARAMS(xp)) {
    const int y = threadIdx.x + blockIdx.x * SIM_BORDER_BLOCK_DIM_X + 1; // NOLINT(*-narrowing-conversions)
    const int z = !reverse_z
                      ? threadIdx.y + blockIdx.y * SIM_BORDER_BLOCK_DIM_Y + 1
                      : size_z - 1 - (threadIdx.y + blockIdx.y * SIM_BORDER_BLOCK_DIM_Y + 1);
    if (!(y < size_y - 1 && ((z < size_z - 1 && !reverse_z) || (z > 0 && reverse_z)))) return;

    const int ynd = -(size_x + pad_x);
    const int ypd = (size_x + pad_x);
    const int znd = !reverse_z ? -((size_x + pad_x) * size_y) : (size_x + pad_x) * size_y;
    const int zpd = !reverse_z ? (size_x + pad_x) * size_y : -((size_x + pad_x) * size_y);
    UPDATE_MACRO(constexpr int x = 0; constexpr int xnd = 1; constexpr int xpd = 1, //
                 UPDATE_MACRO_X1NS, //
                 UPDATE_MACRO_X2NS)
    UPDATE_MACRO(const int x = size_x - 1; constexpr int xnd = -1; constexpr int xpd = -1, //
                 UPDATE_MACRO_X1PS, //
                 UPDATE_MACRO_X2PS)
}

/// Y- and Y+ boundaries single-time-step update handling kernel (including edges shared with X sides)
template<bool reverse_z, int pad_x>
__global__ void __launch_bounds__(SIM_BORDER_BLOCK_DIM_X * SIM_BORDER_BLOCK_DIM_Y)
sim_iter_y(const int size_x, const int size_y, const int size_z, const float *__restrict__ p,
           float *__restrict__ p_aux, KERNEL_BOUNDARY_FORMAL_PARAMS(xn), KERNEL_BOUNDARY_FORMAL_PARAMS(xp),
           KERNEL_BOUNDARY_FORMAL_PARAMS(yn), KERNEL_BOUNDARY_FORMAL_PARAMS(yp)) {
    const int x = threadIdx.x + blockIdx.x * SIM_BORDER_BLOCK_DIM_X; // NOLINT(*-narrowing-conversions)
    const int z = !reverse_z
                      ? threadIdx.y + blockIdx.y * SIM_BORDER_BLOCK_DIM_Y + 1
                      : size_z - 1 - (threadIdx.y + blockIdx.y * SIM_BORDER_BLOCK_DIM_Y + 1);
    if (!(x < size_x && ((z < size_z - 1 && !reverse_z) || (z > 0 && reverse_z)))) return;

    const int znd = !reverse_z ? -((size_x + pad_x) * size_y) : (size_x + pad_x) * size_y;
    const int zpd = !reverse_z ? (size_x + pad_x) * size_y : -((size_x + pad_x) * size_y);
    UPDATE_MACRO(constexpr int y = 0; const int ynd = (size_x + pad_x);
                 const int ypd = (size_x + pad_x), //
                 UPDATE_MACRO_X1N UPDATE_MACRO_X1P UPDATE_MACRO_Y1NS, //
                 UPDATE_MACRO_X2N UPDATE_MACRO_X2P UPDATE_MACRO_Y2NS)
    UPDATE_MACRO(const int y = size_y - 1; const int ynd = -(size_x + pad_x);
                 const int ypd = -(size_x + pad_x), //
                 UPDATE_MACRO_X1N UPDATE_MACRO_X1P UPDATE_MACRO_Y1PS, //
                 UPDATE_MACRO_X2N UPDATE_MACRO_X2P UPDATE_MACRO_Y2PS)
}

/// Z- boundary single-time-step update handling kernel (including all edges)
template<int pad_x>
__global__ void __launch_bounds__(SIM_BORDER_BLOCK_DIM_X * SIM_BORDER_BLOCK_DIM_Y)
sim_iter_z_n(const int size_x, const int size_y, const float *__restrict__ p, float *__restrict__ p_aux,
             KERNEL_BOUNDARY_FORMAL_PARAMS(xn), KERNEL_BOUNDARY_FORMAL_PARAMS(xp),
             KERNEL_BOUNDARY_FORMAL_PARAMS(yn), KERNEL_BOUNDARY_FORMAL_PARAMS(yp),
             KERNEL_BOUNDARY_FORMAL_PARAMS(zn)) {
    const int x = threadIdx.x + blockIdx.x * SIM_BORDER_BLOCK_DIM_X; // NOLINT(*-narrowing-conversions)
    const int y = threadIdx.y + blockIdx.y * SIM_BORDER_BLOCK_DIM_Y; // NOLINT(*-narrowing-conversions)
    if (!(x < size_x && y < size_y)) return;
    UPDATE_MACRO(constexpr int z = 0; const int znd = (size_x + pad_x) * size_y;
                 const int zpd = (size_x + pad_x) * size_y, //
                 UPDATE_MACRO_X1N UPDATE_MACRO_X1P UPDATE_MACRO_Y1N UPDATE_MACRO_Y1P UPDATE_MACRO_Z1NS, //
                 UPDATE_MACRO_X2N UPDATE_MACRO_X2P UPDATE_MACRO_Y2N UPDATE_MACRO_Y2P UPDATE_MACRO_Z2NS)
}

/// Z+ boundary single-time-step update handling kernel (including all edges)
template<int pad_x>
__global__ void __launch_bounds__(SIM_BORDER_BLOCK_DIM_X * SIM_BORDER_BLOCK_DIM_Y)
sim_iter_z_p(const int size_x, const int size_y, const int size_z, const float *__restrict__ p,
             float *__restrict__ p_aux, KERNEL_BOUNDARY_FORMAL_PARAMS(xn), KERNEL_BOUNDARY_FORMAL_PARAMS(xp),
             KERNEL_BOUNDARY_FORMAL_PARAMS(yn), KERNEL_BOUNDARY_FORMAL_PARAMS(yp),
             KERNEL_BOUNDARY_FORMAL_PARAMS(zp)) {
    const int x = threadIdx.x + blockIdx.x * SIM_BORDER_BLOCK_DIM_X; // NOLINT(*-narrowing-conversions)
    const int y = threadIdx.y + blockIdx.y * SIM_BORDER_BLOCK_DIM_Y; // NOLINT(*-narrowing-conversions)
    if (!(x < size_x && y < size_y)) return;
    UPDATE_MACRO(const int z = size_z - 1; const int znd = -(size_x + pad_x) * size_y;
                 const int zpd = -(size_x + pad_x) * size_y, //
                 UPDATE_MACRO_X1N UPDATE_MACRO_X1P UPDATE_MACRO_Y1N UPDATE_MACRO_Y1P UPDATE_MACRO_Z1PS, //
                 UPDATE_MACRO_X2N UPDATE_MACRO_X2P UPDATE_MACRO_Y2N UPDATE_MACRO_Y2P UPDATE_MACRO_Z2PS)
}

/// Center section single-time-step update handling kernel
template<bool reverse_z, int pad_x>
__global__ void __launch_bounds__(SIM_CENTER_BLOCK_DIM_X * SIM_CENTER_BLOCK_DIM_Y)
sim_iter_c(const int size_x, const int size_y, const int size_z, const float *__restrict__ p,
           float *__restrict__ p_aux) {
    // Each thread in each warp computes two adjacent x coordinates (with the first and last threads computing only one)
    const int g_x = threadIdx.x * 2 + blockIdx.x * SIM_CENTER_BLOCK_COMPUTE_DIM_X; // NOLINT(*-narrowing-conversions)
    const int g_y = threadIdx.y + blockIdx.y * SIM_CENTER_BLOCK_DIM_Y + 1; // NOLINT(*-narrowing-conversions)

    // Early exit for out of bounds kernels, the active mask is obtained for later shuffle operations
    if (g_x >= size_x || g_y >= size_y - 1) return;
    const unsigned active = __activemask();

    const int g_z_start =
            !reverse_z ? blockIdx.z * SIM_CENTER_BLOCK_DIM_Z : size_z - blockIdx.z * SIM_CENTER_BLOCK_DIM_Z - 1;

    // To check if each thread needs to write either one of its values, a boolean for each value is checked when writing
    const bool write_1 = threadIdx.x > 0 && g_x < size_x - 1, write_2 = threadIdx.x < 31 && g_x < size_x - 2;

    // p and p_aux are iterated over via pointers arithmetics to increment between xy planes
    const int g_base_xyz = P_INDEX(g_x, g_y, g_z_start);
    const int g_offset_xyz = !reverse_z ? (size_x + pad_x) * size_y : -((size_x + pad_x) * size_y);
    p += g_base_xyz;
    p_aux += g_base_xyz;

    // Loop tiling technique inspired from the stencil computation chapter in
    // Wen-Mei, W. Hwu, David B. Kirk, and Izzat El Hajj. Programming massively parallel processors: a hands-on approach. Morgan Kaufmann, 2026.
    float2 p_prev_r = *reinterpret_cast<const float2 *>(p);
    p += g_offset_xyz;
    float2 p_curr_r = *reinterpret_cast<const float2 *>(p);

    for (int z_i = 1; z_i <= SIM_CENTER_BLOCK_DIM_Z; z_i++) {
        if (!((z_i + g_z_start < size_z - 1 && !reverse_z) || (g_z_start - z_i > 0 && reverse_z))) return;

        // Stencil computation for center section
        const float2 read_p_yn = *reinterpret_cast<const float2 *>(p - size_x - pad_x);
        p_prev_r.x += read_p_yn.x;
        p_prev_r.y += read_p_yn.y;
        p_prev_r.x += __shfl_up_sync(active, p_curr_r.y, 1);
        p_prev_r.x += p_curr_r.y;
        p_prev_r.y += p_curr_r.x;
        p_prev_r.y += __shfl_down_sync(active, p_curr_r.x, 1);
        const float2 read_p_yp = *reinterpret_cast<const float2 *>(p + size_x + pad_x);
        p_prev_r.x += read_p_yp.x;
        p_prev_r.y += read_p_yp.y;
        p += g_offset_xyz;
        const float2 read_p_zp = *reinterpret_cast<const float2 *>(p);
        p_prev_r.x += read_p_zp.x;
        p_prev_r.y += read_p_zp.y;

        // Read and compute new values
        p_aux += g_offset_xyz;
        float2 read_p_aux = *reinterpret_cast<float2 *>(p_aux);
        read_p_aux.x = (D1 * p_prev_r.x - read_p_aux.x);
        read_p_aux.y = (D1 * p_prev_r.y - read_p_aux.y);

        // Write only the correct values (the first and last threads compute at most 1 valid value)
        // This way of writing back to memory has been found to be relatively efficient compared to alternatives
        if (write_1 && write_2) *reinterpret_cast<float2 *>(p_aux) = read_p_aux;
        if (write_1 && !write_2) *p_aux = read_p_aux.x;
        if (!write_1 && write_2) *(p_aux + 1) = read_p_aux.y;

        p_prev_r = p_curr_r;
        p_curr_r = read_p_zp;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Each simulation step is split into an x boundaries section, y boundaries section, z- boundary section,
// z+ boundary section, center section, all running concurrently
template<int pad_x, bool reverse_z>
void dwr3::propagation_medium_kv_2009::kv_2009_graph_builder::add_iteration_nodes_and_record_completion_events_impl(
    cudaEvent_t precondition_event) {
#define BOUNDARY_ACTUAL_PARAMETERS_GRAPH_BUILD(I) pm->b_state[I].g, MACRO_LIST_EXPANSION_0(pm->b_state[I].x[, , , ]), MACRO_LIST_EXPANSION_0(pm->b_state[I].y[, , , ]), pm->b_state[I].dif

    CUDA_THROW_ON_ERROR(cudaStreamWaitEvent(stream_z_hp.get(), precondition_event, 0));
    if (!reverse_z) {
        sim_iter_z_n<pad_x>
                <<<iter_z_grid, dim3(SIM_BORDER_BLOCK_DIM_X, SIM_BORDER_BLOCK_DIM_Y), 0, stream_z_hp.get()>>>(
                    pm->p_info_.size[0], pm->p_info_.size[1], pm->p_alloc_d[0], pm->p_alloc_d[1],
                    BOUNDARY_ACTUAL_PARAMETERS_GRAPH_BUILD(0),
                    BOUNDARY_ACTUAL_PARAMETERS_GRAPH_BUILD(1),
                    BOUNDARY_ACTUAL_PARAMETERS_GRAPH_BUILD(2),
                    BOUNDARY_ACTUAL_PARAMETERS_GRAPH_BUILD(3),
                    BOUNDARY_ACTUAL_PARAMETERS_GRAPH_BUILD(4));
    } else {
        sim_iter_z_p<pad_x>
                <<<iter_z_grid, dim3(SIM_BORDER_BLOCK_DIM_X, SIM_BORDER_BLOCK_DIM_Y), 0, stream_z_hp.get()>>>(
                    pm->p_info_.size[0], pm->p_info_.size[1], pm->p_info_.size[2], pm->p_alloc_d[0],
                    pm->p_alloc_d[1],
                    BOUNDARY_ACTUAL_PARAMETERS_GRAPH_BUILD(0),
                    BOUNDARY_ACTUAL_PARAMETERS_GRAPH_BUILD(1),
                    BOUNDARY_ACTUAL_PARAMETERS_GRAPH_BUILD(2),
                    BOUNDARY_ACTUAL_PARAMETERS_GRAPH_BUILD(3),
                    BOUNDARY_ACTUAL_PARAMETERS_GRAPH_BUILD(5));
    }
    CUDA_THROW_ON_ERROR(cudaEventRecord(event_done_z_hp.get(), stream_z_hp.get()));

    CUDA_THROW_ON_ERROR(cudaStreamWaitEvent(stream_x.get(), precondition_event, 0));
    sim_iter_x<reverse_z, pad_x>
            <<<iter_x_grid, dim3(SIM_BORDER_BLOCK_DIM_X, SIM_BORDER_BLOCK_DIM_Y), 0, stream_x.get()>>>(
                pm->p_info_.size[0], pm->p_info_.size[1], pm->p_info_.size[2], pm->p_alloc_d[0],
                pm->p_alloc_d[1],
                BOUNDARY_ACTUAL_PARAMETERS_GRAPH_BUILD(0),
                BOUNDARY_ACTUAL_PARAMETERS_GRAPH_BUILD(1));
    CUDA_THROW_ON_ERROR(cudaEventRecord(event_done_x.get(), stream_x.get()));

    CUDA_THROW_ON_ERROR(cudaStreamWaitEvent(stream_y.get(), precondition_event, 0));
    sim_iter_y<reverse_z, pad_x> //
            <<<iter_y_grid, dim3(SIM_BORDER_BLOCK_DIM_X, SIM_BORDER_BLOCK_DIM_Y), 0, stream_y.get()>>>(
                pm->p_info_.size[0], pm->p_info_.size[1], pm->p_info_.size[2], pm->p_alloc_d[0],
                pm->p_alloc_d[1],
                BOUNDARY_ACTUAL_PARAMETERS_GRAPH_BUILD(0),
                BOUNDARY_ACTUAL_PARAMETERS_GRAPH_BUILD(1),
                BOUNDARY_ACTUAL_PARAMETERS_GRAPH_BUILD(2),
                BOUNDARY_ACTUAL_PARAMETERS_GRAPH_BUILD(3));
    CUDA_THROW_ON_ERROR(cudaEventRecord(event_done_y.get(), stream_y.get()));

    sim_iter_c<reverse_z, pad_x>
            <<<iter_c_grid, dim3(SIM_CENTER_BLOCK_DIM_X, SIM_CENTER_BLOCK_DIM_Y), 0, main_stream>>>(
                pm->p_info_.size[0], pm->p_info_.size[1], pm->p_info_.size[2], pm->p_alloc_d[0],
                pm->p_alloc_d[1]);

    CUDA_THROW_ON_ERROR(cudaStreamWaitEvent(stream_z_lp.get(), precondition_event, 0));
    if (!reverse_z) {
        sim_iter_z_p<pad_x>
                <<<iter_z_grid, dim3(SIM_BORDER_BLOCK_DIM_X, SIM_BORDER_BLOCK_DIM_Y), 0, stream_z_lp.get()>>>(
                    pm->p_info_.size[0], pm->p_info_.size[1], pm->p_info_.size[2], pm->p_alloc_d[0],
                    pm->p_alloc_d[1],
                    BOUNDARY_ACTUAL_PARAMETERS_GRAPH_BUILD(0),
                    BOUNDARY_ACTUAL_PARAMETERS_GRAPH_BUILD(1),
                    BOUNDARY_ACTUAL_PARAMETERS_GRAPH_BUILD(2),
                    BOUNDARY_ACTUAL_PARAMETERS_GRAPH_BUILD(3),
                    BOUNDARY_ACTUAL_PARAMETERS_GRAPH_BUILD(5));
    } else {
        sim_iter_z_n<pad_x>
                <<<iter_z_grid, dim3(SIM_BORDER_BLOCK_DIM_X, SIM_BORDER_BLOCK_DIM_Y), 0, stream_z_lp.get()>>>(
                    pm->p_info_.size[0], pm->p_info_.size[1], pm->p_alloc_d[0], pm->p_alloc_d[1],
                    BOUNDARY_ACTUAL_PARAMETERS_GRAPH_BUILD(0),
                    BOUNDARY_ACTUAL_PARAMETERS_GRAPH_BUILD(1),
                    BOUNDARY_ACTUAL_PARAMETERS_GRAPH_BUILD(2),
                    BOUNDARY_ACTUAL_PARAMETERS_GRAPH_BUILD(3),
                    BOUNDARY_ACTUAL_PARAMETERS_GRAPH_BUILD(4));
    }
    CUDA_THROW_ON_ERROR(cudaEventRecord(event_done_z_lp.get(), stream_z_lp.get()));

    // Rotate the state pointers
    std::swap(pm->p_alloc_d[0], pm->p_alloc_d[1]);
    for (auto &b: pm->b_state) b.rotate_xy();
#undef BOUNDARY_ACTUAL_PARAMETERS_GRAPH_BUILD
}

dwr3::propagation_medium_kv_2009::kv_2009_graph_builder::kv_2009_graph_builder(
    cudaStream_t main_stream, const stream_priorities sp, propagation_medium_kv_2009 *pm_)
    : main_stream(main_stream), pm(pm_), stream_x(sp.medium()), stream_y(sp.medium()),
      stream_z_lp(sp.low()), stream_z_hp(sp.high()),
      event_vector{event_done_x.get(), event_done_y.get(), event_done_z_lp.get(), event_done_z_hp.get()} {
    iter_x_grid = dim3(((pm->p_info_.size[1] - 2) + SIM_BORDER_BLOCK_DIM_X - 1) / SIM_BORDER_BLOCK_DIM_X,
                       ((pm->p_info_.size[2] - 2) + SIM_BORDER_BLOCK_DIM_Y - 1) / SIM_BORDER_BLOCK_DIM_Y);
    iter_y_grid = dim3((pm->p_info_.size[0] + SIM_BORDER_BLOCK_DIM_X - 1) / SIM_BORDER_BLOCK_DIM_X,
                       ((pm->p_info_.size[2] - 2) + SIM_BORDER_BLOCK_DIM_Y - 1) / SIM_BORDER_BLOCK_DIM_Y);
    iter_z_grid = dim3((pm->p_info_.size[0] + SIM_BORDER_BLOCK_DIM_X - 1) / SIM_BORDER_BLOCK_DIM_X,
                       (pm->p_info_.size[1] + SIM_BORDER_BLOCK_DIM_Y - 1) / SIM_BORDER_BLOCK_DIM_Y);
    iter_c_grid = dim3(
        ((pm->p_info_.size[0] - 2) + SIM_CENTER_BLOCK_COMPUTE_DIM_X - 1) / SIM_CENTER_BLOCK_COMPUTE_DIM_X,
        ((pm->p_info_.size[1] - 2) + SIM_CENTER_BLOCK_DIM_Y - 1) / SIM_CENTER_BLOCK_DIM_Y,
        ((pm->p_info_.size[2] - 2) + SIM_CENTER_BLOCK_DIM_Z - 1) / SIM_CENTER_BLOCK_DIM_Z);
    reverse_z = false;
    pad_x = pm->p_info_.size[0] % 2 != 0;
}

[[nodiscard]] const std::vector<cudaEvent_t> &
dwr3::propagation_medium_kv_2009::kv_2009_graph_builder::completion_events() const { return event_vector; }

void dwr3::propagation_medium_kv_2009::kv_2009_graph_builder::add_iteration_nodes_and_record_completion_events
(cudaEvent_t io_event) {
    if (!pad_x && !reverse_z)
        add_iteration_nodes_and_record_completion_events_impl<0, false>(io_event);
    else if (!pad_x && reverse_z)
        add_iteration_nodes_and_record_completion_events_impl<0, true>(io_event);
    else if (pad_x && !reverse_z)
        add_iteration_nodes_and_record_completion_events_impl<1, false>(io_event);
    else
        add_iteration_nodes_and_record_completion_events_impl<1, true>(io_event);
    /// This is available so to showcase in benchmarks the improvement obtained from this kind of memory access pattern
#ifndef DWR3_DISABLE_UPDATE_UP_DOWN_ITERATION
    reverse_z = !reverse_z;
#endif
}

float *dwr3::propagation_medium_kv_2009::kv_2009_graph_builder::p() {
    // This is called after add_iteration_nodes_and_record_completion_events, which swaps the state pointers,
    // meaning that the latest state is stored in p_alloc_d[0]
    return pm->p_alloc_d[0];
}

std::unique_ptr<dwr3::propagation_medium::graph_builder> dwr3::propagation_medium_kv_2009::create_graph_builder(
    cudaStream_t main_stream, stream_priorities stream_priorities) {
    return std::move(std::make_unique<kv_2009_graph_builder>(main_stream, stream_priorities, this));
}

dwr3::p_info dwr3::propagation_medium_kv_2009::p_info() const noexcept { return p_info_; }

static float *cuda_malloc_throw(const size_t size) {
    float *alloc;
    CUDA_THROW_ON_ERROR(cudaMalloc(&alloc, size));
    return alloc;
}

void dwr3::propagation_medium_kv_2009::boundary_state::init(
    const boundary_reflectance_filter_coefficients *boundary_reflectance_filter, const size_t size) {
    this->size = size;
    if (boundary_reflectance_filter == nullptr) throw std::logic_error("Boundary filter provided null");
    dif_compute_from_boundary_filter(dif, boundary_reflectance_filter);
    g = cuda_malloc_throw(size);
    for (auto &x_: x) x_ = cuda_malloc_throw(size);
    for (auto &y_: y) y_ = cuda_malloc_throw(size);
}

void dwr3::propagation_medium_kv_2009::boundary_state::release() {
    cudaFree(g);
    for (const auto &x_: x) cudaFree(x_);
    for (const auto &y_: y) cudaFree(y_);
}

void dwr3::propagation_medium_kv_2009::boundary_state::reset() const {
    cudaMemset(g, 0, size);
    for (const auto &x_: x)
        CUDA_THROW_ON_ERROR(cudaMemset(x_, 0, size));
    for (const auto &y_: y)
        CUDA_THROW_ON_ERROR(cudaMemset(y_, 0, size));
}

void dwr3::propagation_medium_kv_2009::boundary_state::rotate_xy() {
    float *const l_x = x[DWR3_BOUNDARY_FILTER_ORDER - 1];
    for (int j = DWR3_BOUNDARY_FILTER_ORDER - 1; j > 0; j--) x[j] = x[j - 1];
    x[0] = l_x;
    float *const l_y = y[DWR3_BOUNDARY_FILTER_ORDER - 1];
    for (int j = DWR3_BOUNDARY_FILTER_ORDER - 1; j > 0; j--) y[j] = y[j - 1];
    y[0] = l_y;
}

/// Determine the best mapping between physical and implementation coordinates, aiming to keep the x-axis sides as
/// small as possible and the z-axis are kept as large as possible
template<typename T>
static dwr3::change_of_basis select_change_of_basis(T x, T y, T z) {
    /// If the coordinates mapping is disabled, just return the identity mapping
/// This is available to showcase in benchmarks the improvement for instances that are better handled with mapping
#ifdef DWR3_DISABLE_COORDS_MAPPING
    return dwr3::change_of_basis::xyz;
#endif
    const T xs = y * z;
    const T ys = x * z;
    const T zs = x * y;
    if (xs <= ys && ys <= zs) return dwr3::change_of_basis::xyz;
    if (xs <= zs && zs <= ys) return dwr3::change_of_basis::xzy;
    if (ys <= xs && xs <= zs) return dwr3::change_of_basis::yxz;
    if (ys <= zs && zs <= xs) return dwr3::change_of_basis::yzx;
    if (zs <= xs && xs <= ys) return dwr3::change_of_basis::zxy;
    if (zs <= ys && ys <= xs) return dwr3::change_of_basis::zyx;
    return dwr3::change_of_basis::xyz;
}

dwr3::propagation_medium_kv_2009::propagation_medium_kv_2009(
    instance_info &info, const float size[3], const boundary_reflectance_filter_coefficients *const
    boundary_reflectance_filters[6], const int sample_rate, const int buffer_size) {
    static_assert(buffer_base_size % boundary_filter_order == 0);
    if (buffer_size % boundary_filter_order != 0)
        throw std::logic_error(
            "buffer_size is not a multiple of boundary_filter_order = " + std::to_string(boundary_filter_order));
    if (sample_rate <= 0) throw std::logic_error("sample_rate is less or equal than zero");

    // Compute and apply change of basis
    p_info_.change_of_basis = select_change_of_basis(size[0], size[1], size[2]);
    float size_[3] = {size[0], size[1], size[2]};
    apply_change_of_basis(size_[0], size_[1], size_[2], p_info_.change_of_basis);
    const boundary_reflectance_filter_coefficients * boundary_reflectance_filters_[6] = {
        boundary_reflectance_filters[0], boundary_reflectance_filters[1], boundary_reflectance_filters[2],
        boundary_reflectance_filters[3], boundary_reflectance_filters[4], boundary_reflectance_filters[5]
    };
    apply_change_of_basis(
        boundary_reflectance_filters_[0], boundary_reflectance_filters_[2], boundary_reflectance_filters_[4],
        p_info_.change_of_basis);
    apply_change_of_basis(
        boundary_reflectance_filters_[1], boundary_reflectance_filters_[3], boundary_reflectance_filters_[5],
        p_info_.change_of_basis);

    p_info_.centered_boundary_conditions_scheme = true;
    info.nodes_per_meter = p_info_.nodes_per_meter = L1 * static_cast<float>(sample_rate) / speed_of_sound;
    for (int i = 0; i < 3; i++) {
        if (size_[i] <= 0) throw std::logic_error("size on a axis is less or equal than zero");
        const int ns = static_cast<int>(roundf(size_[i] * p_info_.nodes_per_meter)) + 1;
        if (ns < 3) throw std::runtime_error("Node size on a axis is lesser than 3 nodes");
        info.node_size[i] = p_info_.size[i] = ns;
    }
    // Pad the allocation rows to even length
    p_info_.alloc_stride_x = p_info_.size[0] % 2 == 0 ? p_info_.size[0] : p_info_.size[0] + 1;

    info.pm_memory_size = 0;
    try {
        p_alloc_size = p_info_.alloc_stride_x * p_info_.size[1] * p_info_.size[2] * sizeof(float);
        for (auto &p: p_alloc_d) {
            p = cuda_malloc_throw(p_alloc_size);
            info.pm_memory_size += p_alloc_size;
        }
        const size_t side_area[3] = {
            static_cast<size_t>(p_info_.size[1] * p_info_.size[2]) * sizeof(float),
            static_cast<size_t>(p_info_.size[0] * p_info_.size[2]) * sizeof(float),
            static_cast<size_t>(p_info_.size[0] * p_info_.size[1]) * sizeof(float)
        };
        for (int i = 0; i < 6; i++) {
            b_state[i].init(boundary_reflectance_filters_[i], side_area[i >> 1]);
            info.pm_memory_size += side_area[i >> 1];
        }
        reset();
    } catch (const std::exception &) {
        propagation_medium_kv_2009::~propagation_medium_kv_2009();
        throw;
    }
}

dwr3::propagation_medium_kv_2009::~propagation_medium_kv_2009() noexcept {
    for (const auto &p: p_alloc_d) cudaFree(p);
    for (auto &b: b_state) b.release();
}

void dwr3::propagation_medium_kv_2009::reset() {
    for (const auto &p: p_alloc_d)
        CUDA_THROW_ON_ERROR(cudaMemset(p, 0, p_alloc_size));
    for (const auto &b: b_state) b.reset();
}

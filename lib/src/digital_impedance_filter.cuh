#ifndef DWR3_DIF_CUH
#define DWR3_DIF_CUH

#include "common.cuh"

namespace dwr3 {
    /// Digital impedance filter coefficients, access the values via the provided dif_* functions
    /// @note the a_0 coefficient is normalized to 1, it's not stored in memory, and it cannot be accessed via dif_a
    struct digital_impedance_filter_coefficients {
        float _b_0_rec;
        float _b[DWR3_BOUNDARY_FILTER_ORDER + 1];
        float _a[DWR3_BOUNDARY_FILTER_ORDER];
    };

    __host__ __device__ inline float dif_b_0_rec(const digital_impedance_filter_coefficients &d) {
        return d._b_0_rec;
    }

    static float *dif_b_0_rec_ptr(digital_impedance_filter_coefficients &d) {
        return &d._b_0_rec;
    }

    __host__ __device__ inline float dif_b(const digital_impedance_filter_coefficients &d, const int i) {
        return d._b[i];
    }

    static float *dif_b_ptr(digital_impedance_filter_coefficients &d, const int i) {
        return &d._b[i];
    }

    __host__ __device__ inline float dif_a(const digital_impedance_filter_coefficients &d, const int i) {
        return d._a[i - 1];
    }

    static float *dif_a_ptr(digital_impedance_filter_coefficients &d, const int i) {
        return &d._a[i - 1];
    }

    /// Conversion between boundary reflectance filter coefficients and DIF filter coefficients
    inline void dif_compute_from_boundary_filter(
        digital_impedance_filter_coefficients &df, const boundary_reflectance_filter_coefficients *brf) {
        const double a0 = brf->a[0] - brf->b[0];
        *dif_b_ptr(df, 0) = static_cast<float>((brf->a[0] + brf->b[0]) / a0);
        *dif_b_0_rec_ptr(df) = static_cast<float>(a0 / (brf->a[0] + brf->b[0]));
        for (int i = 1; i <= DWR3_BOUNDARY_FILTER_ORDER; i++) {
            *dif_a_ptr(df, i) = static_cast<float>((brf->a[i] - brf->b[i]) / a0);
            *dif_b_ptr(df, i) = static_cast<float>((brf->a[i] + brf->b[i]) / a0);
        }
    }
}

#endif //DWR3_DIF_CUH

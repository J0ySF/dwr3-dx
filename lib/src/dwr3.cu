#include <dwr3/dwr3.h>

// Compie-time assertions //////////////////////////////////////////////////////////////////////////////////////////////

// Check that dwr3_sample_position_t can be correctly copied to a float4 (with v remapped to w) for use in CUDA kernels
static_assert(std::is_standard_layout_v<dwr3_sample_position_t> == true);
static_assert(std::is_standard_layout_v<float4> == true);
static_assert(sizeof(dwr3_sample_position_t) == sizeof(float4));
static_assert(alignof(dwr3_sample_position_t) == alignof(float4));
static_assert(offsetof(dwr3_sample_position_t, x) == offsetof(float4, x));
static_assert(offsetof(dwr3_sample_position_t, y) == offsetof(float4, y));
static_assert(offsetof(dwr3_sample_position_t, z) == offsetof(float4, z));
static_assert(offsetof(dwr3_sample_position_t, v) == offsetof(float4, w));

// Entrypoints /////////////////////////////////////////////////////////////////////////////////////////////////////////

dwr3_error_t dwr3_create(void **instance, dwr3_instance_info_t *instance_info, float size[3],
                         const dwr3_boundary_reflectance_filter_t *const boundary_reflectance_filters[6],
                         int sample_rate, unsigned int buffer_size, unsigned int input_count_limit,
                         unsigned int output_count_limit) {
    *instance = nullptr;
    if (instance_info) *instance_info = {};

    // TODO: implement
    return DWR3_ERROR_UNKNOWN;
}

dwr3_error_t dwr3_destroy(void *instance) {
    // TODO: implement
    return DWR3_ERROR_UNKNOWN;
}

dwr3_error_t dwr3_reset(void *instance) {
    // TODO: implement
    return DWR3_ERROR_UNKNOWN;
}

dwr3_error_t dwr3_processing_start(void *instance, unsigned int input_count,
                                   const dwr3_sample_position_t *const *input_samples_positions,
                                   unsigned int output_count, const dwr3_sample_position_t *const *output_positions) {
    // TODO: implement
    return DWR3_ERROR_UNKNOWN;
}

int dwr3_processing_started(void *instance) {
    return 0;
}

dwr3_error_t dwr3_processing_retrieve(void *instance, dwr3_sample_position_t *const *output_samples) {
    // TODO: implement
    return DWR3_ERROR_UNKNOWN;
}

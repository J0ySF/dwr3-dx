#include <dwr3/dwr3.hpp>
#include <stdexcept>

namespace dwr3 {
    // Compie-time assertions //////////////////////////////////////////////////////////////////////////////////////////

    // Check that sample_position_value can be correctly copied to a float4 (with v remapped to w) for use in CUDA kernels
    static_assert(std::is_standard_layout_v<sample_position_value> == true);
    static_assert(std::is_standard_layout_v<float4> == true);
    static_assert(sizeof(sample_position_value) == sizeof(float4));
    static_assert(alignof(sample_position_value) == alignof(float4));
    static_assert(offsetof(sample_position_value, x) == offsetof(float4, x));
    static_assert(offsetof(sample_position_value, y) == offsetof(float4, y));
    static_assert(offsetof(sample_position_value, z) == offsetof(float4, z));
    static_assert(offsetof(sample_position_value, v) == offsetof(float4, w));

    // Entrypoint class ////////////////////////////////////////////////////////////////////////////////////////////////

    dwr3::dwr3(
        float size[3], const boundary_reflectance_filter_coefficients *const boundary_coefficients[6],
        int sample_rate, int buffer_size, int input_count_limit, int output_count_limit) {
        throw std::runtime_error("Not implemented yet");
    }

    dwr3::~dwr3() = default;

    instance_info dwr3::info() const {
        throw std::runtime_error("Not implemented yet");
    }

    void dwr3::reset() const {
        throw std::runtime_error("Not implemented yet");
    }

    void dwr3::processing_start(
        const int input_count, const sample_position_value *const *input_samples_positions,
        const int output_count, const sample_position_value *const *output_positions) const {
        throw std::runtime_error("Not implemented yet");
    }

    bool dwr3::processing_started() const {
        throw std::runtime_error("Not implemented yet");
    }

    void dwr3::processing_retrieve(float *const *output_samples) const {
        throw std::runtime_error("Not implemented yet");
    }
}

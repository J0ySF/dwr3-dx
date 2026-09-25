#include "input_output.cuh"

#include <cooperative_groups.h>
#include <cstdint>

// Static assertions ///////////////////////////////////////////////////////////////////////////////////////////////////

// Check that sample_position_value can be correctly copied to a float4 (with v remapped to w) for use in CUDA kernels
// TODO: currently this is not actually needed
static_assert(std::is_standard_layout_v<dwr3::sample_position_value> == true);
static_assert(std::is_standard_layout_v<float4> == true);
static_assert(sizeof(dwr3::sample_position_value) == sizeof(float4));
static_assert(alignof(dwr3::sample_position_value) == alignof(float4));
static_assert(offsetof(dwr3::sample_position_value, x) == offsetof(float4, x));
static_assert(offsetof(dwr3::sample_position_value, y) == offsetof(float4, y));
static_assert(offsetof(dwr3::sample_position_value, z) == offsetof(float4, z));
static_assert(offsetof(dwr3::sample_position_value, v) == offsetof(float4, w));

// CUDA kernels ////////////////////////////////////////////////////////////////////////////////////////////////////////

// Coordinate linearization used to access the p buffer
#define P_INDEX(X, Y, Z) ((X) + alloc_stride_x * ((Y) + size_y * (Z)))

#define IO_BLOCK_SIZE 128

// Aux function from https://developer.nvidia.com/blog/lerp-faster-cuda/
template<typename T>
__device__ T device_lerp(T v0, T v1, T t) { return fma(t, v1, fma(-t, v0, v0)); }

#define IO_THREADS_PER_INPUT 32
// Perform the source excitation with gaussian distribution driving, as presented in
// Tsuchiya, Takao, Yu Teshima, and Shizuko Hiryu. "Three-dimensional finite-difference time-domain simulation of moving
// sound source and receiver with directivity." Japanese Journal of Applied Physics 62.SJ (2023): SJ1015.
__device__ void io_step_input_perform( // TODO: remove size_x?
    const int size_x, const int alloc_stride_x, const int size_y, float *__restrict__ p, // NOLINT(*-non-const-parameter)
    const int iteration_index, const float4 *__restrict__ input_samples_positions, const int input_count) {
    // Check if this warp should perform input excitation
    const int input_index = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x) / IO_THREADS_PER_INPUT;
    if (input_index >= input_count) return;
    const int rank = static_cast<int>(threadIdx.x & (IO_THREADS_PER_INPUT - 1)); // Warp index

    // Input coordinates + sample (w)
    const auto [x, y, z, w] = input_samples_positions[input_index + iteration_index * input_count];
    const float x_floor = floorf(x);
    const float y_floor = floorf(y);
    const float z_floor = floorf(z);

    // Offsets on x and y-axis, between -1 and 2
    const int o_x = ((rank >> 0) & 3) - 1;
    const int o_y = ((rank >> 2) & 3) - 1;
    // Sampling nodes coordinates on x and y-axis
    const int p_x = static_cast<int>(x_floor) + o_x;
    const int p_y = static_cast<int>(y_floor) + o_y;
    // Distances from input coordinate of assigned node on x and y-axis
    const float d_x = x - static_cast<float>(p_x);
    const float d_y = y - static_cast<float>(p_y);

    // The warp's 32 threads handle two distinct positions, offset on the z-axis by two nodes
    const int o_z_base = (rank >> 4) - 1;
    for (int zz_o = 0; zz_o <= 1; zz_o++) {
        const int o_z = o_z_base + zz_o * 2; // Offset on z-axis, between -1 and 2
        const int p_z = static_cast<int>(z_floor) + o_z; // Sampling nodes coordinate on z-axis
        const float d_z = z - static_cast<float>(p_z); // Distances from input coordinate of assigned node on z-axis

        // Compute the node's contribution using the gaussian distribution driving
        // Setting the alpha parameter from the above-mentioned paper to 1.35 (in node units here, not metric units)
        // has been found to keep the total magnitude added more or less constant, regardless of fractional source positions
#define GAUSSIAN_DIST_ALPHA 1.35f
        const float dist = sqrtf(d_x * d_x + d_y * d_y + d_z * d_z);
        const float mul = expf(-GAUSSIAN_DIST_ALPHA * dist * dist);

        // Atomic add is used since other warps may perform other source excitations on the same node at the same time
        atomicAdd(&p[P_INDEX(p_x, p_y, p_z)], w * mul);
    }
}

#define IO_THREADS_PER_OUTPUT 8
// Perform trilinear interpolation sampling for each output
__device__ void io_step_output_perform( // TODO: remove size_x?
    const int size_x, const int alloc_stride_x, const int size_y, const float *__restrict__ const p,
    const int iteration_index, const float4 *__restrict__ output_positions, float *__restrict__ output_samples,
    const int output_count) {
    // Check if this subwarp should perform output sampling
    const int output_index = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x) / IO_THREADS_PER_OUTPUT;
    if (output_index >= output_count) return;
    const int rank = static_cast<int>(threadIdx.x & (IO_THREADS_PER_OUTPUT - 1)); // Subwarp (from 0 to 7) rank

    // Output coordinates
    const float4 c = output_positions[output_index + iteration_index * output_count];
    const float x_floor = floorf(c.x);
    const float x_frac = c.x - x_floor;
    const float y_floor = floorf(c.y);
    const float y_frac = c.y - y_floor;
    const float z_floor = floorf(c.z);
    const float z_frac = c.z - z_floor;

    // Sample each node
    const int p_x = static_cast<int>(x_floor) + ((rank >> 0) & 1); // + 1 on x-axis for ranks 1,3,5,7
    const int p_y = static_cast<int>(y_floor) + ((rank >> 1) & 1); // + 1 on y-axis for ranks 2,3,6,7
    const int p_z = static_cast<int>(z_floor) + ((rank >> 2) & 1); // + 1 on z-axis for ranks 4,5,6,7
    float v = p[P_INDEX(p_x, p_y, p_z)];

    // Compute the trilinear interpolation
    const unsigned active = __activemask();
    v = device_lerp(v, __shfl_down_sync(active, v, 4), z_frac);
    v = device_lerp(v, __shfl_down_sync(active, v, 2), y_frac);
    v = device_lerp(v, __shfl_down_sync(active, v, 1), x_frac);

    // Write the output sample from the subwarp leader
    if (rank == 0) output_samples[output_index + iteration_index * output_count] = v;
}

__global__ void __launch_bounds__(IO_BLOCK_SIZE)
io_step( // TODO: remove size_x?
    const int size_x, const int alloc_stride_x, const int size_y, float *__restrict__ p, // NOLINT(*-non-const-parameter)
    int4 *__restrict__ ctrl, const float4 *__restrict__ input_samples_positions,
    const float4 *__restrict__ output_positions, float *__restrict__ output_samples // NOLINT(*-non-const-parameter)
) {
    const cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    // Control data contains { iteration_index, input_count, output_count, unused }
    const int4 control = *ctrl;
    const int iteration_index = control.x;
    const int input_count = control.y;
    const int output_count = control.z;

    io_step_input_perform(
        size_x, alloc_stride_x, size_y, p, iteration_index, input_samples_positions, input_count);
    grid.sync();
    io_step_output_perform(
        size_x, alloc_stride_x, size_y, p, iteration_index, output_positions, output_samples, output_count);

    if (threadIdx.x == 0 && blockIdx.x == 0) ctrl->x++; // Increase iteration_index
}

// Graph builder ///////////////////////////////////////////////////////////////////////////////////////////////////////

dwr3::input_output::graph_builder::graph_builder(cudaStream_t main_stream_, input_output *io_)
    : main_stream(main_stream_), io(io_) {
    constexpr int inputs_per_block = IO_BLOCK_SIZE / IO_THREADS_PER_INPUT;
    constexpr int outputs_per_block = IO_BLOCK_SIZE / IO_THREADS_PER_OUTPUT;
    // Each launch performs all inputs, synchronizes, then launches all outputs, meaning that the grid must accommodate
    // the largest amount of blocks between inputs and outputs
    grid = dim3(std::max(
        (io->input_count_limit + inputs_per_block - 1) / inputs_per_block,
        (io->output_count_limit + outputs_per_block - 1) / outputs_per_block));

    // Check that the instantiated grid can be executed with a cooperative launch on the CUDA device #0
    int supportsCoopLaunch;
    CUDA_THROW_ON_ERROR(cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, 0));
    if (!supportsCoopLaunch) throw std::runtime_error("CUDA device 0 does not support cooperative launches");
    int max_active_blocks_per_sm;
    CUDA_THROW_ON_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_per_sm, reinterpret_cast<const void*>(io_step), IO_BLOCK_SIZE, 0));
    int sm_count = 0;
    CUDA_THROW_ON_ERROR(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
    if (const int max_cooperative_grid_size = max_active_blocks_per_sm * sm_count; grid.x > max_cooperative_grid_size)
        throw std::runtime_error(
            std::string("Input/output limit exceeds the maximum amount supported on CUDA device 0") +
            "(max inputs: " + std::to_string(max_cooperative_grid_size * inputs_per_block) + ", " +
            "max outputs: " + std::to_string(max_cooperative_grid_size * outputs_per_block) + ")");

    // Starts with this event already completed, since propagation_medium goes first in the graph and depends on this event
    CUDA_THROW_ON_ERROR(cudaEventRecord(event_done.get(), main_stream));
}

cudaEvent_t dwr3::input_output::graph_builder::completion_event() const { return event_done.get(); }

void dwr3::input_output::graph_builder::add_iteration_node_and_record_completion_event(
    const std::vector<cudaEvent_t> &pm_events, float *p) const {
    // Wait for the propagation medium update to complete
    for (const auto &e: pm_events)
        CUDA_THROW_ON_ERROR(cudaStreamWaitEvent(main_stream, e, 0));
    void *params[8] = {
        static_cast<void *>(&io->p_info_.size[0]), static_cast<void *>(&io->p_info_.alloc_stride_x),
        static_cast<void *>(&io->p_info_.size[1]), static_cast<void *>(&p),
        static_cast<void *>(&io->alloc_layout_d.control_data),
        static_cast<void *>(&io->alloc_layout_d.input_samples_positions),
        static_cast<void *>(&io->alloc_layout_d.output_positions),
        static_cast<void *>(&io->alloc_layout_d.output_samples)
    };
    // Perform sources excitation followed by receiver sampling
    CUDA_THROW_ON_ERROR(cudaLaunchCooperativeKernel(io_step, grid, dim3(IO_BLOCK_SIZE), params, 0, main_stream));
    CUDA_THROW_ON_ERROR(cudaEventRecord(event_done.get(), main_stream)); // Record completion event
}

std::unique_ptr<dwr3::input_output::graph_builder> dwr3::input_output::create_graph_builder(cudaStream_t main_stream) {
    return std::move(std::make_unique<graph_builder>(main_stream, this));
}

// io class ////////////////////////////////////////////////////////////////////////////////////////////////////////////

dwr3::input_output::input_output(
    instance_info &info, const int buffer_size_, const int input_count_limit_,
    const int output_count_limit_, const p_info &p_info_)
    : buffer_size(buffer_size_), input_count_limit(input_count_limit_), output_count_limit(output_count_limit_),
      p_info_(p_info_) {
    if (buffer_size <= 0) throw std::logic_error("buffer_size is less or equal than zero");
    if (input_count_limit <= 0) throw std::logic_error("input_count_limit is less or equal than zero");
    if (output_count_limit <= 0) throw std::logic_error("output_count_limit is less or equal than zero");

    // In order to transfer h2d/d2h in a single operation, the data is packed into a single contiguous allocation
    const auto compute_alloc_size_and_layout = [this](void *alloc, alloc_layout &layout) {
        // First there is a h2d data section
        void *h2d_ptr = alloc;
        layout.control_data = static_cast<int4 *>(h2d_ptr);
        h2d_ptr = static_cast<int4 *>(h2d_ptr) + 1;
        layout.input_samples_positions = static_cast<float4 *>(h2d_ptr);
        h2d_ptr = static_cast<float4 *>(h2d_ptr) + input_count_limit * buffer_size;
        layout.output_positions = static_cast<float4 *>(h2d_ptr);
        h2d_ptr = static_cast<float4 *>(h2d_ptr) + output_count_limit * buffer_size;
        alloc_h2d_size = static_cast<size_t>(static_cast<uint8_t *>(h2d_ptr) - static_cast<uint8_t *>(alloc));
        // Then a d2h section
        void *d2h_ptr = h2d_ptr;
        layout.output_samples = static_cast<float *>(d2h_ptr);
        d2h_ptr = static_cast<float *>(d2h_ptr) + output_count_limit * buffer_size;
        alloc_d2h_size = static_cast<size_t>(static_cast<uint8_t *>(d2h_ptr) - static_cast<uint8_t *>(h2d_ptr));
    };
    {
        // First call the function to compute the alloc_h2d_size and alloc_d2h_size members
        alloc_layout _;
        compute_alloc_size_and_layout(nullptr, _);
    }
    // Then allocate the actual memory and fill the respective host/device alloc_layout structs
    CUDA_THROW_ON_ERROR(cudaHostAlloc(&alloc_h, alloc_h2d_size + alloc_h2d_size, cudaHostAllocDefault));
    CUDA_THROW_ON_ERROR(cudaMalloc(&alloc_d, alloc_h2d_size + alloc_h2d_size));
    compute_alloc_size_and_layout(alloc_h, alloc_layout_h);
    compute_alloc_size_and_layout(alloc_d, alloc_layout_d);
    info.io_memory_size = alloc_h2d_size + alloc_h2d_size;
}

dwr3::input_output::~input_output() noexcept {
    cudaFreeHost(alloc_h);
    cudaFree(alloc_d);
}

// ReSharper disable once CppMemberFunctionMayBeStatic
void dwr3::input_output::reset() {
}

// Conversion between physical and propagation medium coordinates clamps out of bounds coordinates inside the propagation medium
template<bool is_input>
static void xyz_physical_to_nodes(const dwr3::p_info &p_info, float &x, float &y, float &z) {
    // Inputs act on a 4x4x4 nodes radius, so an extra unit of padding is needed, compared to the outputs' 2x2x2 radius
    constexpr int padding_i = is_input ? 1 : 0;
    constexpr float padding_f = is_input ? 1.0f : 0.0f;
    if (p_info.centered_boundary_conditions_scheme) {
        // In centered boundary conditions, boundary are considered part of the propagation medium, so the nodes at the
        // extremities of the propagation medium lay exactly on the boundaries
        x = min(max(x * p_info.nodes_per_meter, padding_f), static_cast<float>(p_info.size[0] - 2 - padding_i));
        y = min(max(y * p_info.nodes_per_meter, padding_f), static_cast<float>(p_info.size[1] - 2 - padding_i));
        z = min(max(z * p_info.nodes_per_meter, padding_f), static_cast<float>(p_info.size[2] - 2 - padding_i));
    } else {
        // In non-centered boundary conditions, boundaries are not considered part of the propagation medium
        x = min(max(x * p_info.nodes_per_meter - 0.5f, padding_f), static_cast<float>(p_info.size[0] - 1 - padding_i));
        y = min(max(y * p_info.nodes_per_meter - 0.5f, padding_f), static_cast<float>(p_info.size[1] - 1 - padding_i));
        z = min(max(z * p_info.nodes_per_meter - 0.5f, padding_f), static_cast<float>(p_info.size[2] - 1 - padding_i));
    }
    // Figure 5.17 in
    // Hamilton, Brian. "Finite difference and finite volume methods for wave-based modelling of room acoustics." (2016).
    // showcases the difference between the two cases
}

void dwr3::input_output::prepare_iteration_transfer_h2d(
    int input_count, const sample_position_value *const *input_samples_positions,
    int output_count, const sample_position_value *const *output_positions, cudaStream_t stream) {
    input_count = min(max(input_count, 0), input_count_limit);
    output_count = min(max(output_count, 0), output_count_limit);
    output_count_cached = output_count;

    // Control data contains { iteration_index, input_count, output_count, unused }
    *alloc_layout_h.control_data = {0, input_count, output_count, 0};

    // TODO: switch from host controlled change of basis + clamping to CUDA device change of basis + silencing out of bounds?

    // The h2d data is transposed so that data regarding accessed during a single iteration is all contiguous on device
    for (int i = 0; i < buffer_size; i++) {
        for (int j = 0; j < input_count; j++) {
            auto [x, y, z, v] = input_samples_positions[j][i];
            apply_change_of_basis(x, y, z, p_info_.change_of_basis);
            // Swizzle the x, y and z components to change basis
            xyz_physical_to_nodes<true>(p_info_, x, y, z); // Convert from physical to node coordinates
            alloc_layout_h.input_samples_positions[j + i * input_count] = {x, y, z, v}; // Write transposed
        }
        for (int j = 0; j < output_count; j++) {
            auto [x, y, z, _] = output_positions[j][i];
            apply_change_of_basis(x, y, z, p_info_.change_of_basis);
            // Swizzle the x, y and z components to change basis
            xyz_physical_to_nodes<false>(p_info_, x, y, z); // Convert from physical to node coordinates
            alloc_layout_h.output_positions[j + i * output_count] = {x, y, z, 0}; // Write transposed
        }
    }
    CUDA_THROW_ON_ERROR(cudaMemcpyAsync(alloc_d, alloc_h, alloc_h2d_size, cudaMemcpyHostToDevice, stream));
}

void dwr3::input_output::transfer_d2h(cudaStream_t stream) const {
    CUDA_THROW_ON_ERROR(cudaMemcpyAsync(alloc_layout_h.output_samples, alloc_layout_d.output_samples, alloc_d2h_size,
        cudaMemcpyHostToDevice, stream));
}

void dwr3::input_output::return_output_samples(float *const *output_samples) const noexcept {
    // The samples in output_samples are written in transposed form on device, so they are untransposed back into per-channel signals
    for (int i = 0; i < buffer_size; i++) {
        for (int j = 0; j < output_count_cached; j++) {
            output_samples[j][i] = alloc_layout_h.output_samples[j + i * output_count_cached];
        }
    }
}

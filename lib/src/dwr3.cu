#include <memory>

#include "common.cuh"
#include "propagation_medium.cuh"
#include "input_output.cuh"
#include "colormap.hpp"

namespace dwr3 {
    // Implementation class ////////////////////////////////////////////////////////////////////////////////////////////

    class implementation final {
        instance_info inst_info;
        std::unique_ptr<propagation_medium> pm;
        input_output io;
        int exec_subgraph_iterations;
        stream_priorities sp;
        stream exec_stream;
        graph_exec exec_graph;
        bool processing_started_flag; // The processing_started_flag is used to check the correct call order between
        // processing_start and processing_retrieve

        // Validate and compute the number of subgraph iterations needed
        static int v_exec_subgraph_iterations(const int buffer_size) {
            if (buffer_size % buffer_base_size != 0)
                throw std::logic_error(
                    "buffer_size is not a multiple of buffer_base_size = " + std::to_string(buffer_base_size));
            return buffer_size / buffer_base_size;
        }

        // Build the cudaGraphExec_t used at runtime
        static cudaGraphExec_t v_exec_graph(
            propagation_medium *pm, input_output &io, const stream_priorities &sp, cudaStream_t subgraph_stream) {
            CUDA_THROW_ON_ERROR(cudaStreamBeginCapture(subgraph_stream, cudaStreamCaptureModeGlobal));
            const auto pm_gb = pm->create_graph_builder(subgraph_stream, sp);
            const auto io_gb = io.create_graph_builder(subgraph_stream);
            for (int i = 0; i < buffer_base_size; i++) {
                pm_gb->add_iteration_nodes_and_record_completion_events(io_gb->completion_event());
                io_gb->add_iteration_node_and_record_completion_event(pm_gb->completion_events(), pm_gb->p());
            }
            cudaGraph_t subgraph{};
            CUDA_THROW_ON_ERROR(cudaStreamEndCapture(subgraph_stream, &subgraph));
            try {
                cudaGraphExec_t ge;
                CUDA_THROW_ON_ERROR(
                    cudaGraphInstantiateWithFlags(&ge, subgraph, cudaGraphInstantiateFlagUseNodePriority));
                cudaGraphDestroy(subgraph);
                return ge;
            } catch (const std::exception &) {
                cudaGraphDestroy(subgraph);
                throw;
            }
        }

    public:
        DWR3_NON_COPYABLE(implementation);

        DWR3_DECLARE_NON_MOVEABLE(implementation);

        // Constructors receive a pre-constructed propagation_medium instance, alongsides a pre-filled instance_info
        // with the fields set by the propagation_medium's construction

        implementation(
            const instance_info &info_, std::unique_ptr<propagation_medium> pm_, const int buffer_size,
            const int input_count_limit, const int output_count_limit)
            : inst_info(info_),
              pm(std::move(pm_)),
              io(inst_info, buffer_size, input_count_limit, output_count_limit, pm->p_info()),
              exec_subgraph_iterations(v_exec_subgraph_iterations(buffer_size)),
              exec_stream(sp.medium()),
              exec_graph(v_exec_graph(pm.get(), io, sp, exec_stream.get())),
              processing_started_flag(false) {
        }

        [[nodiscard]] instance_info info() const noexcept { return inst_info; }

        void reset() {
            processing_started_flag = false;
            pm->reset();
            io.reset();
        }

        void processing_start(
            const int input_count, const sample_position_value *const *input_samples_positions,
            const int output_count, const sample_position_value *const *output_positions) {
            if (processing_started_flag)
                throw dwr3::processing_state_error(
                    "processing_start called twice in a row without a preceding processing_retrieve or reset call");
            io.prepare_iteration_transfer_h2d(
                input_count, input_samples_positions, output_count, output_positions, exec_stream.get());
            for (int i = 0; i < exec_subgraph_iterations; i++)
                CUDA_THROW_ON_ERROR(cudaGraphLaunch(exec_graph.get(), exec_stream.get()));
            io.transfer_d2h(exec_stream.get());
            pm->copy_latest_xy_plane_output_data(exec_stream.get());
            processing_started_flag = true;
        }

        [[nodiscard]] bool processing_started() const noexcept { return processing_started_flag; }

        void processing_retrieve(float *const *output_samples) {
            if (!processing_started_flag)
                throw dwr3::processing_state_error(
                    "processing_retrieve called without a preceding processing_start call");
            CUDA_THROW_ON_ERROR(cudaStreamSynchronize(exec_stream.get()));
            processing_started_flag = false;
            io.return_output_samples(output_samples);
        }

        [[nodiscard]] float *xy_plane_output() const noexcept { return pm->xy_plane_output_data(); }
    };

    // Public facing class /////////////////////////////////////////////////////////////////////////////////////////////
    // When exceptions (outsides of processing_state_error) reach the public facing class,
    // this means that an error is unrecoverable, so the instance is deleted turning all future calls into no-ops

    dwr3::dwr3(
        float size[3], const boundary_reflectance_filter_coefficients *const boundary_coefficients[6],
        const int sample_rate, const int buffer_size, const int input_count_limit, const int output_count_limit,
        const bool p_xy_plane_output_enable, const float p_xy_plane_output_z_axis_position) {
        instance_info info{};
        auto pm = std::make_unique<propagation_medium_kv_2009>(
            info, size, boundary_coefficients, sample_rate, buffer_size,
            p_xy_plane_output_enable, p_xy_plane_output_z_axis_position);
        instance = new implementation(info, std::move(pm), buffer_size, input_count_limit, output_count_limit);
    }

    dwr3::~dwr3() {
        delete static_cast<implementation *>(instance);
        instance = nullptr;
    }

    instance_info dwr3::info() const noexcept {
        constexpr instance_info info{};
        if (!instance) return info;
        return static_cast<implementation *>(instance)->info();
    }

    void dwr3::reset() const {
        if (!instance) return;
        try {
            static_cast<implementation *>(instance)->reset();
        } catch (const std::exception &) {
            dwr3::~dwr3();
            throw;
        }
    }

    void dwr3::processing_start(
        const int input_count, const sample_position_value *const *input_samples_positions,
        const int output_count, const sample_position_value *const *output_positions) const {
        if (!instance) return;
        try {
            return static_cast<implementation *>(instance)->processing_start(
                input_count, input_samples_positions, output_count, output_positions);
        } catch (const processing_state_error &) {
            throw;
        } catch (const std::exception &) {
            dwr3::~dwr3();
            throw;
        }
    }

    bool dwr3::processing_started() const noexcept {
        if (!instance) return false;
        return static_cast<implementation *>(instance)->processing_started();
    }

    void dwr3::processing_retrieve(float *const *output_samples) const {
        if (!instance) return;
        try {
            return static_cast<implementation *>(instance)->processing_retrieve(output_samples);
        } catch (const processing_state_error &) {
            throw;
        } catch (const std::exception &) {
            dwr3::~dwr3();
            throw;
        }
    }

    float *dwr3::p_xy_plane_output() const noexcept {
        if (!instance) return nullptr;
        return static_cast<implementation *>(instance)->xy_plane_output();
    }

    void p_xy_plane_output_to_rgb_image_data(
        const float *p_xy_plane_output_data, uint8_t *image_data, const instance_info &info,
        const float clipping_level) noexcept {
        // Handle edge cases
        if (image_data == nullptr) return;
        if (p_xy_plane_output_data == nullptr) {
            memset(image_data, 0, info.node_size[0] * info.node_size[1] * sizeof(uint8_t));
            return;
        }

        for (int y = 0; y < info.node_size[1]; y++) {
            for (int x = 0; x < info.node_size[0]; x++) {
                // Get the value accounting for x-axis stride in memory, normalize with clipping level
                const float n = p_xy_plane_output_data[x] / clipping_level;
                const uint8_t *rgb = colormap_rgb_from_normalized(n);
                image_data[0] = rgb[0];
                image_data[1] = rgb[1];
                image_data[2] = rgb[2];
                image_data += 3;
            }
            p_xy_plane_output_data += info.p_xy_plane_row_stride; // Ignore the values on each row between info.node_size[0] and info.p_xy_plane_row_stride-1
        }
    }
}

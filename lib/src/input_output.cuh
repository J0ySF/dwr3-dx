#ifndef DWR3_INPUT_OUTPUT_CUH
#define DWR3_INPUT_OUTPUT_CUH

#include "common.cuh"

namespace dwr3 {
    // Manages the host<->device transfer of samples and performs the excitation/sampling of values from p buffers
    class input_output final {
        // Instantiated while the execution graph is being recorded to perform each io step
        class graph_builder {
            cudaStream_t main_stream;
            input_output *io;
            dim3 grid;
            event event_done;

        public:
            explicit graph_builder(cudaStream_t main_stream_, input_output *io_);

            [[nodiscard]] cudaEvent_t completion_event() const;

            void add_iteration_node_and_record_completion_event(
                const std::vector<cudaEvent_t> &pm_events, float *p) const;

            DWR3_NON_COPYABLE(graph_builder);

            DWR3_DECLARE_NON_MOVEABLE(graph_builder);
        };

    public:
        std::unique_ptr<graph_builder> create_graph_builder(cudaStream_t main_stream);

    private:
        int buffer_size, input_count_limit, output_count_limit;
        p_info p_info_;

        // The io implementation handles memory on device with a single contiguous memory allocation
        // The single contiguous memory allocation on device has a counterpart on host

        // Pointers to the different data inside the contiguous memory allocation
        struct alloc_layout {
            int4 *control_data{}; // h2d
            float4 *input_samples_positions{}; // h2d
            float4 *output_positions{}; // h2d
            float *output_samples{}; // d2h
        };

        // The allocation has a first section (h2d) containing data that is prepared on host, then transferred from host to device,
        // the second section (d2h) is used to store the data computed on device which is then transferred back to host

        // Sizes of the two allocation sections
        size_t alloc_h2d_size{}, alloc_d2h_size{};

        void *alloc_h = nullptr, *alloc_d = nullptr;
        alloc_layout alloc_layout_h, alloc_layout_d;

        // Value set during prepare_iteration_transfer_h2d to keep track of how many outputs there are
        int output_count_cached{};

    public:
        DWR3_NON_COPYABLE(input_output);

        DWR3_DECLARE_NON_MOVEABLE(input_output);

        input_output(
            instance_info &info, int buffer_size_, int input_count_limit_, int output_count_limit_,
            const p_info &p_info_);

        ~input_output() noexcept;

        void reset();

        void prepare_iteration_transfer_h2d(
            int input_count, const sample_position_value *const *input_samples_positions,
            int output_count, const sample_position_value *const *output_positions, cudaStream_t stream);

        void transfer_d2h(cudaStream_t stream) const;

        void return_output_samples(float *const *output_samples) const noexcept;
    };
}

#endif //DWR3_INPUT_OUTPUT_CUH

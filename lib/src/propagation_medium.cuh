#ifndef DWR3_PROPAGATION_MEDIUM_CUH
#define DWR3_PROPAGATION_MEDIUM_CUH

#include "common.cuh"
#include "digital_impedance_filter.cuh"

namespace dwr3 {
    // Manages the propagation medium simulation
    class propagation_medium {
    public:
        /// Speed of sound constant for air
        static constexpr float speed_of_sound = 343.0f;

        // Instantiated while the execution graph is being recorded to perform each propagation medium simulation step
        class graph_builder {
        public:
            explicit graph_builder() = default;

            virtual ~graph_builder() = default;

            /// Returns a vector reference to one or more events,
            /// since different implementation use different simulation step graph structures
            [[nodiscard]] virtual const std::vector<cudaEvent_t> &completion_events() const = 0;

            virtual void add_iteration_nodes_and_record_completion_events(cudaEvent_t io_event) = 0;

            /// Returns the current simulation instant's p buffer
            virtual float *p() = 0;

            DWR3_NON_COPYABLE(graph_builder);

            DWR3_DECLARE_NON_MOVEABLE(graph_builder);
        };

        virtual std::unique_ptr<graph_builder> create_graph_builder(
            cudaStream_t main_stream, stream_priorities stream_priorities) = 0;

        [[nodiscard]] virtual p_info p_info() const noexcept = 0;

        DWR3_NON_COPYABLE(propagation_medium);

        DWR3_DECLARE_NON_MOVEABLE(propagation_medium);

        explicit propagation_medium() = default;

        virtual ~propagation_medium() noexcept = default;

        virtual void reset() = 0;

        virtual void copy_latest_xy_plane_output_data(cudaStream_t stream) = 0;

        [[nodiscard]] virtual float *xy_plane_output_data() const noexcept = 0;
    };

    /// Implementation based on the (centered boundary conditions) SLF scheme from
    /// K. Kowalczyk and M. van Walstijn, "Room Acoustics Simulation Using 3-D Compact Explicit FDTD Schemes," in
    /// IEEE Transactions on Audio, Speech, and Language Processing, vol. 19, no. 1, pp. 34-46, Jan. 2011, doi: 10.1109/TASL.2010.2045179.
    class propagation_medium_kv_2009 final : public propagation_medium {
        class kv_2009_graph_builder final : public graph_builder {
            cudaStream_t main_stream;
            propagation_medium_kv_2009 *pm;

            dim3 iter_x_grid, iter_y_grid, iter_z_grid, iter_c_grid;

            stream stream_x, stream_y, stream_z_lp, stream_z_hp;
            event event_done_x, event_done_y, event_done_z_lp, event_done_z_hp;
            std::vector<cudaEvent_t> event_vector;

            bool reverse_z;
            bool pad_x;

            template<int pad_x, bool reverse_z>
            void add_iteration_nodes_and_record_completion_events_impl(cudaEvent_t precondition_event);

        public:
            explicit kv_2009_graph_builder(
                cudaStream_t main_stream, stream_priorities sp, propagation_medium_kv_2009 *pm_);

            ~kv_2009_graph_builder() override = default;

            [[nodiscard]] const std::vector<cudaEvent_t> &completion_events() const override;

            void add_iteration_nodes_and_record_completion_events(cudaEvent_t io_event) override;

            float *p() override;
        };

        struct boundary_state {
            size_t size{};

            digital_impedance_filter_coefficients dif{};

            float *g{};
            float *x[DWR3_BOUNDARY_FILTER_ORDER]{}, *y[DWR3_BOUNDARY_FILTER_ORDER]{};

            void init(const boundary_reflectance_filter_coefficients *boundary_reflectance_filter, size_t size);

            void release();

            void reset() const;

            void rotate_xy();
        };

        struct p_info p_info_{};

        size_t p_alloc_size{};
        float *p_alloc_d[2]{};

        boundary_state b_state[6]{};

        float *xy_plane_output_data_alloc{};
        size_t xy_plane_output_data_size{};
        int xy_plane_output_z_axis_position_node{};

    public:
        std::unique_ptr<graph_builder> create_graph_builder(
            cudaStream_t main_stream, stream_priorities stream_priorities) override;

        [[nodiscard]] struct p_info p_info() const noexcept override;

        DWR3_NON_COPYABLE(propagation_medium_kv_2009);

        DWR3_DECLARE_NON_MOVEABLE(propagation_medium_kv_2009);

        propagation_medium_kv_2009(
            instance_info &info, const float size[3],
            const boundary_reflectance_filter_coefficients *const boundary_reflectance_filters[6],
            int sample_rate, int buffer_size, bool xy_plane_output_enable, float xy_plane_output_z_axis_position);

        ~propagation_medium_kv_2009() noexcept override;

        void reset() override;

        void copy_latest_xy_plane_output_data(cudaStream_t stream) override;

        [[nodiscard]] float *xy_plane_output_data() const noexcept override;
    };
}

#endif //DWR3_PROPAGATION_MEDIUM_CUH

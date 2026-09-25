#ifndef DWR3_HPP
#define DWR3_HPP

/**
 * This library uses metric units for units of measure
 */

#include <stdexcept>

#ifndef DWR3_BOUNDARY_FILTER_ORDER
/// The order of the Infinite Impulse Response boundary reflectance filters used by dwr3
#define DWR3_BOUNDARY_FILTER_ORDER 8
#endif

#ifndef DWR3_BUFFER_BASE_SIZE
/// Internal buffer base size used by dwr3
#define DWR3_BUFFER_BASE_SIZE 64
#endif

#define DWR3_BOUNDARY_FILTER_TYPE_REGULAR 0
#define DWR3_BOUNDARY_FILTER_TYPE_HIGHPASS 1
#define DWR3_BOUNDARY_FILTER_TYPE_BANDPASS 2
#ifndef DWR3_BOUNDARY_FILTER_TYPE
/// The boundary type used for boundary filters, see scripts/boundary_reflectance_filters_coefficients_generator.m
#define DWR3_BOUNDARY_FILTER_TYPE DWR3_BOUNDARY_FILTER_TYPE_BANDPASS
#endif

namespace dwr3 {
    /// Boundary filters order used in the implementation
    constexpr int boundary_filter_order = DWR3_BOUNDARY_FILTER_ORDER;
    /// Base buffer size used in the implementation, must be defined as equal or a multiple of boundary_filter_order
    constexpr int buffer_base_size = DWR3_BUFFER_BASE_SIZE;

    /**
     * Instance information
     */
    struct instance_info {
        /// Discrete nodes size used by the propagation medium model
        int node_size[3]{};
        /// Number of discrete nodes per meter used by the propagation medium model
        float nodes_per_meter{};
        /// Bytes of CUDA device memory used by the propagation medium part of the simulation
        size_t pm_memory_size{};
        /// Bytes of CUDA device memory used by the input output part of the simulation
        size_t io_memory_size{};
    };

    /**
     * Infinite Impulse Response boundary reflectance filter coefficients
     * @note The coefficients can be provided in non-normalized form (with a[0] != 1)
     */
    struct boundary_reflectance_filter_coefficients {
        /// b coefficients
        double b[boundary_filter_order + 1];
        /// a coefficients
        double a[boundary_filter_order + 1];
    };

    /**
     * Sample with position information
     */
    struct alignas(16) sample_position_value {
        /// Sample x-axis coordinate
        float x;
        /// Sample y-axis coordinate
        float y;
        /// Sample z-axis coordinate
        float z;
        /// Sample value
        float v;
    };

    /**
     * Class that performs a Finite Difference Time Domain acoustic simulation with movable acoustic sources (inputs) and receivers (outputs).
     * The simulation is processed over blocks of discrete samples, starting with a @p processing_start call, which begins
     * the asynchronous computation on a CUDA enabled device. After calling @p processing_start, @p processing_retrieve is called
     * to read the results from the asynchronous computation, this enables @p processing_start to be called again later.
     * Make sure to refer to the single methods for more information.
     *
     * The type of Finite Difference Time Domain acoustic simulation medium depends on the constructor used, refer to the
     * specific constructors for more information.
     *
     * @note This class is not thread safe.
     * @note Execution is carried out on the default CUDA device (index 0)
     * @note In the case of unrecoverable runtime errors, this class throws exceptions and turns any further calls to its functions into no-ops.
     * Errors related to incorrect @p processing_start/ @p processing_retrieve call order are signaled with @p processing_state_error exceptions,
     * after which the instance state can still be used normally.
     */
    class dwr3 {
    public:
        /**
         * Exception thrown when @p processing_start and @p processing_retrieve are not called with the required ordering rules
         */
        struct processing_state_error : std::logic_error {
            using std::logic_error::logic_error;
        };

        // Delete copy constructors
        dwr3(const dwr3 &) = delete;

        // Delete copy constructors
        dwr3 &operator=(const dwr3 &) = delete;

        // Delete move constructor
        dwr3(dwr3 &&) = delete;

        // Delete move constructor
        dwr3 &operator=(dwr3 &&) = delete;

        /**
         * Creates a rectangular acoustic simulation instance using the SLF scheme from
         * K. Kowalczyk and M. van Walstijn, "Room Acoustics Simulation Using 3-D Compact Explicit FDTD Schemes," in
         * IEEE Transactions on Audio, Speech, and Language Processing, vol. 19, no. 1, pp. 34-46, Jan. 2011, doi: 10.1109/TASL.2010.2045179.
         * @param[in] size Physical size on the x, y and z axes
         * @param[in] boundary_coefficients Boundary reflectance filters coefficients for boundaries in order {x-, x+, y-, y+, z, -z+},
         * the @p boundary_reflectance_filter_material_to_coefficients function can be used to obtain pre-fitted filters
         * @param[in] sample_rate Audio sample rate
         * @param[in] buffer_size Buffer size used during processing calls
         * @param[in] input_count_limit Maximum supported amount of inputs
         * @param[in] output_count_limit Maximum supported amount of outputs
         * @note @p sample_rate, @p buffer_size, @p input_count_limit, @p output_count_limit must be greater than zero
         * @note @p buffer_size must be equal or a multiple of @p dwr3::buffer_base_size
         * @note @p buffer_size must be equal or a multiple of @p dwr3::boundary_filter_order
         * @note each element in @p boundary_coefficients must be not @p NULL
         * @throws std::exception in the case of failure
         */
        explicit dwr3(
            float size[3], const boundary_reflectance_filter_coefficients *const boundary_coefficients[6],
            int sample_rate, int buffer_size, int input_count_limit, int output_count_limit);

        /**
         * @note Waits for any asynchronous processing caused by @p processing_start to terminate
         */
        ~dwr3() noexcept;

        [[nodiscard]] instance_info info() const noexcept;

        /**
         * Resets the instance's state to it's starting state (the same as it is found after construction)
         * @note Waits for any asynchronous processing caused by @p processing_start to terminate
         * @note Resets the computation state relating to the order @p between processing_start and @p processing_retrieve,
         * @p processing_start must be called first after this function returns
         * @throws std::exception in the case of failure
         */
        void reset() const;

        /**
         * Starts processing asynchronously a block of @p buffer_size samples (see @p dwr3::dwr3 's arguments)
         * @param input_count Active inputs count during this block's processing
         * @param input_samples_positions An array of @p input_count buffers, each one containing @p buffer_size samples with position information
         * @param output_count Active outputs count during this block's processing
         * @param output_positions An array of @p output_count buffers, each one containing @p buffer_size samples with position information,
         * (only the {x,y,z} fields are used)
         * @note Requires that @p processing_wait is called after it to retrieve the output samples
         * @note @p input_count and @p output_count are capped respectively to @p input_count_limit and @p output_count_limit if greater
         * (see @p dwr3::dwr3 's arguments)
         * @note @p input_count and @p output_count values less than zero are treated as zero
         * @throws std::exception in the case of failure
        */
        void processing_start(
            int input_count, const sample_position_value *const *input_samples_positions,
            int output_count, const sample_position_value *const *output_positions) const;

        /**
         * @return @p true if @p processing_start was successfully called without a consecutive @p processing_retrieve call,
         * otherwise returns @p false
         */
        [[nodiscard]] bool processing_started() const noexcept;

        /**
         * Waits for the asynchronous processing started by a @p processing_start call to terminate,
         * then provides @p buffer_size output samples for each of the @p output_count outputs (see @p dwr3::processing_start's arguments)
         * @param output_samples An array of @p output_count (see @p dwr3::processing_start 's arguments) buffers,
         * each one containing @p buffer_size samples on success
         * @throws std::exception in the case of failure
         * @note In the case of failure, the data in @p output_samples is not considered valid and should not be used
        */
        void processing_retrieve(float *const *output_samples) const;

    private:
        /// Instance handle
        void *instance{};
    };

    // ReSharper disable once CppUnusedIncludeDirective
#include <dwr3/gen/boundary_reflectance_filters.hpp>
}

#endif //DWR3_HPP

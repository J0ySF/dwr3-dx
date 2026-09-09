#ifndef DWR3_H
#define DWR3_H

#ifdef __cplusplus
extern "C" {
#endif

/// Important information //////////////////////////////////////////////////////////////////////////////////////////////
/// * The public interface of this library is not thread safe, do not call any dwr3 function from multiple threads with
///   the same instance handle, unless you make sure that all calls are mutually exclusive

#ifndef DWR3_BOUNDARY_FILTER_ORDER
/// The order of the Infinite Impulse Response boundary reflectance filters used by dwr3
#define DWR3_BOUNDARY_FILTER_ORDER 8
#endif

/// Infinite Impulse Response boundary reflectance filter coefficients
/// @note The coefficients can be provided in non-normalized form (with a[0] != 1)
typedef struct {
    double b[DWR3_BOUNDARY_FILTER_ORDER + 1], a[DWR3_BOUNDARY_FILTER_ORDER + 1];
} dwr3_boundary_reflectance_filter_t;

/// Sample with position information
typedef struct alignas(16) {
    /// Sample x-axis coordinate
    float x;
    /// Sample y-axis coordinate
    float y;
    /// Sample z-axis coordinate
    float z;
    /// Sample value
    float v;
} dwr3_sample_position_t;

typedef struct {
    // TODO: add fields
} dwr3_instance_info_t;

typedef enum {
    /// No error
    DWR3_ERROR_NONE = 0,
    /// Unknown error
    DWR3_ERROR_UNKNOWN,
} dwr3_error_t;

/**
 * Creates a new dwr3 instance
 * @param[out] instance Pointer set to a valid dwr3 instance handle on success, otherwise set to NULL
 * @param[out] instance_info Information about a successfully created instance
 * @param[in] size Physical size on the x, y and z axes
 * @param[in] boundary_reflectance_filters Boundary reflectance filters for boundaries in order x-, x+, y-, y+, z, -z+
 * @param[in] sample_rate Audio sample rate
 * @param[in] buffer_size Buffer size used during processing calls
 * @param[in] input_count_limit Maximum supported amount of inputs
 * @param[in] output_count_limit Maximum supported amount of outputs
 */
dwr3_error_t dwr3_create(void **instance, dwr3_instance_info_t *instance_info,
                         float size[3],
                         const dwr3_boundary_reflectance_filter_t *const boundary_reflectance_filters[6],
                         int sample_rate, unsigned int buffer_size,
                         unsigned int input_count_limit, unsigned int output_count_limit);

/**
 * Destroys a valid dwr3 instance
 * @note The instance handle can be NULL, in this case nothing happens
 * @note This operation waits for asynchronous processing tasks to terminate
 */
dwr3_error_t dwr3_destroy(void *instance);

/**
 * Resets a valid dwr3 instance's state
 * @param instance A non-NULL pointer to a valid dwr3 instance
 * @note This operation waits for asynchronous processing tasks to terminate, and drops any pending results from previous dwr3_processing_start calls
 */
dwr3_error_t dwr3_reset(void *instance);

/**
 * Starts processing asynchronously a block of buffer_size samples
 * @param instance A non-NULL pointer to a valid dwr3 instance
 * @param input_count Active inputs count during this block's processing, capped to input_count_limit if greater
 * @param input_samples_positions An array of input_count buffers, each one containing buffer_size samples with position information
 * @param output_count Active outputs count during this block's processing, capped to output_count_limit if greater
 * @param output_positions An array of output_count buffers, each one containing buffer_size samples with position information (only fields x,y,z are used, and output samples are stored in output_samples)
 * @note Requires that dwr3_processing_wait is called after it to retrieve the output samples
 */
dwr3_error_t dwr3_processing_start(void *instance,
                                   unsigned int input_count,
                                   const dwr3_sample_position_t *const *input_samples_positions,
                                   unsigned int output_count,
                                   const dwr3_sample_position_t *const *output_positions);

/**
 * @param instance A non-NULL pointer to a valid dwr3 instance
 * @return 1 if data from a dwr3_processing_start call needs to be retrieved before calling it again, otherwise 0
 */
int dwr3_processing_started(void *instance);

/**
 * Waits for the asynchronous processing started by a dwr3_processing_start call to terminate, then provides buffer_size output samples
 * @param instance A non-NULL pointer to a valid dwr3 instance
 * @param output_samples An array of output_count (as set by dwr3_processing_start) buffers, each one containing buffer_size samples on success
 * @note On error, the data possibly written to output_samples is not considered valid and should not be used
 */
dwr3_error_t dwr3_processing_retrieve(void *instance, float *const *output_samples);

#ifdef __cplusplus
}
#endif

#endif //DWR3_H

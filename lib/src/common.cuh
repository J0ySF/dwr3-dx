#ifndef DWR3_COMMON_CUH
#define DWR3_COMMON_CUH

// ReSharper disable CppUnusedIncludeDirective
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <dwr3/dwr3.hpp>
// ReSharper restore CppUnusedIncludeDirective

#define DWR3_STRINGIZE_DETAIL(x) #x
#define DWR3_STRINGIZE(x) DWR3_STRINGIZE_DETAIL(x)

/// Performs a CUDA function call anc check if it returns an error, if so throws a runtime exception with the error details
#define CUDA_THROW_ON_ERROR(CUDA_FUN_CALL)                                                                                             \
    do {                                                                                                                               \
        const cudaError_t result = CUDA_FUN_CALL;                                                                                      \
        if (result != cudaSuccess)                                                                                                     \
            throw std::runtime_error(std::string(DWR3_STRINGIZE(CUDA_FUN_CALL) " failed: ") + std::string(cudaGetErrorString(result)));\
    } while(false)

/// Macro used to delete copy constructors
#define DWR3_NON_COPYABLE(class_name) \
    class_name (const class_name&) = delete;\
    class_name& operator= (const class_name&) = delete;
/// Macro used to delete move constructors
#define DWR3_DECLARE_NON_MOVEABLE(class_name) \
    class_name (class_name&&) = delete;\
    class_name& operator= (class_name&&) = delete;

/// Macro that defines a simple wrapper around an ALREADY ALLOCATED unmanaged resource
/// for automatic destruction when going out of scope
#define DWR3_UNMANAGED_RESOURCE_WRAPPER(name, type, destroyer) class name {\
    type t; \
    public: \
    DWR3_NON_COPYABLE(name) \
    DWR3_DECLARE_NON_MOVEABLE(name) \
    name(type t_) : t(t_) {} \
    ~name() {destroyer(t);} \
    [[nodiscard]] type get() { return t; }; \
};

namespace dwr3 {
    /// Stream priority levels, queries the priority levels on construction from the CUDA API
    class stream_priorities final {
        int low_{}, medium_{}, high_{};

    public:
        explicit stream_priorities();

        [[nodiscard]] int low() const;

        [[nodiscard]] int medium() const;

        [[nodiscard]] int high() const;
    };

    /// Representation of potential changes of base in the coordinate system between what is provides as input
    /// and the propagation medium's orientation in memory
    enum class change_of_basis {
        xyz = 0, // xyz corresponds to identity
        xzy, yxz, yzx, zxy, zyx,
    };

    /// Swaps x, y and z according to change_of_basis
    template<typename T>
    static void apply_change_of_basis(T &x, T &y, T &z, const change_of_basis change_of_basis) {
        const T xc = x, yc = y, zc = z;
        switch (change_of_basis) {
            case change_of_basis::xyz: return;
            case change_of_basis::xzy:
                y = zc;
                z = yc;
                return;
            case change_of_basis::yxz:
                x = yc;
                y = xc;
                return;
            case change_of_basis::yzx:
                x = yc;
                y = zc;
                z = xc;
                return;
            case change_of_basis::zxy:
                x = zc;
                y = xc;
                z = yc;
                return;
            case change_of_basis::zyx:
                x = zc;
                z = xc;
        }
    }

    /// Information about a singular p buffer allocated by the propagation medium implementation
    struct p_info {
        /// Number of discrete nodes per meter used by the propagation medium model
        float nodes_per_meter{};
        /// True if the propagation medium employs a centered boundary conditions scheme,
        /// false if it employs a non-centered boundary conditions cheme
        bool centered_boundary_conditions_scheme{};
        /// Nodes size of the buffer
        int size[3]{};
        /// Stride along the x-axis of the memory allocation, which can be greater or equal than size[0]
        int alloc_stride_x{};
        /// The change of basis applied with respect the physical size provided
        change_of_basis change_of_basis{};
    };

    /// Wrapper around a cudaStream_t instance, allocates and frees a CUDA stream on construction/destruction
    class stream {
        bool valid = false;
        cudaStream_t res{};

    public:
        explicit stream();

        explicit stream(int priority);

        ~stream();

        [[nodiscard]] cudaStream_t get() const;

        DWR3_NON_COPYABLE(stream);

        DWR3_DECLARE_NON_MOVEABLE(stream);
    };


    /// Wrapper around a cudaEvent_t instance, allocates and frees a CUDA stream on construction/destruction
    class event {
        bool valid = false;
        cudaEvent_t res{};

    public:
        explicit event();

        ~event();

        [[nodiscard]] cudaEvent_t get() const;

        DWR3_NON_COPYABLE(event);

        DWR3_DECLARE_NON_MOVEABLE(event);
    };

    /// Wrapper around a valid cudaGraphExec_t instance
    DWR3_UNMANAGED_RESOURCE_WRAPPER(graph_exec, cudaGraphExec_t, cudaGraphExecDestroy)
}

#endif //DWR3_COMMON_CUH

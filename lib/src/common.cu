#include "common.cuh"

dwr3::stream_priorities::stream_priorities() {
    CUDA_THROW_ON_ERROR(cudaDeviceGetStreamPriorityRange(&low_, &high_));
    medium_ = (low_ + high_) / 2;
}

int dwr3::stream_priorities::low() const {
    return low_;
}

int dwr3::stream_priorities::medium() const {
    return medium_;
}

int dwr3::stream_priorities::high() const {
    return high_;
}

dwr3::stream::stream() {
    CUDA_THROW_ON_ERROR(cudaStreamCreate(&res));
    valid = true;
}

dwr3::stream::stream(const int priority) {
    CUDA_THROW_ON_ERROR(cudaStreamCreateWithPriority(&res, cudaStreamDefault, priority));
    valid = true;
}

dwr3::stream::~stream() {
    if (valid) cudaStreamDestroy(res);
}

cudaStream_t dwr3::stream::get() const {
    return res;
}

dwr3::event::event() {
    CUDA_THROW_ON_ERROR(cudaEventCreate(&res));
    valid = true;
}

dwr3::event::~event() {
    if (valid) cudaEventDestroy(res);

}

cudaEvent_t dwr3::event::get() const {
    return res;
}

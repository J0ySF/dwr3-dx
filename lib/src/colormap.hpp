#ifndef DWR3_DX_COLORMAP_HPP
#define DWR3_DX_COLORMAP_HPP

#include <cstdint>

namespace dwr3 {
    // Returns a pointer to the R element in an RGB triplet, given a normalized in [-1,1] range value
    const uint8_t *colormap_rgb_from_normalized(float norm);
}

#endif //DWR3_DX_COLORMAP_HPP

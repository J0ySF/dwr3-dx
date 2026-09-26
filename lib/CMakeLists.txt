cmake_minimum_required(VERSION 3.20)

# This function enables the experimentation with different compile-time options
function(dwr3_variant suffix)
    set(DWR3_SOURCE
            ${CMAKE_SOURCE_DIR}/lib/src/dwr3.cu
            ${CMAKE_SOURCE_DIR}/lib/src/gen/boundary_reflectance_filters.cpp
            ${CMAKE_SOURCE_DIR}/lib/src/common.cu
            ${CMAKE_SOURCE_DIR}/lib/src/input_output.cu
            ${CMAKE_SOURCE_DIR}/lib/src/propagation_medium/kv_2009.cu
            ${CMAKE_SOURCE_DIR}/lib/src/colormap.cpp)
    set(DWR3_INCLUDE ${CMAKE_SOURCE_DIR}/lib/include)
    set(DWR3_VARIANT_NAME "dwr3${suffix}")
    add_library(${DWR3_VARIANT_NAME} STATIC ${DWR3_SOURCE})
    target_include_directories(${DWR3_VARIANT_NAME} PUBLIC ${DWR3_INCLUDE})
    set_property(TARGET ${DWR3_VARIANT_NAME} PROPERTY CXX_STANDARD 20)
    set_property(TARGET ${DWR3_VARIANT_NAME} PROPERTY CUDA_STANDARD 20)
    target_compile_options(${DWR3_VARIANT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            $<$<CONFIG:Debug>:-O0 -lineinfo>
            $<$<CONFIG:RelWithDebInfo>:-O3 -lineinfo>
            $<$<CONFIG:Release>:-O3>
            $<$<CONFIG:MinSizeRel>:-O3>>)
endfunction()

# Default build
dwr3_variant("")
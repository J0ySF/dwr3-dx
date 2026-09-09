cmake_minimum_required(VERSION 3.20)

add_library(dwr3-dx STATIC src/dwr3.cu)
target_include_directories(dwr3-dx PUBLIC include)

cmake_minimum_required(VERSION 3.10)
project(LosslessScalingLayer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
find_program(GLSLC_EXECUTABLE glslc REQUIRED)

# Compile the shader
set(SHADER_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/nis_main.comp")
set(SHADER_SPV "${CMAKE_CURRENT_BINARY_DIR}/nis_main.spv")
add_custom_command(
    OUTPUT ${SHADER_SPV}
    COMMAND ${GLSLC_EXECUTABLE} -o ${SHADER_SPV} ${SHADER_SRC}
    DEPENDS ${SHADER_SRC}
)
add_custom_target(Shaders DEPENDS ${SHADER_SPV})

# Embed the SPIR-V into a C++ header
set(SHADER_HEADER "${CMAKE_CURRENT_BINARY_DIR}/nis_shader.h")
add_custom_command(
    OUTPUT ${SHADER_HEADER}
    COMMAND xxd -i ${SHADER_SPV} > ${SHADER_HEADER}
    DEPENDS ${SHADER_SPV}
)
add_custom_target(ShaderHeader DEPENDS ${SHADER_HEADER})

add_library(lossless_scaling SHARED src/lossless_scaling.cpp)
add_dependencies(lossless_scaling ShaderHeader)

target_include_directories(lossless_scaling PRIVATE
    ${Vulkan_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
)
target_link_libraries(lossless_scaling PRIVATE ${Vulkan_LIBRARIES})

set_target_properties(lossless_scaling PROPERTIES PREFIX "lib")

install(TARGETS lossless_scaling DESTINATION lib)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/lossless_scaling.json DESTINATION share/vulkan/implicit_layers.d)

cmake_minimum_required(VERSION 3.10)
project(LosslessScalingLayer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
find_program(GLSLC_EXECUTABLE glslc REQUIRED)

# Compile the shader
set(SHADER_SRC "${CMAKE_CURRENT_SOURCE_DIR}/shaders/nis_main.comp")
set(SHADER_SPV "${CMAKE_CURRENT_BINARY_DIR}/nis_main.spv")
add_custom_command(
    OUTPUT ${SHADER_SPV}
    COMMAND ${GLSLC_EXECUTABLE} -o ${SHADER_SPV} ${SHADER_SRC} -I${CMAKE_CURRENT_SOURCE_DIR}/include
    DEPENDS ${SHADER_SRC}
)
add_custom_target(Shaders DEPENDS ${SHADER_SPV})

# Embed the SPIR-V into a C++ header
set(SHADER_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/include/nis_shader.h")
add_custom_command(
    OUTPUT ${SHADER_HEADER}
    COMMAND xxd -i -n nis_shader_spirv ${SHADER_SPV} > ${SHADER_HEADER}
    DEPENDS ${SHADER_SPV}
)
add_custom_target(ShaderHeader DEPENDS ${SHADER_HEADER})

# Compile the SGSR shaders
set(SGSR_CONVERT_SRC "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sgsr_convert.comp")
set(SGSR_CONVERT_SPV "${CMAKE_CURRENT_BINARY_DIR}/sgsr_convert.spv")
add_custom_command(
    OUTPUT ${SGSR_CONVERT_SPV}
    COMMAND ${GLSLC_EXECUTABLE} -o ${SGSR_CONVERT_SPV} ${SGSR_CONVERT_SRC}
    DEPENDS ${SGSR_CONVERT_SRC}
)

set(SGSR_UPSCALE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sgsr_upscale.comp")
set(SGSR_UPSCALE_SPV "${CMAKE_CURRENT_BINARY_DIR}/sgsr_upscale.spv")
add_custom_command(
    OUTPUT ${SGSR_UPSCALE_SPV}
    COMMAND ${GLSLC_EXECUTABLE} -o ${SGSR_UPSCALE_SPV} ${SGSR_UPSCALE_SRC}
    DEPENDS ${SGSR_UPSCALE_SRC}
)

add_custom_target(SgsrShaders DEPENDS ${SGSR_CONVERT_SPV} ${SGSR_UPSCALE_SPV})

# Embed the SGSR shaders into C++ headers
set(SGSR_CONVERT_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/include/upscalers/sgsr_convert_shader.h")
add_custom_command(
    OUTPUT ${SGSR_CONVERT_HEADER}
    COMMAND xxd -i -n sgsr_convert_shader_spirv ${SGSR_CONVERT_SPV} > ${SGSR_CONVERT_HEADER}
    DEPENDS ${SGSR_CONVERT_SPV}
)

set(SGSR_UPSCALE_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/include/upscalers/sgsr_upscale_shader.h")
add_custom_command(
    OUTPUT ${SGSR_UPSCALE_HEADER}
    COMMAND xxd -i -n sgsr_upscale_shader_spirv ${SGSR_UPSCALE_SPV} > ${SGSR_UPSCALE_HEADER}
    DEPENDS ${SGSR_UPSCALE_SPV}
)

add_custom_target(SgsrShaderHeaders DEPENDS ${SGSR_CONVERT_HEADER} ${SGSR_UPSCALE_HEADER})

add_library(lossless_scaling SHARED src/lossless_scaling.cpp src/upscalers/nis.cpp src/upscalers/sgsr.cpp)
add_dependencies(lossless_scaling ShaderHeader SgsrShaders SgsrShaderHeaders)

target_include_directories(lossless_scaling PRIVATE
    ${Vulkan_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(lossless_scaling PRIVATE Vulkan::Vulkan)

set_target_properties(lossless_scaling PROPERTIES PREFIX "lib")

install(TARGETS lossless_scaling DESTINATION lib)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/VkLayer_LS_lossless_scaling.json DESTINATION share/vulkan/implicit_layers.d)

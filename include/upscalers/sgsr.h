#pragma once

#include <vulkan/vulkan.h>
#include <memory>

struct SwapchainData;

namespace upscalers {
namespace sgsr {

struct alignas(16) Params {
    uint32_t renderSize[2];
    uint32_t displaySize[2];
    float renderSizeRcp[2];
    float displaySizeRcp[2];
    float jitterOffset[2];
    float padding1[2];
    float clipToPrevClip[16];
    float preExposure;
    float cameraFovAngleHor;
    float cameraNear;
    float MinLerpContribution;
    uint32_t bSameCamera;
    uint32_t reset;
};

struct SgsrData {
    VkPipeline upscale_pipeline;
    VkPipelineLayout upscale_pipeline_layout;

    VkDescriptorSetLayout upscale_descriptor_set_layout;

    VkImage history_image;
    VkImageView history_image_view;
    VkDeviceMemory history_image_memory;

    VkBuffer params_buffer;
    VkDeviceMemory params_buffer_memory;
};

bool init(SwapchainData* swapchain_data);
void cleanup(SwapchainData* swapchain_data);
void upscale(SwapchainData* swapchain_data, uint32_t image_index, VkQueue queue);

}
}

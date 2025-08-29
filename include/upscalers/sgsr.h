#pragma once

#include <vulkan/vulkan.h>
#include <memory>

struct SwapchainData;

namespace upscalers {
namespace sgsr {

bool init(SwapchainData* swapchain_data);
void cleanup(SwapchainData* swapchain_data);
void upscale(SwapchainData* swapchain_data, uint32_t image_index, VkQueue queue);

}
}

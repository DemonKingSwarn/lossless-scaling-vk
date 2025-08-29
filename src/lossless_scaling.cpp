#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>
#include <string.h>
#include <unordered_map>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cstdlib>
#include <cstdio>

#define VK_LAYER_EXPORT
#include <vulkan/vk_layer.h>

#include "lossless_scaling.h"
#include "upscalers/nis.h"
#include "upscalers/sgsr.h"

std::unordered_map<void*, VkLayerInstanceDispatchTable> instance_dispatch_tables;
std::unordered_map<VkPhysicalDevice, VkInstance> physical_device_to_instance_map;
std::unordered_map<void*, VkLayerDeviceDispatchTable> device_dispatch_tables;
std::unordered_map<void*, std::pair<VkInstance, VkPhysicalDevice>> device_data_map;
std::unordered_map<void*, VkDevice> queue_to_device_map;
std::unordered_map<VkSwapchainKHR, std::unique_ptr<SwapchainData>> swapchain_data_map;

uint32_t find_memory_type(VkInstance instance, VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    instance_dispatch_tables.at((void*)instance).GetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("failed to find suitable memory type!");
}

VkShaderModule create_shader_module(VkDevice device, const unsigned char* code, unsigned int code_size) {
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code_size;
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code);

    VkShaderModule shaderModule;
    if (device_dispatch_tables.at((void*)device).CreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shader module!");
    }
    return shaderModule;
}

VKAPI_ATTR VkResult VKAPI_CALL LosslessScaling_CreateInstance(const VkInstanceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkInstance* pInstance) {
    

    VkLayerInstanceCreateInfo* layer_create_info = (VkLayerInstanceCreateInfo*)pCreateInfo->pNext;
    while (layer_create_info && (layer_create_info->sType != VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO || layer_create_info->function != VK_LAYER_LINK_INFO)) {
        layer_create_info = (VkLayerInstanceCreateInfo*)layer_create_info->pNext;
    }
    if (!layer_create_info) return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr next_get_instance_proc_addr = layer_create_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkCreateInstance next_create_instance = (PFN_vkCreateInstance)next_get_instance_proc_addr(VK_NULL_HANDLE, "vkCreateInstance");
    VkResult result = next_create_instance(pCreateInfo, pAllocator, pInstance);

    if (result == VK_SUCCESS) {
        VkLayerInstanceDispatchTable dispatch_table;
        dispatch_table.GetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)next_get_instance_proc_addr(*pInstance, "vkGetInstanceProcAddr");
        dispatch_table.DestroyInstance = (PFN_vkDestroyInstance)next_get_instance_proc_addr(*pInstance, "vkDestroyInstance");
        dispatch_table.CreateDevice = (PFN_vkCreateDevice)next_get_instance_proc_addr(*pInstance, "vkCreateDevice");
        dispatch_table.GetPhysicalDeviceMemoryProperties = (PFN_vkGetPhysicalDeviceMemoryProperties)next_get_instance_proc_addr(*pInstance, "vkGetPhysicalDeviceMemoryProperties");
        dispatch_table.GetPhysicalDeviceDisplayPropertiesKHR = (PFN_vkGetPhysicalDeviceDisplayPropertiesKHR)next_get_instance_proc_addr(*pInstance, "vkGetPhysicalDeviceDisplayPropertiesKHR");
        dispatch_table.EnumeratePhysicalDevices = (PFN_vkEnumeratePhysicalDevices)next_get_instance_proc_addr(*pInstance, "vkEnumeratePhysicalDevices");
        instance_dispatch_tables[(void*)*pInstance] = dispatch_table;
    }
    
    return result;
}

VKAPI_ATTR void VKAPI_CALL LosslessScaling_DestroyInstance(VkInstance instance, const VkAllocationCallbacks* pAllocator) {
    auto* dt = &instance_dispatch_tables.at((void*)instance);
    dt->DestroyInstance(instance, pAllocator);
    instance_dispatch_tables.erase((void*)instance);
}

VKAPI_ATTR VkResult VKAPI_CALL LosslessScaling_CreateDevice(VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDevice* pDevice) {
    

    VkLayerDeviceCreateInfo *layer_create_info = (VkLayerDeviceCreateInfo *)pCreateInfo->pNext;
    while (layer_create_info && (layer_create_info->sType != VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO || layer_create_info->function != VK_LAYER_LINK_INFO)) {
        layer_create_info = (VkLayerDeviceCreateInfo *)layer_create_info->pNext;
    }
    if (!layer_create_info) return VK_ERROR_INITIALIZATION_FAILED;

    auto it = physical_device_to_instance_map.find(physicalDevice);
    if (it == physical_device_to_instance_map.end()) {
        
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    VkInstance instance = it->second;
    PFN_vkGetInstanceProcAddr gipa = layer_create_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkCreateDevice next_create_device = (PFN_vkCreateDevice)gipa(instance, "vkCreateDevice");
    if (!next_create_device) return VK_ERROR_INITIALIZATION_FAILED;

    VkResult result = next_create_device(physicalDevice, pCreateInfo, pAllocator, pDevice);

    if (result == VK_SUCCESS) {
        device_data_map[(void*)*pDevice] = std::make_pair(instance, physicalDevice);
        VkLayerDeviceDispatchTable dispatch_table;
        PFN_vkGetDeviceProcAddr next_get_device_proc_addr = layer_create_info->u.pLayerInfo->pfnNextGetDeviceProcAddr;
        if (next_get_device_proc_addr == NULL) {
            
            return VK_ERROR_INITIALIZATION_FAILED;
        }

#undef GET_DEV_PROC
#define GET_DEV_PROC(func) \
        dispatch_table.func = (PFN_vk##func)next_get_device_proc_addr(*pDevice, "vk" #func); \
        if (dispatch_table.func == NULL) {}


        GET_DEV_PROC(GetDeviceProcAddr);
        GET_DEV_PROC(DestroyDevice);
        GET_DEV_PROC(GetDeviceQueue);
        GET_DEV_PROC(CreateSwapchainKHR);
        GET_DEV_PROC(DestroySwapchainKHR);
        GET_DEV_PROC(GetSwapchainImagesKHR);
        GET_DEV_PROC(QueuePresentKHR);
        GET_DEV_PROC(CreateImage);
        GET_DEV_PROC(DestroyImage);
        GET_DEV_PROC(CreateImageView);
        GET_DEV_PROC(DestroyImageView);
        GET_DEV_PROC(AllocateMemory);
        GET_DEV_PROC(FreeMemory);
        GET_DEV_PROC(BindImageMemory);
        GET_DEV_PROC(CreateSampler);
        GET_DEV_PROC(DestroySampler);
        GET_DEV_PROC(CreateDescriptorSetLayout);
        GET_DEV_PROC(DestroyDescriptorSetLayout);
        GET_DEV_PROC(CreatePipelineLayout);
        GET_DEV_PROC(DestroyPipelineLayout);
        GET_DEV_PROC(CreateComputePipelines);
        GET_DEV_PROC(DestroyPipeline);
        GET_DEV_PROC(CreateDescriptorPool);
        GET_DEV_PROC(DestroyDescriptorPool);
        GET_DEV_PROC(AllocateDescriptorSets);
        GET_DEV_PROC(UpdateDescriptorSets);
        GET_DEV_PROC(CreateCommandPool);
        GET_DEV_PROC(DestroyCommandPool);
        GET_DEV_PROC(AllocateCommandBuffers);
        GET_DEV_PROC(FreeCommandBuffers);
        GET_DEV_PROC(BeginCommandBuffer);
        GET_DEV_PROC(EndCommandBuffer);
        GET_DEV_PROC(CmdPipelineBarrier);
        GET_DEV_PROC(CmdBindPipeline);
        GET_DEV_PROC(CmdBindDescriptorSets);
        GET_DEV_PROC(CmdDispatch);
        GET_DEV_PROC(QueueSubmit);
        GET_DEV_PROC(CreateFence);
        GET_DEV_PROC(DestroyFence);
        GET_DEV_PROC(WaitForFences);
        GET_DEV_PROC(ResetFences);
        GET_DEV_PROC(DeviceWaitIdle);
        GET_DEV_PROC(CreateShaderModule);
        GET_DEV_PROC(DestroyShaderModule);
        GET_DEV_PROC(GetImageMemoryRequirements);
        GET_DEV_PROC(CreateBuffer);
        GET_DEV_PROC(DestroyBuffer);
        GET_DEV_PROC(GetBufferMemoryRequirements);
        GET_DEV_PROC(BindBufferMemory);
        GET_DEV_PROC(MapMemory);
        GET_DEV_PROC(UnmapMemory);
        GET_DEV_PROC(CmdCopyBufferToImage);
        GET_DEV_PROC(QueueWaitIdle);
        device_dispatch_tables[(void*)*pDevice] = dispatch_table;
    }
    
    return result;
}



VKAPI_ATTR void VKAPI_CALL LosslessScaling_DestroyDevice(VkDevice device, const VkAllocationCallbacks* pAllocator) {
    auto* dt = &device_dispatch_tables.at((void*)device);
    dt->DeviceWaitIdle(device);
    dt->DestroyDevice(device, pAllocator);
    device_dispatch_tables.erase((void*)device);
    device_data_map.erase((void*)device);
}

VKAPI_ATTR VkResult VKAPI_CALL LosslessScaling_EnumeratePhysicalDevices(VkInstance instance, uint32_t* pPhysicalDeviceCount, VkPhysicalDevice* pPhysicalDevices) {
    

    auto* dt = &instance_dispatch_tables.at((void*)instance);
    VkResult result = dt->EnumeratePhysicalDevices(instance, pPhysicalDeviceCount, pPhysicalDevices);

    if (result == VK_SUCCESS && pPhysicalDevices) {
        for (uint32_t i = 0; i < *pPhysicalDeviceCount; i++) {
            physical_device_to_instance_map[pPhysicalDevices[i]] = instance;
        }
    }

    
    return result;
}

VKAPI_ATTR void VKAPI_CALL LosslessScaling_GetDeviceQueue(VkDevice device, uint32_t queueFamilyIndex, uint32_t queueIndex, VkQueue* pQueue) {
    

    auto* dt = &device_dispatch_tables.at((void*)device);
    dt->GetDeviceQueue(device, queueFamilyIndex, queueIndex, pQueue);
    if (*pQueue != VK_NULL_HANDLE) {
        queue_to_device_map[(void*)*pQueue] = device;
    }
    
}

VKAPI_ATTR VkResult VKAPI_CALL LosslessScaling_CreateSwapchainKHR(VkDevice device, const VkSwapchainCreateInfoKHR* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSwapchainKHR* pSwapchain) {
    

    auto* dt = &device_dispatch_tables.at((void*)device);

    if (pCreateInfo->imageExtent.width == 0 || pCreateInfo->imageExtent.height == 0) {
        return dt->CreateSwapchainKHR(device, pCreateInfo, pAllocator, pSwapchain);
    }

    VkResult result = dt->CreateSwapchainKHR(device, pCreateInfo, pAllocator, pSwapchain);
    if (result == VK_SUCCESS) {
        auto data = std::make_unique<SwapchainData>();
        data->device = device;
        data->instance = device_data_map.at((void*)device).first;
        data->physical_device = device_data_map.at((void*)device).second;
        data->swapchain = *pSwapchain;
        data->low_res_extent = pCreateInfo->imageExtent;
        uint32_t display_count = 0;
        instance_dispatch_tables.at((void*)data->instance).GetPhysicalDeviceDisplayPropertiesKHR(data->physical_device, &display_count, nullptr);
        std::vector<VkDisplayPropertiesKHR> display_properties(display_count);
        instance_dispatch_tables.at((void*)data->instance).GetPhysicalDeviceDisplayPropertiesKHR(data->physical_device, &display_count, display_properties.data());

        if (display_count > 0) {
            data->high_res_extent = display_properties[0].physicalResolution;
        } else {
            data->high_res_extent = {1920, 1080};
        }
        data->format = pCreateInfo->imageFormat;

        uint32_t image_count;
        dt->GetSwapchainImagesKHR(device, *pSwapchain, &image_count, nullptr);
        data->swapchain_images.resize(image_count);
        dt->GetSwapchainImagesKHR(device, *pSwapchain, &image_count, data->swapchain_images.data());

        data->upscaled_images.resize(image_count);
        data->upscaled_image_memories.resize(image_count);
        data->upscaled_image_views.resize(image_count);
        data->swapchain_image_views.resize(image_count);

        for (uint32_t i = 0; i < image_count; i++) {
            VkImageViewCreateInfo view_info = {};
            view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            view_info.image = data->swapchain_images[i];
            view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view_info.format = data->format;
            view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            view_info.subresourceRange.baseMipLevel = 0;
            view_info.subresourceRange.levelCount = 1;
            view_info.subresourceRange.baseArrayLayer = 0;
            view_info.subresourceRange.layerCount = 1;
            dt->CreateImageView(device, &view_info, nullptr, &data->swapchain_image_views[i]);

            VkImageCreateInfo image_info = {};
            image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            image_info.imageType = VK_IMAGE_TYPE_2D;
            image_info.extent.width = data->high_res_extent.width;
            image_info.extent.height = data->high_res_extent.height;
            image_info.extent.depth = 1;
            image_info.mipLevels = 1;
            image_info.arrayLayers = 1;
            image_info.format = data->format;
            image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
            image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            image_info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            image_info.samples = VK_SAMPLE_COUNT_1_BIT;
            dt->CreateImage(device, &image_info, nullptr, &data->upscaled_images[i]);

            VkMemoryRequirements mem_reqs;
            dt->GetImageMemoryRequirements(device, data->upscaled_images[i], &mem_reqs);
            VkMemoryAllocateInfo alloc_info = {};
            alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            alloc_info.allocationSize = mem_reqs.size;
            alloc_info.memoryTypeIndex = find_memory_type(data->instance, data->physical_device, mem_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            dt->AllocateMemory(device, &alloc_info, nullptr, &data->upscaled_image_memories[i]);
            dt->BindImageMemory(device, data->upscaled_images[i], data->upscaled_image_memories[i], 0);

            view_info.image = data->upscaled_images[i];
            dt->CreateImageView(device, &view_info, nullptr, &data->upscaled_image_views[i]);
        }

        VkSamplerCreateInfo sampler_info = {};
        sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampler_info.magFilter = VK_FILTER_LINEAR;
        sampler_info.minFilter = VK_FILTER_LINEAR;
        sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler_info.anisotropyEnable = VK_FALSE;
        sampler_info.maxAnisotropy = 1.0f;
        sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        sampler_info.unnormalizedCoordinates = VK_FALSE;
        sampler_info.compareEnable = VK_FALSE;
        sampler_info.compareOp = VK_COMPARE_OP_ALWAYS;
        sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sampler_info.mipLodBias = 0.0f;
        sampler_info.minLod = 0.0f;
        sampler_info.maxLod = 0.0f;
        dt->CreateSampler(device, &sampler_info, nullptr, &data->sampler);

        const char* upscaler = std::getenv("UPSCALER");
        if (upscaler && (strcmp(upscaler, "nis") == 0 || strcmp(upscaler, "sgsr") == 0)) {
            printf("[lossless-scaling-vk] Upscaling using %s (%ux%u) -> (%ux%u)\n", upscaler, data->low_res_extent.width, data->low_res_extent.height, data->high_res_extent.width, data->high_res_extent.height);
        }
        if (upscaler && strcmp(upscaler, "nis") == 0) {
            upscalers::nis::init(data.get());
        } else if (upscaler && strcmp(upscaler, "sgsr") == 0) {
            upscalers::sgsr::init(data.get());
        }

        

        VkCommandPoolCreateInfo cmd_pool_info = {};
        cmd_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cmd_pool_info.queueFamilyIndex = 0;
        dt->CreateCommandPool(device, &cmd_pool_info, nullptr, &data->command_pool);

        VkCommandBufferAllocateInfo cmd_buf_alloc_info = {};
        cmd_buf_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmd_buf_alloc_info.commandPool = data->command_pool;
        cmd_buf_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmd_buf_alloc_info.commandBufferCount = image_count;
        data->command_buffers.resize(image_count);
        dt->AllocateCommandBuffers(device, &cmd_buf_alloc_info, data->command_buffers.data());

        data->fences.resize(image_count);
        VkFenceCreateInfo fence_info = {};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        for (uint32_t i = 0; i < image_count; i++) {
            dt->CreateFence(device, &fence_info, nullptr, &data->fences[i]);
        }

        swapchain_data_map[*pSwapchain] = std::move(data);
    }
    
    return result;
}

VKAPI_ATTR void VKAPI_CALL LosslessScaling_DestroySwapchainKHR(VkDevice device, VkSwapchainKHR swapchain, const VkAllocationCallbacks* pAllocator) {
    

    auto it = swapchain_data_map.find(swapchain);
    if (it == swapchain_data_map.end()) {
        device_dispatch_tables.at((void*)device).DestroySwapchainKHR(device, swapchain, pAllocator);
        return;
    }

    std::unique_ptr<SwapchainData> data = std::move(it->second);
    swapchain_data_map.erase(it);

    auto* dt = &device_dispatch_tables.at((void*)device);
    dt->DeviceWaitIdle(device);

    const char* upscaler = std::getenv("UPSCALER");
    if (upscaler && strcmp(upscaler, "nis") == 0) {
        upscalers::nis::cleanup(data.get());
    }

    for (auto& fence : data->fences) dt->DestroyFence(device, fence, nullptr);
    dt->FreeCommandBuffers(device, data->command_pool, static_cast<uint32_t>(data->command_buffers.size()), data->command_buffers.data());
    dt->DestroyCommandPool(device, data->command_pool, nullptr);
    dt->DestroyDescriptorPool(device, data->descriptor_pool, nullptr);
    dt->DestroyPipeline(device, data->compute_pipeline, nullptr);
    dt->DestroyPipelineLayout(device, data->pipeline_layout, nullptr);
    dt->DestroyDescriptorSetLayout(device, data->descriptor_set_layout, nullptr);
    dt->DestroySampler(device, data->sampler, nullptr);

    for (auto& view : data->swapchain_image_views) dt->DestroyImageView(device, view, nullptr);
    for (auto& view : data->upscaled_image_views) dt->DestroyImageView(device, view, nullptr);
    for (auto& image : data->upscaled_images) dt->DestroyImage(device, image, nullptr);
    for (auto& mem : data->upscaled_image_memories) dt->FreeMemory(device, mem, nullptr);

    dt->DestroySwapchainKHR(device, swapchain, pAllocator);
    
}

VKAPI_ATTR VkResult VKAPI_CALL LosslessScaling_QueuePresentKHR(VkQueue queue, const VkPresentInfoKHR* pPresentInfo) {
    

    auto device = queue_to_device_map.at((void*)queue);
    auto* dt = &device_dispatch_tables.at((void*)device);

    const char* upscaler = std::getenv("UPSCALER");
    bool enabled = (upscaler != nullptr);

    if (!enabled) {
        return dt->QueuePresentKHR(queue, pPresentInfo);
    }

    VkSwapchainKHR current_swapchain = pPresentInfo->pSwapchains[0];
    auto it = swapchain_data_map.find(current_swapchain);
    if (it == swapchain_data_map.end()) {
        return dt->QueuePresentKHR(queue, pPresentInfo);
    }
    SwapchainData* data = it->second.get();

    uint32_t image_index = pPresentInfo->pImageIndices[0];

    if (strcmp(upscaler, "nis") == 0) {
        upscalers::nis::upscale(data, image_index, queue);
    } else if (strcmp(upscaler, "sgsr") == 0) {
        upscalers::sgsr::upscale(data, image_index, queue);
    }

    VkPresentInfoKHR modified_present_info = *pPresentInfo;
    uint32_t local_image_index = image_index;
    modified_present_info.pImageIndices = &local_image_index;
    modified_present_info.swapchainCount = 1;
    modified_present_info.pSwapchains = &data->swapchain;
    modified_present_info.waitSemaphoreCount = 0;
    modified_present_info.pWaitSemaphores = nullptr;

    VkResult present_result = dt->QueuePresentKHR(queue, &modified_present_info);
    
    return present_result;
}

extern "C" {

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL LosslessScaling_GetInstanceProcAddr(VkInstance instance, const char* pName) {
    if (strcmp(pName, "vkCreateInstance") == 0) return (PFN_vkVoidFunction)LosslessScaling_CreateInstance;
    if (strcmp(pName, "vkDestroyInstance") == 0) return (PFN_vkVoidFunction)LosslessScaling_DestroyInstance;
    if (strcmp(pName, "vkCreateDevice") == 0) return (PFN_vkVoidFunction)LosslessScaling_CreateDevice;
    if (strcmp(pName, "vkEnumeratePhysicalDevices") == 0) return (PFN_vkVoidFunction)LosslessScaling_EnumeratePhysicalDevices;
    if (instance == VK_NULL_HANDLE) return nullptr;
    return instance_dispatch_tables.at((void*)instance).GetInstanceProcAddr(instance, pName);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL LosslessScaling_GetDeviceProcAddr(VkDevice device, const char* pName) {
    if (strcmp(pName, "vkDestroyDevice") == 0) return (PFN_vkVoidFunction)LosslessScaling_DestroyDevice;
    if (strcmp(pName, "vkGetDeviceQueue") == 0) return (PFN_vkVoidFunction)LosslessScaling_GetDeviceQueue;
    if (strcmp(pName, "vkCreateSwapchainKHR") == 0) return (PFN_vkVoidFunction)LosslessScaling_CreateSwapchainKHR;
    if (strcmp(pName, "vkDestroySwapchainKHR") == 0) return (PFN_vkVoidFunction)LosslessScaling_DestroySwapchainKHR;
    if (strcmp(pName, "vkQueuePresentKHR") == 0) return (PFN_vkVoidFunction)LosslessScaling_QueuePresentKHR;
    if (device == VK_NULL_HANDLE) return nullptr;
    return device_dispatch_tables.at((void*)device).GetDeviceProcAddr(device, pName);
}

VKAPI_ATTR VkResult VKAPI_CALL vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface *pVersionStruct) {
    
    if (pVersionStruct == NULL || pVersionStruct->loaderLayerInterfaceVersion < 2) {

        return VK_ERROR_INITIALIZATION_FAILED;
    }
    pVersionStruct->pfnGetInstanceProcAddr = LosslessScaling_GetInstanceProcAddr;
    pVersionStruct->pfnGetDeviceProcAddr = LosslessScaling_GetDeviceProcAddr;
    pVersionStruct->pfnGetPhysicalDeviceProcAddr = NULL;
    return VK_SUCCESS;
}

}
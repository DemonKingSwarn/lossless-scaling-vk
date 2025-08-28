#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>
#include <string.h>
#include <unordered_map>
#include <vector>
#include <memory>
#include <stdexcept>

#define VK_LAYER_EXPORT
#include <vulkan/vk_layer.h>

#include "nis_shader.h"

struct VkLayerInstanceDispatchTable {
    PFN_vkGetInstanceProcAddr GetInstanceProcAddr;
    PFN_vkDestroyInstance DestroyInstance;
    PFN_vkCreateDevice CreateDevice;
    PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties;
    PFN_vkGetPhysicalDeviceDisplayPropertiesKHR GetPhysicalDeviceDisplayPropertiesKHR;
};

struct VkLayerDeviceDispatchTable {
    PFN_vkGetDeviceProcAddr GetDeviceProcAddr;
    PFN_vkDestroyDevice DestroyDevice;
    PFN_vkGetDeviceQueue GetDeviceQueue;
    PFN_vkCreateSwapchainKHR CreateSwapchainKHR;
    PFN_vkDestroySwapchainKHR DestroySwapchainKHR;
    PFN_vkGetSwapchainImagesKHR GetSwapchainImagesKHR;
    PFN_vkQueuePresentKHR QueuePresentKHR;
    PFN_vkCreateImage CreateImage;
    PFN_vkDestroyImage DestroyImage;
    PFN_vkCreateImageView CreateImageView;
    PFN_vkDestroyImageView DestroyImageView;
    PFN_vkAllocateMemory AllocateMemory;
    PFN_vkFreeMemory FreeMemory;
    PFN_vkBindImageMemory BindImageMemory;
    PFN_vkCreateSampler CreateSampler;
    PFN_vkDestroySampler DestroySampler;
    PFN_vkCreateDescriptorSetLayout CreateDescriptorSetLayout;
    PFN_vkDestroyDescriptorSetLayout DestroyDescriptorSetLayout;
    PFN_vkCreatePipelineLayout CreatePipelineLayout;
    PFN_vkDestroyPipelineLayout DestroyPipelineLayout;
    PFN_vkCreateComputePipelines CreateComputePipelines;
    PFN_vkDestroyPipeline DestroyPipeline;
    PFN_vkCreateDescriptorPool CreateDescriptorPool;
    PFN_vkDestroyDescriptorPool DestroyDescriptorPool;
    PFN_vkAllocateDescriptorSets AllocateDescriptorSets;
    PFN_vkUpdateDescriptorSets UpdateDescriptorSets;
    PFN_vkCreateCommandPool CreateCommandPool;
    PFN_vkDestroyCommandPool DestroyCommandPool;
    PFN_vkAllocateCommandBuffers AllocateCommandBuffers;
    PFN_vkFreeCommandBuffers FreeCommandBuffers;
    PFN_vkBeginCommandBuffer BeginCommandBuffer;
    PFN_vkEndCommandBuffer EndCommandBuffer;
    PFN_vkCmdPipelineBarrier CmdPipelineBarrier;
    PFN_vkCmdBindPipeline CmdBindPipeline;
    PFN_vkCmdBindDescriptorSets CmdBindDescriptorSets;
    PFN_vkCmdDispatch CmdDispatch;
    PFN_vkQueueSubmit QueueSubmit;
    PFN_vkCreateFence CreateFence;
    PFN_vkDestroyFence DestroyFence;
    PFN_vkWaitForFences WaitForFences;
    PFN_vkResetFences ResetFences;
    PFN_vkDeviceWaitIdle DeviceWaitIdle;
    PFN_vkCreateShaderModule CreateShaderModule;
    PFN_vkDestroyShaderModule DestroyShaderModule;
    PFN_vkGetImageMemoryRequirements GetImageMemoryRequirements;
    PFN_vkCreateBuffer CreateBuffer;
    PFN_vkDestroyBuffer DestroyBuffer;
    PFN_vkGetBufferMemoryRequirements GetBufferMemoryRequirements;
    PFN_vkBindBufferMemory BindBufferMemory;
    PFN_vkMapMemory MapMemory;
    PFN_vkUnmapMemory UnmapMemory;
    PFN_vkCmdCopyBufferToImage CmdCopyBufferToImage;
    PFN_vkQueueWaitIdle QueueWaitIdle;
};

#include "NIS_Config.h"

struct SwapchainData {
    VkDevice device;
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkSwapchainKHR swapchain;
    VkExtent2D low_res_extent;
    VkExtent2D high_res_extent;
    VkFormat format;

    std::vector<VkImage> swapchain_images;
    std::vector<VkImageView> swapchain_image_views;
    std::vector<VkImage> upscaled_images;
    std::vector<VkDeviceMemory> upscaled_image_memories;
    std::vector<VkImageView> upscaled_image_views;

    VkDescriptorSetLayout descriptor_set_layout;
    VkPipelineLayout pipeline_layout;
    VkPipeline compute_pipeline;
    VkDescriptorPool descriptor_pool;
    std::vector<VkDescriptorSet> descriptor_sets;

    VkSampler sampler;

    // NIS specific resources
    VkBuffer uniform_buffer;
    VkDeviceMemory uniform_buffer_memory;
    VkImage coef_scaler_image;
    VkDeviceMemory coef_scaler_image_memory;
    VkImageView coef_scaler_image_view;
    VkImage coef_usm_image;
    VkDeviceMemory coef_usm_image_memory;
    VkImageView coef_usm_image_view;

    VkCommandPool command_pool;
    std::vector<VkCommandBuffer> command_buffers;
    std::vector<VkFence> fences;
};

namespace {
    std::unordered_map<void*, VkLayerInstanceDispatchTable> instance_dispatch_tables;
    std::unordered_map<void*, VkLayerDeviceDispatchTable> device_dispatch_tables;
    std::unordered_map<void*, std::pair<VkInstance, VkPhysicalDevice>> device_data_map;
    std::unordered_map<void*, VkDevice> queue_to_device_map;
    std::unordered_map<VkSwapchainKHR, std::unique_ptr<SwapchainData>> swapchain_data_map;
}

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
    fprintf(stderr, "[LosslessScaling Layer] LosslessScaling_CreateInstance called.\n");

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
    VkInstance instance = VK_NULL_HANDLE;
    for(auto const& [inst_handle, table] : instance_dispatch_tables) {
        instance = (VkInstance)inst_handle;
        break;
    }

    PFN_vkCreateDevice next_create_device = instance_dispatch_tables.at((void*)instance).CreateDevice;
    VkResult result = next_create_device(physicalDevice, pCreateInfo, pAllocator, pDevice);

    if (result == VK_SUCCESS) {
        device_data_map[(void*)*pDevice] = {instance, physicalDevice}; // Store instance and physical device
        VkLayerDeviceDispatchTable dispatch_table;
        PFN_vkGetInstanceProcAddr gipa = instance_dispatch_tables.at((void*)instance).GetInstanceProcAddr;
        PFN_vkGetDeviceProcAddr next_get_device_proc_addr = (PFN_vkGetDeviceProcAddr)gipa(instance, "vkGetDeviceProcAddr");
#define GET_DEV_PROC(func) dispatch_table.func = (PFN_vk##func)next_get_device_proc_addr(*pDevice, "vk" #func)
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
#undef GET_DEV_PROC
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

VKAPI_ATTR void VKAPI_CALL LosslessScaling_GetDeviceQueue(VkDevice device, uint32_t queueFamilyIndex, uint32_t queueIndex, VkQueue* pQueue) {
    auto* dt = &device_dispatch_tables.at((void*)device);
    dt->GetDeviceQueue(device, queueFamilyIndex, queueIndex, pQueue);
    if (*pQueue != VK_NULL_HANDLE) {
        queue_to_device_map[(void*)*pQueue] = device;
    }
}

VKAPI_ATTR VkResult VKAPI_CALL LosslessScaling_CreateSwapchainKHR(VkDevice device, const VkSwapchainCreateInfoKHR* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSwapchainKHR* pSwapchain) {
    fprintf(stderr, "[LosslessScaling Layer] LosslessScaling_CreateSwapchainKHR called.\n");

    auto* dt = &device_dispatch_tables.at((void*)device);

    if (pCreateInfo->imageExtent.width == 0 || pCreateInfo->imageExtent.height == 0) {
        fprintf(stderr, "[LosslessScaling Layer] ERROR: imageExtent is zero.\n");
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
            // Fallback to hardcoded value if no displays are found
            data->high_res_extent = {1920, 1080};
        }
        data->format = pCreateInfo->imageFormat;

        uint32_t image_count;
        dt->GetSwapchainImagesKHR(device, *pSwapchain, &image_count, nullptr);
        data->swapchain_images.resize(image_count);
        dt->GetSwapchainImagesKHR(device, *pSwapchain, &image_count, data->swapchain_images.data());

        // Create resources for each swapchain image
        data->upscaled_images.resize(image_count);
        data->upscaled_image_memories.resize(image_count);
        data->upscaled_image_views.resize(image_count);
        data->swapchain_image_views.resize(image_count);

        for (uint32_t i = 0; i < image_count; i++) {
            // Create image view for the source swapchain image
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

            // Create the destination upscaled image
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

            // Create image view for the destination upscaled image
            view_info.image = data->upscaled_images[i];
            dt->CreateImageView(device, &view_info, nullptr, &data->upscaled_image_views[i]);
        }

        // Create Uniform Buffer
        VkDeviceSize buffer_size = 256; // Placeholder size, needs to match const_buffer in shader
        VkBufferCreateInfo buffer_info = {};
        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.size = buffer_size;
        buffer_info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        dt->CreateBuffer(device, &buffer_info, nullptr, &data->uniform_buffer);

        VkMemoryRequirements ub_mem_reqs;
        dt->GetBufferMemoryRequirements(device, data->uniform_buffer, &ub_mem_reqs);
        VkMemoryAllocateInfo ub_alloc_info = {};
        ub_alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ub_alloc_info.allocationSize = ub_mem_reqs.size;
        ub_alloc_info.memoryTypeIndex = find_memory_type(data->instance, data->physical_device, ub_mem_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        dt->AllocateMemory(device, &ub_alloc_info, nullptr, &data->uniform_buffer_memory);
        dt->BindBufferMemory(device, data->uniform_buffer, data->uniform_buffer_memory, 0);

        // Populate Uniform Buffer with actual NIS SDK data
        NISConfig nis_config = {};
        NVScalerUpdateConfig(nis_config, 0.5f, // sharpness (0.0 to 1.0)
                             0, 0, data->low_res_extent.width, data->low_res_extent.height, // input viewport
                             data->low_res_extent.width, data->low_res_extent.height, // input texture
                             0, 0, data->high_res_extent.width, data->high_res_extent.height, // output viewport
                             data->high_res_extent.width, data->high_res_extent.height, // output texture
                             NISHDRMode::None);

        void* mapped_data;
        dt->MapMemory(device, data->uniform_buffer_memory, 0, sizeof(NISConfig), 0, &mapped_data);
        memcpy(mapped_data, &nis_config, sizeof(NISConfig));
        dt->UnmapMemory(device, data->uniform_buffer_memory);

        // Create Coefficient Textures (coef_scaler and coef_usm)
        // Dimensions are 2x64 based on NIS_Scaler.h analysis
        VkImageCreateInfo coef_image_info = {};
        coef_image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        coef_image_info.imageType = VK_IMAGE_TYPE_2D;
        coef_image_info.extent.width = 2;
        coef_image_info.extent.height = 64;
        coef_image_info.extent.depth = 1;
        coef_image_info.mipLevels = 1;
        coef_image_info.arrayLayers = 1;
        coef_image_info.format = VK_FORMAT_R32G32B32A32_SFLOAT; // Assuming float4 for coefficients
        coef_image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        coef_image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        coef_image_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        coef_image_info.samples = VK_SAMPLE_COUNT_1_BIT;

        // coef_scaler
        dt->CreateImage(device, &coef_image_info, nullptr, &data->coef_scaler_image);
        VkMemoryRequirements cs_mem_reqs;
        dt->GetImageMemoryRequirements(device, data->coef_scaler_image, &cs_mem_reqs);
        VkMemoryAllocateInfo cs_alloc_info = {};
        cs_alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        cs_alloc_info.allocationSize = cs_mem_reqs.size;
        cs_alloc_info.memoryTypeIndex = find_memory_type(data->instance, data->physical_device, cs_mem_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        dt->AllocateMemory(device, &cs_alloc_info, nullptr, &data->coef_scaler_image_memory);
        dt->BindImageMemory(device, data->coef_scaler_image, data->coef_scaler_image_memory, 0);

        VkImageViewCreateInfo coef_view_info = {};
        coef_view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        coef_view_info.image = data->coef_scaler_image;
        coef_view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        coef_view_info.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        coef_view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        coef_view_info.subresourceRange.baseMipLevel = 0;
        coef_view_info.subresourceRange.levelCount = 1;
        coef_view_info.subresourceRange.baseArrayLayer = 0;
        coef_view_info.subresourceRange.layerCount = 1;
        dt->CreateImageView(device, &coef_view_info, nullptr, &data->coef_scaler_image_view);

        // coef_usm
        dt->CreateImage(device, &coef_image_info, nullptr, &data->coef_usm_image);
        VkMemoryRequirements cu_mem_reqs;
        dt->GetImageMemoryRequirements(device, data->coef_usm_image, &cu_mem_reqs);
        VkMemoryAllocateInfo cu_alloc_info = {};
        cu_alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        cu_alloc_info.allocationSize = cu_mem_reqs.size;
        cu_alloc_info.memoryTypeIndex = find_memory_type(data->instance, data->physical_device, cu_mem_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        dt->AllocateMemory(device, &cu_alloc_info, nullptr, &data->coef_usm_image_memory);
        dt->BindImageMemory(device, data->coef_usm_image, data->coef_usm_image_memory, 0);

        coef_view_info.image = data->coef_usm_image;
        dt->CreateImageView(device, &coef_view_info, nullptr, &data->coef_usm_image_view);

        // Populate Coefficient Textures with actual NIS SDK data
        // For coef_scaler (2x64 float4)
        // coef_scale is float[64][8], which is 64 rows of 8 floats. Each float4 takes 4 floats.
        // So, each row of 8 floats corresponds to 2 float4s.
        // The texture is 2x64 (width x height) of R32G32B32A32_SFLOAT.
        // We need to copy 64 * 8 floats for each.
        std::vector<float> real_coef_scaler_data(64 * 8);
        for (int i = 0; i < 64; ++i) {
            for (int j = 0; j < 8; ++j) {
                real_coef_scaler_data[i * 8 + j] = coef_scale[i][j];
            }
        }

        // For coef_usm (2x64 float4)
        std::vector<float> real_coef_usm_data(64 * 8);
        for (int i = 0; i < 64; ++i) {
            for (int j = 0; j < 8; ++j) {
                real_coef_usm_data[i * 8 + j] = coef_usm[i][j];
            }
        }

        VkDeviceSize scaler_data_size = real_coef_scaler_data.size() * sizeof(float);
        VkDeviceSize usm_data_size = real_coef_usm_data.size() * sizeof(float);

        // Create staging buffer
        VkBuffer staging_buffer;
        VkDeviceMemory staging_buffer_memory;
        VkBufferCreateInfo staging_buffer_info = {};
        staging_buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        staging_buffer_info.size = scaler_data_size + usm_data_size;
        staging_buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        staging_buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        dt->CreateBuffer(device, &staging_buffer_info, nullptr, &staging_buffer);

        VkMemoryRequirements staging_mem_reqs;
        dt->GetBufferMemoryRequirements(device, staging_buffer, &staging_mem_reqs);
        VkMemoryAllocateInfo staging_alloc_info = {};
        staging_alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        staging_alloc_info.allocationSize = staging_mem_reqs.size;
        staging_alloc_info.memoryTypeIndex = find_memory_type(data->instance, data->physical_device, staging_mem_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        dt->AllocateMemory(device, &staging_alloc_info, nullptr, &staging_buffer_memory);
        dt->BindBufferMemory(device, staging_buffer, staging_buffer_memory, 0);

        void* staging_mapped_data;
        dt->MapMemory(device, staging_buffer_memory, 0, staging_buffer_info.size, 0, &staging_mapped_data);
        memcpy(staging_mapped_data, real_coef_scaler_data.data(), scaler_data_size);
        memcpy(static_cast<char*>(staging_mapped_data) + scaler_data_size, real_coef_usm_data.data(), usm_data_size);
        dt->UnmapMemory(device, staging_buffer_memory);

        // Create a one-time command buffer for transfer
        VkCommandPoolCreateInfo transfer_cmd_pool_info = {};
        transfer_cmd_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        transfer_cmd_pool_info.queueFamilyIndex = 0; // Assuming queue family 0 is suitable for transfer
        transfer_cmd_pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT; // For short-lived command buffers
        VkCommandPool transfer_command_pool;
        dt->CreateCommandPool(device, &transfer_cmd_pool_info, nullptr, &transfer_command_pool);

        VkCommandBufferAllocateInfo transfer_cmd_buf_alloc_info = {};
        transfer_cmd_buf_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        transfer_cmd_buf_alloc_info.commandPool = transfer_command_pool;
        transfer_cmd_buf_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        transfer_cmd_buf_alloc_info.commandBufferCount = 1;
        VkCommandBuffer transfer_command_buffer;
        dt->AllocateCommandBuffers(device, &transfer_cmd_buf_alloc_info, &transfer_command_buffer);

        VkCommandBufferBeginInfo begin_info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
        dt->BeginCommandBuffer(transfer_command_buffer, &begin_info);

        // Image barriers to transition images to TRANSFER_DST_OPTIMAL
        VkImageMemoryBarrier barrier_scaler_transfer = {};
        barrier_scaler_transfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier_scaler_transfer.srcAccessMask = 0;
        barrier_scaler_transfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier_scaler_transfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier_scaler_transfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier_scaler_transfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier_scaler_transfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier_scaler_transfer.image = data->coef_scaler_image;
        barrier_scaler_transfer.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkImageMemoryBarrier barrier_usm_transfer = {};
        barrier_usm_transfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier_usm_transfer.srcAccessMask = 0;
        barrier_usm_transfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier_usm_transfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier_usm_transfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier_usm_transfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier_usm_transfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier_usm_transfer.image = data->coef_usm_image;
        barrier_usm_transfer.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        std::vector<VkImageMemoryBarrier> transfer_barriers = {barrier_scaler_transfer, barrier_usm_transfer};
        dt->CmdPipelineBarrier(transfer_command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, static_cast<uint32_t>(transfer_barriers.size()), transfer_barriers.data());

        // Copy buffer to image
        VkBufferImageCopy region_scaler = {};
        region_scaler.bufferOffset = 0;
        region_scaler.bufferRowLength = 0;
        region_scaler.bufferImageHeight = 0;
        region_scaler.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region_scaler.imageSubresource.mipLevel = 0;
        region_scaler.imageSubresource.baseArrayLayer = 0;
        region_scaler.imageSubresource.layerCount = 1;
        region_scaler.imageOffset = {0, 0, 0};
        region_scaler.imageExtent = {2, 64, 1};
        dt->CmdCopyBufferToImage(transfer_command_buffer, staging_buffer, data->coef_scaler_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region_scaler);

        VkBufferImageCopy region_usm = {};
        region_usm.bufferOffset = scaler_data_size;
        region_usm.bufferRowLength = 0;
        region_usm.bufferImageHeight = 0;
        region_usm.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region_usm.imageSubresource.mipLevel = 0;
        region_usm.imageSubresource.baseArrayLayer = 0;
        region_usm.imageSubresource.layerCount = 1;
        region_usm.imageOffset = {0, 0, 0};
        region_usm.imageExtent = {2, 64, 1};
        dt->CmdCopyBufferToImage(transfer_command_buffer, staging_buffer, data->coef_usm_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region_usm);

        // Image barriers to transition images to SHADER_READ_ONLY_OPTIMAL
        VkImageMemoryBarrier barrier_scaler_shader_read = {};
        barrier_scaler_shader_read.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier_scaler_shader_read.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier_scaler_shader_read.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier_scaler_shader_read.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier_scaler_shader_read.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier_scaler_shader_read.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier_scaler_shader_read.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier_scaler_shader_read.image = data->coef_scaler_image;
        barrier_scaler_shader_read.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkImageMemoryBarrier barrier_usm_shader_read = {};
        barrier_usm_shader_read.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier_usm_shader_read.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier_usm_shader_read.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier_usm_shader_read.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier_usm_shader_read.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier_usm_shader_read.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier_usm_shader_read.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier_usm_shader_read.image = data->coef_usm_image;
        barrier_usm_shader_read.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        std::vector<VkImageMemoryBarrier> shader_read_barriers = {barrier_scaler_shader_read, barrier_usm_shader_read};
        dt->CmdPipelineBarrier(transfer_command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, static_cast<uint32_t>(shader_read_barriers.size()), shader_read_barriers.data());

        dt->EndCommandBuffer(transfer_command_buffer);

        // Submit and wait
        VkSubmitInfo submit_info = {};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &transfer_command_buffer;
        VkQueue graphics_queue; // Assuming graphics queue for submission
        dt->GetDeviceQueue(device, 0, 0, &graphics_queue);
        dt->QueueSubmit(graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
        dt->QueueWaitIdle(graphics_queue);

        // Clean up staging buffer and command pool
        dt->DestroyBuffer(device, staging_buffer, nullptr);
        dt->FreeMemory(device, staging_buffer_memory, nullptr);
        dt->FreeCommandBuffers(device, transfer_command_pool, 1, &transfer_command_buffer);
        dt->DestroyCommandPool(device, transfer_command_pool, nullptr);

        // Create Sampler
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

        // Create Descriptor Set Layout
        std::vector<VkDescriptorSetLayoutBinding> bindings(6);

        // Binding 0: Uniform Buffer (const_buffer)
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[0].pImmutableSamplers = nullptr;

        // Binding 1: Sampler (samplerLinearClamp)
        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[1].pImmutableSamplers = &data->sampler; // Use the created sampler

        // Binding 2: Sampled Image (in_texture)
        bindings[2].binding = 2;
        bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[2].pImmutableSamplers = nullptr;

        // Binding 3: Storage Image (out_texture)
        bindings[3].binding = 3;
        bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[3].descriptorCount = 1;
        bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[3].pImmutableSamplers = nullptr;

        // Binding 4: Sampled Image (coef_scaler)
        bindings[4].binding = 4;
        bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[4].descriptorCount = 1;
        bindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[4].pImmutableSamplers = nullptr;

        // Binding 5: Sampled Image (coef_usm)
        bindings[5].binding = 5;
        bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[5].descriptorCount = 1;
        bindings[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[5].pImmutableSamplers = nullptr;

        VkDescriptorSetLayoutCreateInfo layout_info = {};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
        layout_info.pBindings = bindings.data();
        dt->CreateDescriptorSetLayout(device, &layout_info, nullptr, &data->descriptor_set_layout);

        // Create Pipeline Layout
        VkPipelineLayoutCreateInfo pipeline_layout_info = {};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &data->descriptor_set_layout;
        dt->CreatePipelineLayout(device, &pipeline_layout_info, nullptr, &data->pipeline_layout);

        // Create Compute Pipeline
        VkShaderModule compute_shader_module = create_shader_module(device, _home_swarn_dox_code_LS_lossless_scaling_build_nis_main_spv, _home_swarn_dox_code_LS_lossless_scaling_build_nis_main_spv_len);

        VkPipelineShaderStageCreateInfo shader_stage_info = {};
        shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shader_stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shader_stage_info.module = compute_shader_module;
        shader_stage_info.pName = "main";

        VkComputePipelineCreateInfo pipeline_info = {};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.layout = data->pipeline_layout;
        pipeline_info.stage = shader_stage_info;
        dt->CreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &data->compute_pipeline);

        dt->DestroyShaderModule(device, compute_shader_module, nullptr);

        // Create Descriptor Pool
        std::vector<VkDescriptorPoolSize> pool_sizes(3);
        pool_sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        pool_sizes[0].descriptorCount = image_count; // One uniform buffer per set
        pool_sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        pool_sizes[1].descriptorCount = image_count * 4; // Sampler, in_texture, coef_scaler, coef_usm
        pool_sizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        pool_sizes[2].descriptorCount = image_count; // out_texture

        VkDescriptorPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
        pool_info.pPoolSizes = pool_sizes.data();
        pool_info.maxSets = image_count;
        dt->CreateDescriptorPool(device, &pool_info, nullptr, &data->descriptor_pool);

        // Allocate and Update Descriptor Sets
        data->descriptor_sets.resize(image_count);
        for (uint32_t i = 0; i < image_count; i++) {
            VkDescriptorSetAllocateInfo alloc_set_info = {};
            alloc_set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            alloc_set_info.descriptorPool = data->descriptor_pool;
            alloc_set_info.descriptorSetCount = 1;
            alloc_set_info.pSetLayouts = &data->descriptor_set_layout;
            dt->AllocateDescriptorSets(device, &alloc_set_info, &data->descriptor_sets[i]);

            // Binding 0: Uniform Buffer
            VkDescriptorBufferInfo buffer_info = {};
            buffer_info.buffer = data->uniform_buffer;
            buffer_info.offset = 0;
            buffer_info.range = VK_WHOLE_SIZE;

            // Binding 1: Sampler
            VkDescriptorImageInfo sampler_info = {};
            sampler_info.sampler = data->sampler;

            // Binding 2: Input Texture
            VkDescriptorImageInfo input_image_info = {};
            input_image_info.imageView = data->swapchain_image_views[i];
            input_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; // Changed layout

            // Binding 3: Output Texture
            VkDescriptorImageInfo output_image_info = {};
            output_image_info.imageView = data->upscaled_image_views[i];
            output_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            // Binding 4: Coef Scaler
            VkDescriptorImageInfo coef_scaler_info = {};
            coef_scaler_info.imageView = data->coef_scaler_image_view;
            coef_scaler_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            // Binding 5: Coef USM
            VkDescriptorImageInfo coef_usm_info = {};
            coef_usm_info.imageView = data->coef_usm_image_view;
            coef_usm_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            std::vector<VkWriteDescriptorSet> descriptor_writes(6);

            // Write for Binding 0 (Uniform Buffer)
            descriptor_writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptor_writes[0].dstSet = data->descriptor_sets[i];
            descriptor_writes[0].dstBinding = 0;
            descriptor_writes[0].dstArrayElement = 0;
            descriptor_writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptor_writes[0].descriptorCount = 1;
            descriptor_writes[0].pBufferInfo = &buffer_info;

            // Write for Binding 1 (Sampler)
            descriptor_writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptor_writes[1].dstSet = data->descriptor_sets[i];
            descriptor_writes[1].dstBinding = 1;
            descriptor_writes[1].dstArrayElement = 0;
            descriptor_writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptor_writes[1].descriptorCount = 1;
            descriptor_writes[1].pImageInfo = &sampler_info;

            // Write for Binding 2 (Input Texture)
            descriptor_writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptor_writes[2].dstSet = data->descriptor_sets[i];
            descriptor_writes[2].dstBinding = 2;
            descriptor_writes[2].dstArrayElement = 0;
            descriptor_writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptor_writes[2].descriptorCount = 1;
            descriptor_writes[2].pImageInfo = &input_image_info;

            // Write for Binding 3 (Output Texture)
            descriptor_writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptor_writes[3].dstSet = data->descriptor_sets[i];
            descriptor_writes[3].dstBinding = 3;
            descriptor_writes[3].dstArrayElement = 0;
            descriptor_writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptor_writes[3].descriptorCount = 1;
            descriptor_writes[3].pImageInfo = &output_image_info;

            // Write for Binding 4 (Coef Scaler)
            descriptor_writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptor_writes[4].dstSet = data->descriptor_sets[i];
            descriptor_writes[4].dstBinding = 4;
            descriptor_writes[4].dstArrayElement = 0;
            descriptor_writes[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptor_writes[4].descriptorCount = 1;
            descriptor_writes[4].pImageInfo = &coef_scaler_info;

            // Write for Binding 5 (Coef USM)
            descriptor_writes[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptor_writes[5].dstSet = data->descriptor_sets[i];
            descriptor_writes[5].dstBinding = 5;
            descriptor_writes[5].dstArrayElement = 0;
            descriptor_writes[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptor_writes[5].descriptorCount = 1;
            descriptor_writes[5].pImageInfo = &coef_usm_info;

            dt->UpdateDescriptorSets(device, static_cast<uint32_t>(descriptor_writes.size()), descriptor_writes.data(), 0, nullptr);
        }

        // Create Command Pool and Command Buffers
        VkCommandPoolCreateInfo cmd_pool_info = {};
        cmd_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cmd_pool_info.queueFamilyIndex = 0; // Assuming queue family 0 is suitable for compute
        dt->CreateCommandPool(device, &cmd_pool_info, nullptr, &data->command_pool);

        VkCommandBufferAllocateInfo cmd_buf_alloc_info = {};
        cmd_buf_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmd_buf_alloc_info.commandPool = data->command_pool;
        cmd_buf_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmd_buf_alloc_info.commandBufferCount = image_count;
        data->command_buffers.resize(image_count);
        dt->AllocateCommandBuffers(device, &cmd_buf_alloc_info, data->command_buffers.data());

        // Create Fences
        data->fences.resize(image_count);
        VkFenceCreateInfo fence_info = {};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT; // Start in signaled state
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
        // Should not happen, but handle gracefully
        device_dispatch_tables.at((void*)device).DestroySwapchainKHR(device, swapchain, pAllocator);
        return;
    }

    std::unique_ptr<SwapchainData> data = std::move(it->second);
    swapchain_data_map.erase(it);

    auto* dt = &device_dispatch_tables.at((void*)device);
    dt->DeviceWaitIdle(device);

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

    // NIS specific cleanup
    dt->DestroyBuffer(device, data->uniform_buffer, nullptr);
    dt->FreeMemory(device, data->uniform_buffer_memory, nullptr);
    dt->DestroyImageView(device, data->coef_scaler_image_view, nullptr);
    dt->DestroyImage(device, data->coef_scaler_image, nullptr);
    dt->FreeMemory(device, data->coef_scaler_image_memory, nullptr);
    dt->DestroyImageView(device, data->coef_usm_image_view, nullptr);
    dt->DestroyImage(device, data->coef_usm_image, nullptr);
    dt->FreeMemory(device, data->coef_usm_image_memory, nullptr);

    dt->DestroySwapchainKHR(device, swapchain, pAllocator);
}

VKAPI_ATTR VkResult VKAPI_CALL LosslessScaling_QueuePresentKHR(VkQueue queue, const VkPresentInfoKHR* pPresentInfo) {
    fprintf(stderr, "[LosslessScaling Layer] LosslessScaling_QueuePresentKHR called.\n");

    auto device = queue_to_device_map.at((void*)queue);
    auto* dt = &device_dispatch_tables.at((void*)device);

    // Check if the layer is enabled via environment variable
    const char* env_var = getenv("LOSSLESS_SCALING");
    bool enabled = (env_var != nullptr && strcmp(env_var, "1") == 0);

    if (!enabled) {
        // If not enabled, just pass through the original present call
        return dt->QueuePresentKHR(queue, pPresentInfo);
    }



    // Find the swapchain data for the current swapchain being presented
    VkSwapchainKHR current_swapchain = pPresentInfo->pSwapchains[0]; // Assuming single swapchain for simplicity
    auto it = swapchain_data_map.find(current_swapchain);
    if (it == swapchain_data_map.end()) {
        // Data not found, pass through
        return dt->QueuePresentKHR(queue, pPresentInfo);
    }
    SwapchainData* data = it->second.get();

    fprintf(stderr, "[LosslessScaling Layer] Original swapchain: %p\n", (void*)pPresentInfo->pSwapchains[0]);
    fprintf(stderr, "[LosslessScaling Layer] Modified swapchain: %p\n", (void*)data->swapchain);

    uint32_t image_index = pPresentInfo->pImageIndices[0];

    // Wait for the fence of the current image to ensure it's not in use
    dt->WaitForFences(device, 1, &data->fences[image_index], VK_TRUE, UINT64_MAX);
    dt->ResetFences(device, 1, &data->fences[image_index]);

    VkCommandBuffer command_buffer = data->command_buffers[image_index];
    VkCommandBufferBeginInfo begin_info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
    dt->BeginCommandBuffer(command_buffer, &begin_info);

    // Image memory barrier to transition swapchain image to shader read optimal layout
    VkImageMemoryBarrier barrier_in = {};
    barrier_in.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier_in.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    barrier_in.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier_in.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    barrier_in.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier_in.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier_in.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier_in.image = data->swapchain_images[image_index];
    barrier_in.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    dt->CmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier_in);

    // Image memory barrier to transition upscaled image to general layout for shader write
    VkImageMemoryBarrier barrier_out = {};
    barrier_out.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier_out.srcAccessMask = 0;
    barrier_out.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier_out.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier_out.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier_out.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier_out.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier_out.image = data->upscaled_images[image_index];
    barrier_out.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    dt->CmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier_out);

    // Image memory barrier for coef_scaler_image to shader read optimal layout
    VkImageMemoryBarrier barrier_coef_scaler = {};
    barrier_coef_scaler.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier_coef_scaler.srcAccessMask = 0;
    barrier_coef_scaler.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier_coef_scaler.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier_coef_scaler.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier_coef_scaler.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier_coef_scaler.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier_coef_scaler.image = data->coef_scaler_image;
    barrier_coef_scaler.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    dt->CmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier_coef_scaler);

    // Image memory barrier for coef_usm_image to shader read optimal layout
    VkImageMemoryBarrier barrier_coef_usm = {};
    barrier_coef_usm.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier_coef_usm.srcAccessMask = 0;
    barrier_coef_usm.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier_coef_usm.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier_coef_usm.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier_coef_usm.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier_coef_usm.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier_coef_usm.image = data->coef_usm_image;
    barrier_coef_usm.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    dt->CmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier_coef_usm);

    // Bind pipeline and descriptor sets
    dt->CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, data->compute_pipeline);
    dt->CmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, data->pipeline_layout, 0, 1, &data->descriptor_sets[image_index], 0, nullptr);

    fprintf(stderr, "[LosslessScaling Layer] Dispatching NIS compute shader...\n");
    // Dispatch compute shader
    const uint32_t NIS_BLOCK_WIDTH = 32;
    const uint32_t NIS_BLOCK_HEIGHT = 24;
    dt->CmdDispatch(command_buffer, (data->high_res_extent.width + NIS_BLOCK_WIDTH - 1) / NIS_BLOCK_WIDTH, (data->high_res_extent.height + NIS_BLOCK_HEIGHT - 1) / NIS_BLOCK_HEIGHT, 1);
    fprintf(stderr, "[LosslessScaling Layer] NIS compute shader dispatched.\n");

    // Image memory barrier to transition upscaled image to present layout
    VkImageMemoryBarrier barrier_present = {};
    barrier_present.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier_present.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier_present.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    barrier_present.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier_present.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    barrier_present.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier_present.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier_present.image = data->upscaled_images[image_index];
    barrier_present.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    dt->CmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier_present);

    dt->EndCommandBuffer(command_buffer);

    // Submit command buffer
    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    dt->QueueSubmit(queue, 1, &submit_info, data->fences[image_index]);

    // Modify present info to use the upscaled image
    VkPresentInfoKHR modified_present_info = *pPresentInfo;
    uint32_t local_image_index = image_index; // Use a local variable for the index
    modified_present_info.pImageIndices = &local_image_index; // Point to the local variable
    modified_present_info.swapchainCount = 1; // Corrected from imageCount
    modified_present_info.pSwapchains = &data->swapchain; // Ensure it points to the original swapchain
    modified_present_info.waitSemaphoreCount = 0;
    modified_present_info.pWaitSemaphores = nullptr;

    // Call next layer's present with the upscaled image
    VkResult present_result = dt->QueuePresentKHR(queue, &modified_present_info);
    return present_result;
}

// --- Main entry points ---
extern "C" {

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL LosslessScaling_GetInstanceProcAddr(VkInstance instance, const char* pName) {
    if (strcmp(pName, "vkCreateInstance") == 0) return (PFN_vkVoidFunction)LosslessScaling_CreateInstance;
    if (strcmp(pName, "vkDestroyInstance") == 0) return (PFN_vkVoidFunction)LosslessScaling_DestroyInstance;
    if (strcmp(pName, "vkCreateDevice") == 0) return (PFN_vkVoidFunction)LosslessScaling_CreateDevice;
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

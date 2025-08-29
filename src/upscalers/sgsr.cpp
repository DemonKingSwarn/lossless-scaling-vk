#include "upscalers/sgsr.h"

#include <vulkan/vulkan.h>
#include <string.h>
#include <vector>
#include <memory>
#include <stdexcept>

#include "../lossless_scaling.h"
#include "upscalers/sgsr_upscale_shader.h"

extern std::unordered_map<void*, VkLayerInstanceDispatchTable> instance_dispatch_tables;
extern std::unordered_map<void*, VkLayerDeviceDispatchTable> device_dispatch_tables;

namespace upscalers {
namespace sgsr {

bool init(SwapchainData* data) {
    auto* dt = &device_dispatch_tables.at((void*)data->device);
    auto sgsr_data = std::make_unique<SgsrData>();

    VkImageCreateInfo image_info = {};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.extent.width = data->high_res_extent.width;
    image_info.extent.height = data->high_res_extent.height;
    image_info.extent.depth = 1;
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;

    dt->CreateImage(data->device, &image_info, nullptr, &sgsr_data->history_image);

    VkMemoryRequirements mem_reqs;
    dt->GetImageMemoryRequirements(data->device, sgsr_data->history_image, &mem_reqs);

    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = find_memory_type(data->instance, data->physical_device, mem_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    dt->AllocateMemory(data->device, &alloc_info, nullptr, &sgsr_data->history_image_memory);
    dt->BindImageMemory(data->device, sgsr_data->history_image, sgsr_data->history_image_memory, 0);

    VkImageViewCreateInfo view_info = {};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = sgsr_data->history_image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;

    dt->CreateImageView(data->device, &view_info, nullptr, &sgsr_data->history_image_view);

    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = sizeof(Params);
    buffer_info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    dt->CreateBuffer(data->device, &buffer_info, nullptr, &sgsr_data->params_buffer);

    VkMemoryRequirements ub_mem_reqs;
    dt->GetBufferMemoryRequirements(data->device, sgsr_data->params_buffer, &ub_mem_reqs);
    VkMemoryAllocateInfo ub_alloc_info = {};
    ub_alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ub_alloc_info.allocationSize = ub_mem_reqs.size;
    ub_alloc_info.memoryTypeIndex = find_memory_type(data->instance, data->physical_device, ub_mem_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    dt->AllocateMemory(data->device, &ub_alloc_info, nullptr, &sgsr_data->params_buffer_memory);
    dt->BindBufferMemory(data->device, sgsr_data->params_buffer, sgsr_data->params_buffer_memory, 0);

    std::vector<VkDescriptorSetLayoutBinding> upscale_bindings(6);
    upscale_bindings[0].binding = 0;
    upscale_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    upscale_bindings[0].descriptorCount = 1;
    upscale_bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    upscale_bindings[0].pImmutableSamplers = nullptr;

    upscale_bindings[1].binding = 1;
    upscale_bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    upscale_bindings[1].descriptorCount = 1;
    upscale_bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    upscale_bindings[1].pImmutableSamplers = &data->sampler;

    upscale_bindings[2].binding = 2;
    upscale_bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    upscale_bindings[2].descriptorCount = 1;
    upscale_bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    upscale_bindings[2].pImmutableSamplers = nullptr;

    upscale_bindings[3].binding = 3;
    upscale_bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    upscale_bindings[3].descriptorCount = 1;
    upscale_bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    upscale_bindings[3].pImmutableSamplers = nullptr;

    upscale_bindings[4].binding = 4;
    upscale_bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    upscale_bindings[4].descriptorCount = 1;
    upscale_bindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    upscale_bindings[4].pImmutableSamplers = nullptr;

    upscale_bindings[5].binding = 5;
    upscale_bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    upscale_bindings[5].descriptorCount = 1;
    upscale_bindings[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    upscale_bindings[5].pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutCreateInfo upscale_layout_info = {};
    upscale_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    upscale_layout_info.bindingCount = static_cast<uint32_t>(upscale_bindings.size());
    upscale_layout_info.pBindings = upscale_bindings.data();
    dt->CreateDescriptorSetLayout(data->device, &upscale_layout_info, nullptr, &sgsr_data->upscale_descriptor_set_layout);

    VkPipelineLayoutCreateInfo upscale_pipeline_layout_info = {};
    upscale_pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    upscale_pipeline_layout_info.setLayoutCount = 1;
    upscale_pipeline_layout_info.pSetLayouts = &sgsr_data->upscale_descriptor_set_layout;
    dt->CreatePipelineLayout(data->device, &upscale_pipeline_layout_info, nullptr, &sgsr_data->upscale_pipeline_layout);

    VkShaderModule upscale_shader_module = create_shader_module(data->device, sgsr_upscale_shader_spirv, sgsr_upscale_shader_spirv_len);

    VkPipelineShaderStageCreateInfo upscale_shader_stage_info = {};
    upscale_shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    upscale_shader_stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    upscale_shader_stage_info.module = upscale_shader_module;
    upscale_shader_stage_info.pName = "main";

    VkComputePipelineCreateInfo upscale_pipeline_info = {};
    upscale_pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    upscale_pipeline_info.layout = sgsr_data->upscale_pipeline_layout;
    upscale_pipeline_info.stage = upscale_shader_stage_info;
    dt->CreateComputePipelines(data->device, VK_NULL_HANDLE, 1, &upscale_pipeline_info, nullptr, &sgsr_data->upscale_pipeline);

    dt->DestroyShaderModule(data->device, upscale_shader_module, nullptr);

    uint32_t image_count = data->swapchain_images.size();
    std::vector<VkDescriptorPoolSize> pool_sizes(3);
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool_sizes[0].descriptorCount = image_count;
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_sizes[1].descriptorCount = image_count * 3;
    pool_sizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pool_sizes[2].descriptorCount = image_count * 2;

    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
    pool_info.pPoolSizes = pool_sizes.data();
    pool_info.maxSets = image_count;
    dt->CreateDescriptorPool(data->device, &pool_info, nullptr, &data->descriptor_pool);

    data->descriptor_sets.resize(image_count);
    for (uint32_t i = 0; i < image_count; i++) {
        VkDescriptorSetAllocateInfo alloc_set_info = {};
        alloc_set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_set_info.descriptorPool = data->descriptor_pool;
        alloc_set_info.descriptorSetCount = 1;
        alloc_set_info.pSetLayouts = &sgsr_data->upscale_descriptor_set_layout;
        dt->AllocateDescriptorSets(data->device, &alloc_set_info, &data->descriptor_sets[i]);

        VkDescriptorBufferInfo buffer_info_desc = {};
        buffer_info_desc.buffer = sgsr_data->params_buffer;
        buffer_info_desc.offset = 0;
        buffer_info_desc.range = VK_WHOLE_SIZE;

        VkDescriptorImageInfo history_image_info = {};
        history_image_info.sampler = data->sampler;
        history_image_info.imageView = sgsr_data->history_image_view;
        history_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo input_image_info = {};
        input_image_info.sampler = data->sampler;
        input_image_info.imageView = data->swapchain_image_views[i];
        input_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo motion_depth_clip_alpha_info = {};
        motion_depth_clip_alpha_info.sampler = data->sampler;
        motion_depth_clip_alpha_info.imageView = data->swapchain_image_views[i]; // Placeholder
        motion_depth_clip_alpha_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo history_output_info = {};
        history_output_info.imageView = sgsr_data->history_image_view;
        history_output_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo scene_color_output_info = {};
        scene_color_output_info.imageView = data->upscaled_image_views[i];
        scene_color_output_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        std::vector<VkWriteDescriptorSet> descriptor_writes(6);

        descriptor_writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptor_writes[0].dstSet = data->descriptor_sets[i];
        descriptor_writes[0].dstBinding = 0;
        descriptor_writes[0].dstArrayElement = 0;
        descriptor_writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptor_writes[0].descriptorCount = 1;
        descriptor_writes[0].pBufferInfo = &buffer_info_desc;

        descriptor_writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptor_writes[1].dstSet = data->descriptor_sets[i];
        descriptor_writes[1].dstBinding = 1;
        descriptor_writes[1].dstArrayElement = 0;
        descriptor_writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptor_writes[1].descriptorCount = 1;
        descriptor_writes[1].pImageInfo = &history_image_info;

        descriptor_writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptor_writes[2].dstSet = data->descriptor_sets[i];
        descriptor_writes[2].dstBinding = 2;
        descriptor_writes[2].dstArrayElement = 0;
        descriptor_writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptor_writes[2].descriptorCount = 1;
        descriptor_writes[2].pImageInfo = &motion_depth_clip_alpha_info;

        descriptor_writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptor_writes[3].dstSet = data->descriptor_sets[i];
        descriptor_writes[3].dstBinding = 3;
        descriptor_writes[3].dstArrayElement = 0;
        descriptor_writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptor_writes[3].descriptorCount = 1;
        descriptor_writes[3].pImageInfo = &input_image_info;

        descriptor_writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptor_writes[4].dstSet = data->descriptor_sets[i];
        descriptor_writes[4].dstBinding = 4;
        descriptor_writes[4].dstArrayElement = 0;
        descriptor_writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptor_writes[4].descriptorCount = 1;
        descriptor_writes[4].pImageInfo = &history_output_info;

        descriptor_writes[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptor_writes[5].dstSet = data->descriptor_sets[i];
        descriptor_writes[5].dstBinding = 5;
        descriptor_writes[5].dstArrayElement = 0;
        descriptor_writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptor_writes[5].descriptorCount = 1;
        descriptor_writes[5].pImageInfo = &scene_color_output_info;

        dt->UpdateDescriptorSets(data->device, static_cast<uint32_t>(descriptor_writes.size()), descriptor_writes.data(), 0, nullptr);
    }

    data->upscaler_data = sgsr_data.release();
    return true;
}

void cleanup(SwapchainData* data) {
    if (!data->upscaler_data) {
        return;
    }
    auto* dt = &device_dispatch_tables.at((void*)data->device);
    auto* sgsr_data = static_cast<SgsrData*>(data->upscaler_data);

    dt->DestroyPipeline(data->device, sgsr_data->upscale_pipeline, nullptr);
    dt->DestroyPipelineLayout(data->device, sgsr_data->upscale_pipeline_layout, nullptr);
    dt->DestroyDescriptorSetLayout(data->device, sgsr_data->upscale_descriptor_set_layout, nullptr);

    dt->DestroyImageView(data->device, sgsr_data->history_image_view, nullptr);
    dt->DestroyImage(data->device, sgsr_data->history_image, nullptr);
    dt->FreeMemory(data->device, sgsr_data->history_image_memory, nullptr);

    dt->DestroyBuffer(data->device, sgsr_data->params_buffer, nullptr);
    dt->FreeMemory(data->device, sgsr_data->params_buffer_memory, nullptr);

    delete sgsr_data;
    data->upscaler_data = nullptr;
}

void upscale(SwapchainData* data, uint32_t image_index, VkQueue queue) {
    auto* dt = &device_dispatch_tables.at((void*)data->device);
    auto* sgsr_data = static_cast<SgsrData*>(data->upscaler_data);

    // Update params
    {
        Params params = {};
        params.renderSize[0] = data->low_res_extent.width;
        params.renderSize[1] = data->low_res_extent.height;
        params.displaySize[0] = data->high_res_extent.width;
        params.displaySize[1] = data->high_res_extent.height;
        params.renderSizeRcp[0] = 1.0f / data->low_res_extent.width;
        params.renderSizeRcp[1] = 1.0f / data->low_res_extent.height;
        params.displaySizeRcp[0] = 1.0f / data->high_res_extent.width;
        params.displaySizeRcp[1] = 1.0f / data->high_res_extent.height;
        // TODO: Get jitter from application
        params.jitterOffset[0] = 0.0f;
        params.jitterOffset[1] = 0.0f;
        params.preExposure = 1.0f;
        // TODO: Get camera data from application
        params.cameraFovAngleHor = 1.5708f;
        params.cameraNear = 0.1f;
        params.reset = 1;

        void* mapped_data;
        dt->MapMemory(data->device, sgsr_data->params_buffer_memory, 0, sizeof(Params), 0, &mapped_data);
        memcpy(mapped_data, &params, sizeof(Params));
        dt->UnmapMemory(data->device, sgsr_data->params_buffer_memory);
    }

    dt->WaitForFences(data->device, 1, &data->fences[image_index], VK_TRUE, UINT64_MAX);
    dt->ResetFences(data->device, 1, &data->fences[image_index]);

    VkCommandBuffer command_buffer = data->command_buffers[image_index];
    VkCommandBufferBeginInfo begin_info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
    dt->BeginCommandBuffer(command_buffer, &begin_info);

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

    VkImageMemoryBarrier barrier_history = {};
    barrier_history.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier_history.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier_history.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier_history.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier_history.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier_history.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier_history.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier_history.image = sgsr_data->history_image;
    barrier_history.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

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

    std::vector<VkImageMemoryBarrier> barriers = {barrier_in, barrier_history, barrier_out};
    dt->CmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, static_cast<uint32_t>(barriers.size()), barriers.data());

    dt->CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, sgsr_data->upscale_pipeline);
    dt->CmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, sgsr_data->upscale_pipeline_layout, 0, 1, &data->descriptor_sets[image_index], 0, nullptr);
    dt->CmdDispatch(command_buffer, (data->high_res_extent.width + 7) / 8, (data->high_res_extent.height + 7) / 8, 1);

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

    VkImageMemoryBarrier barrier_history_out = {};
    barrier_history_out.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier_history_out.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier_history_out.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier_history_out.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier_history_out.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier_history_out.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier_history_out.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier_history_out.image = sgsr_data->history_image;
    barrier_history_out.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    std::vector<VkImageMemoryBarrier> out_barriers = {barrier_present, barrier_history_out};
    dt->CmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, static_cast<uint32_t>(out_barriers.size()), out_barriers.data());

    dt->EndCommandBuffer(command_buffer);

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    dt->QueueSubmit(queue, 1, &submit_info, data->fences[image_index]);
}

}
}

#include "upscalers/nis.h"

#include <vulkan/vulkan.h>
#include <string.h>
#include <vector>
#include <memory>
#include <stdexcept>

#include "nis_shader.h"
#include "NIS_Config.h"

#include "../lossless_scaling.h"

extern std::unordered_map<void*, VkLayerInstanceDispatchTable> instance_dispatch_tables;
extern std::unordered_map<void*, VkLayerDeviceDispatchTable> device_dispatch_tables;

namespace upscalers {
namespace nis {

struct NisData {
    VkBuffer uniform_buffer;
    VkDeviceMemory uniform_buffer_memory;
    VkImage coef_scaler_image;
    VkDeviceMemory coef_scaler_image_memory;
    VkImageView coef_scaler_image_view;
    VkImage coef_usm_image;
    VkDeviceMemory coef_usm_image_memory;
    VkImageView coef_usm_image_view;
};



bool init(SwapchainData* data) {
    auto* dt = &device_dispatch_tables.at((void*)data->device);
    auto nis_data = std::make_unique<NisData>();

    VkDeviceSize buffer_size = 256;
    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = buffer_size;
    buffer_info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    dt->CreateBuffer(data->device, &buffer_info, nullptr, &nis_data->uniform_buffer);

    VkMemoryRequirements ub_mem_reqs;
    dt->GetBufferMemoryRequirements(data->device, nis_data->uniform_buffer, &ub_mem_reqs);
    VkMemoryAllocateInfo ub_alloc_info = {};
    ub_alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ub_alloc_info.allocationSize = ub_mem_reqs.size;
    ub_alloc_info.memoryTypeIndex = find_memory_type(data->instance, data->physical_device, ub_mem_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    dt->AllocateMemory(data->device, &ub_alloc_info, nullptr, &nis_data->uniform_buffer_memory);
    dt->BindBufferMemory(data->device, nis_data->uniform_buffer, nis_data->uniform_buffer_memory, 0);

    NISConfig nis_config = {};
    NVScalerUpdateConfig(nis_config, 0.5f,
                         0, 0, data->low_res_extent.width, data->low_res_extent.height,
                         data->low_res_extent.width, data->low_res_extent.height,
                         0, 0, data->high_res_extent.width, data->high_res_extent.height,
                         data->high_res_extent.width, data->high_res_extent.height,
                         NISHDRMode::None);

    void* mapped_data;
    dt->MapMemory(data->device, nis_data->uniform_buffer_memory, 0, sizeof(NISConfig), 0, &mapped_data);
    memcpy(mapped_data, &nis_config, sizeof(NISConfig));
    dt->UnmapMemory(data->device, nis_data->uniform_buffer_memory);

    VkImageCreateInfo coef_image_info = {};
    coef_image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    coef_image_info.imageType = VK_IMAGE_TYPE_2D;
    coef_image_info.extent.width = 2;
    coef_image_info.extent.height = 64;
    coef_image_info.extent.depth = 1;
    coef_image_info.mipLevels = 1;
    coef_image_info.arrayLayers = 1;
    coef_image_info.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    coef_image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    coef_image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    coef_image_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    coef_image_info.samples = VK_SAMPLE_COUNT_1_BIT;

    dt->CreateImage(data->device, &coef_image_info, nullptr, &nis_data->coef_scaler_image);
    VkMemoryRequirements cs_mem_reqs;
    dt->GetImageMemoryRequirements(data->device, nis_data->coef_scaler_image, &cs_mem_reqs);
    VkMemoryAllocateInfo cs_alloc_info = {};
    cs_alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    cs_alloc_info.allocationSize = cs_mem_reqs.size;
    cs_alloc_info.memoryTypeIndex = find_memory_type(data->instance, data->physical_device, cs_mem_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    dt->AllocateMemory(data->device, &cs_alloc_info, nullptr, &nis_data->coef_scaler_image_memory);
    dt->BindImageMemory(data->device, nis_data->coef_scaler_image, nis_data->coef_scaler_image_memory, 0);

    VkImageViewCreateInfo coef_view_info = {};
    coef_view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    coef_view_info.image = nis_data->coef_scaler_image;
    coef_view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    coef_view_info.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    coef_view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    coef_view_info.subresourceRange.baseMipLevel = 0;
    coef_view_info.subresourceRange.levelCount = 1;
    coef_view_info.subresourceRange.baseArrayLayer = 0;
    coef_view_info.subresourceRange.layerCount = 1;
    dt->CreateImageView(data->device, &coef_view_info, nullptr, &nis_data->coef_scaler_image_view);

    dt->CreateImage(data->device, &coef_image_info, nullptr, &nis_data->coef_usm_image);
    VkMemoryRequirements cu_mem_reqs;
    dt->GetImageMemoryRequirements(data->device, nis_data->coef_usm_image, &cu_mem_reqs);
    VkMemoryAllocateInfo cu_alloc_info = {};
    cu_alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    cu_alloc_info.allocationSize = cu_mem_reqs.size;
    cu_alloc_info.memoryTypeIndex = find_memory_type(data->instance, data->physical_device, cu_mem_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    dt->AllocateMemory(data->device, &cu_alloc_info, nullptr, &nis_data->coef_usm_image_memory);
    dt->BindImageMemory(data->device, nis_data->coef_usm_image, nis_data->coef_usm_image_memory, 0);

    coef_view_info.image = nis_data->coef_usm_image;
    dt->CreateImageView(data->device, &coef_view_info, nullptr, &nis_data->coef_usm_image_view);

    std::vector<float> real_coef_scaler_data(64 * 8);
    for (int i = 0; i < 64; ++i) {
        for (int j = 0; j < 8; ++j) {
            real_coef_scaler_data[i * 8 + j] = coef_scale[i][j];
        }
    }

    std::vector<float> real_coef_usm_data(64 * 8);
    for (int i = 0; i < 64; ++i) {
        for (int j = 0; j < 8; ++j) {
            real_coef_usm_data[i * 8 + j] = coef_usm[i][j];
        }
    }

    VkDeviceSize scaler_data_size = real_coef_scaler_data.size() * sizeof(float);
    VkDeviceSize usm_data_size = real_coef_usm_data.size() * sizeof(float);

    VkBuffer staging_buffer;
    VkDeviceMemory staging_buffer_memory;
    VkBufferCreateInfo staging_buffer_info = {};
    staging_buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    staging_buffer_info.size = scaler_data_size + usm_data_size;
    staging_buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    staging_buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    dt->CreateBuffer(data->device, &staging_buffer_info, nullptr, &staging_buffer);

    VkMemoryRequirements staging_mem_reqs;
    dt->GetBufferMemoryRequirements(data->device, staging_buffer, &staging_mem_reqs);
    VkMemoryAllocateInfo staging_alloc_info = {};
    staging_alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    staging_alloc_info.allocationSize = staging_mem_reqs.size;
    staging_alloc_info.memoryTypeIndex = find_memory_type(data->instance, data->physical_device, staging_mem_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    dt->AllocateMemory(data->device, &staging_alloc_info, nullptr, &staging_buffer_memory);
    dt->BindBufferMemory(data->device, staging_buffer, staging_buffer_memory, 0);

    void* staging_mapped_data;
    dt->MapMemory(data->device, staging_buffer_memory, 0, staging_buffer_info.size, 0, &staging_mapped_data);
    memcpy(staging_mapped_data, real_coef_scaler_data.data(), scaler_data_size);
    memcpy(static_cast<char*>(staging_mapped_data) + scaler_data_size, real_coef_usm_data.data(), usm_data_size);
    dt->UnmapMemory(data->device, staging_buffer_memory);

    VkCommandPoolCreateInfo transfer_cmd_pool_info = {};
    transfer_cmd_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    transfer_cmd_pool_info.queueFamilyIndex = 0;
    transfer_cmd_pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    VkCommandPool transfer_command_pool;
    dt->CreateCommandPool(data->device, &transfer_cmd_pool_info, nullptr, &transfer_command_pool);

    VkCommandBufferAllocateInfo transfer_cmd_buf_alloc_info = {};
    transfer_cmd_buf_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    transfer_cmd_buf_alloc_info.commandPool = transfer_command_pool;
    transfer_cmd_buf_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    transfer_cmd_buf_alloc_info.commandBufferCount = 1;
    VkCommandBuffer transfer_command_buffer;
    dt->AllocateCommandBuffers(data->device, &transfer_cmd_buf_alloc_info, &transfer_command_buffer);

    VkCommandBufferBeginInfo begin_info = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
    dt->BeginCommandBuffer(transfer_command_buffer, &begin_info);

    VkImageMemoryBarrier barrier_scaler_transfer = {};
    barrier_scaler_transfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier_scaler_transfer.srcAccessMask = 0;
    barrier_scaler_transfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier_scaler_transfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier_scaler_transfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier_scaler_transfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier_scaler_transfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier_scaler_transfer.image = nis_data->coef_scaler_image;
    barrier_scaler_transfer.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkImageMemoryBarrier barrier_usm_transfer = {};
    barrier_usm_transfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier_usm_transfer.srcAccessMask = 0;
    barrier_usm_transfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier_usm_transfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier_usm_transfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier_usm_transfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier_usm_transfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier_usm_transfer.image = nis_data->coef_usm_image;
    barrier_usm_transfer.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    std::vector<VkImageMemoryBarrier> transfer_barriers = {barrier_scaler_transfer, barrier_usm_transfer};
    dt->CmdPipelineBarrier(transfer_command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, static_cast<uint32_t>(transfer_barriers.size()), transfer_barriers.data());

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
    dt->CmdCopyBufferToImage(transfer_command_buffer, staging_buffer, nis_data->coef_scaler_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region_scaler);

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
    dt->CmdCopyBufferToImage(transfer_command_buffer, staging_buffer, nis_data->coef_usm_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region_usm);

    VkImageMemoryBarrier barrier_scaler_shader_read = {};
    barrier_scaler_shader_read.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier_scaler_shader_read.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier_scaler_shader_read.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier_scaler_shader_read.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier_scaler_shader_read.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier_scaler_shader_read.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier_scaler_shader_read.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier_scaler_shader_read.image = nis_data->coef_scaler_image;
    barrier_scaler_shader_read.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkImageMemoryBarrier barrier_usm_shader_read = {};
    barrier_usm_shader_read.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier_usm_shader_read.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier_usm_shader_read.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier_usm_shader_read.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier_usm_shader_read.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier_usm_shader_read.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier_usm_shader_read.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier_usm_shader_read.image = nis_data->coef_usm_image;
    barrier_usm_shader_read.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    std::vector<VkImageMemoryBarrier> shader_read_barriers = {barrier_scaler_shader_read, barrier_usm_shader_read};
    dt->CmdPipelineBarrier(transfer_command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, static_cast<uint32_t>(shader_read_barriers.size()), shader_read_barriers.data());

    dt->EndCommandBuffer(transfer_command_buffer);

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &transfer_command_buffer;
    VkQueue graphics_queue;
    dt->GetDeviceQueue(data->device, 0, 0, &graphics_queue);
    dt->QueueSubmit(graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
    dt->QueueWaitIdle(graphics_queue);

    dt->DestroyBuffer(data->device, staging_buffer, nullptr);
    dt->FreeMemory(data->device, staging_buffer_memory, nullptr);
    dt->FreeCommandBuffers(data->device, transfer_command_pool, 1, &transfer_command_buffer);
    dt->DestroyCommandPool(data->device, transfer_command_pool, nullptr);

    std::vector<VkDescriptorSetLayoutBinding> bindings(6);

    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[0].pImmutableSamplers = nullptr;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].pImmutableSamplers = &data->sampler;

    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[2].pImmutableSamplers = nullptr;

    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[3].pImmutableSamplers = nullptr;

    bindings[4].binding = 4;
    bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[4].descriptorCount = 1;
    bindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[4].pImmutableSamplers = nullptr;

    bindings[5].binding = 5;
    bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[5].descriptorCount = 1;
    bindings[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[5].pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutCreateInfo layout_info = {};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
    layout_info.pBindings = bindings.data();
    dt->CreateDescriptorSetLayout(data->device, &layout_info, nullptr, &data->descriptor_set_layout);

    VkPipelineLayoutCreateInfo pipeline_layout_info = {};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &data->descriptor_set_layout;
    dt->CreatePipelineLayout(data->device, &pipeline_layout_info, nullptr, &data->pipeline_layout);

    VkShaderModule compute_shader_module = create_shader_module(data->device, nis_shader_spirv, nis_shader_spirv_len);

    VkPipelineShaderStageCreateInfo shader_stage_info = {};
    shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shader_stage_info.module = compute_shader_module;
    shader_stage_info.pName = "main";

    VkComputePipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.layout = data->pipeline_layout;
    pipeline_info.stage = shader_stage_info;
    dt->CreateComputePipelines(data->device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &data->compute_pipeline);

    dt->DestroyShaderModule(data->device, compute_shader_module, nullptr);

    uint32_t image_count = data->swapchain_images.size();
    std::vector<VkDescriptorPoolSize> pool_sizes(3);
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool_sizes[0].descriptorCount = image_count;
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_sizes[1].descriptorCount = image_count * 4;
    pool_sizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pool_sizes[2].descriptorCount = image_count;

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
        alloc_set_info.pSetLayouts = &data->descriptor_set_layout;
        dt->AllocateDescriptorSets(data->device, &alloc_set_info, &data->descriptor_sets[i]);

        VkDescriptorBufferInfo buffer_info_desc = {};
        buffer_info_desc.buffer = nis_data->uniform_buffer;
        buffer_info_desc.offset = 0;
        buffer_info_desc.range = VK_WHOLE_SIZE;

        VkDescriptorImageInfo sampler_info_desc = {};
        sampler_info_desc.sampler = data->sampler;

        VkDescriptorImageInfo input_image_info = {};
        input_image_info.imageView = data->swapchain_image_views[i];
        input_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo output_image_info = {};
        output_image_info.imageView = data->upscaled_image_views[i];
        output_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo coef_scaler_info = {};
        coef_scaler_info.imageView = nis_data->coef_scaler_image_view;
        coef_scaler_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo coef_usm_info = {};
        coef_usm_info.imageView = nis_data->coef_usm_image_view;
        coef_usm_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

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
        descriptor_writes[1].pImageInfo = &sampler_info_desc;

        descriptor_writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptor_writes[2].dstSet = data->descriptor_sets[i];
        descriptor_writes[2].dstBinding = 2;
        descriptor_writes[2].dstArrayElement = 0;
        descriptor_writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptor_writes[2].descriptorCount = 1;
        descriptor_writes[2].pImageInfo = &input_image_info;

        descriptor_writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptor_writes[3].dstSet = data->descriptor_sets[i];
        descriptor_writes[3].dstBinding = 3;
        descriptor_writes[3].dstArrayElement = 0;
        descriptor_writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptor_writes[3].descriptorCount = 1;
        descriptor_writes[3].pImageInfo = &output_image_info;

        descriptor_writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptor_writes[4].dstSet = data->descriptor_sets[i];
        descriptor_writes[4].dstBinding = 4;
        descriptor_writes[4].dstArrayElement = 0;
        descriptor_writes[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptor_writes[4].descriptorCount = 1;
        descriptor_writes[4].pImageInfo = &coef_scaler_info;

        descriptor_writes[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptor_writes[5].dstSet = data->descriptor_sets[i];
        descriptor_writes[5].dstBinding = 5;
        descriptor_writes[5].dstArrayElement = 0;
        descriptor_writes[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptor_writes[5].descriptorCount = 1;
        descriptor_writes[5].pImageInfo = &coef_usm_info;

        dt->UpdateDescriptorSets(data->device, static_cast<uint32_t>(descriptor_writes.size()), descriptor_writes.data(), 0, nullptr);
    }

    data->upscaler_data = nis_data.release();
    return true;
}

void cleanup(SwapchainData* data) {
    if (!data->upscaler_data) {
        return;
    }
    auto* dt = &device_dispatch_tables.at((void*)data->device);
    auto* nis_data = static_cast<NisData*>(data->upscaler_data);

    dt->DestroyBuffer(data->device, nis_data->uniform_buffer, nullptr);
    dt->FreeMemory(data->device, nis_data->uniform_buffer_memory, nullptr);
    dt->DestroyImageView(data->device, nis_data->coef_scaler_image_view, nullptr);
    dt->DestroyImage(data->device, nis_data->coef_scaler_image, nullptr);
    dt->FreeMemory(data->device, nis_data->coef_scaler_image_memory, nullptr);
    dt->DestroyImageView(data->device, nis_data->coef_usm_image_view, nullptr);
    dt->DestroyImage(data->device, nis_data->coef_usm_image, nullptr);
    dt->FreeMemory(data->device, nis_data->coef_usm_image_memory, nullptr);

    delete nis_data;
    data->upscaler_data = nullptr;
}

void upscale(SwapchainData* data, uint32_t image_index, VkQueue queue) {
    auto* dt = &device_dispatch_tables.at((void*)data->device);
    auto* nis_data = static_cast<NisData*>(data->upscaler_data);

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
    dt->CmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier_in);

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

    VkImageMemoryBarrier barrier_coef_scaler = {};
    barrier_coef_scaler.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier_coef_scaler.srcAccessMask = 0;
    barrier_coef_scaler.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier_coef_scaler.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier_coef_scaler.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier_coef_scaler.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier_coef_scaler.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier_coef_scaler.image = nis_data->coef_scaler_image;
    barrier_coef_scaler.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    dt->CmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier_coef_scaler);

    VkImageMemoryBarrier barrier_coef_usm = {};
    barrier_coef_usm.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier_coef_usm.srcAccessMask = 0;
    barrier_coef_usm.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier_coef_usm.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier_coef_usm.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier_coef_usm.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier_coef_usm.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier_coef_usm.image = nis_data->coef_usm_image;
    barrier_coef_usm.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    dt->CmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier_coef_usm);

    dt->CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, data->compute_pipeline);
    dt->CmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, data->pipeline_layout, 0, 1, &data->descriptor_sets[image_index], 0, nullptr);

    const uint32_t NIS_BLOCK_WIDTH = 32;
    const uint32_t NIS_BLOCK_HEIGHT = 24;
    dt->CmdDispatch(command_buffer, (data->high_res_extent.width + NIS_BLOCK_WIDTH - 1) / NIS_BLOCK_WIDTH, (data->high_res_extent.height + NIS_BLOCK_HEIGHT - 1) / NIS_BLOCK_HEIGHT, 1);

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

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    dt->QueueSubmit(queue, 1, &submit_info, data->fences[image_index]);
}

}
}

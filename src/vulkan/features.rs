//! The feature bits of each core version, the structs that carry them, and the
//! extensions each version promoted to core.
//!
//! Field names and their order follow the C declarations (`vk.xml`): a
//! [`Chain`] block is read and written by index, so the order is the layout.

use std::ffi::CStr;

use super::ApiVersion;

/// One `VkBool32` feature member.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Field {
    /// The member name as the spec spells it (`drawIndirectCount`).
    pub name: &'static str,
    /// The extension it came from before it was core, if any.
    pub from: Option<&'static CStr>,
}

const fn f(name: &'static str) -> Field {
    Field { name, from: None }
}

const fn e(name: &'static str, from: &'static CStr) -> Field {
    Field {
        name,
        from: Some(from),
    }
}

/// A feature struct that goes into a `pNext` chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Chain {
    /// `VkPhysicalDeviceFeatures2`, whose `features` are the 1.0 bits. Always
    /// the head of a chain.
    Core10,
    /// `VkPhysicalDeviceVulkan11Features`: needs 1.2 (1.1 itself has the
    /// single structs below).
    Vulkan11,
    Vulkan12,
    Vulkan13,
    Vulkan14,
    // Core 1.1 single structs, or their KHR extensions before 1.1.
    Storage16Bit,
    Multiview,
    VariablePointers,
    ProtectedMemory,
    SamplerYcbcrConversion,
    ShaderDrawParameters,
    // Promoted to 1.2.
    Storage8Bit,
    Float16Int8,
    BufferDeviceAddress,
    DescriptorIndexing,
    ScalarBlockLayout,
    TimelineSemaphore,
    HostQueryReset,
    // Promoted to 1.3.
    Synchronization2,
    SubgroupSizeControl,
    DemoteToHelperInvocation,
    IntegerDotProduct,
    DynamicRendering,
    // Promoted to 1.4.
    IndexTypeUint8,
    // Extensions only.
    MeshShader,
    MemoryPriority,
    PageableDeviceLocalMemory,
    FifoLatestReady,
}

impl Chain {
    /// The struct's `VkStructureType` value.
    #[must_use]
    pub const fn s_type(self) -> i32 {
        match self {
            Self::Core10 => 1_000_059_000,
            Self::Vulkan11 => 49,
            Self::Vulkan12 => 51,
            Self::Vulkan13 => 53,
            Self::Vulkan14 => 55,
            Self::Storage16Bit => 1_000_083_000,
            Self::Multiview => 1_000_053_001,
            Self::VariablePointers => 1_000_120_000,
            Self::ProtectedMemory => 1_000_145_001,
            Self::SamplerYcbcrConversion => 1_000_156_004,
            Self::ShaderDrawParameters => 1_000_063_000,
            Self::Storage8Bit => 1_000_177_000,
            Self::Float16Int8 => 1_000_082_000,
            Self::BufferDeviceAddress => 1_000_257_000,
            Self::DescriptorIndexing => 1_000_161_001,
            Self::ScalarBlockLayout => 1_000_221_000,
            Self::TimelineSemaphore => 1_000_207_000,
            Self::HostQueryReset => 1_000_261_000,
            Self::Synchronization2 => 1_000_314_007,
            Self::SubgroupSizeControl => 1_000_225_002,
            Self::DemoteToHelperInvocation => 1_000_276_000,
            Self::IntegerDotProduct => 1_000_280_000,
            Self::DynamicRendering => 1_000_044_003,
            Self::IndexTypeUint8 => 1_000_265_000,
            Self::MeshShader => 1_000_328_000,
            Self::MemoryPriority => 1_000_238_000,
            Self::PageableDeviceLocalMemory => 1_000_412_000,
            Self::FifoLatestReady => 1_000_361_000,
        }
    }

    /// The C struct name, for logs.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Core10 => "VkPhysicalDeviceFeatures",
            Self::Vulkan11 => "VkPhysicalDeviceVulkan11Features",
            Self::Vulkan12 => "VkPhysicalDeviceVulkan12Features",
            Self::Vulkan13 => "VkPhysicalDeviceVulkan13Features",
            Self::Vulkan14 => "VkPhysicalDeviceVulkan14Features",
            Self::Storage16Bit => "VkPhysicalDevice16BitStorageFeatures",
            Self::Multiview => "VkPhysicalDeviceMultiviewFeatures",
            Self::VariablePointers => "VkPhysicalDeviceVariablePointersFeatures",
            Self::ProtectedMemory => "VkPhysicalDeviceProtectedMemoryFeatures",
            Self::SamplerYcbcrConversion => "VkPhysicalDeviceSamplerYcbcrConversionFeatures",
            Self::ShaderDrawParameters => "VkPhysicalDeviceShaderDrawParametersFeatures",
            Self::Storage8Bit => "VkPhysicalDevice8BitStorageFeatures",
            Self::Float16Int8 => "VkPhysicalDeviceShaderFloat16Int8Features",
            Self::BufferDeviceAddress => "VkPhysicalDeviceBufferDeviceAddressFeatures",
            Self::DescriptorIndexing => "VkPhysicalDeviceDescriptorIndexingFeatures",
            Self::ScalarBlockLayout => "VkPhysicalDeviceScalarBlockLayoutFeatures",
            Self::TimelineSemaphore => "VkPhysicalDeviceTimelineSemaphoreFeatures",
            Self::HostQueryReset => "VkPhysicalDeviceHostQueryResetFeatures",
            Self::Synchronization2 => "VkPhysicalDeviceSynchronization2Features",
            Self::SubgroupSizeControl => "VkPhysicalDeviceSubgroupSizeControlFeatures",
            Self::DemoteToHelperInvocation => {
                "VkPhysicalDeviceShaderDemoteToHelperInvocationFeatures"
            }
            Self::IntegerDotProduct => "VkPhysicalDeviceShaderIntegerDotProductFeatures",
            Self::DynamicRendering => "VkPhysicalDeviceDynamicRenderingFeatures",
            Self::IndexTypeUint8 => "VkPhysicalDeviceIndexTypeUint8Features",
            Self::MeshShader => "VkPhysicalDeviceMeshShaderFeaturesEXT",
            Self::MemoryPriority => "VkPhysicalDeviceMemoryPriorityFeaturesEXT",
            Self::PageableDeviceLocalMemory => {
                "VkPhysicalDevicePageableDeviceLocalMemoryFeaturesEXT"
            }
            Self::FifoLatestReady => "VkPhysicalDevicePresentModeFifoLatestReadyFeaturesKHR",
        }
    }

    /// The struct's members after `sType` and `pNext`, in declaration order.
    #[must_use]
    pub const fn fields(self) -> &'static [Field] {
        match self {
            Self::Core10 => CORE_1_0,
            Self::Vulkan11 => CORE_1_1,
            Self::Vulkan12 => CORE_1_2,
            Self::Vulkan13 => CORE_1_3,
            Self::Vulkan14 => CORE_1_4,
            Self::Storage16Bit => STORAGE_16BIT,
            Self::Multiview => MULTIVIEW,
            Self::VariablePointers => VARIABLE_POINTERS,
            Self::ProtectedMemory => PROTECTED_MEMORY_FIELDS,
            Self::SamplerYcbcrConversion => SAMPLER_YCBCR_CONVERSION_FIELDS,
            Self::ShaderDrawParameters => SHADER_DRAW_PARAMETERS_FIELDS,
            Self::Storage8Bit => STORAGE_8BIT,
            Self::Float16Int8 => FLOAT16_INT8_FIELDS,
            Self::BufferDeviceAddress => BUFFER_DEVICE_ADDRESS,
            Self::DescriptorIndexing => DESCRIPTOR_INDEXING,
            Self::ScalarBlockLayout => SCALAR_BLOCK_LAYOUT_FIELDS,
            Self::TimelineSemaphore => TIMELINE_SEMAPHORE_FIELDS,
            Self::HostQueryReset => HOST_QUERY_RESET_FIELDS,
            Self::Synchronization2 => SYNCHRONIZATION2_FIELDS,
            Self::SubgroupSizeControl => SUBGROUP_SIZE_CONTROL_FIELDS,
            Self::DemoteToHelperInvocation => DEMOTE_TO_HELPER_INVOCATION_FIELDS,
            Self::IntegerDotProduct => INTEGER_DOT_PRODUCT_FIELDS,
            Self::DynamicRendering => DYNAMIC_RENDERING_FIELDS,
            Self::IndexTypeUint8 => INDEX_TYPE_UINT8_FIELDS,
            Self::MeshShader => MESH_SHADER,
            Self::MemoryPriority => MEMORY_PRIORITY_FIELDS,
            Self::PageableDeviceLocalMemory => PAGEABLE_DEVICE_LOCAL_MEMORY_FIELDS,
            Self::FifoLatestReady => FIFO_LATEST_READY_FIELDS,
        }
    }

    /// The index of member `name`.
    #[must_use]
    pub fn index(self, name: &str) -> Option<usize> {
        self.fields().iter().position(|field| field.name == name)
    }
}

/// What one core version added.
#[derive(Debug, Clone, Copy)]
pub struct Version {
    pub version: ApiVersion,
    /// The struct holding the version's bits ([`Chain::Core10`] for 1.0).
    pub chain: Chain,
    /// Its feature bits, in declaration order.
    pub fields: &'static [Field],
    /// The extensions this version made core.
    pub promoted: &'static [&'static CStr],
}

/// Every core version these tables know, oldest first.
pub const VERSIONS: [Version; 5] = [
    Version {
        version: ApiVersion::V1_0,
        chain: Chain::Core10,
        fields: CORE_1_0,
        promoted: &[],
    },
    Version {
        version: ApiVersion::V1_1,
        chain: Chain::Vulkan11,
        fields: CORE_1_1,
        promoted: PROMOTED_1_1,
    },
    Version {
        version: ApiVersion::V1_2,
        chain: Chain::Vulkan12,
        fields: CORE_1_2,
        promoted: PROMOTED_1_2,
    },
    Version {
        version: ApiVersion::V1_3,
        chain: Chain::Vulkan13,
        fields: CORE_1_3,
        promoted: PROMOTED_1_3,
    },
    Version {
        version: ApiVersion::V1_4,
        chain: Chain::Vulkan14,
        fields: CORE_1_4,
        promoted: PROMOTED_1_4,
    },
];

/// The table of `version` (patch ignored), if known.
#[must_use]
pub fn of(version: ApiVersion) -> Option<&'static Version> {
    VERSIONS
        .iter()
        .find(|v| v.version.major == version.major && v.version.minor == version.minor)
}

/// The version that made extension `name` core, if one did.
#[must_use]
pub fn promoted_in(name: &CStr) -> Option<ApiVersion> {
    VERSIONS
        .iter()
        .find(|v| v.promoted.contains(&name))
        .map(|v| v.version)
}

/// `VkPhysicalDeviceFeatures`.
pub const CORE_1_0: &[Field] = &[
    f("robustBufferAccess"),
    f("fullDrawIndexUint32"),
    f("imageCubeArray"),
    f("independentBlend"),
    f("geometryShader"),
    f("tessellationShader"),
    f("sampleRateShading"),
    f("dualSrcBlend"),
    f("logicOp"),
    f("multiDrawIndirect"),
    f("drawIndirectFirstInstance"),
    f("depthClamp"),
    f("depthBiasClamp"),
    f("fillModeNonSolid"),
    f("depthBounds"),
    f("wideLines"),
    f("largePoints"),
    f("alphaToOne"),
    f("multiViewport"),
    f("samplerAnisotropy"),
    f("textureCompressionETC2"),
    f("textureCompressionASTC_LDR"),
    f("textureCompressionBC"),
    f("occlusionQueryPrecise"),
    f("pipelineStatisticsQuery"),
    f("vertexPipelineStoresAndAtomics"),
    f("fragmentStoresAndAtomics"),
    f("shaderTessellationAndGeometryPointSize"),
    f("shaderImageGatherExtended"),
    f("shaderStorageImageExtendedFormats"),
    f("shaderStorageImageMultisample"),
    f("shaderStorageImageReadWithoutFormat"),
    f("shaderStorageImageWriteWithoutFormat"),
    f("shaderUniformBufferArrayDynamicIndexing"),
    f("shaderSampledImageArrayDynamicIndexing"),
    f("shaderStorageBufferArrayDynamicIndexing"),
    f("shaderStorageImageArrayDynamicIndexing"),
    f("shaderClipDistance"),
    f("shaderCullDistance"),
    f("shaderFloat64"),
    f("shaderInt64"),
    f("shaderInt16"),
    f("shaderResourceResidency"),
    f("shaderResourceMinLod"),
    f("sparseBinding"),
    f("sparseResidencyBuffer"),
    f("sparseResidencyImage2D"),
    f("sparseResidencyImage3D"),
    f("sparseResidency2Samples"),
    f("sparseResidency4Samples"),
    f("sparseResidency8Samples"),
    f("sparseResidency16Samples"),
    f("sparseResidencyAliased"),
    f("variableMultisampleRate"),
    f("inheritedQueries"),
];

const KHR_16BIT_STORAGE: &CStr = c"VK_KHR_16bit_storage";
const KHR_MULTIVIEW: &CStr = c"VK_KHR_multiview";
const KHR_VARIABLE_POINTERS: &CStr = c"VK_KHR_variable_pointers";

/// `VkPhysicalDeviceVulkan11Features`.
pub const CORE_1_1: &[Field] = &[
    e("storageBuffer16BitAccess", KHR_16BIT_STORAGE),
    e("uniformAndStorageBuffer16BitAccess", KHR_16BIT_STORAGE),
    e("storagePushConstant16", KHR_16BIT_STORAGE),
    e("storageInputOutput16", KHR_16BIT_STORAGE),
    e("multiview", KHR_MULTIVIEW),
    e("multiviewGeometryShader", KHR_MULTIVIEW),
    e("multiviewTessellationShader", KHR_MULTIVIEW),
    e("variablePointersStorageBuffer", KHR_VARIABLE_POINTERS),
    e("variablePointers", KHR_VARIABLE_POINTERS),
    f("protectedMemory"),
    e("samplerYcbcrConversion", c"VK_KHR_sampler_ycbcr_conversion"),
    e("shaderDrawParameters", c"VK_KHR_shader_draw_parameters"),
];

const STORAGE_16BIT: &[Field] = &[
    f("storageBuffer16BitAccess"),
    f("uniformAndStorageBuffer16BitAccess"),
    f("storagePushConstant16"),
    f("storageInputOutput16"),
];

const MULTIVIEW: &[Field] = &[
    f("multiview"),
    f("multiviewGeometryShader"),
    f("multiviewTessellationShader"),
];

const VARIABLE_POINTERS: &[Field] = &[f("variablePointersStorageBuffer"), f("variablePointers")];

const KHR_8BIT_STORAGE: &CStr = c"VK_KHR_8bit_storage";
const KHR_ATOMIC_INT64: &CStr = c"VK_KHR_shader_atomic_int64";
const KHR_FLOAT16_INT8: &CStr = c"VK_KHR_shader_float16_int8";
const EXT_DESCRIPTOR_INDEXING: &CStr = c"VK_EXT_descriptor_indexing";
const KHR_BUFFER_DEVICE_ADDRESS: &CStr = c"VK_KHR_buffer_device_address";
const KHR_MEMORY_MODEL: &CStr = c"VK_KHR_vulkan_memory_model";
const EXT_VIEWPORT_INDEX_LAYER: &CStr = c"VK_EXT_shader_viewport_index_layer";

/// `VkPhysicalDeviceVulkan12Features`.
pub const CORE_1_2: &[Field] = &[
    e(
        "samplerMirrorClampToEdge",
        c"VK_KHR_sampler_mirror_clamp_to_edge",
    ),
    e("drawIndirectCount", c"VK_KHR_draw_indirect_count"),
    e("storageBuffer8BitAccess", KHR_8BIT_STORAGE),
    e("uniformAndStorageBuffer8BitAccess", KHR_8BIT_STORAGE),
    e("storagePushConstant8", KHR_8BIT_STORAGE),
    e("shaderBufferInt64Atomics", KHR_ATOMIC_INT64),
    e("shaderSharedInt64Atomics", KHR_ATOMIC_INT64),
    e("shaderFloat16", KHR_FLOAT16_INT8),
    e("shaderInt8", KHR_FLOAT16_INT8),
    f("descriptorIndexing"),
    e(
        "shaderInputAttachmentArrayDynamicIndexing",
        EXT_DESCRIPTOR_INDEXING,
    ),
    e(
        "shaderUniformTexelBufferArrayDynamicIndexing",
        EXT_DESCRIPTOR_INDEXING,
    ),
    e(
        "shaderStorageTexelBufferArrayDynamicIndexing",
        EXT_DESCRIPTOR_INDEXING,
    ),
    e(
        "shaderUniformBufferArrayNonUniformIndexing",
        EXT_DESCRIPTOR_INDEXING,
    ),
    e(
        "shaderSampledImageArrayNonUniformIndexing",
        EXT_DESCRIPTOR_INDEXING,
    ),
    e(
        "shaderStorageBufferArrayNonUniformIndexing",
        EXT_DESCRIPTOR_INDEXING,
    ),
    e(
        "shaderStorageImageArrayNonUniformIndexing",
        EXT_DESCRIPTOR_INDEXING,
    ),
    e(
        "shaderInputAttachmentArrayNonUniformIndexing",
        EXT_DESCRIPTOR_INDEXING,
    ),
    e(
        "shaderUniformTexelBufferArrayNonUniformIndexing",
        EXT_DESCRIPTOR_INDEXING,
    ),
    e(
        "shaderStorageTexelBufferArrayNonUniformIndexing",
        EXT_DESCRIPTOR_INDEXING,
    ),
    e(
        "descriptorBindingUniformBufferUpdateAfterBind",
        EXT_DESCRIPTOR_INDEXING,
    ),
    e(
        "descriptorBindingSampledImageUpdateAfterBind",
        EXT_DESCRIPTOR_INDEXING,
    ),
    e(
        "descriptorBindingStorageImageUpdateAfterBind",
        EXT_DESCRIPTOR_INDEXING,
    ),
    e(
        "descriptorBindingStorageBufferUpdateAfterBind",
        EXT_DESCRIPTOR_INDEXING,
    ),
    e(
        "descriptorBindingUniformTexelBufferUpdateAfterBind",
        EXT_DESCRIPTOR_INDEXING,
    ),
    e(
        "descriptorBindingStorageTexelBufferUpdateAfterBind",
        EXT_DESCRIPTOR_INDEXING,
    ),
    e(
        "descriptorBindingUpdateUnusedWhilePending",
        EXT_DESCRIPTOR_INDEXING,
    ),
    e("descriptorBindingPartiallyBound", EXT_DESCRIPTOR_INDEXING),
    e(
        "descriptorBindingVariableDescriptorCount",
        EXT_DESCRIPTOR_INDEXING,
    ),
    e("runtimeDescriptorArray", EXT_DESCRIPTOR_INDEXING),
    e("samplerFilterMinmax", c"VK_EXT_sampler_filter_minmax"),
    e("scalarBlockLayout", c"VK_EXT_scalar_block_layout"),
    e("imagelessFramebuffer", c"VK_KHR_imageless_framebuffer"),
    e(
        "uniformBufferStandardLayout",
        c"VK_KHR_uniform_buffer_standard_layout",
    ),
    e(
        "shaderSubgroupExtendedTypes",
        c"VK_KHR_shader_subgroup_extended_types",
    ),
    e(
        "separateDepthStencilLayouts",
        c"VK_KHR_separate_depth_stencil_layouts",
    ),
    e("hostQueryReset", c"VK_EXT_host_query_reset"),
    e("timelineSemaphore", c"VK_KHR_timeline_semaphore"),
    e("bufferDeviceAddress", KHR_BUFFER_DEVICE_ADDRESS),
    e(
        "bufferDeviceAddressCaptureReplay",
        KHR_BUFFER_DEVICE_ADDRESS,
    ),
    e("bufferDeviceAddressMultiDevice", KHR_BUFFER_DEVICE_ADDRESS),
    e("vulkanMemoryModel", KHR_MEMORY_MODEL),
    e("vulkanMemoryModelDeviceScope", KHR_MEMORY_MODEL),
    e(
        "vulkanMemoryModelAvailabilityVisibilityChains",
        KHR_MEMORY_MODEL,
    ),
    e("shaderOutputViewportIndex", EXT_VIEWPORT_INDEX_LAYER),
    e("shaderOutputLayer", EXT_VIEWPORT_INDEX_LAYER),
    f("subgroupBroadcastDynamicId"),
];

const STORAGE_8BIT: &[Field] = &[
    f("storageBuffer8BitAccess"),
    f("uniformAndStorageBuffer8BitAccess"),
    f("storagePushConstant8"),
];

const BUFFER_DEVICE_ADDRESS: &[Field] = &[
    f("bufferDeviceAddress"),
    f("bufferDeviceAddressCaptureReplay"),
    f("bufferDeviceAddressMultiDevice"),
];

const DESCRIPTOR_INDEXING: &[Field] = &[
    f("shaderInputAttachmentArrayDynamicIndexing"),
    f("shaderUniformTexelBufferArrayDynamicIndexing"),
    f("shaderStorageTexelBufferArrayDynamicIndexing"),
    f("shaderUniformBufferArrayNonUniformIndexing"),
    f("shaderSampledImageArrayNonUniformIndexing"),
    f("shaderStorageBufferArrayNonUniformIndexing"),
    f("shaderStorageImageArrayNonUniformIndexing"),
    f("shaderInputAttachmentArrayNonUniformIndexing"),
    f("shaderUniformTexelBufferArrayNonUniformIndexing"),
    f("shaderStorageTexelBufferArrayNonUniformIndexing"),
    f("descriptorBindingUniformBufferUpdateAfterBind"),
    f("descriptorBindingSampledImageUpdateAfterBind"),
    f("descriptorBindingStorageImageUpdateAfterBind"),
    f("descriptorBindingStorageBufferUpdateAfterBind"),
    f("descriptorBindingUniformTexelBufferUpdateAfterBind"),
    f("descriptorBindingStorageTexelBufferUpdateAfterBind"),
    f("descriptorBindingUpdateUnusedWhilePending"),
    f("descriptorBindingPartiallyBound"),
    f("descriptorBindingVariableDescriptorCount"),
    f("runtimeDescriptorArray"),
];

const EXT_INLINE_UNIFORM_BLOCK: &CStr = c"VK_EXT_inline_uniform_block";
const EXT_SUBGROUP_SIZE_CONTROL: &CStr = c"VK_EXT_subgroup_size_control";

/// `VkPhysicalDeviceVulkan13Features`.
pub const CORE_1_3: &[Field] = &[
    e("robustImageAccess", c"VK_EXT_image_robustness"),
    e("inlineUniformBlock", EXT_INLINE_UNIFORM_BLOCK),
    e(
        "descriptorBindingInlineUniformBlockUpdateAfterBind",
        EXT_INLINE_UNIFORM_BLOCK,
    ),
    e(
        "pipelineCreationCacheControl",
        c"VK_EXT_pipeline_creation_cache_control",
    ),
    e("privateData", c"VK_EXT_private_data"),
    e(
        "shaderDemoteToHelperInvocation",
        c"VK_EXT_shader_demote_to_helper_invocation",
    ),
    e(
        "shaderTerminateInvocation",
        c"VK_KHR_shader_terminate_invocation",
    ),
    e("subgroupSizeControl", EXT_SUBGROUP_SIZE_CONTROL),
    e("computeFullSubgroups", EXT_SUBGROUP_SIZE_CONTROL),
    e("synchronization2", c"VK_KHR_synchronization2"),
    e(
        "textureCompressionASTC_HDR",
        c"VK_EXT_texture_compression_astc_hdr",
    ),
    e(
        "shaderZeroInitializeWorkgroupMemory",
        c"VK_KHR_zero_initialize_workgroup_memory",
    ),
    e("dynamicRendering", c"VK_KHR_dynamic_rendering"),
    e(
        "shaderIntegerDotProduct",
        c"VK_KHR_shader_integer_dot_product",
    ),
    e("maintenance4", c"VK_KHR_maintenance4"),
];

const KHR_SUBGROUP_ROTATE: &CStr = c"VK_KHR_shader_subgroup_rotate";
const KHR_LINE_RASTERIZATION: &CStr = c"VK_KHR_line_rasterization";
const KHR_VERTEX_ATTRIBUTE_DIVISOR: &CStr = c"VK_KHR_vertex_attribute_divisor";

/// `VkPhysicalDeviceVulkan14Features`.
pub const CORE_1_4: &[Field] = &[
    e("globalPriorityQuery", c"VK_KHR_global_priority"),
    e("shaderSubgroupRotate", KHR_SUBGROUP_ROTATE),
    e("shaderSubgroupRotateClustered", KHR_SUBGROUP_ROTATE),
    e("shaderFloatControls2", c"VK_KHR_shader_float_controls2"),
    e("shaderExpectAssume", c"VK_KHR_shader_expect_assume"),
    e("rectangularLines", KHR_LINE_RASTERIZATION),
    e("bresenhamLines", KHR_LINE_RASTERIZATION),
    e("smoothLines", KHR_LINE_RASTERIZATION),
    e("stippledRectangularLines", KHR_LINE_RASTERIZATION),
    e("stippledBresenhamLines", KHR_LINE_RASTERIZATION),
    e("stippledSmoothLines", KHR_LINE_RASTERIZATION),
    e(
        "vertexAttributeInstanceRateDivisor",
        KHR_VERTEX_ATTRIBUTE_DIVISOR,
    ),
    e(
        "vertexAttributeInstanceRateZeroDivisor",
        KHR_VERTEX_ATTRIBUTE_DIVISOR,
    ),
    e("indexTypeUint8", c"VK_KHR_index_type_uint8"),
    e(
        "dynamicRenderingLocalRead",
        c"VK_KHR_dynamic_rendering_local_read",
    ),
    e("maintenance5", c"VK_KHR_maintenance5"),
    e("maintenance6", c"VK_KHR_maintenance6"),
    e(
        "pipelineProtectedAccess",
        c"VK_EXT_pipeline_protected_access",
    ),
    e("pipelineRobustness", c"VK_EXT_pipeline_robustness"),
    e("hostImageCopy", c"VK_EXT_host_image_copy"),
    e("pushDescriptor", c"VK_KHR_push_descriptor"),
];

const PROTECTED_MEMORY_FIELDS: &[Field] = &[f("protectedMemory")];
const SAMPLER_YCBCR_CONVERSION_FIELDS: &[Field] = &[f("samplerYcbcrConversion")];
const SHADER_DRAW_PARAMETERS_FIELDS: &[Field] = &[f("shaderDrawParameters")];
const FLOAT16_INT8_FIELDS: &[Field] = &[f("shaderFloat16"), f("shaderInt8")];
const SCALAR_BLOCK_LAYOUT_FIELDS: &[Field] = &[f("scalarBlockLayout")];
const TIMELINE_SEMAPHORE_FIELDS: &[Field] = &[f("timelineSemaphore")];
const HOST_QUERY_RESET_FIELDS: &[Field] = &[f("hostQueryReset")];
const SYNCHRONIZATION2_FIELDS: &[Field] = &[f("synchronization2")];
const SUBGROUP_SIZE_CONTROL_FIELDS: &[Field] =
    &[f("subgroupSizeControl"), f("computeFullSubgroups")];
const DEMOTE_TO_HELPER_INVOCATION_FIELDS: &[Field] = &[f("shaderDemoteToHelperInvocation")];
const INTEGER_DOT_PRODUCT_FIELDS: &[Field] = &[f("shaderIntegerDotProduct")];
const DYNAMIC_RENDERING_FIELDS: &[Field] = &[f("dynamicRendering")];
const INDEX_TYPE_UINT8_FIELDS: &[Field] = &[f("indexTypeUint8")];
const MEMORY_PRIORITY_FIELDS: &[Field] = &[f("memoryPriority")];
const PAGEABLE_DEVICE_LOCAL_MEMORY_FIELDS: &[Field] = &[f("pageableDeviceLocalMemory")];
const FIFO_LATEST_READY_FIELDS: &[Field] = &[f("presentModeFifoLatestReady")];

const MESH_SHADER: &[Field] = &[
    f("taskShader"),
    f("meshShader"),
    f("multiviewMeshShader"),
    f("primitiveFragmentShadingRateMeshShader"),
    f("meshShaderQueries"),
];

const PROMOTED_1_1: &[&CStr] = &[
    KHR_16BIT_STORAGE,
    c"VK_KHR_bind_memory2",
    c"VK_KHR_dedicated_allocation",
    c"VK_KHR_descriptor_update_template",
    c"VK_KHR_device_group",
    c"VK_KHR_device_group_creation",
    c"VK_KHR_external_fence",
    c"VK_KHR_external_fence_capabilities",
    c"VK_KHR_external_memory",
    c"VK_KHR_external_memory_capabilities",
    c"VK_KHR_external_semaphore",
    c"VK_KHR_external_semaphore_capabilities",
    c"VK_KHR_get_memory_requirements2",
    c"VK_KHR_get_physical_device_properties2",
    c"VK_KHR_maintenance1",
    c"VK_KHR_maintenance2",
    c"VK_KHR_maintenance3",
    KHR_MULTIVIEW,
    c"VK_KHR_relaxed_block_layout",
    c"VK_KHR_sampler_ycbcr_conversion",
    c"VK_KHR_shader_draw_parameters",
    c"VK_KHR_storage_buffer_storage_class",
    KHR_VARIABLE_POINTERS,
];

const PROMOTED_1_2: &[&CStr] = &[
    KHR_8BIT_STORAGE,
    KHR_BUFFER_DEVICE_ADDRESS,
    c"VK_KHR_create_renderpass2",
    c"VK_KHR_depth_stencil_resolve",
    c"VK_KHR_draw_indirect_count",
    c"VK_KHR_driver_properties",
    c"VK_KHR_image_format_list",
    c"VK_KHR_imageless_framebuffer",
    c"VK_KHR_sampler_mirror_clamp_to_edge",
    c"VK_KHR_separate_depth_stencil_layouts",
    KHR_ATOMIC_INT64,
    KHR_FLOAT16_INT8,
    c"VK_KHR_shader_float_controls",
    c"VK_KHR_shader_subgroup_extended_types",
    c"VK_KHR_spirv_1_4",
    c"VK_KHR_timeline_semaphore",
    c"VK_KHR_uniform_buffer_standard_layout",
    KHR_MEMORY_MODEL,
    EXT_DESCRIPTOR_INDEXING,
    c"VK_EXT_host_query_reset",
    c"VK_EXT_sampler_filter_minmax",
    c"VK_EXT_scalar_block_layout",
    c"VK_EXT_separate_stencil_usage",
    EXT_VIEWPORT_INDEX_LAYER,
];

const PROMOTED_1_3: &[&CStr] = &[
    c"VK_KHR_copy_commands2",
    c"VK_KHR_dynamic_rendering",
    c"VK_KHR_format_feature_flags2",
    c"VK_KHR_maintenance4",
    c"VK_KHR_shader_integer_dot_product",
    c"VK_KHR_shader_non_semantic_info",
    c"VK_KHR_shader_terminate_invocation",
    c"VK_KHR_synchronization2",
    c"VK_KHR_zero_initialize_workgroup_memory",
    c"VK_EXT_4444_formats",
    c"VK_EXT_extended_dynamic_state",
    c"VK_EXT_extended_dynamic_state2",
    c"VK_EXT_image_robustness",
    EXT_INLINE_UNIFORM_BLOCK,
    c"VK_EXT_pipeline_creation_cache_control",
    c"VK_EXT_pipeline_creation_feedback",
    c"VK_EXT_private_data",
    c"VK_EXT_shader_demote_to_helper_invocation",
    EXT_SUBGROUP_SIZE_CONTROL,
    c"VK_EXT_texel_buffer_alignment",
    c"VK_EXT_texture_compression_astc_hdr",
    c"VK_EXT_tooling_info",
    c"VK_EXT_ycbcr_2plane_444_formats",
];

/// The spec's version appendix, "Vulkan Version 1.4".
const PROMOTED_1_4: &[&CStr] = &[
    c"VK_KHR_dynamic_rendering_local_read",
    c"VK_KHR_global_priority",
    c"VK_KHR_index_type_uint8",
    KHR_LINE_RASTERIZATION,
    c"VK_KHR_load_store_op_none",
    c"VK_KHR_maintenance5",
    c"VK_KHR_maintenance6",
    c"VK_KHR_map_memory2",
    c"VK_KHR_push_descriptor",
    c"VK_KHR_shader_expect_assume",
    c"VK_KHR_shader_float_controls2",
    KHR_SUBGROUP_ROTATE,
    KHR_VERTEX_ATTRIBUTE_DIVISOR,
    c"VK_EXT_host_image_copy",
    c"VK_EXT_pipeline_protected_access",
    c"VK_EXT_pipeline_robustness",
];

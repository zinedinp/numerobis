//! Runs GEMMs on the matrix engine (XMX, Tensor Cores, WMMA) through Vulkan
//! cooperative matrices, and sets the driver's tiles next to
//! [`numerobis::matrix`]'s tables.
//!
//! For every device with `VK_KHR_cooperative_matrix`:
//!
//! 1. the cooperative matrices the driver reports, read into the engine
//!    model and compared with the engine's instructions (what matches, what
//!    the hardware has but the driver does not expose, what the table does
//!    not explain);
//! 2. each reported type the shader is built for (`kernels/vulkan/
//!    gemm_coopmat.comp`: `f16 -> f32`, `f16 -> f16`, `i8 -> i32`,
//!    `u8 -> i32`) as a `gemm` with numerobis's layout and `alpha`/`beta`,
//!    compared bit for bit with the mock backend's `ops::gemm` (the inputs
//!    are chosen so that every sum is exact in any order);
//! 3. a larger `f16 -> f32` and `i8 -> i32` GEMM timed with GPU timestamps
//!    (checked on sampled entries).
//!
//! `cargo run --release --example matrix_probe --features vulkan`
//! (`NUMEROBIS_NO_VALIDATION=1` skips the validation layer,
//! `NUMEROBIS_MATRIX_SIZE=4096` sets the timed size). Exits non-zero on a
//! wrong result, an entry the table does not explain, or a validation error.
//! Windows: tested (Intel Arc B580). Linux: untested.

#![allow(
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::struct_field_names,
    reason = "an example that reads top to bottom: sizes are small and known,               integers are reinterpreted on purpose (push constants, i8 data),               m, n, k are named as in the math"
)]

use std::ffi::{CStr, CString, c_char, c_void};
use std::sync::atomic::{AtomicUsize, Ordering};

use miraculix::formats::{MiniFloat, Overflow};
use numerobis::elem::{Acc, Elem, F16, F32, Format, I8, I32, U8};
use numerobis::matrix::vulkan::{Properties, compare};
use numerobis::matrix::{Engine, Scope};
use numerobis::vulkan::features::Chain;
use numerobis::vulkan::{ApiVersion, Capability, DeviceInfo, Offer, Probe};
use numerobis::{Choice, Device, GemmShape, ops};

type Void = unsafe extern "system" fn();
type GetInstanceProcAddr = unsafe extern "system" fn(*mut c_void, *const c_char) -> Option<Void>;
type Handle = *mut c_void;

const SUCCESS: i32 = 0;

#[cfg(windows)]
fn open_loader() -> Option<GetInstanceProcAddr> {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryA(name: *const c_char) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const c_char) -> Option<Void>;
    }
    // SAFETY: NUL-terminated names; the library stays loaded.
    unsafe {
        let module = LoadLibraryA(c"vulkan-1.dll".as_ptr());
        if module.is_null() {
            return None;
        }
        let f = GetProcAddress(module, c"vkGetInstanceProcAddr".as_ptr())?;
        Some(std::mem::transmute::<Void, GetInstanceProcAddr>(f))
    }
}

#[cfg(target_os = "linux")]
fn open_loader() -> Option<GetInstanceProcAddr> {
    // SAFETY: NUL-terminated names; the library stays loaded.
    unsafe {
        let module = libc::dlopen(
            c"libvulkan.so.1".as_ptr(),
            libc::RTLD_NOW | libc::RTLD_LOCAL,
        );
        if module.is_null() {
            return None;
        }
        let f = libc::dlsym(module, c"vkGetInstanceProcAddr".as_ptr());
        (!f.is_null()).then(|| std::mem::transmute::<*mut c_void, GetInstanceProcAddr>(f))
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
fn open_loader() -> Option<GetInstanceProcAddr> {
    None
}

/// Loads one entry point as `F`.
///
/// # Safety
///
/// `F` must be the function's real signature.
unsafe fn load<F: Copy>(gipa: GetInstanceProcAddr, instance: Handle, name: &CStr) -> Option<F> {
    assert_eq!(size_of::<F>(), size_of::<Void>());
    // SAFETY: the caller's promise about `F`; a fn pointer of the same size.
    unsafe {
        let f = gipa(instance, name.as_ptr())?;
        Some(std::mem::transmute_copy::<Void, F>(&f))
    }
}

/// The Vulkan entry points used here, loaded through the instance.
macro_rules! entry_points {
    ($($field:ident = $name:literal: fn($($arg:ty),*) $(-> $ret:ty)?;)*) => {
        struct Vk { $($field: unsafe extern "system" fn($($arg),*) $(-> $ret)?,)* }
        impl Vk {
            /// # Safety
            ///
            /// The signatures above are the real ones.
            unsafe fn load(gipa: GetInstanceProcAddr, instance: Handle) -> Self {
                // SAFETY: the caller's promise.
                unsafe { Self { $($field: load(gipa, instance, $name).expect(stringify!($field)),)* } }
            }
        }
    };
}

entry_points! {
    enumerate_physical_devices = c"vkEnumeratePhysicalDevices": fn(Handle, *mut u32, *mut Handle) -> i32;
    physical_properties = c"vkGetPhysicalDeviceProperties": fn(Handle, *mut c_void);
    physical_features2 = c"vkGetPhysicalDeviceFeatures2": fn(Handle, *mut c_void);
    physical_properties2 = c"vkGetPhysicalDeviceProperties2": fn(Handle, *mut c_void);
    physical_memory2 = c"vkGetPhysicalDeviceMemoryProperties2": fn(Handle, *mut c_void);
    physical_memory = c"vkGetPhysicalDeviceMemoryProperties": fn(Handle, *mut MemoryProperties);
    queue_families = c"vkGetPhysicalDeviceQueueFamilyProperties": fn(Handle, *mut u32, *mut QueueFamily);
    device_extensions = c"vkEnumerateDeviceExtensionProperties": fn(Handle, *const c_char, *mut u32, *mut ExtensionProperties) -> i32;
    create_device = c"vkCreateDevice": fn(Handle, *const DeviceCreateInfo, *const c_void, *mut Handle) -> i32;
    destroy_device = c"vkDestroyDevice": fn(Handle, *const c_void);
    get_device_queue = c"vkGetDeviceQueue": fn(Handle, u32, u32, *mut Handle);
    create_buffer = c"vkCreateBuffer": fn(Handle, *const BufferCreateInfo, *const c_void, *mut u64) -> i32;
    destroy_buffer = c"vkDestroyBuffer": fn(Handle, u64, *const c_void);
    buffer_requirements = c"vkGetBufferMemoryRequirements": fn(Handle, u64, *mut MemoryRequirements);
    allocate_memory = c"vkAllocateMemory": fn(Handle, *const MemoryAllocateInfo, *const c_void, *mut u64) -> i32;
    free_memory = c"vkFreeMemory": fn(Handle, u64, *const c_void);
    bind_buffer_memory = c"vkBindBufferMemory": fn(Handle, u64, u64, u64) -> i32;
    map_memory = c"vkMapMemory": fn(Handle, u64, u64, u64, u32, *mut *mut c_void) -> i32;
    create_set_layout = c"vkCreateDescriptorSetLayout": fn(Handle, *const SetLayoutCreateInfo, *const c_void, *mut u64) -> i32;
    destroy_set_layout = c"vkDestroyDescriptorSetLayout": fn(Handle, u64, *const c_void);
    create_pipeline_layout = c"vkCreatePipelineLayout": fn(Handle, *const PipelineLayoutCreateInfo, *const c_void, *mut u64) -> i32;
    destroy_pipeline_layout = c"vkDestroyPipelineLayout": fn(Handle, u64, *const c_void);
    create_shader_module = c"vkCreateShaderModule": fn(Handle, *const ShaderModuleCreateInfo, *const c_void, *mut u64) -> i32;
    destroy_shader_module = c"vkDestroyShaderModule": fn(Handle, u64, *const c_void);
    create_compute_pipelines = c"vkCreateComputePipelines": fn(Handle, u64, u32, *const ComputePipelineCreateInfo, *const c_void, *mut u64) -> i32;
    destroy_pipeline = c"vkDestroyPipeline": fn(Handle, u64, *const c_void);
    create_descriptor_pool = c"vkCreateDescriptorPool": fn(Handle, *const DescriptorPoolCreateInfo, *const c_void, *mut u64) -> i32;
    destroy_descriptor_pool = c"vkDestroyDescriptorPool": fn(Handle, u64, *const c_void);
    allocate_descriptor_sets = c"vkAllocateDescriptorSets": fn(Handle, *const DescriptorSetAllocateInfo, *mut u64) -> i32;
    update_descriptor_sets = c"vkUpdateDescriptorSets": fn(Handle, u32, *const WriteDescriptorSet, u32, *const c_void);
    create_command_pool = c"vkCreateCommandPool": fn(Handle, *const CommandPoolCreateInfo, *const c_void, *mut u64) -> i32;
    destroy_command_pool = c"vkDestroyCommandPool": fn(Handle, u64, *const c_void);
    allocate_command_buffers = c"vkAllocateCommandBuffers": fn(Handle, *const CommandBufferAllocateInfo, *mut Handle) -> i32;
    begin_command_buffer = c"vkBeginCommandBuffer": fn(Handle, *const CommandBufferBeginInfo) -> i32;
    end_command_buffer = c"vkEndCommandBuffer": fn(Handle) -> i32;
    cmd_bind_pipeline = c"vkCmdBindPipeline": fn(Handle, u32, u64);
    cmd_bind_descriptor_sets = c"vkCmdBindDescriptorSets": fn(Handle, u32, u64, u32, u32, *const u64, u32, *const u32);
    cmd_push_constants = c"vkCmdPushConstants": fn(Handle, u64, u32, u32, u32, *const c_void);
    cmd_dispatch = c"vkCmdDispatch": fn(Handle, u32, u32, u32);
    cmd_pipeline_barrier = c"vkCmdPipelineBarrier": fn(Handle, u32, u32, u32, u32, *const MemoryBarrier, u32, *const c_void, u32, *const c_void);
    cmd_reset_query_pool = c"vkCmdResetQueryPool": fn(Handle, u64, u32, u32);
    cmd_write_timestamp = c"vkCmdWriteTimestamp": fn(Handle, u32, u64, u32);
    create_query_pool = c"vkCreateQueryPool": fn(Handle, *const QueryPoolCreateInfo, *const c_void, *mut u64) -> i32;
    destroy_query_pool = c"vkDestroyQueryPool": fn(Handle, u64, *const c_void);
    query_results = c"vkGetQueryPoolResults": fn(Handle, u64, u32, u32, usize, *mut c_void, u64, u32) -> i32;
    queue_submit = c"vkQueueSubmit": fn(Handle, u32, *const SubmitInfo, u64) -> i32;
    create_fence = c"vkCreateFence": fn(Handle, *const FenceCreateInfo, *const c_void, *mut u64) -> i32;
    destroy_fence = c"vkDestroyFence": fn(Handle, u64, *const c_void);
    wait_for_fences = c"vkWaitForFences": fn(Handle, u32, *const u64, u32, u64) -> i32;
    reset_fences = c"vkResetFences": fn(Handle, u32, *const u64) -> i32;
    cooperative_matrices = c"vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR": fn(Handle, *mut u32, *mut Properties) -> i32;
}

#[repr(C)]
struct ExtensionProperties {
    name: [c_char; 256],
    spec_version: u32,
}

#[repr(C)]
struct ApplicationInfo {
    s_type: i32,
    p_next: *const c_void,
    application_name: *const c_char,
    application_version: u32,
    engine_name: *const c_char,
    engine_version: u32,
    api_version: u32,
}

#[repr(C)]
struct InstanceCreateInfo {
    s_type: i32,
    p_next: *const c_void,
    flags: u32,
    application_info: *const ApplicationInfo,
    layer_count: u32,
    layers: *const *const c_char,
    extension_count: u32,
    extensions: *const *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct QueueFamily {
    flags: u32,
    count: u32,
    timestamp_valid_bits: u32,
    min_image_transfer_granularity: [u32; 3],
}

#[repr(C)]
struct DeviceQueueCreateInfo {
    s_type: i32,
    p_next: *const c_void,
    flags: u32,
    family: u32,
    count: u32,
    priorities: *const f32,
}

#[repr(C)]
struct DeviceCreateInfo {
    s_type: i32,
    p_next: *const c_void,
    flags: u32,
    queue_info_count: u32,
    queue_infos: *const DeviceQueueCreateInfo,
    layer_count: u32,
    layers: *const *const c_char,
    extension_count: u32,
    extensions: *const *const c_char,
    enabled_features: *const c_void,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MemoryType {
    flags: u32,
    heap: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MemoryHeap {
    size: u64,
    flags: u32,
}

#[repr(C)]
struct MemoryProperties {
    type_count: u32,
    types: [MemoryType; 32],
    heap_count: u32,
    heaps: [MemoryHeap; 16],
}

#[repr(C)]
struct BufferCreateInfo {
    s_type: i32,
    p_next: *const c_void,
    flags: u32,
    size: u64,
    usage: u32,
    sharing_mode: u32,
    family_count: u32,
    families: *const u32,
}

#[repr(C)]
#[derive(Default)]
struct MemoryRequirements {
    size: u64,
    alignment: u64,
    type_bits: u32,
}

#[repr(C)]
struct MemoryAllocateInfo {
    s_type: i32,
    p_next: *const c_void,
    size: u64,
    type_index: u32,
}

#[repr(C)]
struct SetLayoutBinding {
    binding: u32,
    descriptor_type: u32,
    count: u32,
    stages: u32,
    samplers: *const c_void,
}

#[repr(C)]
struct SetLayoutCreateInfo {
    s_type: i32,
    p_next: *const c_void,
    flags: u32,
    binding_count: u32,
    bindings: *const SetLayoutBinding,
}

#[repr(C)]
struct PushConstantRange {
    stages: u32,
    offset: u32,
    size: u32,
}

#[repr(C)]
struct PipelineLayoutCreateInfo {
    s_type: i32,
    p_next: *const c_void,
    flags: u32,
    set_layout_count: u32,
    set_layouts: *const u64,
    push_range_count: u32,
    push_ranges: *const PushConstantRange,
}

#[repr(C)]
struct ShaderModuleCreateInfo {
    s_type: i32,
    p_next: *const c_void,
    flags: u32,
    code_size: usize,
    code: *const u32,
}

#[repr(C)]
struct SpecializationEntry {
    id: u32,
    offset: u32,
    size: usize,
}

#[repr(C)]
struct SpecializationInfo {
    entry_count: u32,
    entries: *const SpecializationEntry,
    data_size: usize,
    data: *const c_void,
}

#[repr(C)]
struct RequiredSubgroupSize {
    s_type: i32,
    p_next: *const c_void,
    size: u32,
}

#[repr(C)]
struct ShaderStageCreateInfo {
    s_type: i32,
    p_next: *const c_void,
    flags: u32,
    stage: u32,
    module: u64,
    name: *const c_char,
    specialization: *const SpecializationInfo,
}

#[repr(C)]
struct ComputePipelineCreateInfo {
    s_type: i32,
    p_next: *const c_void,
    flags: u32,
    stage: ShaderStageCreateInfo,
    layout: u64,
    base: u64,
    base_index: i32,
}

#[repr(C)]
struct DescriptorPoolSize {
    descriptor_type: u32,
    count: u32,
}

#[repr(C)]
struct DescriptorPoolCreateInfo {
    s_type: i32,
    p_next: *const c_void,
    flags: u32,
    max_sets: u32,
    size_count: u32,
    sizes: *const DescriptorPoolSize,
}

#[repr(C)]
struct DescriptorSetAllocateInfo {
    s_type: i32,
    p_next: *const c_void,
    pool: u64,
    count: u32,
    layouts: *const u64,
}

#[repr(C)]
struct DescriptorBufferInfo {
    buffer: u64,
    offset: u64,
    range: u64,
}

#[repr(C)]
struct WriteDescriptorSet {
    s_type: i32,
    p_next: *const c_void,
    set: u64,
    binding: u32,
    array_element: u32,
    count: u32,
    descriptor_type: u32,
    images: *const c_void,
    buffers: *const DescriptorBufferInfo,
    texel_views: *const c_void,
}

#[repr(C)]
struct CommandPoolCreateInfo {
    s_type: i32,
    p_next: *const c_void,
    flags: u32,
    family: u32,
}

#[repr(C)]
struct CommandBufferAllocateInfo {
    s_type: i32,
    p_next: *const c_void,
    pool: u64,
    level: u32,
    count: u32,
}

#[repr(C)]
struct CommandBufferBeginInfo {
    s_type: i32,
    p_next: *const c_void,
    flags: u32,
    inheritance: *const c_void,
}

#[repr(C)]
struct MemoryBarrier {
    s_type: i32,
    p_next: *const c_void,
    src_access: u32,
    dst_access: u32,
}

#[repr(C)]
struct QueryPoolCreateInfo {
    s_type: i32,
    p_next: *const c_void,
    flags: u32,
    query_type: u32,
    count: u32,
    statistics: u32,
}

#[repr(C)]
struct SubmitInfo {
    s_type: i32,
    p_next: *const c_void,
    wait_count: u32,
    waits: *const u64,
    wait_stages: *const u32,
    command_buffer_count: u32,
    command_buffers: *const Handle,
    signal_count: u32,
    signals: *const u64,
}

#[repr(C)]
struct FenceCreateInfo {
    s_type: i32,
    p_next: *const c_void,
    flags: u32,
}

#[repr(C)]
struct DebugCallbackData {
    s_type: i32,
    p_next: *const c_void,
    flags: u32,
    message_id_name: *const c_char,
    message_id_number: i32,
    message: *const c_char,
}

type DebugCallback =
    unsafe extern "system" fn(u32, u32, *const DebugCallbackData, *mut c_void) -> u32;

#[repr(C)]
struct DebugMessengerCreateInfo {
    s_type: i32,
    p_next: *const c_void,
    flags: u32,
    severity: u32,
    types: u32,
    callback: DebugCallback,
    user_data: *mut c_void,
}

static VALIDATION_ERRORS: AtomicUsize = AtomicUsize::new(0);

unsafe extern "system" fn on_message(
    severity: u32,
    _types: u32,
    data: *const DebugCallbackData,
    _user: *mut c_void,
) -> u32 {
    // SAFETY: the layer passes a valid callback data struct.
    let message = unsafe { CStr::from_ptr((*data).message) }.to_string_lossy();
    if severity & 0x1000 != 0 {
        VALIDATION_ERRORS.fetch_add(1, Ordering::Relaxed);
        eprintln!("VALIDATION ERROR: {message}");
    } else {
        eprintln!("validation warning: {message}");
    }
    0
}

const STORAGE_BUFFER: u32 = 7;
const COMPUTE_STAGE: u32 = 0x20;
const BIND_POINT_COMPUTE: u32 = 1;
const STAGE_TOP: u32 = 0x1;
const STAGE_COMPUTE: u32 = 0x800;
const STAGE_BOTTOM: u32 = 0x2000;
const ACCESS_SHADER_READ: u32 = 0x20;
const ACCESS_SHADER_WRITE: u32 = 0x40;
const HOST_VISIBLE: u32 = 0x2;
const HOST_COHERENT: u32 = 0x4;
const DEVICE_LOCAL: u32 = 0x1;
/// `VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT`.
const REQUIRE_FULL_SUBGROUPS: u32 = 0x2;
/// Blocking of `gemm_coopmat.comp` (`RM`, `RN`).
const RM: u32 = 4;
const RN: u32 = 2;

/// One SPIR-V of `kernels/vulkan/gemm_coopmat.comp` and what it computes.
struct Kernel {
    a: Format,
    acc: Format,
    spirv: &'static [u8],
}

const KERNELS: [Kernel; 4] = [
    Kernel {
        a: F16::FORMAT,
        acc: F32::FORMAT,
        spirv: include_bytes!("../kernels/bin/spirv/gemm_coopmat_f16_f16_f32.spv"),
    },
    Kernel {
        a: F16::FORMAT,
        acc: F16::FORMAT,
        spirv: include_bytes!("../kernels/bin/spirv/gemm_coopmat_f16_f16_f16.spv"),
    },
    Kernel {
        a: I8::FORMAT,
        acc: I32::FORMAT,
        spirv: include_bytes!("../kernels/bin/spirv/gemm_coopmat_i8_i8_i32.spv"),
    },
    Kernel {
        a: U8::FORMAT,
        acc: I32::FORMAT,
        spirv: include_bytes!("../kernels/bin/spirv/gemm_coopmat_u8_u8_i32.spv"),
    },
];

/// A small deterministic generator (xorshift).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// In `lo..=hi`.
    fn int(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % (hi - lo + 1) as u64) as i64
    }
}

fn f16_bits(v: f32) -> u16 {
    MiniFloat::F16.encode(v, Overflow::Saturate) as u16
}

fn bytes_of<T: Copy>(v: &[T]) -> Vec<u8> {
    // SAFETY: plain numbers, read as bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
        .to_vec()
}

/// A GEMM problem with host data: A, B, initial C as device bytes.
struct Problem {
    m: u32,
    n: u32,
    k: u32,
    a: Vec<u8>,
    b: Vec<u8>,
    c: Vec<u8>,
    /// `alpha`, `beta` as the push constants hold them (f32 or i32 bits).
    alpha: u32,
    beta: u32,
}

/// Inputs whose every partial sum is exact (in f32, f16 or i32), so any
/// summation order gives the same bits.
fn problem(kernel: &Kernel, m: u32, n: u32, k: u32, seed: u64) -> Problem {
    let mut rng = Rng(seed | 1);
    let (ma, mb, mc) = ((m * k) as usize, (n * k) as usize, (m * n) as usize);
    match (kernel.a, kernel.acc) {
        // Multiples of 1/4 in [-2, 2]: products are multiples of 1/16, sums
        // stay below 2^14 * 4: exact in f32 for K up to 2^16.
        (a, acc) if a == F16::FORMAT && acc == F32::FORMAT => {
            let mut q = |len: usize| -> Vec<u16> {
                (0..len)
                    .map(|_| f16_bits(rng.int(-8, 8) as f32 / 4.0))
                    .collect()
            };
            let (av, bv) = (q(ma), q(mb));
            let cv: Vec<f32> = (0..mc).map(|_| rng.int(-64, 64) as f32 / 8.0).collect();
            Problem {
                m,
                n,
                k,
                a: bytes_of(&av),
                b: bytes_of(&bv),
                c: bytes_of(&cv),
                alpha: 0.5f32.to_bits(),
                beta: 2.0f32.to_bits(),
            }
        }
        // -1, 0, 1: sums are integers up to K, exact in f16 up to 2048.
        (a, _) if a == F16::FORMAT => {
            let mut q = |len: usize| -> Vec<u16> {
                (0..len).map(|_| f16_bits(rng.int(-1, 1) as f32)).collect()
            };
            let (av, bv) = (q(ma), q(mb));
            Problem {
                m,
                n,
                k,
                a: bytes_of(&av),
                b: bytes_of(&bv),
                c: bytes_of(&vec![0u16; mc]),
                alpha: 1.0f32.to_bits(),
                beta: 0,
            }
        }
        (a, _) => {
            let signed = a == I8::FORMAT;
            let mut q = |len: usize| -> Vec<u8> {
                (0..len)
                    .map(|_| {
                        if signed {
                            rng.int(-128, 127) as i8 as u8
                        } else {
                            rng.int(0, 255) as u8
                        }
                    })
                    .collect()
            };
            let (av, bv) = (q(ma), q(mb));
            let cv: Vec<i32> = (0..mc).map(|_| rng.int(-1000, 1000) as i32).collect();
            Problem {
                m,
                n,
                k,
                a: av,
                b: bv,
                c: bytes_of(&cv),
                alpha: 3i32 as u32,
                beta: (-1i32) as u32,
            }
        }
    }
}

/// The mock backend's `ops::gemm` on the same problem: the reference bits.
fn reference(kernel: &Kernel, p: &Problem) -> numerobis::Result<Vec<u8>> {
    fn run<A: Elem, C: Acc>(
        p: &Problem,
        alpha: C::Scalar,
        beta: C::Scalar,
    ) -> numerobis::Result<Vec<u8>> {
        let dev = Device::mock();
        let mut s = dev.stream();
        let a = s.upload_packed::<A>((p.m * p.k) as usize, &p.a)?;
        let b = s.upload_packed::<A>((p.n * p.k) as usize, &p.b)?;
        let mut c = s.upload_packed::<C>((p.m * p.n) as usize, &p.c)?;
        let shape = GemmShape::new(p.m as usize, p.n as usize, p.k as usize);
        ops::gemm(&mut s, Choice::Up, shape, alpha, &a, &b, beta, &mut c)?.wait()?;
        Ok(s.download(&c)?.wait()?.as_bytes().to_vec())
    }
    let (fa, fb) = (f32::from_bits(p.alpha), f32::from_bits(p.beta));
    let (ia, ib) = (p.alpha as i32, p.beta as i32);
    match (kernel.a, kernel.acc) {
        (a, acc) if a == F16::FORMAT && acc == F32::FORMAT => run::<F16, F32>(p, fa, fb),
        (a, _) if a == F16::FORMAT => run::<F16, F16>(p, fa, fb),
        (a, _) if a == I8::FORMAT => run::<I8, I32>(p, ia, ib),
        _ => run::<U8, I32>(p, ia, ib),
    }
}

/// A device with a compute queue, and what the GEMMs need of it.
struct Gpu<'a> {
    vk: &'a Vk,
    device: Handle,
    queue: Handle,
    family: u32,
    memory: MemoryProperties,
    timestamp_period: f32,
    lanes: u32,
    /// Ask the pipeline for `lanes` (subgroup size control).
    require_lanes: bool,
}

struct Buffer {
    buffer: u64,
    memory: u64,
    ptr: *mut u8,
    size: u64,
}

impl Gpu<'_> {
    /// A host-visible buffer (device-local where the BAR allows).
    unsafe fn buffer(&self, size: u64) -> Buffer {
        let vk = self.vk;
        // SAFETY: valid create infos; the handles are this device's.
        unsafe {
            let info = BufferCreateInfo {
                s_type: 12,
                p_next: std::ptr::null(),
                flags: 0,
                size,
                usage: 0x20, // STORAGE_BUFFER
                sharing_mode: 0,
                family_count: 0,
                families: std::ptr::null(),
            };
            let mut buffer = 0;
            assert_eq!(
                (vk.create_buffer)(
                    self.device,
                    &raw const info,
                    std::ptr::null(),
                    &raw mut buffer
                ),
                SUCCESS
            );
            let mut req = MemoryRequirements::default();
            (vk.buffer_requirements)(self.device, buffer, &raw mut req);
            let types = &self.memory.types[..self.memory.type_count as usize];
            let pick = |want: u32| {
                types
                    .iter()
                    .enumerate()
                    .position(|(i, t)| req.type_bits & (1 << i) != 0 && t.flags & want == want)
            };
            let index = pick(HOST_VISIBLE | HOST_COHERENT | DEVICE_LOCAL)
                .or_else(|| pick(HOST_VISIBLE | HOST_COHERENT))
                .expect("host-visible memory");
            let alloc = MemoryAllocateInfo {
                s_type: 5,
                p_next: std::ptr::null(),
                size: req.size,
                type_index: index as u32,
            };
            let mut memory = 0;
            assert_eq!(
                (vk.allocate_memory)(
                    self.device,
                    &raw const alloc,
                    std::ptr::null(),
                    &raw mut memory
                ),
                SUCCESS,
                "vkAllocateMemory {size}"
            );
            assert_eq!(
                (vk.bind_buffer_memory)(self.device, buffer, memory, 0),
                SUCCESS
            );
            let mut ptr = std::ptr::null_mut();
            assert_eq!(
                (vk.map_memory)(self.device, memory, 0, u64::MAX, 0, &raw mut ptr),
                SUCCESS
            );
            Buffer {
                buffer,
                memory,
                ptr: ptr.cast(),
                size,
            }
        }
    }

    unsafe fn free(&self, b: &Buffer) {
        // SAFETY: no work uses `b` any more.
        unsafe {
            (self.vk.destroy_buffer)(self.device, b.buffer, std::ptr::null());
            (self.vk.free_memory)(self.device, b.memory, std::ptr::null());
        }
    }

    /// Runs `kernel` on `p` `repeat` times; returns C's bytes and the GPU
    /// time of one run in seconds.
    unsafe fn gemm(
        &self,
        kernel: &Kernel,
        tile: (u32, u32, u32),
        p: &Problem,
        repeat: u32,
    ) -> (Vec<u8>, f64) {
        let vk = self.vk;
        let dev = self.device;
        let null = std::ptr::null();
        // SAFETY: every struct handed over lives across its call; all
        // handles are this device's and destroyed after the fence.
        unsafe {
            let bufs = [
                self.buffer(p.a.len() as u64),
                self.buffer(p.b.len() as u64),
                self.buffer(p.c.len() as u64),
            ];
            for (b, data) in bufs.iter().zip([&p.a, &p.b, &p.c]) {
                std::ptr::copy_nonoverlapping(data.as_ptr(), b.ptr, data.len());
            }

            let bindings: Vec<SetLayoutBinding> = (0..3)
                .map(|i| SetLayoutBinding {
                    binding: i,
                    descriptor_type: STORAGE_BUFFER,
                    count: 1,
                    stages: COMPUTE_STAGE,
                    samplers: null,
                })
                .collect();
            let set_info = SetLayoutCreateInfo {
                s_type: 32,
                p_next: null,
                flags: 0,
                binding_count: 3,
                bindings: bindings.as_ptr(),
            };
            let mut set_layout = 0;
            assert_eq!(
                (vk.create_set_layout)(dev, &raw const set_info, null, &raw mut set_layout),
                SUCCESS
            );
            let range = PushConstantRange {
                stages: COMPUTE_STAGE,
                offset: 0,
                size: 20,
            };
            let layout_info = PipelineLayoutCreateInfo {
                s_type: 30,
                p_next: null,
                flags: 0,
                set_layout_count: 1,
                set_layouts: &raw const set_layout,
                push_range_count: 1,
                push_ranges: &raw const range,
            };
            let mut layout = 0;
            assert_eq!(
                (vk.create_pipeline_layout)(dev, &raw const layout_info, null, &raw mut layout),
                SUCCESS
            );
            let code: Vec<u32> = kernel
                .spirv
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&w| u32::from_le_bytes(w))
                .collect();
            let module_info = ShaderModuleCreateInfo {
                s_type: 16,
                p_next: null,
                flags: 0,
                code_size: kernel.spirv.len(),
                code: code.as_ptr(),
            };
            let mut module = 0;
            assert_eq!(
                (vk.create_shader_module)(dev, &raw const module_info, null, &raw mut module),
                SUCCESS
            );
            let constants = [tile.0, tile.1, tile.2, self.lanes];
            let entries: Vec<SpecializationEntry> = (0..4)
                .map(|i| SpecializationEntry {
                    id: i,
                    offset: i * 4,
                    size: 4,
                })
                .collect();
            let spec = SpecializationInfo {
                entry_count: 4,
                entries: entries.as_ptr(),
                data_size: 16,
                data: constants.as_ptr().cast(),
            };
            let required = RequiredSubgroupSize {
                s_type: 1_000_225_001,
                p_next: null,
                size: self.lanes,
            };
            let pipeline_info = ComputePipelineCreateInfo {
                s_type: 29,
                p_next: null,
                flags: 0,
                stage: ShaderStageCreateInfo {
                    s_type: 18,
                    p_next: if self.require_lanes {
                        (&raw const required).cast()
                    } else {
                        null
                    },
                    flags: if self.require_lanes {
                        REQUIRE_FULL_SUBGROUPS
                    } else {
                        0
                    },
                    stage: COMPUTE_STAGE,
                    module,
                    name: c"main".as_ptr(),
                    specialization: &raw const spec,
                },
                layout,
                base: 0,
                base_index: -1,
            };
            let mut pipeline = 0;
            assert_eq!(
                (vk.create_compute_pipelines)(
                    dev,
                    0,
                    1,
                    &raw const pipeline_info,
                    null,
                    &raw mut pipeline
                ),
                SUCCESS,
                "vkCreateComputePipelines"
            );

            let pool_size = DescriptorPoolSize {
                descriptor_type: STORAGE_BUFFER,
                count: 3,
            };
            let pool_info = DescriptorPoolCreateInfo {
                s_type: 33,
                p_next: null,
                flags: 0,
                max_sets: 1,
                size_count: 1,
                sizes: &raw const pool_size,
            };
            let mut pool = 0;
            assert_eq!(
                (vk.create_descriptor_pool)(dev, &raw const pool_info, null, &raw mut pool),
                SUCCESS
            );
            let alloc = DescriptorSetAllocateInfo {
                s_type: 34,
                p_next: null,
                pool,
                count: 1,
                layouts: &raw const set_layout,
            };
            let mut set = 0;
            assert_eq!(
                (vk.allocate_descriptor_sets)(dev, &raw const alloc, &raw mut set),
                SUCCESS
            );
            let infos: Vec<DescriptorBufferInfo> = bufs
                .iter()
                .map(|b| DescriptorBufferInfo {
                    buffer: b.buffer,
                    offset: 0,
                    range: b.size,
                })
                .collect();
            let writes: Vec<WriteDescriptorSet> = (0..3)
                .map(|i| WriteDescriptorSet {
                    s_type: 35,
                    p_next: null,
                    set,
                    binding: i,
                    array_element: 0,
                    count: 1,
                    descriptor_type: STORAGE_BUFFER,
                    images: null,
                    buffers: &raw const infos[i as usize],
                    texel_views: null,
                })
                .collect();
            (vk.update_descriptor_sets)(dev, 3, writes.as_ptr(), 0, null);

            let query_info = QueryPoolCreateInfo {
                s_type: 11,
                p_next: null,
                flags: 0,
                query_type: 2, // TIMESTAMP
                count: 2,
                statistics: 0,
            };
            let mut queries = 0;
            assert_eq!(
                (vk.create_query_pool)(dev, &raw const query_info, null, &raw mut queries),
                SUCCESS
            );
            let cmd_pool_info = CommandPoolCreateInfo {
                s_type: 39,
                p_next: null,
                flags: 0,
                family: self.family,
            };
            let mut cmd_pool = 0;
            assert_eq!(
                (vk.create_command_pool)(dev, &raw const cmd_pool_info, null, &raw mut cmd_pool),
                SUCCESS
            );
            let cb_info = CommandBufferAllocateInfo {
                s_type: 40,
                p_next: null,
                pool: cmd_pool,
                level: 0,
                count: 1,
            };
            let mut cb = std::ptr::null_mut();
            assert_eq!(
                (vk.allocate_command_buffers)(dev, &raw const cb_info, &raw mut cb),
                SUCCESS
            );
            let begin = CommandBufferBeginInfo {
                s_type: 42,
                p_next: null,
                flags: 1, // ONE_TIME_SUBMIT
                inheritance: null,
            };
            assert_eq!((vk.begin_command_buffer)(cb, &raw const begin), SUCCESS);
            (vk.cmd_reset_query_pool)(cb, queries, 0, 2);
            (vk.cmd_bind_pipeline)(cb, BIND_POINT_COMPUTE, pipeline);
            (vk.cmd_bind_descriptor_sets)(
                cb,
                BIND_POINT_COMPUTE,
                layout,
                0,
                1,
                &raw const set,
                0,
                std::ptr::null(),
            );
            let push = [p.m, p.n, p.k, p.alpha, p.beta];
            (vk.cmd_push_constants)(cb, layout, COMPUTE_STAGE, 0, 20, push.as_ptr().cast());
            (vk.cmd_write_timestamp)(cb, STAGE_TOP, queries, 0);
            let groups = (p.n / (RN * tile.1), p.m / (RM * tile.0));
            let barrier = MemoryBarrier {
                s_type: 46,
                p_next: null,
                src_access: ACCESS_SHADER_WRITE | ACCESS_SHADER_READ,
                dst_access: ACCESS_SHADER_WRITE | ACCESS_SHADER_READ,
            };
            for i in 0..repeat {
                if i > 0 {
                    // Same C again: beta must be 0 for repeats to agree.
                    (vk.cmd_pipeline_barrier)(
                        cb,
                        STAGE_COMPUTE,
                        STAGE_COMPUTE,
                        0,
                        1,
                        &raw const barrier,
                        0,
                        null,
                        0,
                        null,
                    );
                }
                (vk.cmd_dispatch)(cb, groups.0, groups.1, 1);
            }
            (vk.cmd_write_timestamp)(cb, STAGE_BOTTOM, queries, 1);
            assert_eq!((vk.end_command_buffer)(cb), SUCCESS);

            let fence_info = FenceCreateInfo {
                s_type: 8,
                p_next: null,
                flags: 0,
            };
            let mut fence = 0;
            assert_eq!(
                (vk.create_fence)(dev, &raw const fence_info, null, &raw mut fence),
                SUCCESS
            );
            let submit = SubmitInfo {
                s_type: 4,
                p_next: null,
                wait_count: 0,
                waits: std::ptr::null(),
                wait_stages: std::ptr::null(),
                command_buffer_count: 1,
                command_buffers: &raw const cb,
                signal_count: 0,
                signals: std::ptr::null(),
            };
            assert_eq!(
                (vk.queue_submit)(self.queue, 1, &raw const submit, fence),
                SUCCESS
            );
            assert_eq!(
                (vk.wait_for_fences)(dev, 1, &raw const fence, 1, 60_000_000_000),
                SUCCESS,
                "GEMM did not finish in 60 s"
            );
            let mut stamps = [0u64; 2];
            assert_eq!(
                (vk.query_results)(
                    dev,
                    queries,
                    0,
                    2,
                    16,
                    stamps.as_mut_ptr().cast(),
                    8,
                    0x1 | 0x2
                ),
                SUCCESS
            );
            let seconds = (stamps[1] - stamps[0]) as f64 * f64::from(self.timestamp_period)
                / 1e9
                / f64::from(repeat);
            let out = std::slice::from_raw_parts(bufs[2].ptr, p.c.len()).to_vec();

            assert_eq!((vk.reset_fences)(dev, 1, &raw const fence), SUCCESS);
            (vk.destroy_fence)(dev, fence, null);
            (vk.destroy_command_pool)(dev, cmd_pool, null);
            (vk.destroy_query_pool)(dev, queries, null);
            (vk.destroy_descriptor_pool)(dev, pool, null);
            (vk.destroy_pipeline)(dev, pipeline, null);
            (vk.destroy_shader_module)(dev, module, null);
            (vk.destroy_pipeline_layout)(dev, layout, null);
            (vk.destroy_set_layout)(dev, set_layout, null);
            for b in &bufs {
                self.free(b);
            }
            (out, seconds)
        }
    }
}

/// C[i][j] of `gemm` computed on the host (exact for [`problem`]'s inputs).
fn entry(kernel: &Kernel, p: &Problem, i: usize, j: usize) -> Vec<u8> {
    let k = p.k as usize;
    let (n, m) = (p.n as usize, p.m as usize);
    let _ = m;
    if kernel.a == F16::FORMAT {
        let read = |v: &[u8], at: usize| {
            MiniFloat::F16.decode(u32::from(u16::from_le_bytes([v[2 * at], v[2 * at + 1]])))
        };
        let sum: f32 = (0..k)
            .map(|x| read(&p.a, i * k + x) * read(&p.b, j * k + x))
            .sum();
        if kernel.acc == F32::FORMAT {
            let c = f32::from_le_bytes(p.c[4 * (i * n + j)..][..4].try_into().expect("4"));
            let d = f32::from_bits(p.alpha) * sum + f32::from_bits(p.beta) * c;
            d.to_le_bytes().to_vec()
        } else {
            f16_bits(sum).to_le_bytes().to_vec()
        }
    } else {
        let signed = kernel.a == I8::FORMAT;
        let read = |v: &[u8], at: usize| {
            if signed {
                i32::from(v[at] as i8)
            } else {
                i32::from(v[at])
            }
        };
        let sum: i32 = (0..k)
            .map(|x| read(&p.a, i * k + x) * read(&p.b, j * k + x))
            .sum();
        let c = i32::from_le_bytes(p.c[4 * (i * n + j)..][..4].try_into().expect("4"));
        let d = (p.alpha as i32)
            .wrapping_mul(sum)
            .wrapping_add((p.beta as i32).wrapping_mul(c));
        d.to_le_bytes().to_vec()
    }
}

fn main() {
    let Some(gipa) = open_loader() else {
        println!("no Vulkan loader on this system");
        return;
    };
    let null = std::ptr::null_mut();
    let timed: u32 = std::env::var("NUMEROBIS_MATRIX_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096);
    let mut failed = false;
    // SAFETY: every entry point is loaded with its real signature; every
    // struct handed to Vulkan lives across the call.
    unsafe {
        type CreateInstance =
            unsafe extern "system" fn(*const InstanceCreateInfo, *const c_void, *mut Handle) -> i32;
        type EnumerateLayers = unsafe extern "system" fn(*mut u32, *mut [u8; 520]) -> i32;
        let layers = load::<EnumerateLayers>(gipa, null, c"vkEnumerateInstanceLayerProperties")
            .map_or_else(Vec::new, |f| {
                let mut n = 0;
                f(&raw mut n, null.cast());
                let mut v = vec![[0u8; 520]; n as usize];
                f(&raw mut n, v.as_mut_ptr());
                v
            });
        let validation = std::env::var_os("NUMEROBIS_NO_VALIDATION").is_none()
            && layers
                .iter()
                .any(|l| CStr::from_ptr(l.as_ptr().cast()) == c"VK_LAYER_KHRONOS_validation");
        let layer_ptrs: Vec<*const c_char> = if validation {
            vec![c"VK_LAYER_KHRONOS_validation".as_ptr()]
        } else {
            Vec::new()
        };
        let extension_ptrs: Vec<*const c_char> = if validation {
            vec![c"VK_EXT_debug_utils".as_ptr()]
        } else {
            Vec::new()
        };
        let instance_api = ApiVersion::V1_3;
        let app = ApplicationInfo {
            s_type: 0,
            p_next: std::ptr::null(),
            application_name: c"numerobis matrix_probe".as_ptr(),
            application_version: 1,
            engine_name: c"numerobis".as_ptr(),
            engine_version: 1,
            api_version: instance_api.raw(),
        };
        let create_info = InstanceCreateInfo {
            s_type: 1,
            p_next: std::ptr::null(),
            flags: 0,
            application_info: &raw const app,
            layer_count: layer_ptrs.len() as u32,
            layers: layer_ptrs.as_ptr(),
            extension_count: extension_ptrs.len() as u32,
            extensions: extension_ptrs.as_ptr(),
        };
        let mut instance = null;
        let r = load::<CreateInstance>(gipa, null, c"vkCreateInstance").expect("vkCreateInstance")(
            &raw const create_info,
            std::ptr::null(),
            &raw mut instance,
        );
        if r != SUCCESS {
            println!("vkCreateInstance (1.3) failed: {r}");
            std::process::exit(1);
        }
        println!(
            "instance {instance_api}, validation {}",
            if validation { "on" } else { "off" }
        );
        type CreateMessenger = unsafe extern "system" fn(
            Handle,
            *const DebugMessengerCreateInfo,
            *const c_void,
            *mut u64,
        ) -> i32;
        type DestroyMessenger = unsafe extern "system" fn(Handle, u64, *const c_void);
        let messenger = validation.then(|| {
            let info = DebugMessengerCreateInfo {
                s_type: 1_000_128_004,
                p_next: std::ptr::null(),
                flags: 0,
                severity: 0x100 | 0x1000,
                types: 0x1 | 0x2,
                callback: on_message,
                user_data: null,
            };
            let mut handle = 0;
            load::<CreateMessenger>(gipa, instance, c"vkCreateDebugUtilsMessengerEXT")
                .expect("debug utils")(
                instance,
                &raw const info,
                std::ptr::null(),
                &raw mut handle,
            );
            handle
        });
        let vk = Vk::load(gipa, instance);

        let mut count = 0;
        (vk.enumerate_physical_devices)(instance, &raw mut count, std::ptr::null_mut());
        let mut physical = vec![null; count as usize];
        (vk.enumerate_physical_devices)(instance, &raw mut count, physical.as_mut_ptr());

        for &pd in &physical {
            let mut properties = [0u64; 128];
            (vk.physical_properties)(pd, properties.as_mut_ptr().cast());
            let info = DeviceInfo::from_properties(properties.as_ptr().cast()).expect("filled");
            // VkPhysicalDeviceLimits::timestampPeriod, at 296 + 424 with a
            // 64-bit size_t; 32-bit builds skip the timing.
            let raw: &[u8] = std::slice::from_raw_parts(properties.as_ptr().cast(), 1024);
            let timestamp_period = if cfg!(target_pointer_width = "64") {
                f32::from_le_bytes(raw[720..724].try_into().expect("4"))
            } else {
                0.0
            };
            let engine = Engine::from_gpu(info.gpu());
            println!(
                "\n{} ({}): matrix engine {}",
                info.name,
                info.gpu()
                    .architecture
                    .name()
                    .unwrap_or("unknown architecture"),
                match engine {
                    Some(e) => e.to_string(),
                    None if Engine::known_absent(info.gpu()) => "none".to_string(),
                    None => "unknown".to_string(),
                }
            );

            let mut n = 0;
            (vk.device_extensions)(pd, std::ptr::null(), &raw mut n, std::ptr::null_mut());
            let mut ext = (0..n)
                .map(|_| ExtensionProperties {
                    name: [0; 256],
                    spec_version: 0,
                })
                .collect::<Vec<_>>();
            (vk.device_extensions)(pd, std::ptr::null(), &raw mut n, ext.as_mut_ptr());
            let extensions: Vec<CString> = ext
                .iter()
                .map(|e| CStr::from_ptr(e.name.as_ptr()).to_owned())
                .collect();
            let offer = Offer::new(
                instance_api,
                info.api.unwrap_or(ApiVersion::V1_0),
                extensions.iter().map(CString::as_c_str),
            );
            let wanted = [
                Capability::CooperativeMatrix,
                Capability::VulkanMemoryModel,
                Capability::Float16,
                Capability::Int8,
                Capability::Storage16Bit,
                Capability::Storage8Bit,
                // The shader's subgroup-sized workgroup is `LocalSizeId`.
                Capability::Maintenance4,
                Capability::SubgroupSizeControl,
            ];
            let mut probe = Probe::new(offer, &wanted);
            (vk.physical_features2)(pd, probe.features2());
            (vk.physical_properties2)(pd, probe.properties2());
            (vk.physical_memory2)(pd, probe.memory_properties2());
            let plan = probe.plan();
            let lacking = plan.lacking(&wanted[..7]);
            if !lacking.is_empty() {
                let names: Vec<&str> = lacking.iter().map(|c| c.name()).collect();
                println!("  skipped: lacks {}", names.join(", "));
                continue;
            }

            // 1. The driver's cooperative matrices next to the engine table.
            let mut n = 0;
            (vk.cooperative_matrices)(pd, &raw mut n, std::ptr::null_mut());
            let mut reported = vec![Properties::empty(); n as usize];
            (vk.cooperative_matrices)(pd, &raw mut n, reported.as_mut_ptr());
            // The subgroup the engine's instructions use; the default otherwise.
            let default_lanes = plan.subgroup().map_or(32, |s| s.size);
            let lanes = engine
                .and_then(|e| e.mmas().next())
                .and_then(|m| match m.scope {
                    Scope::Subgroup { lanes } => Some(lanes),
                    _ => None,
                })
                .unwrap_or(default_lanes);
            let mmas: Vec<_> = reported.iter().filter_map(|p| p.mma(lanes)).collect();
            println!(
                "  driver: {} cooperative matrices ({} in the engine model)",
                reported.len(),
                mmas.len()
            );
            if let Some(engine) = engine {
                let c = compare(engine, &mmas);
                println!("  {}", c.to_string().trim_end().replace('\n', "\n  "));
                if !c.unexplained.is_empty() {
                    failed = true;
                }
            } else {
                for m in &mmas {
                    println!("  {m}");
                }
            }

            // 2. Device with a compute queue and the features the shader needs.
            let mut families_n = 0;
            (vk.queue_families)(pd, &raw mut families_n, std::ptr::null_mut());
            let mut families = vec![QueueFamily::default(); families_n as usize];
            (vk.queue_families)(pd, &raw mut families_n, families.as_mut_ptr());
            let Some(family) = families
                .iter()
                .position(|f| f.flags & 0x2 != 0 && f.timestamp_valid_bits > 0)
            else {
                println!("  skipped: no compute queue with timestamps");
                continue;
            };
            let mut chain = plan.device_chain();
            let require_lanes = plan.has(Capability::SubgroupSizeControl)
                && lanes != default_lanes
                && plan.supported(Chain::Vulkan13, "computeFullSubgroups") == Some(true)
                && chain
                    .enable(Chain::Vulkan13, "computeFullSubgroups")
                    .is_ok();
            let names: Vec<*const c_char> = plan.extensions().iter().map(|e| e.as_ptr()).collect();
            let priority = 1.0f32;
            let queue_info = DeviceQueueCreateInfo {
                s_type: 2,
                p_next: std::ptr::null(),
                flags: 0,
                family: family as u32,
                count: 1,
                priorities: &raw const priority,
            };
            let device_info = DeviceCreateInfo {
                s_type: 3,
                p_next: chain.features2().cast_const(),
                flags: 0,
                queue_info_count: 1,
                queue_infos: &raw const queue_info,
                layer_count: 0,
                layers: std::ptr::null(),
                extension_count: names.len() as u32,
                extensions: names.as_ptr(),
                enabled_features: std::ptr::null(),
            };
            let mut device = null;
            let r = (vk.create_device)(
                pd,
                &raw const device_info,
                std::ptr::null(),
                &raw mut device,
            );
            if r != SUCCESS {
                println!("  vkCreateDevice FAILED: {r}");
                failed = true;
                continue;
            }
            let mut queue = null;
            (vk.get_device_queue)(device, family as u32, 0, &raw mut queue);
            let mut memory: MemoryProperties = std::mem::zeroed();
            (vk.physical_memory)(pd, &raw mut memory);
            let gpu = Gpu {
                vk: &vk,
                device,
                queue,
                family: family as u32,
                memory,
                timestamp_period,
                lanes,
                require_lanes,
            };
            println!(
                "  pipelines: subgroup of {lanes}{}",
                if require_lanes {
                    " (required, full subgroups)"
                } else {
                    ""
                }
            );

            // 3. Each reported type the shader has, against the mock oracle.
            for kernel in &KERNELS {
                let Some(m) = mmas
                    .iter()
                    .find(|m| m.a == kernel.a && m.b == kernel.a && m.acc == kernel.acc)
                else {
                    continue;
                };
                let tile = (m.shape.m, m.shape.n, m.shape.k);
                let (bm, bn) = (RM * tile.0, RN * tile.1);
                let size = |base: u32, step: u32| base.div_ceil(step) * step;
                let p = problem(kernel, size(256, bm), size(192, bn), size(320, tile.2), 7);
                let (got, _) = gpu.gemm(kernel, tile, &p, 1);
                let want = reference(kernel, &p).expect("mock gemm");
                let name = format!(
                    "gemm_{}_{}_{}",
                    kernel.a.name(),
                    kernel.a.name(),
                    kernel.acc.name()
                );
                let width = got.len() / (p.m * p.n) as usize;
                let wrong = got
                    .chunks(width)
                    .zip(want.chunks(width))
                    .filter(|(g, w)| g != w)
                    .count();
                println!(
                    "  {name} {}x{}x{} on {} tiles: {}",
                    p.m,
                    p.n,
                    p.k,
                    m.shape,
                    if wrong == 0 {
                        "bit-identical to the mock backend".to_string()
                    } else {
                        failed = true;
                        format!("{wrong} of {} entries WRONG", p.m * p.n)
                    }
                );
            }

            // 4. Throughput at a larger size (beta = 0 so repeats agree).
            let timed_kernels = KERNELS
                .iter()
                .filter(|k| k.acc != F16::FORMAT && k.a != U8::FORMAT && timestamp_period > 0.0);
            for kernel in timed_kernels {
                let Some(m) = mmas
                    .iter()
                    .find(|m| m.a == kernel.a && m.b == kernel.a && m.acc == kernel.acc)
                else {
                    continue;
                };
                let tile = (m.shape.m, m.shape.n, m.shape.k);
                let s = timed.div_ceil(256) * 256;
                let mut p = problem(kernel, s, s, s, 11);
                p.beta = 0;
                p.c.fill(0);
                gpu.gemm(kernel, tile, &p, 1); // warm-up: pipeline, caches, clocks
                let (got, seconds) = gpu.gemm(kernel, tile, &p, 10);
                let mut rng = Rng(99);
                let width = got.len() / (s * s) as usize;
                let mut wrong = 0;
                for _ in 0..256 {
                    let (i, j) = (
                        rng.int(0, i64::from(s) - 1) as usize,
                        rng.int(0, i64::from(s) - 1) as usize,
                    );
                    let at = (i * s as usize + j) * width;
                    if got[at..at + width] != entry(kernel, &p, i, j)[..] {
                        wrong += 1;
                    }
                }
                if wrong > 0 {
                    failed = true;
                }
                let ops = 2.0 * f64::from(s).powi(3);
                println!(
                    "  timed gemm_{0}_{0}_{1} {s}x{s}x{s}: {2:.2} ms, {3:.1} {4} ({5})",
                    kernel.a.name(),
                    kernel.acc.name(),
                    seconds * 1e3,
                    ops / seconds / 1e12,
                    if kernel.a.is_integer() {
                        "TOPS"
                    } else {
                        "TFLOPS"
                    },
                    if wrong == 0 {
                        "256 sampled entries exact".to_string()
                    } else {
                        format!("{wrong} of 256 sampled entries WRONG")
                    }
                );
            }
            (vk.destroy_device)(device, std::ptr::null());
        }

        if let Some(handle) = messenger {
            load::<DestroyMessenger>(gipa, instance, c"vkDestroyDebugUtilsMessengerEXT")
                .expect("debug utils")(instance, handle, std::ptr::null());
        }
        type DestroyInstance = unsafe extern "system" fn(Handle, *const c_void);
        load::<DestroyInstance>(gipa, instance, c"vkDestroyInstance").expect("vkDestroyInstance")(
            instance,
            std::ptr::null(),
        );
    }
    let errors = VALIDATION_ERRORS.load(Ordering::Relaxed);
    println!("\nvalidation errors: {errors}");
    if failed || errors > 0 {
        std::process::exit(1);
    }
}

//! Runs the Vulkan waterfall against the installed driver: every capability
//! is planned, the device is created with the plan's feature chain and
//! extensions (under the validation layer when it is installed), and the
//! memory Vulkan reports is set next to the PCI side from `pci::bar`.
//!
//! `cargo run --example vulkan_probe` (`NUMEROBIS_NO_VALIDATION=1` skips the
//! layer, `NUMEROBIS_VK_API=1.1` caps the instance version so the extension
//! routes are used). Exits non-zero on a validation error or a failed `vkCreateDevice`.
//!
//! No bindings: the loader is opened by hand (`vulkan-1.dll` on Windows,
//! `libvulkan.so.1` on Linux). Windows: tested (Intel Arc B580). Linux:
//! untested.

#![allow(
    clippy::too_many_lines,
    reason = "an example that reads top to bottom, one Vulkan call after another"
)]

use std::ffi::{CStr, CString, c_char, c_void};
use std::sync::atomic::{AtomicUsize, Ordering};

use numerobis::pci::bar;
use numerobis::vulkan::{ApiVersion, Capability, DeviceInfo, FeatureQuery, Instance, Offer, Probe};

type Void = unsafe extern "system" fn();
type GetInstanceProcAddr = unsafe extern "system" fn(*mut c_void, *const c_char) -> Option<Void>;

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

#[repr(C)]
struct ExtensionProperties {
    name: [c_char; 256],
    spec_version: u32,
}

#[repr(C)]
struct LayerProperties {
    name: [c_char; 256],
    spec_version: u32,
    implementation_version: u32,
    description: [c_char; 256],
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
struct QueueFamilyProperties {
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

const SEVERITY_WARNING: u32 = 0x100;
const SEVERITY_ERROR: u32 = 0x1000;
const TYPE_GENERAL: u32 = 0x1;
const TYPE_VALIDATION: u32 = 0x2;

static VALIDATION_ERRORS: AtomicUsize = AtomicUsize::new(0);

unsafe extern "system" fn on_message(
    severity: u32,
    _types: u32,
    data: *const DebugCallbackData,
    _user: *mut c_void,
) -> u32 {
    // SAFETY: the layer passes a valid callback data struct.
    let message = unsafe { CStr::from_ptr((*data).message) }.to_string_lossy();
    if severity & SEVERITY_ERROR != 0 {
        VALIDATION_ERRORS.fetch_add(1, Ordering::Relaxed);
        eprintln!("VALIDATION ERROR: {message}");
    } else {
        eprintln!("validation warning: {message}");
    }
    0
}

/// Loads one entry point as `F`.
///
/// # Safety
///
/// `F` must be the function's real signature.
unsafe fn load<F: Copy>(
    gipa: GetInstanceProcAddr,
    instance: *mut c_void,
    name: &CStr,
) -> Option<F> {
    assert_eq!(size_of::<F>(), size_of::<Void>());
    // SAFETY: the caller's promise about `F`; a fn pointer of the same size.
    unsafe {
        let f = gipa(instance, name.as_ptr())?;
        Some(std::mem::transmute_copy::<Void, F>(&f))
    }
}

type EnumerateInstanceVersion = unsafe extern "system" fn(*mut u32) -> i32;
type EnumerateInstanceExtensions =
    unsafe extern "system" fn(*const c_char, *mut u32, *mut ExtensionProperties) -> i32;
type EnumerateInstanceLayers = unsafe extern "system" fn(*mut u32, *mut LayerProperties) -> i32;
type CreateInstance =
    unsafe extern "system" fn(*const InstanceCreateInfo, *const c_void, *mut *mut c_void) -> i32;
type DestroyInstance = unsafe extern "system" fn(*mut c_void, *const c_void);
type EnumeratePhysicalDevices =
    unsafe extern "system" fn(*mut c_void, *mut u32, *mut *mut c_void) -> i32;
/// `vkGetPhysicalDevice{Features,Properties,MemoryProperties}[2]`.
type PhysicalDeviceQuery = unsafe extern "system" fn(*mut c_void, *mut c_void);
type EnumerateDeviceExtensions = unsafe extern "system" fn(
    *mut c_void,
    *const c_char,
    *mut u32,
    *mut ExtensionProperties,
) -> i32;
type GetQueueFamilies =
    unsafe extern "system" fn(*mut c_void, *mut u32, *mut QueueFamilyProperties);
type CreateDevice = unsafe extern "system" fn(
    *mut c_void,
    *const DeviceCreateInfo,
    *const c_void,
    *mut *mut c_void,
) -> i32;
type DestroyDevice = unsafe extern "system" fn(*mut c_void, *const c_void);
type CreateDebugMessenger = unsafe extern "system" fn(
    *mut c_void,
    *const DebugMessengerCreateInfo,
    *const c_void,
    *mut u64,
) -> i32;
type DestroyDebugMessenger = unsafe extern "system" fn(*mut c_void, u64, *const c_void);

fn name_of(raw: &[c_char; 256]) -> CString {
    // SAFETY: Vulkan NUL-terminates names within the array.
    unsafe { CStr::from_ptr(raw.as_ptr()) }.to_owned()
}

/// Calls a two-step enumeration (count, then fill).
fn enumerate<T>(mut call: impl FnMut(*mut u32, *mut T) -> i32, empty: impl Fn() -> T) -> Vec<T> {
    let mut count = 0;
    if call(&raw mut count, std::ptr::null_mut()) != SUCCESS {
        return Vec::new();
    }
    let mut out: Vec<T> = (0..count).map(|_| empty()).collect();
    if call(&raw mut count, out.as_mut_ptr()) < SUCCESS {
        return Vec::new();
    }
    out.truncate(count as usize);
    out
}

fn empty_extension() -> ExtensionProperties {
    ExtensionProperties {
        name: [0; 256],
        spec_version: 0,
    }
}

fn main() {
    let Some(gipa) = open_loader() else {
        println!("no Vulkan loader on this system");
        return;
    };
    let null = std::ptr::null_mut();
    // SAFETY: every `load` below names the entry point's real signature;
    // every struct handed to Vulkan is alive for the call.
    let failed = unsafe {
        let loader = load::<EnumerateInstanceVersion>(gipa, null, c"vkEnumerateInstanceVersion")
            .and_then(|f| {
                let mut v = 0;
                (f(&raw mut v) == SUCCESS)
                    .then(|| ApiVersion::from_raw(v))
                    .flatten()
            });
        let enumerate_extensions = load::<EnumerateInstanceExtensions>(
            gipa,
            null,
            c"vkEnumerateInstanceExtensionProperties",
        )
        .expect("vkEnumerateInstanceExtensionProperties");
        let available: Vec<CString> = enumerate(
            |n, p| enumerate_extensions(std::ptr::null(), n, p),
            empty_extension,
        )
        .iter()
        .map(|e| name_of(&e.name))
        .collect();
        let has = |name: &CStr| available.iter().any(|e| e.as_c_str() == name);
        let layers: Vec<CString> =
            load::<EnumerateInstanceLayers>(gipa, null, c"vkEnumerateInstanceLayerProperties")
                .map_or_else(Vec::new, |f| {
                    enumerate(
                        |n, p| f(n, p),
                        || LayerProperties {
                            name: [0; 256],
                            spec_version: 0,
                            implementation_version: 0,
                            description: [0; 256],
                        },
                    )
                })
                .iter()
                .map(|l| name_of(&l.name))
                .collect();
        let validation = std::env::var_os("NUMEROBIS_NO_VALIDATION").is_none()
            && layers
                .iter()
                .any(|l| l.as_c_str() == c"VK_LAYER_KHRONOS_validation");

        // NUMEROBIS_VK_API=1.1 caps the instance version: the extension
        // routes then take over from the core ones.
        let ceiling = std::env::var("NUMEROBIS_VK_API")
            .ok()
            .and_then(|v| {
                let (major, minor) = v.split_once('.')?;
                Some(ApiVersion::new(major.parse().ok()?, minor.parse().ok()?))
            })
            .unwrap_or(ApiVersion::NEWEST);
        let instance_setup =
            Instance::new(loader, ceiling, available.iter().map(CString::as_c_str));
        // The host's own: debug messages, and the surface extensions the
        // present_id2 / present_wait2 routes depend on.
        let mut instance_extensions: Vec<&CStr> = instance_setup.extensions().to_vec();
        let debug = validation && has(c"VK_EXT_debug_utils");
        for (name, wanted) in [
            (c"VK_EXT_debug_utils", debug),
            (c"VK_KHR_surface", has(c"VK_KHR_surface")),
            (
                c"VK_KHR_get_surface_capabilities2",
                has(c"VK_KHR_get_surface_capabilities2"),
            ),
        ] {
            if wanted {
                instance_extensions.push(name);
            }
        }
        let extension_ptrs: Vec<*const c_char> =
            instance_extensions.iter().map(|e| e.as_ptr()).collect();
        let layer_ptrs: Vec<*const c_char> = if validation {
            vec![c"VK_LAYER_KHRONOS_validation".as_ptr()]
        } else {
            Vec::new()
        };
        let app = ApplicationInfo {
            s_type: 0,
            p_next: std::ptr::null(),
            application_name: c"numerobis vulkan_probe".as_ptr(),
            application_version: 1,
            engine_name: c"numerobis".as_ptr(),
            engine_version: 1,
            api_version: instance_setup.api.raw(),
        };
        let create_info = InstanceCreateInfo {
            s_type: 1,
            p_next: std::ptr::null(),
            flags: instance_setup.flags,
            application_info: &raw const app,
            layer_count: u32::try_from(layer_ptrs.len()).expect("few layers"),
            layers: layer_ptrs.as_ptr(),
            extension_count: u32::try_from(extension_ptrs.len()).expect("few extensions"),
            extensions: extension_ptrs.as_ptr(),
        };
        let create_instance =
            load::<CreateInstance>(gipa, null, c"vkCreateInstance").expect("vkCreateInstance");
        let mut instance = null;
        let r = create_instance(&raw const create_info, std::ptr::null(), &raw mut instance);
        if r != SUCCESS {
            println!("vkCreateInstance failed: {r}");
            std::process::exit(1);
        }
        println!(
            "loader {}, instance {}, validation {}, instance extensions: {}",
            loader.map_or_else(|| "1.0".into(), |v| v.to_string()),
            instance_setup.api,
            if validation { "on" } else { "off" },
            instance_extensions
                .iter()
                .map(|e| e.to_string_lossy())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let messenger = if debug {
            let create =
                load::<CreateDebugMessenger>(gipa, instance, c"vkCreateDebugUtilsMessengerEXT")
                    .expect("debug utils");
            let info = DebugMessengerCreateInfo {
                s_type: 1_000_128_004,
                p_next: std::ptr::null(),
                flags: 0,
                severity: SEVERITY_WARNING | SEVERITY_ERROR,
                types: TYPE_GENERAL | TYPE_VALIDATION,
                callback: on_message,
                user_data: null,
            };
            let mut handle = 0;
            (create(instance, &raw const info, std::ptr::null(), &raw mut handle) == SUCCESS)
                .then_some(handle)
        } else {
            None
        };

        let get = |name: &CStr| load::<PhysicalDeviceQuery>(gipa, instance, name);
        let enumerate_devices =
            load::<EnumeratePhysicalDevices>(gipa, instance, c"vkEnumeratePhysicalDevices")
                .expect("vkEnumeratePhysicalDevices");
        let device_extensions = load::<EnumerateDeviceExtensions>(
            gipa,
            instance,
            c"vkEnumerateDeviceExtensionProperties",
        )
        .expect("vkEnumerateDeviceExtensionProperties");
        let queue_families =
            load::<GetQueueFamilies>(gipa, instance, c"vkGetPhysicalDeviceQueueFamilyProperties")
                .expect("queue families");
        let create_device =
            load::<CreateDevice>(gipa, instance, c"vkCreateDevice").expect("vkCreateDevice");
        let destroy_device =
            load::<DestroyDevice>(gipa, instance, c"vkDestroyDevice").expect("vkDestroyDevice");
        let physical = enumerate(|n, p| enumerate_devices(instance, n, p), || null);
        let gpus = bar::scan().unwrap_or_default();
        let mut failed = false;

        for &pd in &physical {
            let mut properties = [0u64; 128];
            get(c"vkGetPhysicalDeviceProperties").expect("properties")(
                pd,
                properties.as_mut_ptr().cast(),
            );
            let info = DeviceInfo::from_properties(properties.as_ptr().cast()).expect("filled");
            let extensions: Vec<CString> = enumerate(
                |n, p| device_extensions(pd, std::ptr::null(), n, p),
                empty_extension,
            )
            .iter()
            .map(|e| name_of(&e.name))
            .collect();
            let offer = Offer::new(
                instance_setup.api,
                info.api.unwrap_or(ApiVersion::V1_0),
                extensions.iter().map(CString::as_c_str),
            )
            .with_instance(instance_extensions.iter().copied());
            let mut probe = Probe::new(offer, Capability::ALL);
            let khr = probe.query() == FeatureQuery::Features2Khr;
            let entry = |core: &'static CStr, ext: &'static CStr| get(if khr { ext } else { core });
            if probe.query().has_chain() {
                entry(
                    c"vkGetPhysicalDeviceFeatures2",
                    c"vkGetPhysicalDeviceFeatures2KHR",
                )
                .expect("features2")(pd, probe.features2());
                entry(
                    c"vkGetPhysicalDeviceProperties2",
                    c"vkGetPhysicalDeviceProperties2KHR",
                )
                .expect("properties2")(pd, probe.properties2());
                entry(
                    c"vkGetPhysicalDeviceMemoryProperties2",
                    c"vkGetPhysicalDeviceMemoryProperties2KHR",
                )
                .expect("memory properties2")(pd, probe.memory_properties2());
            } else {
                get(c"vkGetPhysicalDeviceFeatures").expect("features")(pd, probe.core_features());
                get(c"vkGetPhysicalDeviceMemoryProperties").expect("memory")(
                    pd,
                    probe.memory_properties(),
                );
            }
            let plan = probe.plan();
            println!("\n{plan}");

            // Create the device exactly as planned.
            let device_extensions: Vec<&CStr> = plan.extensions().to_vec();
            let names: Vec<*const c_char> = device_extensions.iter().map(|e| e.as_ptr()).collect();
            let families = enumerate(
                |n, p| {
                    queue_families(pd, n, p);
                    SUCCESS
                },
                QueueFamilyProperties::default,
            );
            let family = families.iter().position(|f| f.count > 0).unwrap_or(0);
            let priority = 1.0f32;
            let queue = DeviceQueueCreateInfo {
                s_type: 2,
                p_next: std::ptr::null(),
                flags: 0,
                family: u32::try_from(family).expect("few families"),
                count: 1,
                priorities: &raw const priority,
            };
            let mut chain = plan.device_chain();
            let (p_next, enabled_features) = if plan.query.has_chain() {
                (chain.features2().cast_const(), std::ptr::null())
            } else {
                (std::ptr::null(), chain.core_features())
            };
            let device_info = DeviceCreateInfo {
                s_type: 3,
                p_next,
                flags: 0,
                queue_info_count: 1,
                queue_infos: &raw const queue,
                layer_count: 0,
                layers: std::ptr::null(),
                extension_count: u32::try_from(names.len()).expect("few extensions"),
                extensions: names.as_ptr(),
                enabled_features,
            };
            let mut device = null;
            let r = create_device(
                pd,
                &raw const device_info,
                std::ptr::null(),
                &raw mut device,
            );
            let granted: Vec<&str> = plan
                .outcomes()
                .iter()
                .filter(|o| o.granted())
                .map(|o| o.cap.name())
                .collect();
            if r == SUCCESS {
                println!(
                    "  vkCreateDevice: ok with {} extensions and {} feature bits",
                    names.len(),
                    plan.enabled().count()
                );
                destroy_device(device, std::ptr::null());
            } else {
                println!("  vkCreateDevice FAILED: {r}");
                failed = true;
            }
            println!("  granted ({}): {}", granted.len(), granted.join(" "));

            // The PCI side of the same card.
            let pci = gpus.iter().find(|g| match (g.address, info.pci) {
                (Some(a), Some(b)) => a == b,
                _ => g.ids.vendor.id() == info.vendor_id && g.ids.device == info.device_id,
            });
            match (pci, plan.memory()) {
                (Some(gpu), Some(memory)) => println!(
                    "  pci::bar: {}\n  CPU-visible VRAM: Vulkan {}, PCI aperture {}",
                    gpu.to_string().replace('\n', "\n  "),
                    numerobis::vulkan::memory::bytes(memory.cpu_visible_vram()),
                    gpu.cpu_visible()
                        .map_or_else(|| "unknown".into(), numerobis::vulkan::memory::bytes)
                ),
                (None, _) => println!("  pci::bar: no PCI device matches (not a PCI GPU?)"),
                (_, None) => println!("  memory properties not read"),
            }
        }

        if let Some(handle) = messenger {
            load::<DestroyDebugMessenger>(gipa, instance, c"vkDestroyDebugUtilsMessengerEXT")
                .expect("debug utils")(instance, handle, std::ptr::null());
        }
        load::<DestroyInstance>(gipa, instance, c"vkDestroyInstance").expect("vkDestroyInstance")(
            instance,
            std::ptr::null(),
        );
        failed
    };
    let errors = VALIDATION_ERRORS.load(Ordering::Relaxed);
    println!("\nvalidation errors: {errors}");
    if failed || errors > 0 {
        std::process::exit(1);
    }
}

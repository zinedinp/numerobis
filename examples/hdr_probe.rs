//! Presents HDR on the local display: a window, a swapchain in the colour
//! space [`swapchain::output_format`] picks, test patches of known
//! luminance encoded by keinbonus and copied into the swapchain images, and
//! `vkSetHdrMetadataEXT` for HDR10. Under the validation layer when it is
//! installed.
//!
//! `cargo run --example hdr_probe --features vulkan [-- hdr10|scrgb|hlg|p3|sdr]`
//! (default `hdr10`; `NUMEROBIS_HDR_SECONDS=20` how long it stays up, `0`
//! until the window is closed).
//!
//! What the window shows, left to right on top: 100, 203 (SDR white,
//! BT.2408), 400, 600, 1000 and 4000 cd/m² grey; below, a ramp even in PQ
//! from black to 10 000 cd/m². In HDR the patches get brighter up to the
//! display's peak and the ramp stops changing there; in SDR everything from
//! 203 up is the same white.
//!
//! Windows only (Win32 surface). No bindings: the loader is opened by hand.

#![allow(
    clippy::too_many_lines,
    clippy::items_after_statements,
    reason = "an example that reads top to bottom, one Vulkan call after another,               each entry point's type next to its use"
)]
#![allow(
    clippy::many_single_char_names,
    reason = "pixel coordinates and channels"
)]

#[cfg(not(windows))]
fn main() {
    println!("hdr_probe: Windows only (a Win32 window and surface)");
}

#[cfg(windows)]
fn main() {
    win::main();
}

#[cfg(windows)]
mod win {
    use std::ffi::{CStr, CString, c_char, c_void};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    use numerobis::keinbonus::Colour;
    use numerobis::keinbonus::hdr::formats::Hdr10;
    use numerobis::keinbonus::hdr::{self, ContentLight, Luminance, MasteringDisplay};
    use numerobis::keinbonus::quant::{self, BitDepth, Range, Signal};
    use numerobis::keinbonus::transfer::Curve;
    use numerobis::keinbonus::{cicp::ColourPrimaries, half};
    use numerobis::vulkan::alloc::Usage;
    use numerobis::vulkan::color_space::{ColorSpace, HdrMetadata};
    use numerobis::vulkan::format;
    use numerobis::vulkan::swapchain::{
        self, Encoding, Extent, Output, SurfaceCapabilities, SurfaceFormat,
    };
    use numerobis::vulkan::{ApiVersion, Capability, DeviceInfo, Instance, Offer, Probe};

    type Void = unsafe extern "system" fn();
    type Gipa = unsafe extern "system" fn(*mut c_void, *const c_char) -> Option<Void>;
    type Handle = *mut c_void;
    type NonDispatch = u64;

    const SUCCESS: i32 = 0;

    // ---- Win32 ----

    type WndProc = unsafe extern "system" fn(Handle, u32, usize, isize) -> isize;

    #[repr(C)]
    struct WndClassExW {
        size: u32,
        style: u32,
        proc_: WndProc,
        cls_extra: i32,
        wnd_extra: i32,
        instance: Handle,
        icon: Handle,
        cursor: Handle,
        background: Handle,
        menu_name: *const u16,
        class_name: *const u16,
        icon_small: Handle,
    }

    #[repr(C)]
    struct Msg {
        hwnd: Handle,
        message: u32,
        wparam: usize,
        lparam: isize,
        time: u32,
        pt: [i32; 2],
        private: u32,
    }

    #[repr(C)]
    struct Rect {
        left: i32,
        top: i32,
        right: i32,
        bottom: i32,
    }

    #[link(name = "user32")]
    unsafe extern "system" {
        fn RegisterClassExW(class: *const WndClassExW) -> u16;
        fn CreateWindowExW(
            ex_style: u32,
            class: *const u16,
            title: *const u16,
            style: u32,
            x: i32,
            y: i32,
            w: i32,
            h: i32,
            parent: Handle,
            menu: Handle,
            instance: Handle,
            param: *mut c_void,
        ) -> Handle;
        fn DefWindowProcW(hwnd: Handle, msg: u32, w: usize, l: isize) -> isize;
        fn PeekMessageW(msg: *mut Msg, hwnd: Handle, min: u32, max: u32, remove: u32) -> i32;
        fn TranslateMessage(msg: *const Msg) -> i32;
        fn DispatchMessageW(msg: *const Msg) -> isize;
        fn DestroyWindow(hwnd: Handle) -> i32;
        fn AdjustWindowRect(rect: *mut Rect, style: u32, menu: i32) -> i32;
        fn SetProcessDpiAwarenessContext(context: isize) -> i32;
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetModuleHandleW(name: *const u16) -> Handle;
        fn LoadLibraryA(name: *const c_char) -> Handle;
        fn GetProcAddress(module: Handle, name: *const c_char) -> Option<Void>;
    }

    const WM_CLOSE: u32 = 0x0010;
    const WS_OVERLAPPED_FIXED: u32 = 0x00C0_0000 | 0x0008_0000 | 0x0002_0000; // caption, sysmenu, minimize
    const WS_VISIBLE: u32 = 0x1000_0000;
    const PM_REMOVE: u32 = 1;
    const DPI_PER_MONITOR_V2: isize = -4;

    static CLOSED: AtomicBool = AtomicBool::new(false);

    unsafe extern "system" fn window_proc(hwnd: Handle, msg: u32, w: usize, l: isize) -> isize {
        if msg == WM_CLOSE {
            CLOSED.store(true, Ordering::Relaxed);
            return 0;
        }
        // SAFETY: the arguments as Windows passed them.
        unsafe { DefWindowProcW(hwnd, msg, w, l) }
    }

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain([0]).collect()
    }

    /// Dispatches what is queued; `false` once the window was closed.
    fn pump() -> bool {
        let mut msg = Msg {
            hwnd: std::ptr::null_mut(),
            message: 0,
            wparam: 0,
            lparam: 0,
            time: 0,
            pt: [0; 2],
            private: 0,
        };
        // SAFETY: `msg` outlives the calls.
        unsafe {
            while PeekMessageW(&raw mut msg, std::ptr::null_mut(), 0, 0, PM_REMOVE) != 0 {
                TranslateMessage(&raw const msg);
                DispatchMessageW(&raw const msg);
            }
        }
        !CLOSED.load(Ordering::Relaxed)
    }

    // ---- Vulkan structs (1.0 and KHR) ----

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
        granularity: [u32; 3],
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

    #[repr(C)]
    struct Win32SurfaceCreateInfo {
        s_type: i32,
        p_next: *const c_void,
        flags: u32,
        hinstance: Handle,
        hwnd: Handle,
    }

    #[repr(C)]
    struct SwapchainCreateInfo {
        s_type: i32,
        p_next: *const c_void,
        flags: u32,
        surface: NonDispatch,
        min_image_count: u32,
        image_format: u32,
        image_color_space: u32,
        image_extent: Extent,
        image_array_layers: u32,
        image_usage: u32,
        image_sharing_mode: u32,
        queue_family_index_count: u32,
        queue_family_indices: *const u32,
        pre_transform: u32,
        composite_alpha: u32,
        present_mode: u32,
        clipped: u32,
        old_swapchain: NonDispatch,
    }

    #[repr(C)]
    struct BufferCreateInfo {
        s_type: i32,
        p_next: *const c_void,
        flags: u32,
        size: u64,
        usage: u32,
        sharing_mode: u32,
        queue_family_index_count: u32,
        queue_family_indices: *const u32,
    }

    #[repr(C)]
    #[derive(Default)]
    struct MemoryRequirements {
        size: u64,
        alignment: u64,
        memory_type_bits: u32,
    }

    #[repr(C)]
    struct MemoryAllocateInfo {
        s_type: i32,
        p_next: *const c_void,
        size: u64,
        memory_type: u32,
    }

    #[repr(C)]
    struct MappedMemoryRange {
        s_type: i32,
        p_next: *const c_void,
        memory: NonDispatch,
        offset: u64,
        size: u64,
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
        pool: NonDispatch,
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
    struct SubresourceRange {
        aspect: u32,
        base_mip: u32,
        mip_count: u32,
        base_layer: u32,
        layer_count: u32,
    }

    #[repr(C)]
    struct ImageMemoryBarrier {
        s_type: i32,
        p_next: *const c_void,
        src_access: u32,
        dst_access: u32,
        old_layout: u32,
        new_layout: u32,
        src_family: u32,
        dst_family: u32,
        image: NonDispatch,
        range: SubresourceRange,
    }

    #[repr(C)]
    struct BufferImageCopy {
        buffer_offset: u64,
        buffer_row_length: u32,
        buffer_image_height: u32,
        aspect: u32,
        mip: u32,
        base_layer: u32,
        layer_count: u32,
        image_offset: [i32; 3],
        image_extent: [u32; 3],
    }

    #[repr(C)]
    struct SemaphoreCreateInfo {
        s_type: i32,
        p_next: *const c_void,
        flags: u32,
    }

    #[repr(C)]
    struct FenceCreateInfo {
        s_type: i32,
        p_next: *const c_void,
        flags: u32,
    }

    #[repr(C)]
    struct SubmitInfo {
        s_type: i32,
        p_next: *const c_void,
        wait_count: u32,
        wait: *const NonDispatch,
        wait_stages: *const u32,
        command_buffer_count: u32,
        command_buffers: *const Handle,
        signal_count: u32,
        signal: *const NonDispatch,
    }

    #[repr(C)]
    struct PresentInfo {
        s_type: i32,
        p_next: *const c_void,
        wait_count: u32,
        wait: *const NonDispatch,
        swapchain_count: u32,
        swapchains: *const NonDispatch,
        image_indices: *const u32,
        results: *mut i32,
    }

    const SEVERITY_WARNING: u32 = 0x100;
    const SEVERITY_ERROR: u32 = 0x1000;
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

    const TRANSFER_DST: u32 = 0x2;
    const TRANSFER_SRC_BUFFER: u32 = 0x1;
    const LAYOUT_UNDEFINED: u32 = 0;
    const LAYOUT_TRANSFER_DST: u32 = 7;
    const LAYOUT_PRESENT_SRC: u32 = 1_000_001_002;
    const ACCESS_TRANSFER_WRITE: u32 = 0x1000;
    const STAGE_TRANSFER: u32 = 0x1000;
    const STAGE_BOTTOM: u32 = 0x2000;
    const ASPECT_COLOR: u32 = 0x1;
    const QUEUE_GRAPHICS: u32 = 0x1;
    const FAMILY_IGNORED: u32 = u32::MAX;

    /// Every entry point used, loaded once.
    struct Vk {
        gipa: Gipa,
        instance: Handle,
    }

    impl Vk {
        /// # Safety
        ///
        /// `F` is the function's real signature.
        unsafe fn get<F: Copy>(&self, name: &CStr) -> F {
            assert_eq!(size_of::<F>(), size_of::<Void>());
            // SAFETY: the caller's promise about `F`.
            unsafe {
                let f = (self.gipa)(self.instance, name.as_ptr())
                    .unwrap_or_else(|| panic!("{} missing", name.to_string_lossy()));
                std::mem::transmute_copy::<Void, F>(&f)
            }
        }
    }

    fn name_of(raw: &[c_char; 256]) -> CString {
        // SAFETY: Vulkan NUL-terminates names within the array.
        unsafe { CStr::from_ptr(raw.as_ptr()) }.to_owned()
    }

    fn enumerate<T>(
        mut call: impl FnMut(*mut u32, *mut T) -> i32,
        empty: impl Fn() -> T,
    ) -> Vec<T> {
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

    // ---- The test image ----

    /// The patches' grey levels, cd/m².
    const PATCHES: [f64; 6] = [100.0, 203.0, 400.0, 600.0, 1000.0, 4000.0];

    /// Display light at a pixel: the patches on the top 60 %, the PQ ramp
    /// below.
    fn nits_at(x: u32, y: u32, extent: Extent) -> f64 {
        let (w, h) = (f64::from(extent.width), f64::from(extent.height));
        let (fx, fy) = (f64::from(x) / w, f64::from(y) / h);
        if fy < 0.6 {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "0..6"
            )]
            let i = ((fx * 6.0) as usize).min(5);
            // A dark border between patches.
            let inside = (fx * 6.0).fract();
            if !(0.04..0.96).contains(&inside) || !(0.05..0.55).contains(&fy) {
                return 0.0;
            }
            PATCHES[i]
        } else if fy < 0.65 {
            0.0
        } else {
            10_000.0 * Curve::Pq.decode(fx)
        }
    }

    /// How a pixel of `nits` grey is written to `format` in `space`.
    struct Encoder {
        colour: Colour,
        lum: Luminance,
        luma: [f64; 3],
        format: u32,
    }

    impl Encoder {
        fn new(space: ColorSpace, format: u32) -> Self {
            let cicp = space.cicp().expect("a described colour space");
            let colour = Colour::new(&cicp).expect("resolves");
            let lum = Luminance {
                sdr_white: space.reference_white().unwrap_or(203.0),
                ..Luminance::default()
            };
            Self {
                luma: colour.primaries.luminance(),
                colour,
                lum,
                format,
            }
        }

        fn bytes_per_pixel(&self) -> usize {
            if self.format == format::R16G16B16A16_SFLOAT {
                8
            } else {
                4
            }
        }

        fn write(&self, nits: f64, out: &mut [u8]) {
            let e = hdr::from_nits(self.colour.curve, [nits; 3], &self.lum, self.luma);
            let code = |v: f64, depth| quant::quantize(v, depth, Range::Full, Signal::Luma);
            match self.format {
                format::R16G16B16A16_SFLOAT => {
                    for (i, v) in e.iter().chain(&[1.0]).enumerate() {
                        out[2 * i..2 * i + 2].copy_from_slice(&half::from_f64(*v).to_le_bytes());
                    }
                }
                format::A2B10G10R10_UNORM_PACK32 | format::A2R10G10B10_UNORM_PACK32 => {
                    let [r, g, b] = e.map(|v| u32::from(code(v, BitDepth::B10)));
                    let packed = if self.format == format::A2B10G10R10_UNORM_PACK32 {
                        r | g << 10 | b << 20 | 3 << 30
                    } else {
                        b | g << 10 | r << 20 | 3 << 30
                    };
                    out[..4].copy_from_slice(&packed.to_le_bytes());
                }
                f => {
                    #[allow(clippy::cast_possible_truncation, reason = "8-bit codes")]
                    let [r, g, b] = e.map(|v| code(v, BitDepth::B8) as u8);
                    let bgra = matches!(f, format::B8G8R8A8_UNORM | format::B8G8R8A8_SRGB);
                    out[..4].copy_from_slice(&if bgra { [b, g, r, 255] } else { [r, g, b, 255] });
                }
            }
        }

        /// The first code written for `nits`, for the log.
        fn describe(&self, nits: f64) -> String {
            let e = hdr::from_nits(self.colour.curve, [nits; 3], &self.lum, self.luma)[0];
            if self.format == format::R16G16B16A16_SFLOAT {
                format!("{e:.4} (f16 {:#06x})", half::from_f64(e))
            } else {
                let depth = if self.bytes_per_pixel() == 4 && self.is_10bit() {
                    BitDepth::B10
                } else {
                    BitDepth::B8
                };
                format!(
                    "code {}",
                    quant::quantize(e, depth, Range::Full, Signal::Luma)
                )
            }
        }

        fn is_10bit(&self) -> bool {
            matches!(
                self.format,
                format::A2B10G10R10_UNORM_PACK32 | format::A2R10G10B10_UNORM_PACK32
            )
        }
    }

    fn wishes(mode: &str) -> Vec<Output> {
        let sdr = Output::Sdr(Encoding::Unorm);
        match mode {
            "scrgb" => vec![Output::ScRgb, sdr],
            "hlg" => vec![Output::Hlg, sdr],
            "p3" => vec![Output::DisplayP3(Encoding::Unorm), sdr],
            "sdr" => vec![sdr],
            _ => vec![Output::Hdr10, sdr],
        }
    }

    pub fn main() {
        let mode = std::env::args().nth(1).unwrap_or_else(|| "hdr10".into());
        let seconds: u64 = std::env::var("NUMEROBIS_HDR_SECONDS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(20);
        let failed = run(&mode, seconds);
        let errors = VALIDATION_ERRORS.load(Ordering::Relaxed);
        println!("validation errors: {errors}");
        if failed || errors > 0 {
            std::process::exit(1);
        }
    }

    fn run(mode: &str, seconds: u64) -> bool {
        // SAFETY: every entry point is loaded with its real signature, every
        // struct handed over outlives the call, and handles are destroyed
        // after the device is idle, children before parents.
        unsafe {
            SetProcessDpiAwarenessContext(DPI_PER_MONITOR_V2);
            let module = LoadLibraryA(c"vulkan-1.dll".as_ptr());
            if module.is_null() {
                println!("no Vulkan loader");
                return true;
            }
            let gipa: Gipa = std::mem::transmute::<Void, Gipa>(
                GetProcAddress(module, c"vkGetInstanceProcAddr".as_ptr()).expect("gipa"),
            );
            let null: Handle = std::ptr::null_mut();
            let global = Vk {
                gipa,
                instance: null,
            };

            // Instance: the surface extensions, the colour spaces, validation.
            type EnumExt =
                unsafe extern "system" fn(*const c_char, *mut u32, *mut ExtensionProperties) -> i32;
            type EnumLayers = unsafe extern "system" fn(*mut u32, *mut LayerProperties) -> i32;
            type EnumVersion = unsafe extern "system" fn(*mut u32) -> i32;
            let enum_ext: EnumExt = global.get(c"vkEnumerateInstanceExtensionProperties");
            let available: Vec<CString> =
                enumerate(|n, p| enum_ext(std::ptr::null(), n, p), empty_extension)
                    .iter()
                    .map(|e| name_of(&e.name))
                    .collect();
            let has = |n: &CStr| available.iter().any(|e| e.as_c_str() == n);
            let enum_layers: EnumLayers = global.get(c"vkEnumerateInstanceLayerProperties");
            let validation = std::env::var_os("NUMEROBIS_NO_VALIDATION").is_none()
                && enumerate(
                    |n, p| enum_layers(n, p),
                    || LayerProperties {
                        name: [0; 256],
                        spec_version: 0,
                        implementation_version: 0,
                        description: [0; 256],
                    },
                )
                .iter()
                .any(|l| name_of(&l.name).as_c_str() == c"VK_LAYER_KHRONOS_validation");
            let loader = {
                let f: EnumVersion = global.get(c"vkEnumerateInstanceVersion");
                let mut v = 0;
                (f(&raw mut v) == SUCCESS)
                    .then(|| ApiVersion::from_raw(v))
                    .flatten()
            };
            let setup = Instance::new(
                loader,
                ApiVersion::NEWEST,
                available.iter().map(CString::as_c_str),
            )
            .with_color_spaces(available.iter().map(CString::as_c_str));
            let mut extensions: Vec<&CStr> = setup.extensions().to_vec();
            extensions.extend([c"VK_KHR_surface", c"VK_KHR_win32_surface"]);
            let debug = validation && has(c"VK_EXT_debug_utils");
            if debug {
                extensions.push(c"VK_EXT_debug_utils");
            }
            let ext_ptrs: Vec<*const c_char> = extensions.iter().map(|e| e.as_ptr()).collect();
            let layer_ptrs: Vec<*const c_char> = if validation {
                vec![c"VK_LAYER_KHRONOS_validation".as_ptr()]
            } else {
                Vec::new()
            };
            let app = ApplicationInfo {
                s_type: 0,
                p_next: std::ptr::null(),
                application_name: c"numerobis hdr_probe".as_ptr(),
                application_version: 1,
                engine_name: c"numerobis".as_ptr(),
                engine_version: 1,
                api_version: setup.api.raw(),
            };
            let info = InstanceCreateInfo {
                s_type: 1,
                p_next: std::ptr::null(),
                flags: setup.flags,
                application_info: &raw const app,
                layer_count: u32::try_from(layer_ptrs.len()).expect("few"),
                layers: layer_ptrs.as_ptr(),
                extension_count: u32::try_from(ext_ptrs.len()).expect("few"),
                extensions: ext_ptrs.as_ptr(),
            };
            type CreateInstance = unsafe extern "system" fn(
                *const InstanceCreateInfo,
                *const c_void,
                *mut Handle,
            ) -> i32;
            let create_instance: CreateInstance = global.get(c"vkCreateInstance");
            let mut instance = null;
            let r = create_instance(&raw const info, std::ptr::null(), &raw mut instance);
            if r != SUCCESS {
                println!("vkCreateInstance failed: {r}");
                return true;
            }
            let vk = Vk { gipa, instance };
            println!(
                "instance {}, validation {}, extensions: {}",
                setup.api,
                if validation { "on" } else { "off" },
                extensions
                    .iter()
                    .map(|e| e.to_string_lossy())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            let mut messenger: NonDispatch = 0;
            if debug {
                type Create = unsafe extern "system" fn(
                    Handle,
                    *const DebugMessengerCreateInfo,
                    *const c_void,
                    *mut NonDispatch,
                ) -> i32;
                let create: Create = vk.get(c"vkCreateDebugUtilsMessengerEXT");
                let info = DebugMessengerCreateInfo {
                    s_type: 1_000_128_004,
                    p_next: std::ptr::null(),
                    flags: 0,
                    severity: SEVERITY_WARNING | SEVERITY_ERROR,
                    types: 0x1 | 0x2,
                    callback: on_message,
                    user_data: null,
                };
                create(
                    instance,
                    &raw const info,
                    std::ptr::null(),
                    &raw mut messenger,
                );
            }

            // The window and its surface.
            let hinstance = GetModuleHandleW(std::ptr::null());
            let class = wide("numerobis_hdr_probe");
            let wc = WndClassExW {
                size: u32::try_from(size_of::<WndClassExW>()).expect("small"),
                style: 0,
                proc_: window_proc,
                cls_extra: 0,
                wnd_extra: 0,
                instance: hinstance,
                icon: null,
                cursor: null,
                background: null,
                menu_name: std::ptr::null(),
                class_name: class.as_ptr(),
                icon_small: null,
            };
            RegisterClassExW(&raw const wc);
            let mut rect = Rect {
                left: 0,
                top: 0,
                right: 1280,
                bottom: 720,
            };
            AdjustWindowRect(&raw mut rect, WS_OVERLAPPED_FIXED, 0);
            let title = wide(&format!("numerobis hdr_probe ({mode})"));
            let hwnd = CreateWindowExW(
                0,
                class.as_ptr(),
                title.as_ptr(),
                WS_OVERLAPPED_FIXED | WS_VISIBLE,
                100,
                100,
                rect.right - rect.left,
                rect.bottom - rect.top,
                null,
                null,
                hinstance,
                null,
            );
            assert!(!hwnd.is_null(), "CreateWindowExW");
            type CreateSurface = unsafe extern "system" fn(
                Handle,
                *const Win32SurfaceCreateInfo,
                *const c_void,
                *mut NonDispatch,
            ) -> i32;
            let create_surface: CreateSurface = vk.get(c"vkCreateWin32SurfaceKHR");
            let sinfo = Win32SurfaceCreateInfo {
                s_type: 1_000_009_000,
                p_next: std::ptr::null(),
                flags: 0,
                hinstance,
                hwnd,
            };
            let mut surface: NonDispatch = 0;
            assert_eq!(
                create_surface(
                    instance,
                    &raw const sinfo,
                    std::ptr::null(),
                    &raw mut surface
                ),
                SUCCESS
            );

            // The device: the first with a queue that draws and presents here.
            type EnumDevices = unsafe extern "system" fn(Handle, *mut u32, *mut Handle) -> i32;
            type Query = unsafe extern "system" fn(Handle, *mut c_void);
            type EnumDevExt = unsafe extern "system" fn(
                Handle,
                *const c_char,
                *mut u32,
                *mut ExtensionProperties,
            ) -> i32;
            type Families = unsafe extern "system" fn(Handle, *mut u32, *mut QueueFamilyProperties);
            type SurfaceSupport =
                unsafe extern "system" fn(Handle, u32, NonDispatch, *mut u32) -> i32;
            let enum_devices: EnumDevices = vk.get(c"vkEnumeratePhysicalDevices");
            let properties: Query = vk.get(c"vkGetPhysicalDeviceProperties");
            let enum_dev_ext: EnumDevExt = vk.get(c"vkEnumerateDeviceExtensionProperties");
            let families: Families = vk.get(c"vkGetPhysicalDeviceQueueFamilyProperties");
            let support: SurfaceSupport = vk.get(c"vkGetPhysicalDeviceSurfaceSupportKHR");
            let mut chosen = None;
            for pd in enumerate(|n, p| enum_devices(instance, n, p), || null) {
                let fams = enumerate(
                    |n, p| {
                        families(pd, n, p);
                        SUCCESS
                    },
                    QueueFamilyProperties::default,
                );
                let family = (0..fams.len()).find(|&i| {
                    let mut ok = 0;
                    let i32_ = u32::try_from(i).expect("few");
                    fams[i].flags & QUEUE_GRAPHICS != 0
                        && support(pd, i32_, surface, &raw mut ok) == SUCCESS
                        && ok != 0
                });
                if let Some(family) = family {
                    chosen = Some((pd, u32::try_from(family).expect("few")));
                    break;
                }
            }
            let Some((pd, family)) = chosen else {
                println!("no device presents to this window");
                return true;
            };
            let mut props = [0u64; 128];
            properties(pd, props.as_mut_ptr().cast());
            let device_info = DeviceInfo::from_properties(props.as_ptr().cast()).expect("filled");
            println!("device: {}", device_info.name);

            // What the surface offers.
            type Formats =
                unsafe extern "system" fn(Handle, NonDispatch, *mut u32, *mut SurfaceFormat) -> i32;
            type Caps =
                unsafe extern "system" fn(Handle, NonDispatch, *mut SurfaceCapabilities) -> i32;
            let formats_of: Formats = vk.get(c"vkGetPhysicalDeviceSurfaceFormatsKHR");
            let caps_of: Caps = vk.get(c"vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
            let offered = enumerate(
                |n, p| formats_of(pd, surface, n, p),
                || SurfaceFormat::new(0),
            );
            println!("surface formats:");
            for f in &offered {
                let space = f.space().map_or_else(
                    || format!("colour space {}", f.color_space),
                    |s| s.to_string(),
                );
                println!("  format {:>3} in {space}", f.format);
            }
            let wishes = wishes(mode);
            let Some((output, surface_format)) = swapchain::output_format(&offered, &wishes) else {
                println!("none of {wishes:?} offered");
                return true;
            };
            let space = surface_format.space().expect("a known colour space");
            println!(
                "chosen: {output:?}: format {} in {space}",
                surface_format.format
            );
            if output != wishes[0] {
                if offered
                    .iter()
                    .any(|f| f.space().is_some_and(ColorSpace::is_hdr))
                {
                    println!(
                        "  {:?} is not offered here; fell back to {output:?}",
                        wishes[0]
                    );
                } else {
                    println!(
                        "  no HDR colour space offered: is HDR on in Windows' display settings for this monitor?"
                    );
                }
            }

            // Plan the device: the swapchain, and HDR metadata where offered.
            let dev_exts: Vec<CString> = enumerate(
                |n, p| enum_dev_ext(pd, std::ptr::null(), n, p),
                empty_extension,
            )
            .iter()
            .map(|e| name_of(&e.name))
            .collect();
            let offer = Offer::new(
                setup.api,
                device_info.api.unwrap_or(ApiVersion::V1_0),
                dev_exts.iter().map(CString::as_c_str),
            )
            .with_instance(extensions.iter().copied());
            let wanted_caps = [Capability::HdrMetadata, Capability::SwapchainColorSpace];
            let mut probe = Probe::new(offer, &wanted_caps);
            if probe.query().has_chain() {
                let f2: Query = vk.get(c"vkGetPhysicalDeviceFeatures2");
                let m2: Query = vk.get(c"vkGetPhysicalDeviceMemoryProperties2");
                f2(pd, probe.features2());
                m2(pd, probe.memory_properties2());
            } else {
                let m: Query = vk.get(c"vkGetPhysicalDeviceMemoryProperties");
                m(pd, probe.memory_properties());
            }
            let plan = probe.plan();
            println!("{plan}");
            let memory = plan.memory().expect("memory properties").clone();
            let mut dev_ext: Vec<&CStr> = plan.extensions().to_vec();
            if !dev_ext.contains(&c"VK_KHR_swapchain") {
                dev_ext.push(c"VK_KHR_swapchain");
            }
            let names: Vec<*const c_char> = dev_ext.iter().map(|e| e.as_ptr()).collect();
            let priority = 1.0f32;
            let queue_info = DeviceQueueCreateInfo {
                s_type: 2,
                p_next: std::ptr::null(),
                flags: 0,
                family,
                count: 1,
                priorities: &raw const priority,
            };
            let dinfo = DeviceCreateInfo {
                s_type: 3,
                p_next: std::ptr::null(),
                flags: 0,
                queue_info_count: 1,
                queue_infos: &raw const queue_info,
                layer_count: 0,
                layers: std::ptr::null(),
                extension_count: u32::try_from(names.len()).expect("few"),
                extensions: names.as_ptr(),
                enabled_features: std::ptr::null(),
            };
            type CreateDevice = unsafe extern "system" fn(
                Handle,
                *const DeviceCreateInfo,
                *const c_void,
                *mut Handle,
            ) -> i32;
            let create_device: CreateDevice = vk.get(c"vkCreateDevice");
            let mut device = null;
            let r = create_device(pd, &raw const dinfo, std::ptr::null(), &raw mut device);
            if r != SUCCESS {
                println!("vkCreateDevice failed: {r}");
                return true;
            }
            // Same signature as vkGetInstanceProcAddr, with the device.
            let device_proc: Gipa = vk.get(c"vkGetDeviceProcAddr");
            let dev = Vk {
                gipa: device_proc,
                instance: device,
            };

            type GetQueue = unsafe extern "system" fn(Handle, u32, u32, *mut Handle);
            let get_queue: GetQueue = dev.get(c"vkGetDeviceQueue");
            let mut queue = null;
            get_queue(device, family, 0, &raw mut queue);

            // The swapchain.
            let mut caps = SurfaceCapabilities::default();
            assert_eq!(caps_of(pd, surface, &raw mut caps), SUCCESS);
            let Some(extent) = caps.extent(Extent::new(1280, 720)) else {
                println!("the window has no size");
                return true;
            };
            if !caps.supports_usage(TRANSFER_DST) {
                println!("swapchain images can't be copied to");
                return true;
            }
            let sc_info = SwapchainCreateInfo {
                s_type: 1_000_001_000,
                p_next: std::ptr::null(),
                flags: 0,
                surface,
                min_image_count: caps.image_count(3),
                image_format: surface_format.format,
                image_color_space: surface_format.color_space,
                image_extent: extent,
                image_array_layers: 1,
                image_usage: TRANSFER_DST,
                image_sharing_mode: 0,
                queue_family_index_count: 0,
                queue_family_indices: std::ptr::null(),
                pre_transform: caps.current_transform,
                composite_alpha: caps.composite_alpha(),
                present_mode: swapchain::PresentMode::Fifo.raw(),
                clipped: 1,
                old_swapchain: 0,
            };
            type CreateSwapchain = unsafe extern "system" fn(
                Handle,
                *const SwapchainCreateInfo,
                *const c_void,
                *mut NonDispatch,
            ) -> i32;
            type GetImages =
                unsafe extern "system" fn(Handle, NonDispatch, *mut u32, *mut NonDispatch) -> i32;
            let create_swapchain: CreateSwapchain = dev.get(c"vkCreateSwapchainKHR");
            let get_images: GetImages = dev.get(c"vkGetSwapchainImagesKHR");
            let mut sc: NonDispatch = 0;
            let r = create_swapchain(device, &raw const sc_info, std::ptr::null(), &raw mut sc);
            if r != SUCCESS {
                println!("vkCreateSwapchainKHR failed: {r}");
                return true;
            }
            let images = enumerate(|n, p| get_images(device, sc, n, p), || 0);
            println!("swapchain: {extent}, {} images", images.len());

            // The test image, encoded by keinbonus.
            let enc = Encoder::new(space, surface_format.format);
            let bpp = enc.bytes_per_pixel();
            let size = extent.width as usize * extent.height as usize * bpp;
            let mut pixels = vec![0u8; size];
            let mut light_sum = 0.0;
            for y in 0..extent.height {
                for x in 0..extent.width {
                    let l = nits_at(x, y, extent);
                    light_sum += l.min(10_000.0);
                    let at = (y as usize * extent.width as usize + x as usize) * bpp;
                    enc.write(l, &mut pixels[at..at + bpp]);
                }
            }
            println!("patches:");
            for l in PATCHES {
                println!("  {l:>6} cd/m²: {}", enc.describe(l));
            }

            // HDR10 metadata: as mastered on a P3 1000 cd/m² display, the
            // content's light levels as drawn (MaxCLL from the ramp's end).
            if plan.has(Capability::HdrMetadata) && space == ColorSpace::Hdr10St2084 {
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "cd/m²"
                )]
                let max_fall = (light_sum / f64::from(extent.width * extent.height)).round() as u16;
                let hdr10 = Hdr10 {
                    mastering: MasteringDisplay {
                        primaries: ColourPrimaries::Smpte432.chromaticities().expect("P3"),
                        max_luminance: 1000.0,
                        min_luminance: 0.0001,
                    },
                    content: ContentLight {
                        max_cll: 10_000,
                        max_fall,
                    },
                };
                let metadata = HdrMetadata::from(&hdr10);
                type SetHdr =
                    unsafe extern "system" fn(Handle, u32, *const NonDispatch, *const HdrMetadata);
                let set_hdr: SetHdr = dev.get(c"vkSetHdrMetadataEXT");
                set_hdr(device, 1, &raw const sc, &raw const metadata);
                println!(
                    "vkSetHdrMetadataEXT: mastering 1000 / 0.0001 cd/m² (P3), MaxCLL {}, MaxFALL {}",
                    hdr10.content.max_cll, hdr10.content.max_fall
                );
            }

            // A host buffer holding the image, copied into each frame.
            type CreateBuffer = unsafe extern "system" fn(
                Handle,
                *const BufferCreateInfo,
                *const c_void,
                *mut NonDispatch,
            ) -> i32;
            type BufferReqs =
                unsafe extern "system" fn(Handle, NonDispatch, *mut MemoryRequirements);
            type Allocate = unsafe extern "system" fn(
                Handle,
                *const MemoryAllocateInfo,
                *const c_void,
                *mut NonDispatch,
            ) -> i32;
            type Bind = unsafe extern "system" fn(Handle, NonDispatch, NonDispatch, u64) -> i32;
            type Map = unsafe extern "system" fn(
                Handle,
                NonDispatch,
                u64,
                u64,
                u32,
                *mut *mut c_void,
            ) -> i32;
            type Flush = unsafe extern "system" fn(Handle, u32, *const MappedMemoryRange) -> i32;
            let create_buffer: CreateBuffer = dev.get(c"vkCreateBuffer");
            let buffer_reqs: BufferReqs = dev.get(c"vkGetBufferMemoryRequirements");
            let allocate: Allocate = dev.get(c"vkAllocateMemory");
            let bind: Bind = dev.get(c"vkBindBufferMemory");
            let map: Map = dev.get(c"vkMapMemory");
            let flush: Flush = dev.get(c"vkFlushMappedMemoryRanges");
            let binfo = BufferCreateInfo {
                s_type: 12,
                p_next: std::ptr::null(),
                flags: 0,
                size: size as u64,
                usage: TRANSFER_SRC_BUFFER,
                sharing_mode: 0,
                queue_family_index_count: 0,
                queue_family_indices: std::ptr::null(),
            };
            let mut buffer: NonDispatch = 0;
            assert_eq!(
                create_buffer(device, &raw const binfo, std::ptr::null(), &raw mut buffer),
                SUCCESS
            );
            let mut reqs = MemoryRequirements::default();
            buffer_reqs(device, buffer, &raw mut reqs);
            let mut buffer_memory: NonDispatch = 0;
            for t in memory.types_for(reqs.memory_type_bits, Usage::Upload, false) {
                let ainfo = MemoryAllocateInfo {
                    s_type: 5,
                    p_next: std::ptr::null(),
                    size: reqs.size,
                    memory_type: t,
                };
                if allocate(
                    device,
                    &raw const ainfo,
                    std::ptr::null(),
                    &raw mut buffer_memory,
                ) == SUCCESS
                {
                    break;
                }
            }
            assert_ne!(buffer_memory, 0, "no upload memory");
            assert_eq!(bind(device, buffer, buffer_memory, 0), SUCCESS);
            let mut mapped: *mut c_void = std::ptr::null_mut();
            assert_eq!(
                map(device, buffer_memory, 0, u64::MAX, 0, &raw mut mapped),
                SUCCESS
            );
            std::ptr::copy_nonoverlapping(pixels.as_ptr(), mapped.cast::<u8>(), size);
            let range = MappedMemoryRange {
                s_type: 6,
                p_next: std::ptr::null(),
                memory: buffer_memory,
                offset: 0,
                size: u64::MAX,
            };
            assert_eq!(flush(device, 1, &raw const range), SUCCESS);

            // One command buffer per image: barrier, copy, barrier.
            type CreatePool = unsafe extern "system" fn(
                Handle,
                *const CommandPoolCreateInfo,
                *const c_void,
                *mut NonDispatch,
            ) -> i32;
            type AllocCmd = unsafe extern "system" fn(
                Handle,
                *const CommandBufferAllocateInfo,
                *mut Handle,
            ) -> i32;
            type Begin = unsafe extern "system" fn(Handle, *const CommandBufferBeginInfo) -> i32;
            type End = unsafe extern "system" fn(Handle) -> i32;
            type Barrier = unsafe extern "system" fn(
                Handle,
                u32,
                u32,
                u32,
                u32,
                *const c_void,
                u32,
                *const c_void,
                u32,
                *const ImageMemoryBarrier,
            );
            type Copy = unsafe extern "system" fn(
                Handle,
                NonDispatch,
                NonDispatch,
                u32,
                u32,
                *const BufferImageCopy,
            );
            let create_pool: CreatePool = dev.get(c"vkCreateCommandPool");
            let alloc_cmd: AllocCmd = dev.get(c"vkAllocateCommandBuffers");
            let begin: Begin = dev.get(c"vkBeginCommandBuffer");
            let end: End = dev.get(c"vkEndCommandBuffer");
            let barrier: Barrier = dev.get(c"vkCmdPipelineBarrier");
            let copy: Copy = dev.get(c"vkCmdCopyBufferToImage");
            let pinfo = CommandPoolCreateInfo {
                s_type: 39,
                p_next: std::ptr::null(),
                flags: 0,
                family,
            };
            let mut pool: NonDispatch = 0;
            assert_eq!(
                create_pool(device, &raw const pinfo, std::ptr::null(), &raw mut pool),
                SUCCESS
            );
            let mut cmds = vec![null; images.len()];
            let cinfo = CommandBufferAllocateInfo {
                s_type: 40,
                p_next: std::ptr::null(),
                pool,
                level: 0,
                count: u32::try_from(cmds.len()).expect("few"),
            };
            assert_eq!(
                alloc_cmd(device, &raw const cinfo, cmds.as_mut_ptr()),
                SUCCESS
            );
            let range = || SubresourceRange {
                aspect: ASPECT_COLOR,
                base_mip: 0,
                mip_count: 1,
                base_layer: 0,
                layer_count: 1,
            };
            for (&cmd, &image) in cmds.iter().zip(&images) {
                let bi = CommandBufferBeginInfo {
                    s_type: 42,
                    p_next: std::ptr::null(),
                    flags: 0,
                    inheritance: std::ptr::null(),
                };
                assert_eq!(begin(cmd, &raw const bi), SUCCESS);
                let to_dst = ImageMemoryBarrier {
                    s_type: 45,
                    p_next: std::ptr::null(),
                    src_access: 0,
                    dst_access: ACCESS_TRANSFER_WRITE,
                    old_layout: LAYOUT_UNDEFINED,
                    new_layout: LAYOUT_TRANSFER_DST,
                    src_family: FAMILY_IGNORED,
                    dst_family: FAMILY_IGNORED,
                    image,
                    range: range(),
                };
                barrier(
                    cmd,
                    STAGE_TRANSFER,
                    STAGE_TRANSFER,
                    0,
                    0,
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    1,
                    &raw const to_dst,
                );
                let region = BufferImageCopy {
                    buffer_offset: 0,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    aspect: ASPECT_COLOR,
                    mip: 0,
                    base_layer: 0,
                    layer_count: 1,
                    image_offset: [0; 3],
                    image_extent: [extent.width, extent.height, 1],
                };
                copy(
                    cmd,
                    buffer,
                    image,
                    LAYOUT_TRANSFER_DST,
                    1,
                    &raw const region,
                );
                let to_present = ImageMemoryBarrier {
                    src_access: ACCESS_TRANSFER_WRITE,
                    dst_access: 0,
                    old_layout: LAYOUT_TRANSFER_DST,
                    new_layout: LAYOUT_PRESENT_SRC,
                    ..to_dst
                };
                barrier(
                    cmd,
                    STAGE_TRANSFER,
                    STAGE_BOTTOM,
                    0,
                    0,
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    1,
                    &raw const to_present,
                );
                assert_eq!(end(cmd), SUCCESS);
            }

            // Present until the time is up or the window closes; one frame
            // in flight.
            type CreateSemaphore = unsafe extern "system" fn(
                Handle,
                *const SemaphoreCreateInfo,
                *const c_void,
                *mut NonDispatch,
            ) -> i32;
            type CreateFence = unsafe extern "system" fn(
                Handle,
                *const FenceCreateInfo,
                *const c_void,
                *mut NonDispatch,
            ) -> i32;
            type Wait = unsafe extern "system" fn(Handle, u32, *const NonDispatch, u32, u64) -> i32;
            type Reset = unsafe extern "system" fn(Handle, u32, *const NonDispatch) -> i32;
            type Acquire = unsafe extern "system" fn(
                Handle,
                NonDispatch,
                u64,
                NonDispatch,
                NonDispatch,
                *mut u32,
            ) -> i32;
            type Submit =
                unsafe extern "system" fn(Handle, u32, *const SubmitInfo, NonDispatch) -> i32;
            type Present = unsafe extern "system" fn(Handle, *const PresentInfo) -> i32;
            let create_semaphore: CreateSemaphore = dev.get(c"vkCreateSemaphore");
            let create_fence: CreateFence = dev.get(c"vkCreateFence");
            let wait: Wait = dev.get(c"vkWaitForFences");
            let reset: Reset = dev.get(c"vkResetFences");
            let acquire: Acquire = dev.get(c"vkAcquireNextImageKHR");
            let submit: Submit = dev.get(c"vkQueueSubmit");
            let present: Present = dev.get(c"vkQueuePresentKHR");
            let seminfo = SemaphoreCreateInfo {
                s_type: 9,
                p_next: std::ptr::null(),
                flags: 0,
            };
            let mut acquired: NonDispatch = 0;
            assert_eq!(
                create_semaphore(
                    device,
                    &raw const seminfo,
                    std::ptr::null(),
                    &raw mut acquired
                ),
                SUCCESS
            );
            // One "rendered" semaphore per image: present may still hold one.
            let mut rendered = vec![0; images.len()];
            for s in &mut rendered {
                assert_eq!(
                    create_semaphore(device, &raw const seminfo, std::ptr::null(), s),
                    SUCCESS
                );
            }
            let finfo = FenceCreateInfo {
                s_type: 8,
                p_next: std::ptr::null(),
                flags: 1,
            };
            let mut fence: NonDispatch = 0;
            assert_eq!(
                create_fence(device, &raw const finfo, std::ptr::null(), &raw mut fence),
                SUCCESS
            );

            let start = Instant::now();
            let mut frames = 0u64;
            let mut failed = false;
            while pump() && (seconds == 0 || start.elapsed() < Duration::from_secs(seconds)) {
                assert_eq!(wait(device, 1, &raw const fence, 1, u64::MAX), SUCCESS);
                let mut index = 0;
                let r = acquire(device, sc, u64::MAX, acquired, 0, &raw mut index);
                match swapchain::recreate(r, extent, || {
                    let mut now = SurfaceCapabilities::default();
                    (caps_of(pd, surface, &raw mut now) == SUCCESS)
                        .then(|| now.current())
                        .flatten()
                }) {
                    Ok(false) => {}
                    Ok(true) => {
                        println!(
                            "the swapchain went out of date (the window is fixed in size): stopping"
                        );
                        break;
                    }
                    Err(e) => {
                        println!("vkAcquireNextImageKHR: {e}");
                        failed = true;
                        break;
                    }
                }
                if r != SUCCESS && r != swapchain::result::SUBOPTIMAL {
                    continue;
                }
                assert_eq!(reset(device, 1, &raw const fence), SUCCESS);
                let stage = STAGE_TRANSFER;
                let i = index as usize;
                let sinfo = SubmitInfo {
                    s_type: 4,
                    p_next: std::ptr::null(),
                    wait_count: 1,
                    wait: &raw const acquired,
                    wait_stages: &raw const stage,
                    command_buffer_count: 1,
                    command_buffers: &raw const cmds[i],
                    signal_count: 1,
                    signal: &raw const rendered[i],
                };
                assert_eq!(submit(queue, 1, &raw const sinfo, fence), SUCCESS);
                let pinfo = PresentInfo {
                    s_type: 1_000_001_001,
                    p_next: std::ptr::null(),
                    wait_count: 1,
                    wait: &raw const rendered[i],
                    swapchain_count: 1,
                    swapchains: &raw const sc,
                    image_indices: &raw const index,
                    results: std::ptr::null_mut(),
                };
                let r = present(queue, &raw const pinfo);
                if let Err(e) = swapchain::recreate(r, extent, || None) {
                    println!("vkQueuePresentKHR: {e}");
                    failed = true;
                    break;
                }
                frames += 1;
                // The acquire semaphore is reused: wait for this frame first.
                assert_eq!(wait(device, 1, &raw const fence, 1, u64::MAX), SUCCESS);
            }
            println!(
                "presented {frames} frames in {:.1} s",
                start.elapsed().as_secs_f64()
            );

            // Tear down, children first.
            type DeviceWaitIdle = unsafe extern "system" fn(Handle) -> i32;
            type DestroyChild = unsafe extern "system" fn(Handle, NonDispatch, *const c_void);
            type DestroyRoot = unsafe extern "system" fn(Handle, *const c_void);
            type DestroyOfInstance = unsafe extern "system" fn(Handle, NonDispatch, *const c_void);
            let idle: DeviceWaitIdle = dev.get(c"vkDeviceWaitIdle");
            idle(device);
            let destroy = |name: &CStr, h: NonDispatch| {
                let f: DestroyChild = dev.get(name);
                f(device, h, std::ptr::null());
            };
            destroy(c"vkDestroyFence", fence);
            destroy(c"vkDestroySemaphore", acquired);
            for s in rendered {
                destroy(c"vkDestroySemaphore", s);
            }
            destroy(c"vkDestroyCommandPool", pool);
            destroy(c"vkDestroyBuffer", buffer);
            destroy(c"vkFreeMemory", buffer_memory);
            destroy(c"vkDestroySwapchainKHR", sc);
            let destroy_device: DestroyRoot = dev.get(c"vkDestroyDevice");
            destroy_device(device, std::ptr::null());
            let destroy_surface: DestroyOfInstance = vk.get(c"vkDestroySurfaceKHR");
            destroy_surface(instance, surface, std::ptr::null());
            if messenger != 0 {
                let f: DestroyOfInstance = vk.get(c"vkDestroyDebugUtilsMessengerEXT");
                f(instance, messenger, std::ptr::null());
            }
            let destroy_instance: DestroyRoot = vk.get(c"vkDestroyInstance");
            destroy_instance(instance, std::ptr::null());
            DestroyWindow(hwnd);
            failed
        }
    }
}

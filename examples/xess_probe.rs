//! XeSS 2 against the installed driver: Intel's AI upscaler, its network on
//! the XMX engines (the driver's implementation) or, without them, on DP4a.
//!
//! The Intel driver carries XeSS 2 next to its user-mode driver:
//! `igxess2.dll` (super resolution, the public `xess*` API), `igxess_fg.dll`
//! (frame generation) and `igxell.dll` (low latency). The probe finds that
//! folder through D3DKMT (`KMTQAITYPE_UMDRIVERNAME`), creates a D3D12
//! device on the adapter, a XeSS context on it, and reads what XeSS reports
//! next to [`numerobis::intel::xess`]: versions, whether the driver is the
//! optimal one, the input resolution per quality preset (current and
//! legacy factors), the memory it needs, and an `xessD3D12Init` (which
//! builds the network's pipelines) with the XeSS libraries the process then
//! holds.
//!
//! `cargo run --example xess_probe --features directx`
//! (`NUMEROBIS_XESS_DLL=path\to\libxess.dll` uses a game's or the SDK's
//! library instead). Windows only: tested (Intel Arc B580). Exits non-zero
//! when XeSS answers otherwise than the module expects.

#[cfg(not(windows))]
fn main() {
    println!("XeSS is a Windows library; nothing to probe here");
}

#[cfg(windows)]
fn main() {
    if let Err(e) = windows::run() {
        println!("FAILED: {e}");
        std::process::exit(1);
    }
}

#[cfg(windows)]
mod windows {
    use std::ffi::{CStr, c_char, c_void};

    use numerobis::directx::generated::iid;
    use numerobis::directx::modern::dxgi::{Adapter, AdapterDesc1};
    use numerobis::intel::xess::{
        self, Component, D3d12InitParams, Path, Properties, Quality, Size, Version, result,
    };
    use numerobis::pci::Gpu;

    type Hresult = i32;
    type Com = *mut c_void;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryA(name: *const c_char) -> *mut c_void;
        fn LoadLibraryW(name: *const u16) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
        fn GetCurrentProcess() -> *mut c_void;
        fn K32EnumProcessModules(
            process: *mut c_void,
            modules: *mut *mut c_void,
            bytes: u32,
            needed: *mut u32,
        ) -> i32;
        fn GetModuleFileNameW(module: *mut c_void, name: *mut u16, size: u32) -> u32;
    }

    /// File names of the loaded modules whose name contains one of `parts`.
    fn loaded_modules(parts: &[&str]) -> Vec<String> {
        let mut modules = vec![std::ptr::null_mut(); 1024];
        let mut needed = 0;
        // SAFETY: buffers of the sizes passed; the pseudo handle needs no close.
        unsafe {
            let bytes = u32::try_from(modules.len() * size_of::<*mut c_void>()).expect("small");
            if K32EnumProcessModules(
                GetCurrentProcess(),
                modules.as_mut_ptr(),
                bytes,
                &raw mut needed,
            ) == 0
            {
                return Vec::new();
            }
            modules.truncate(needed as usize / size_of::<*mut c_void>());
            let mut out: Vec<String> = modules
                .iter()
                .filter_map(|&m| {
                    let mut name = [0u16; 520];
                    let n = GetModuleFileNameW(m, name.as_mut_ptr(), 520) as usize;
                    let path = String::from_utf16_lossy(&name[..n]);
                    let file = path.rsplit('\\').next()?.to_string();
                    let is_dll = std::path::Path::new(&file)
                        .extension()
                        .is_some_and(|e| e.eq_ignore_ascii_case("dll"));
                    let lower = file.to_ascii_lowercase();
                    (is_dll && parts.iter().any(|p| lower.contains(p))).then_some(file)
                })
                .collect();
            out.sort();
            out
        }
    }

    #[link(name = "gdi32")]
    unsafe extern "system" {
        fn D3DKMTOpenAdapterFromLuid(info: *mut OpenAdapterFromLuid) -> i32;
        fn D3DKMTQueryAdapterInfo(info: *const QueryAdapterInfo) -> i32;
        fn D3DKMTCloseAdapter(info: *const u32) -> i32;
    }

    /// `D3DKMT_OPENADAPTERFROMLUID`.
    #[repr(C)]
    struct OpenAdapterFromLuid {
        luid: [u32; 2],
        adapter: u32,
    }

    /// `D3DKMT_QUERYADAPTERINFO`.
    #[repr(C)]
    struct QueryAdapterInfo {
        adapter: u32,
        kind: u32,
        data: *mut c_void,
        size: u32,
    }

    /// `KMTQAITYPE_UMDRIVERNAME`, `KMTUMDVERSION_DX12`.
    const UMD_DRIVER_NAME: u32 = 1;
    const UMD_DX12: u32 = 3;

    /// `D3DKMT_UMDFILENAMEINFO`.
    #[repr(C)]
    struct UmdFileName {
        version: u32,
        name: [u16; 260],
    }

    fn symbol<F: Copy>(module: *mut c_void, name: &CStr) -> Option<F> {
        assert_eq!(size_of::<F>(), size_of::<*mut c_void>());
        // SAFETY: a NUL-terminated name; the caller names `F` as the
        // function's signature.
        unsafe {
            let f = GetProcAddress(module, name.as_ptr());
            (!f.is_null()).then(|| std::mem::transmute_copy(&f))
        }
    }

    fn library(name: &CStr) -> Option<*mut c_void> {
        // SAFETY: a NUL-terminated name.
        let m = unsafe { LoadLibraryA(name.as_ptr()) };
        (!m.is_null()).then_some(m)
    }

    fn library_w(path: &std::path::Path) -> Option<*mut c_void> {
        use std::os::windows::ffi::OsStrExt;
        let wide: Vec<u16> = path.as_os_str().encode_wide().chain([0]).collect();
        // SAFETY: a NUL-terminated wide path.
        let m = unsafe { LoadLibraryW(wide.as_ptr()) };
        (!m.is_null()).then_some(m)
    }

    /// # Safety
    ///
    /// `obj` is a live COM object whose interface has `F` at `index`.
    unsafe fn method<F: Copy>(obj: Com, index: usize) -> F {
        // SAFETY: the caller's promise; a COM object starts with its vtable.
        unsafe {
            let vtbl = *obj.cast::<*const *const c_void>();
            std::mem::transmute_copy(&*vtbl.add(index))
        }
    }

    fn release(obj: Com) {
        if !obj.is_null() {
            // SAFETY: IUnknown::Release is entry 2 of every vtable.
            unsafe { method::<unsafe extern "system" fn(Com) -> u32>(obj, 2)(obj) };
        }
    }

    /// The folder of the adapter's D3D12 user-mode driver (the driver store).
    fn driver_folder(luid: [u32; 2]) -> Option<std::path::PathBuf> {
        let mut open = OpenAdapterFromLuid { luid, adapter: 0 };
        // SAFETY: valid structs for the calls; the adapter is closed after.
        unsafe {
            if D3DKMTOpenAdapterFromLuid(&raw mut open) != 0 {
                return None;
            }
            let mut info = UmdFileName {
                version: UMD_DX12,
                name: [0; 260],
            };
            let query = QueryAdapterInfo {
                adapter: open.adapter,
                kind: UMD_DRIVER_NAME,
                data: (&raw mut info).cast(),
                size: u32::try_from(size_of::<UmdFileName>()).expect("524 bytes"),
            };
            let status = D3DKMTQueryAdapterInfo(&raw const query);
            D3DKMTCloseAdapter(&raw const open.adapter);
            if status != 0 {
                return None;
            }
            let len = info.name.iter().position(|&c| c == 0).unwrap_or(260);
            let path = std::path::PathBuf::from(String::from_utf16_lossy(&info.name[..len]));
            path.parent().map(std::path::Path::to_path_buf)
        }
    }

    type XessGetVersion = unsafe extern "C" fn(*mut Version) -> i32;
    type XessCreateContext = unsafe extern "C" fn(Com, *mut *mut c_void) -> i32;
    type XessContextCall = unsafe extern "C" fn(*mut c_void) -> i32;
    type XessContextVersion = unsafe extern "C" fn(*mut c_void, *mut Version) -> i32;
    type XessOptimalInput =
        unsafe extern "C" fn(*mut c_void, *const Size, i32, *mut Size, *mut Size, *mut Size) -> i32;
    type XessInput = unsafe extern "C" fn(*mut c_void, *const Size, i32, *mut Size) -> i32;
    type XessProperties = unsafe extern "C" fn(*mut c_void, *const Size, *mut Properties) -> i32;
    type XessForceLegacy = unsafe extern "C" fn(*mut c_void, bool) -> i32;
    type XessInit = unsafe extern "C" fn(*mut c_void, *const D3d12InitParams) -> i32;

    #[allow(
        clippy::too_many_lines,
        reason = "an example that reads top to bottom, one XeSS call after another"
    )]
    pub fn run() -> Result<(), String> {
        let mut failures = Vec::new();
        // SAFETY: every function is loaded with its real signature (from
        // the SDK headers and the Windows SDK); COM objects are released.
        unsafe {
            type CreateFactory = unsafe extern "system" fn(*const c_void, *mut Com) -> Hresult;
            type EnumAdapters1 = unsafe extern "system" fn(Com, u32, *mut Com) -> Hresult;
            type GetDesc1 = unsafe extern "system" fn(Com, *mut AdapterDesc1) -> Hresult;
            type CreateDevice =
                unsafe extern "system" fn(Com, u32, *const c_void, *mut Com) -> Hresult;
            let dxgi = library(c"dxgi.dll").ok_or("no dxgi.dll")?;
            let d3d12 = library(c"d3d12.dll").ok_or("no d3d12.dll")?;
            let create_factory =
                symbol::<CreateFactory>(dxgi, c"CreateDXGIFactory1").ok_or("CreateDXGIFactory1")?;
            let create_device =
                symbol::<CreateDevice>(d3d12, c"D3D12CreateDevice").ok_or("D3D12CreateDevice")?;
            let (factory_iid, device_iid) = (iid::IDXGIFactory1, iid::ID3D12Device);
            let mut factory = std::ptr::null_mut();
            if create_factory((&raw const factory_iid).cast(), &raw mut factory) < 0 {
                return Err("CreateDXGIFactory1 failed".into());
            }

            let mut probed = 0;
            for index in 0.. {
                let mut adapter = std::ptr::null_mut();
                if method::<EnumAdapters1>(factory, 12)(factory, index, &raw mut adapter) < 0 {
                    break;
                }
                let mut desc: AdapterDesc1 = std::mem::zeroed();
                method::<GetDesc1>(adapter, 10)(adapter, &raw mut desc);
                let info = Adapter::from_desc1(&desc);
                // DXGI_ADAPTER_FLAG_SOFTWARE: WARP has no XeSS.
                if desc.flags & 2 != 0 {
                    release(adapter);
                    continue;
                }
                let gpu = Gpu::from_pci(desc.vendor_id, desc.device_id);
                let path = Path::of(gpu);
                println!("\n{}: XeSS path {path}", info.describe());

                let mut device = std::ptr::null_mut();
                let hr = create_device(
                    adapter,
                    0xb000, // D3D_FEATURE_LEVEL_11_0
                    (&raw const device_iid).cast(),
                    &raw mut device,
                );
                if hr < 0 {
                    println!("  D3D12CreateDevice: {hr:#x}, skipped");
                    release(adapter);
                    continue;
                }

                // XeSS 2 from the driver folder (Intel), or the given library.
                let folder = driver_folder(desc.adapter_luid);
                let pick = |component: Component| {
                    if component == Component::SuperResolution
                        && let Some(p) = std::env::var_os("NUMEROBIS_XESS_DLL")
                    {
                        return library_w(std::path::Path::new(&p)).map(|m| (m, p.into()));
                    }
                    let folder = folder.as_ref()?;
                    let file = folder.join(component.driver_dll());
                    library_w(&file).map(|m| (m, file))
                };
                match &folder {
                    Some(f) => println!("  driver folder: {}", f.display()),
                    None => println!("  driver folder: not found (D3DKMT)"),
                }
                for component in Component::ALL {
                    let Some((module, file)) = pick(component) else {
                        println!("  {component}: no {}", component.driver_dll());
                        continue;
                    };
                    let get = symbol::<XessGetVersion>(module, component.version_entry());
                    let mut v = Version::default();
                    let r = get.map_or(result::ERROR_NOT_IMPLEMENTED, |f| f(&raw mut v));
                    println!(
                        "  {component}: {} {v} ({})",
                        file.display(),
                        result::name(r)
                    );
                    if component == Component::SuperResolution {
                        let missing = xess::ENTRY_POINTS
                            .iter()
                            .filter(|e| {
                                symbol::<XessGetVersion>(
                                    module,
                                    &std::ffi::CString::new(**e).expect("name"),
                                )
                                .is_none()
                            })
                            .count();
                        if missing > 0 {
                            failures.push(format!("{missing} XeSS entry points missing"));
                        }
                    }
                }

                // Super resolution on this device.
                let Some((sr, _)) = pick(Component::SuperResolution) else {
                    release(device);
                    release(adapter);
                    continue;
                };
                let call = |name: &CStr| symbol::<XessContextCall>(sr, name);
                let create = symbol::<XessCreateContext>(sr, c"xessD3D12CreateContext")
                    .ok_or("xessD3D12CreateContext")?;
                let mut context = std::ptr::null_mut();
                let r = create(device, &raw mut context);
                println!("  xessD3D12CreateContext: {} ({r})", result::name(r));
                if r < 0 {
                    // DXGI lists the B580 a second time (another LUID) that
                    // takes no XeSS context: only no context at all fails.
                    release(device);
                    release(adapter);
                    continue;
                }
                probed += 1;
                let optimal = call(c"xessIsOptimalDriver")
                    .map_or(result::ERROR_NOT_IMPLEMENTED, |f| f(context));
                println!("  xessIsOptimalDriver: {}", result::name(optimal));
                let xefx_version = || {
                    let mut v = Version::default();
                    if let Some(f) = symbol::<XessContextVersion>(sr, c"xessGetIntelXeFXVersion") {
                        f(context, &raw mut v);
                    }
                    v
                };
                println!("  Intel XeFX before init: {}", xefx_version());

                let optimal_input =
                    symbol::<XessOptimalInput>(sr, c"xessGetOptimalInputResolution")
                        .ok_or("optimal input")?;
                let input = symbol::<XessInput>(sr, c"xessGetInputResolution");
                let legacy = symbol::<XessForceLegacy>(sr, c"xessForceLegacyScaleFactors");
                for out in [
                    Size::new(1920, 1080),
                    Size::new(2560, 1440),
                    Size::new(3840, 2160),
                ] {
                    println!("  output {out}:");
                    for q in Quality::ALL {
                        let (mut opt, mut min, mut max) =
                            (Size::default(), Size::default(), Size::default());
                        let r = optimal_input(
                            context,
                            &raw const out,
                            q.raw(),
                            &raw mut opt,
                            &raw mut min,
                            &raw mut max,
                        );
                        let want = xess::input_resolution(out, q);
                        let mut fixed = Size::default();
                        let rf = input.map_or(result::ERROR_NOT_IMPLEMENTED, |f| {
                            f(context, &raw const out, q.raw(), &raw mut fixed)
                        });
                        println!(
                            "    {:<20} {:.1}x: optimal {opt} (range {min} .. {max}), fixed {fixed}{}",
                            q.name(),
                            q.scale(),
                            if r < 0 || rf < 0 {
                                format!(" [{} / {}]", result::name(r), result::name(rf))
                            } else {
                                String::new()
                            },
                        );
                        if r >= 0 && opt != want {
                            failures.push(format!("{q:?} at {out}: XeSS {opt}, module {want}"));
                        }
                    }
                }
                if let Some(force) = legacy {
                    let r = force(context, true);
                    let out = Size::new(3840, 2160);
                    let mut line = Vec::new();
                    for q in Quality::ALL {
                        let (mut opt, mut min, mut max) =
                            (Size::default(), Size::default(), Size::default());
                        optimal_input(
                            context,
                            &raw const out,
                            q.raw(),
                            &raw mut opt,
                            &raw mut min,
                            &raw mut max,
                        );
                        line.push(format!("{} {opt}", q.name()));
                        let want = xess::legacy_input_resolution(out, q);
                        if want != opt {
                            failures.push(format!("legacy {q:?}: XeSS {opt}, module {want}"));
                        }
                    }
                    println!(
                        "  legacy factors ({}): {}",
                        result::name(r),
                        line.join(", ")
                    );
                    force(context, false);
                }
                if let Some(f) = symbol::<XessProperties>(sr, c"xessGetProperties") {
                    for out in [Size::new(1920, 1080), Size::new(3840, 2160)] {
                        let mut p = Properties::default();
                        let r = f(context, &raw const out, &raw mut p);
                        println!(
                            "  xessGetProperties {out}: {} descriptors, buffers {}, textures {} ({})",
                            p.required_descriptor_count,
                            numerobis::pci::bytes(p.temp_buffer_heap_size),
                            numerobis::pci::bytes(p.temp_texture_heap_size),
                            result::name(r)
                        );
                    }
                }
                // Init builds the network's pipelines (blocking).
                if let Some(init) = symbol::<XessInit>(sr, c"xessD3D12Init") {
                    let params = D3d12InitParams::new(Size::new(1920, 1080), Quality::Quality);
                    let start = std::time::Instant::now();
                    let r = init(context, &raw const params);
                    println!(
                        "  xessD3D12Init 1920x1080 quality: {} in {:.0} ms",
                        result::name(r),
                        start.elapsed().as_secs_f64() * 1e3
                    );
                    // xessGetIntelXeFXVersion stays 0.0.0 on the B580 (driver
                    // 2.0.2 and SDK 1.3.1 alike): no evidence either way.
                    // Which XeSS libraries the process holds now says more.
                    println!(
                        "  Intel XeFX after init: {}; XeSS libraries loaded: {}",
                        xefx_version(),
                        loaded_modules(&["xess", "xefx", "xell"]).join(", ")
                    );
                    if r < 0 {
                        failures.push(format!("xessD3D12Init: {}", result::name(r)));
                    }
                }
                if let Some(destroy) = call(c"xessDestroyContext") {
                    destroy(context);
                }
                release(device);
                release(adapter);
            }
            release(factory);
            if probed == 0 {
                failures.push("no adapter took a XeSS context".into());
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures.join("; "))
        }
    }
}

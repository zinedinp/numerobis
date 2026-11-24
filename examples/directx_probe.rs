//! Runs the Direct3D tables against the installed drivers, for every DXGI
//! adapter and every selected version:
//!
//! - DXGI: the adapter next to `pci::bar`, the video memory budget, DXVK.
//! - Direct3D 12 under the debug layer: the feature level, every
//!   capability through `CheckFeatureSupport`, a command queue at the
//!   asked priority, vkd3d-proton.
//! - Direct3D 11 under the debug layer, Direct3D 10.1: level and
//!   capabilities.
//! - Direct3D 9: the caps, and Direct3D 8's as d3d8to9 / DXVK convert them
//!   (Windows has no 64-bit `d3d8.dll`); a 32-bit build reads the real
//!   Direct3D 8 caps too and compares.
//!
//! `cargo run --example directx_probe --features directx`
//! (`NUMEROBIS_DX=12` pins a version, `auto-11` detects down to 11, default
//! `auto`; `NUMEROBIS_DX_PRIORITY=realtime|high|normal`, default `high`;
//! `NUMEROBIS_NO_VALIDATION=1` skips the debug layers). Exits non-zero on a
//! debug layer error or a version that was pinned and failed.
//!
//! No bindings: the DLLs are opened by hand and COM is called through the
//! vtables, with the method indices of the SDK headers. Windows only;
//! tested on an Intel Arc B580.

#![allow(
    clippy::too_many_lines,
    clippy::items_after_statements,
    reason = "an example that reads top to bottom, one COM call after another,               each signature next to its call"
)]

#[cfg(not(windows))]
fn main() {
    eprintln!("directx_probe: Windows only (Direct3D is reached through Wine/Proton elsewhere)");
}

#[cfg(windows)]
fn main() {
    windows::main();
}

#[cfg(windows)]
mod windows {
    use std::ffi::{CStr, c_char, c_void};
    use std::ptr::null_mut;

    use numerobis::directx::generated::iid;
    use numerobis::directx::legacy::{self, Caps9};
    use numerobis::directx::modern::queue::{self, ListType, QueuePriority};
    use numerobis::directx::modern::{
        self, Adapter, BudgetCache, Capability, FeatureLevels, Plan, Probe, VideoMemory, cache,
        dxgi,
    };
    use numerobis::directx::{Api, FeatureLevel, Guid, LegacyLevel, Runtime, Select};
    use numerobis::pci::bytes;

    type Hresult = i32;
    type Com = *mut c_void;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryA(name: *const c_char) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
    }

    /// A function from a system DLL, or `None` when it isn't there.
    fn load<F: Copy>(dll: &CStr, name: &CStr) -> Option<F> {
        assert_eq!(size_of::<F>(), size_of::<*mut c_void>());
        // SAFETY: plain Win32 calls with NUL-terminated names; the caller
        // names `F` as the function's signature.
        unsafe {
            let module = LoadLibraryA(dll.as_ptr());
            if module.is_null() {
                return None;
            }
            let f = GetProcAddress(module, name.as_ptr());
            (!f.is_null()).then(|| std::mem::transmute_copy(&f))
        }
    }

    /// Method `index` of a COM object's vtable, typed as `F`.
    ///
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

    /// `QueryInterface`; the new reference is released at once, the answer
    /// is only whether the object has it.
    fn has(obj: Com, id: &Guid) -> bool {
        let mut out = null_mut();
        // SAFETY: IUnknown::QueryInterface is entry 0.
        let hr = unsafe {
            method::<unsafe extern "system" fn(Com, *const Guid, *mut Com) -> Hresult>(obj, 0)(
                obj,
                id,
                &raw mut out,
            )
        };
        release(out);
        hr >= 0
    }

    fn query(obj: Com, id: &Guid) -> Option<Com> {
        let mut out = null_mut();
        // SAFETY: as in `has`.
        let hr = unsafe {
            method::<unsafe extern "system" fn(Com, *const Guid, *mut Com) -> Hresult>(obj, 0)(
                obj,
                id,
                &raw mut out,
            )
        };
        (hr >= 0 && !out.is_null()).then_some(out)
    }

    /// The layer an object belongs to, by the interfaces only it answers.
    fn layer(obj: Com, kind: &str) -> Option<Runtime> {
        Runtime::ALL.into_iter().find(|r| {
            r.interfaces()
                .iter()
                .any(|(on, _, id)| *on == kind && has(obj, id))
        })
    }

    /// `D3D12_MESSAGE` / `D3D11_MESSAGE`.
    #[repr(C)]
    struct Message {
        category: i32,
        severity: i32,
        id: i32,
        description: *const c_char,
        length: usize,
    }

    /// Errors (corruption, error) an `ID3D12InfoQueue` / `ID3D11InfoQueue`
    /// holds, printed. Both have `GetMessage` at 5 and
    /// `GetNumStoredMessages` at 8.
    fn info_errors(queue: Com, what: &str) -> usize {
        type Count = unsafe extern "system" fn(Com) -> u64;
        type Get = unsafe extern "system" fn(Com, u64, *mut Message, *mut usize) -> Hresult;
        let mut errors = 0;
        // SAFETY: the info queue vtables as the SDK headers list them; the
        // message buffer is as long as the first call says.
        unsafe {
            let n = method::<Count>(queue, 8)(queue);
            for i in 0..n {
                let mut len = 0usize;
                method::<Get>(queue, 5)(queue, i, null_mut(), &raw mut len);
                let mut buf = vec![0u64; len.div_ceil(8)];
                let msg = buf.as_mut_ptr().cast::<Message>();
                if method::<Get>(queue, 5)(queue, i, msg, &raw mut len) < 0 {
                    continue;
                }
                let severity = (*msg).severity;
                if severity <= 1 {
                    errors += 1;
                    let text = CStr::from_ptr((*msg).description).to_string_lossy();
                    println!("  {what} debug layer ERROR {}: {text}", (*msg).id);
                }
            }
        }
        errors
    }

    #[repr(C)]
    struct PrivilegeSet {
        count: u32,
        control: u32,
        luid: [u32; 2],
        attributes: u32,
    }

    /// Whether this process holds `privilege`, enabling it in the token
    /// first (administrators have `SeIncreaseBasePriorityPrivilege`, off).
    fn has_privilege(privilege: &str) -> bool {
        #[link(name = "advapi32")]
        unsafe extern "system" {
            fn OpenProcessToken(process: *mut c_void, access: u32, token: *mut *mut c_void) -> i32;
            fn LookupPrivilegeValueA(
                system: *const c_char,
                name: *const c_char,
                luid: *mut [u32; 2],
            ) -> i32;
            fn AdjustTokenPrivileges(
                token: *mut c_void,
                disable_all: i32,
                new: *const PrivilegeSet,
                len: u32,
                old: *mut c_void,
                ret: *mut u32,
            ) -> i32;
            fn PrivilegeCheck(token: *mut c_void, set: *mut PrivilegeSet, result: *mut i32) -> i32;
        }
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetCurrentProcess() -> *mut c_void;
            fn CloseHandle(handle: *mut c_void) -> i32;
        }
        const TOKEN_ADJUST_PRIVILEGES: u32 = 0x20;
        const TOKEN_QUERY: u32 = 0x8;
        const SE_PRIVILEGE_ENABLED: u32 = 0x2;
        let Ok(name) = std::ffi::CString::new(privilege) else {
            return false;
        };
        // SAFETY: Win32 token calls on this process's token, closed at the end.
        unsafe {
            let mut token = null_mut();
            if OpenProcessToken(
                GetCurrentProcess(),
                TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY,
                &raw mut token,
            ) == 0
            {
                return false;
            }
            let mut set = PrivilegeSet {
                count: 1,
                control: 1,
                luid: [0; 2],
                attributes: SE_PRIVILEGE_ENABLED,
            };
            let mut held = 0;
            if LookupPrivilegeValueA(std::ptr::null(), name.as_ptr(), &raw mut set.luid) != 0 {
                // Fails quietly (ERROR_NOT_ALL_ASSIGNED) when the token lacks it.
                AdjustTokenPrivileges(token, 0, &raw const set, 0, null_mut(), null_mut());
                PrivilegeCheck(token, &raw mut set, &raw mut held);
            }
            CloseHandle(token);
            held != 0
        }
    }

    fn env(name: &str) -> Option<String> {
        std::env::var(name).ok().filter(|v| !v.is_empty())
    }

    pub fn main() {
        let select: Select = env("NUMEROBIS_DX").map_or(Select::DETECT, |s| {
            s.parse().unwrap_or_else(|e| panic!("NUMEROBIS_DX: {e}"))
        });
        let priority: QueuePriority =
            env("NUMEROBIS_DX_PRIORITY").map_or(QueuePriority::High, |s| {
                s.parse()
                    .unwrap_or_else(|e| panic!("NUMEROBIS_DX_PRIORITY: {e}"))
            });
        let validate = env("NUMEROBIS_NO_VALIDATION").is_none();
        println!(
            "selection: {select}, queue priority {priority}, debug layers {}",
            if validate { "on" } else { "off" }
        );
        numerobis::pci::bar::warm_up();

        let mut failed = false;
        let mut errors = 0;
        let mut plans: Vec<Plan> = Vec::new();
        let mut created: Vec<Api> = Vec::new();

        // DXGI.
        type CreateFactory = unsafe extern "system" fn(*const Guid, *mut Com) -> Hresult;
        let create_factory: CreateFactory =
            load(c"dxgi.dll", c"CreateDXGIFactory1").expect("dxgi.dll");
        let mut factory = null_mut();
        // SAFETY: CreateDXGIFactory1(riid, ppFactory).
        let hr = unsafe { create_factory(&iid::IDXGIFactory1, &raw mut factory) };
        assert!(hr >= 0, "CreateDXGIFactory1: {hr:#x}");
        if let Some(r) = layer(factory, "IDXGIFactory") {
            println!("DXGI is {r}");
        }

        if validate && select.order().contains(&Api::D3D12) {
            type GetDebug = unsafe extern "system" fn(*const Guid, *mut Com) -> Hresult;
            if let Some(get) = load::<GetDebug>(c"d3d12.dll", c"D3D12GetDebugInterface") {
                let mut debug = null_mut();
                // SAFETY: D3D12GetDebugInterface, then ID3D12Debug::EnableDebugLayer (3).
                unsafe {
                    if get(&iid::ID3D12Debug, &raw mut debug) >= 0 {
                        method::<unsafe extern "system" fn(Com)>(debug, 3)(debug);
                        println!("D3D12 debug layer enabled");
                    } else {
                        println!("D3D12 debug layer not installed (Graphics Tools)");
                    }
                }
                release(debug);
            }
        }

        let mut index = 0;
        loop {
            let mut adapter: Com = null_mut();
            // SAFETY: IDXGIFactory1::EnumAdapters1 (12).
            let hr = unsafe {
                method::<unsafe extern "system" fn(Com, u32, *mut Com) -> Hresult>(factory, 12)(
                    factory,
                    index,
                    &raw mut adapter,
                )
            };
            if hr < 0 {
                break; // DXGI_ERROR_NOT_FOUND
            }
            index += 1;
            let mut desc = dxgi::AdapterDesc1::default();
            // SAFETY: IDXGIAdapter1::GetDesc1 (10).
            unsafe {
                method::<unsafe extern "system" fn(Com, *mut dxgi::AdapterDesc1) -> Hresult>(
                    adapter, 10,
                )(adapter, &raw mut desc);
            }
            let info = Adapter::from_desc1(&desc);
            println!("\nadapter {}: {}", index - 1, info.describe());
            if let Some(bar) = info.bar() {
                println!("  PCI: {}", bar_line(bar));
            }
            if let Some(a3) = query(adapter, &iid::IDXGIAdapter3) {
                for (group, name) in [
                    (dxgi::SegmentGroup::Local, "local"),
                    (dxgi::SegmentGroup::NonLocal, "non-local"),
                ] {
                    let mut budget = BudgetCache::new(std::time::Duration::ZERO);
                    let memory = budget.refresh(|m: &mut VideoMemory| {
                        // SAFETY: IDXGIAdapter3::QueryVideoMemoryInfo (14).
                        unsafe {
                            method::<
                                unsafe extern "system" fn(
                                    Com,
                                    u32,
                                    u32,
                                    *mut VideoMemory,
                                )
                                    -> Hresult,
                            >(a3, 14)(a3, 0, group as u32, m)
                                >= 0
                        }
                    });
                    if let Some(m) = memory {
                        println!("  video memory {name}: {m}");
                    }
                }
                release(a3);
            }

            for api in select.order() {
                let result = match api {
                    Api::D3D12 => d3d12(adapter, &info, priority, validate, &mut errors),
                    Api::D3D11 => d3d11(adapter, &info, validate, &mut errors),
                    Api::D3D10 => d3d10(adapter, &info),
                    // The legacy versions enumerate adapters themselves.
                    Api::D3D9 | Api::D3D8 => continue,
                };
                match result {
                    Some(plan) => {
                        if !created.contains(&api) {
                            created.push(api);
                        }
                        plans.push(plan);
                    }
                    None => println!("  {api}: not created"),
                }
            }
            release(adapter);
        }
        release(factory);

        let legacy = select
            .order()
            .into_iter()
            .filter(|a| a.era() == numerobis::directx::Era::Legacy);
        for api in legacy {
            if api == Api::D3D9 && d3d9(select) {
                created.push(Api::D3D9);
            }
            if api == Api::D3D8 && d3d8() {
                created.push(Api::D3D8);
            }
        }

        if let Select::Pin(api) = select
            && !created.contains(&api)
        {
            println!("\n{api} was pinned and could not be created");
            failed = true;
        }
        let picked = select.first(|api| created.contains(&api).then_some(()));
        println!(
            "\nselection {select} picks: {}",
            picked.map_or_else(|| "nothing".into(), |(a, ())| a.to_string())
        );

        // The process cache, as a program would keep it.
        let kept = cache::get_or_init(|| plans);
        let required = [Capability::Raytracing11, Capability::MeshShader];
        match cache::best(&required) {
            Some((i, plan)) => println!(
                "cache: {} plans, best with raytracing 1.1 + mesh shaders: #{i}, {} on {}",
                kept.len(),
                plan.api,
                plan.adapter().map_or("?", |a| a.name.as_str())
            ),
            None => println!(
                "cache: {} plans, none with raytracing 1.1 + mesh shaders",
                kept.len()
            ),
        }

        if errors > 0 {
            println!("{errors} debug layer error(s)");
            failed = true;
        }
        if failed {
            std::process::exit(1);
        }
    }

    fn d3d12(
        adapter: Com,
        info: &Adapter,
        priority: QueuePriority,
        validate: bool,
        errors: &mut usize,
    ) -> Option<Plan> {
        type Create = unsafe extern "system" fn(Com, u32, *const Guid, *mut Com) -> Hresult;
        type Check = unsafe extern "system" fn(Com, u32, *mut c_void, u32) -> Hresult;
        let create: Create = load(c"d3d12.dll", c"D3D12CreateDevice")?;
        let mut device = null_mut();
        // SAFETY: D3D12CreateDevice(adapter, 11_0, riid, ppDevice).
        let hr = unsafe {
            create(
                adapter,
                FeatureLevel::L11_0.raw(),
                &iid::ID3D12Device,
                &raw mut device,
            )
        };
        if hr < 0 {
            println!("  D3D12CreateDevice: {hr:#x}");
            return None;
        }
        // SAFETY: ID3D12Device::CheckFeatureSupport (13) for each call.
        let check = |feature: u32, data: *mut c_void, size: u32| unsafe {
            method::<Check>(device, 13)(device, feature, data, size) >= 0
        };
        static LEVELS: [u32; 5] = modern::D3D12_LEVELS;
        let mut levels = FeatureLevels::new(&LEVELS);
        check(
            modern::D3D12_FEATURE_FEATURE_LEVELS,
            (&raw mut levels).cast(),
            u32::try_from(size_of::<FeatureLevels>()).expect("small"),
        );
        let level = levels.max().unwrap_or(FeatureLevel::L11_0);
        let mut probe = Probe::new(Api::D3D12, level, Capability::ALL).with_adapter(info.clone());
        probe.fill(check);
        let plan = probe.plan();
        if let Some(r) = layer(device, "ID3D12Device") {
            println!("  Direct3D 12 is {r}");
        }
        print!("  {}", indent(&plan.to_string()));
        if plan.has(Capability::GpuUploadHeap) {
            let bar = info.bar().map_or_else(|| "no PCI entry".into(), bar_line);
            println!("  GPU upload heap granted; PCI: {bar}");
        }

        // A queue at the asked priority, stepping down when refused.
        let order = queue::candidates(ListType::Direct, priority, check, |privilege| {
            let held = has_privilege(privilege);
            if !held {
                println!(
                    "  global realtime needs {privilege}, which this process lacks: not tried"
                );
            }
            held
        });
        let names: Vec<&str> = QueuePriority::ALL
            .iter()
            .filter(|p| queue::supported(ListType::Direct, **p, check))
            .map(|p| p.name())
            .collect();
        println!("  direct queue priorities supported: {}", names.join(", "));
        type CreateQueue =
            unsafe extern "system" fn(Com, *const [u32; 4], *const Guid, *mut Com) -> Hresult;
        let made = queue::create_with_fallback(&order, |p| {
            // D3D12_COMMAND_QUEUE_DESC { Type, Priority, Flags, NodeMask }.
            let desc = [ListType::Direct.raw(), p.raw(), 0, 0];
            let mut q = null_mut();
            // SAFETY: ID3D12Device::CreateCommandQueue (8).
            let hr = unsafe {
                method::<CreateQueue>(device, 8)(
                    device,
                    &raw const desc,
                    &iid::ID3D12CommandQueue,
                    &raw mut q,
                )
            };
            if hr < 0 {
                println!("  queue priority {p}: {hr:#x}, trying lower");
                Err(hr)
            } else {
                Ok(q)
            }
        });
        match made {
            Ok((q, got)) => {
                println!("  CreateCommandQueue: ok, priority {got} (asked {priority})");
                release(q);
            }
            Err(hr) => println!("  CreateCommandQueue FAILED: {hr:#x}"),
        }

        if validate && let Some(iq) = query(device, &iid::ID3D12InfoQueue) {
            *errors += info_errors(iq, "D3D12");
            release(iq);
        }
        release(device);
        Some(plan)
    }

    fn d3d11(adapter: Com, info: &Adapter, validate: bool, errors: &mut usize) -> Option<Plan> {
        type Create = unsafe extern "system" fn(
            Com,
            u32,
            *mut c_void,
            u32,
            *const u32,
            u32,
            u32,
            *mut Com,
            *mut u32,
            *mut Com,
        ) -> Hresult;
        type Check = unsafe extern "system" fn(Com, u32, *mut c_void, u32) -> Hresult;
        let create: Create = load(c"d3d11.dll", c"D3D11CreateDevice")?;
        let levels = FeatureLevel::down_from(FeatureLevel::L12_1, FeatureLevel::L9_1);
        let attempt = |flags: u32| {
            let mut device = null_mut();
            let mut level = 0u32;
            // SAFETY: D3D11CreateDevice(adapter, D3D_DRIVER_TYPE_UNKNOWN, no
            // software module, flags, levels, D3D11_SDK_VERSION, ...).
            let hr = unsafe {
                create(
                    adapter,
                    0,
                    null_mut(),
                    flags,
                    levels.as_ptr(),
                    u32::try_from(levels.len()).expect("few"),
                    Api::D3D11.sdk_version().expect("11"),
                    &raw mut device,
                    &raw mut level,
                    null_mut(),
                )
            };
            (hr >= 0).then_some((device, level)).ok_or(hr)
        };
        // D3D11_CREATE_DEVICE_DEBUG (2) needs the SDK layers.
        let (device, raw_level) = match attempt(if validate { 2 } else { 0 }) {
            Ok(made) => {
                if validate {
                    println!("  D3D11 debug layer on");
                }
                made
            }
            Err(_) if validate => {
                println!("  D3D11 debug layer not installed");
                attempt(0).ok()?
            }
            Err(hr) => {
                println!("  D3D11CreateDevice: {hr:#x}");
                return None;
            }
        };
        let level = FeatureLevel::from_raw(raw_level)?;
        let mut probe = Probe::new(Api::D3D11, level, Capability::ALL).with_adapter(info.clone());
        // SAFETY: ID3D11Device::CheckFeatureSupport (33).
        probe.fill(|feature, data, size| unsafe {
            method::<Check>(device, 33)(device, feature, data, size) >= 0
        });
        let plan = probe.plan();
        if let Some(r) = layer(device, "ID3D11Device") {
            println!("  Direct3D 11 is {r}");
        }
        print!("  {}", indent(&plan.to_string()));
        if validate && let Some(iq) = query(device, &iid::ID3D11InfoQueue) {
            *errors += info_errors(iq, "D3D11");
            release(iq);
        }
        release(device);
        Some(plan)
    }

    fn d3d10(adapter: Com, info: &Adapter) -> Option<Plan> {
        type Create =
            unsafe extern "system" fn(Com, u32, *mut c_void, u32, u32, u32, *mut Com) -> Hresult;
        let create: Create = load(c"d3d10_1.dll", c"D3D10CreateDevice1")?;
        for &level in Api::D3D10.levels() {
            let mut device = null_mut();
            // SAFETY: D3D10CreateDevice1(adapter, D3D10_DRIVER_TYPE_HARDWARE,
            // no module, no flags, HardwareLevel, D3D10_1_SDK_VERSION, ...).
            let hr = unsafe {
                create(
                    adapter,
                    0,
                    null_mut(),
                    0,
                    level.raw(),
                    Api::D3D10.sdk_version().expect("10"),
                    &raw mut device,
                )
            };
            if hr < 0 {
                continue;
            }
            // SAFETY: ID3D10Device1::GetFeatureLevel (100).
            let got =
                unsafe { method::<unsafe extern "system" fn(Com) -> u32>(device, 100)(device) };
            release(device);
            let level = FeatureLevel::from_raw(got).unwrap_or(level);
            let plan = Probe::new(Api::D3D10, level, Capability::ALL)
                .with_adapter(info.clone())
                .plan();
            println!(
                "  Direct3D 10.1 at feature level {level}: {} of {} granted",
                plan.outcomes().iter().filter(|o| o.granted()).count(),
                plan.outcomes().len()
            );
            return Some(plan);
        }
        None
    }

    /// `D3DADAPTER_IDENTIFIER9`.
    #[repr(C)]
    struct Identifier9 {
        driver: [u8; 512],
        description: [u8; 512],
        device_name: [u8; 32],
        driver_version: i64,
        vendor_id: u32,
        device_id: u32,
        sub_sys_id: u32,
        revision: u32,
        device_identifier: Guid,
        whql_level: u32,
    }

    const _: () = assert!(size_of::<Identifier9>() == 1104);

    fn d3d9(select: Select) -> bool {
        type Create = unsafe extern "system" fn(u32) -> Com;
        let Some(create) = load::<Create>(c"d3d9.dll", c"Direct3DCreate9") else {
            println!("\nDirect3D 9: no d3d9.dll");
            return false;
        };
        // SAFETY: Direct3DCreate9(D3D_SDK_VERSION).
        let d3d = unsafe { create(Api::D3D9.sdk_version().expect("9")) };
        if d3d.is_null() {
            println!("\nDirect3D 9: Direct3DCreate9 failed");
            return false;
        }
        if let Some(r) = layer(d3d, "IDirect3D9") {
            println!("\nDirect3D 9 is {r}");
        }
        // SAFETY: IDirect3D9::GetAdapterCount (4).
        let count = unsafe { method::<unsafe extern "system" fn(Com) -> u32>(d3d, 4)(d3d) };
        for adapter in 0..count {
            // SAFETY: GetAdapterIdentifier (5), GetDeviceCaps (14) with HAL.
            let (id, caps) = unsafe {
                let mut id: Identifier9 = std::mem::zeroed();
                method::<unsafe extern "system" fn(Com, u32, u32, *mut Identifier9) -> i32>(d3d, 5)(
                    d3d,
                    adapter,
                    0,
                    &raw mut id,
                );
                let mut caps = Caps9::default();
                let hr = method::<unsafe extern "system" fn(Com, u32, u32, *mut Caps9) -> i32>(
                    d3d, 14,
                )(d3d, adapter, legacy::D3DDEVTYPE_HAL, &raw mut caps);
                if hr < 0 {
                    println!("\nDirect3D 9 adapter {adapter}: GetDeviceCaps {hr:#x}");
                    continue;
                }
                (id, caps)
            };
            let name = CStr::from_bytes_until_nul(&id.description)
                .map_or_else(|_| "?".into(), |c| c.to_string_lossy());
            println!(
                "\nDirect3D 9 adapter {adapter}: {name} [{:04x}:{:04x}]",
                id.vendor_id, id.device_id
            );
            let plan = legacy::plan(Api::D3D9, &caps, legacy::Capability::ALL);
            print!("  {}", indent(&plan.to_string()));
            if select.order().contains(&Api::D3D8) {
                let caps8 = caps.as_d3d8();
                let plan8 = legacy::plan(Api::D3D8, &caps8, legacy::Capability::ALL);
                println!("  as Direct3D 8 through d3d8to9 / DXVK's conversion:");
                print!("  {}", indent(&plan8.to_string()));
                *CAPS8_CONVERTED.lock().expect("lock") = Some(caps8);
            }
        }
        release(d3d);
        count > 0
    }

    static CAPS8_CONVERTED: std::sync::Mutex<Option<Caps9>> = std::sync::Mutex::new(None);

    /// The real Direct3D 8 runtime: 32-bit only.
    fn d3d8() -> bool {
        if !Api::D3D8.has_64bit_runtime() && cfg!(target_pointer_width = "64") {
            println!(
                "\nDirect3D 8: no 64-bit d3d8.dll; the caps above are the d3d8to9 / DXVK conversion"
            );
            return CAPS8_CONVERTED.lock().expect("lock").is_some();
        }
        type Create = unsafe extern "system" fn(u32) -> Com;
        let Some(create) = load::<Create>(c"d3d8.dll", c"Direct3DCreate8") else {
            println!("\nDirect3D 8: no d3d8.dll");
            return false;
        };
        // SAFETY: Direct3DCreate8(220).
        let d3d = unsafe { create(legacy::D3D8_SDK_VERSION) };
        if d3d.is_null() {
            println!("\nDirect3D 8: Direct3DCreate8 failed");
            return false;
        }
        let mut caps = Caps9::default();
        // SAFETY: IDirect3D8::GetDeviceCaps (13) writes the D3DCAPS8 prefix.
        let hr = unsafe {
            method::<unsafe extern "system" fn(Com, u32, u32, *mut Caps9) -> i32>(d3d, 13)(
                d3d,
                0,
                legacy::D3DDEVTYPE_HAL,
                &raw mut caps,
            )
        };
        release(d3d);
        if hr < 0 {
            println!("\nDirect3D 8: GetDeviceCaps {hr:#x}");
            return false;
        }
        let plan = legacy::plan(Api::D3D8, &caps, legacy::Capability::ALL);
        println!("\nDirect3D 8 (Microsoft's 32-bit runtime), adapter 0:");
        print!("  {}", indent(&plan.to_string()));
        if let Some(converted) = *CAPS8_CONVERTED.lock().expect("lock") {
            let differ: Vec<&str> = (0..legacy::CAPS8_WORDS)
                .filter(|&i| caps.words()[i] != converted.words()[i])
                .map(|i| legacy::CAPS9_MEMBERS[i])
                .collect();
            println!(
                "  real vs converted D3DCAPS8: {} of {} words differ{}{}",
                differ.len(),
                legacy::CAPS8_WORDS,
                if differ.is_empty() { "" } else { ": " },
                differ.join(", ")
            );
            let level = LegacyLevel::of(&converted);
            println!("  levels: real {}, converted {level}", plan.level);
        }
        true
    }

    fn bar_line(gpu: &numerobis::pci::bar::Gpu) -> String {
        format!(
            "resizable BAR {:?}, CPU-visible VRAM {}",
            gpu.resizable_bar(),
            gpu.cpu_visible().map_or_else(|| "unknown".into(), bytes)
        )
    }

    fn indent(s: &str) -> String {
        s.trim_end().replace('\n', "\n  ") + "\n"
    }
}

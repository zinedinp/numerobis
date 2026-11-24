# numerobis

Raw accelerator computing on GPU matrix engines, NPUs and accelerator cards,
with the number formats of [miraculix](https://crates.io/crates/miraculix).

numerobis talks to the **kernel driver UAPI directly** (ioctl and mmap): no CUDA,
ROCm, Level Zero or tt-metal user space is needed at run time. The kernels are
written once in OpenCL C (`kernels/src`), compiled ahead of time for every target
and checked in under `kernels/bin`. Every result is checked **bit for bit**
against a Rust reference built on the miraculix codecs.

Data lives on the device. `DeviceBuffer<E>` has no host view; the only ways in
and out are `Stream::upload` / `upload_packed` and `Stream::download`.

## Status

| Backend | Feature | Access | Hardware | Tiers with kernels | State |
|---|---|---|---|---|---|
| mock (reference) | `mock` | host | none | Emulated | done, tests only |
| NVIDIA | `nvidia` | raw: RM + UVM ioctls, GPFIFO, QMD | Ampere (sm_80/86), Ada (sm_89) | Emulated | **working**, bit-exact on RTX 3080 |
| Tenstorrent | `tt` | raw: tt-kmd, TLB windows, NoC | Blackhole | Emulated (one Tensix core, soft-float) | implemented; kernel code verified under qemu, on-card validation pending |
| Intel GPU | `intel` | raw: `xe` | Xe-LP ... Xe2 | - | planned |
| AMD GPU | `amd` | raw: KFD | gfx908 ... gfx12 | - | planned (HSACO kernels already built) |
| NPUs | `accel`, `xdna` | raw: `accel` subsystem | Intel NPU, AMD XDNA, RK3588, Ethos-U | - | planned |
| Raspberry Pi | `v3d` | raw: `v3d` | Pi 4/5 | - | planned |
| Vulkan, CUDA, HIP | `vulkan`, `cuda`, `hip` | comfort (vendor or portable user space) | any | - | planned (the Vulkan feature waterfall for graphics hosts is done, see below) |

"Emulated" is the portable tier: formats are decoded with integer bit logic and
the math runs in F32 with explicit FMA. It runs on every target. Native matrix
tiers (tensor cores, XMX, MFMA, Tensix FPU) are added per vendor on top of it.

## Cargo features

```toml
[dependencies]
numerobis = { version = "0.4.0-beta", features = ["nvidia", "tt"] }
```

| Feature | Enables |
|---|---|
| `nvidia` | raw NVIDIA backend (Linux, open kernel modules) |
| `tt` | raw Tenstorrent backend (Linux, tt-kmd 2.x) |
| `mock` | host reference backend (`Device::mock()`), for tests |
| `vulkan` | `numerobis::vulkan`: the Vulkan capability waterfall (and later the comfort compute backend) |
| `directx` | `numerobis::directx`: Direct3D 8 to 12 capabilities, legacy and modern apart |
| `intel`, `amd`, `xdna`, `v3d`, `accel`, `cuda`, `hip` | reserved, no code yet |

`vulkan` and `directx` build without each other; a host takes the one it
needs. With both, `directx::vulkan` converts between them.

Without any feature, `Device::all()` is empty and the crate still builds
(type layer, ops API, ELF loader, `pci`).

## Feature detection

`Device::all()` opens every device of the enabled backends; devices that fail to
open are skipped. Each device describes itself with a `FeatureSet`:

```rust
use numerobis::{Device, Engine};

for dev in Device::all() {
    let f = dev.features();
    println!("{} ({:?}), driver {}", f.name, f.class, f.driver);
    println!("  matrix engine via {:?}", f.access(Engine::Matrix)); // Some(Raw)
    for t in &f.tiers {
        println!("  tier {}", t.name()); // "emulated", "nvidia-sm80", ...
    }
}
```

- `class`: `Gpu`, `Npu` or `Accel`.
- `engines`: which engines are reachable and how: `Access::Raw` (kernel UAPI),
  `Access::Comfort(..)` (vendor user space) or `Access::Host` (mock).
- `tiers`: the matrix ladder of the device, lowest first. A tier listed there
  is a hardware capability; whether a kernel exists for a given op is asked per
  op (see below).

`cargo run --example probe --features nvidia,tt` prints this, plus the reason
when a device did not open:

```text
[0] NVIDIA GeForce RTX 3080 (Gpu, driver nvidia 620.70.06 (raw RM))
    engine Matrix: Raw
    engine Copy: Raw
    tiers: emulated < nvidia-sm80
```

### Choosing a device

```rust
use numerobis::{Device, Placement};

let fastest = Device::open(Placement::Fastest)?;   // highest tier, GPUs and cards first
let frugal  = Device::open(Placement::LowPower)?;  // NPUs first
let second  = Device::open(Placement::Device(1))?; // by index in Device::all()
```

### Choosing a tier

Every op takes a `Choice`:

- `Choice::Up`: the highest tier of the device that has a kernel for this op
  and these formats (falls back to `Emulated`).
- `Choice::Down(tier)`: exactly that tier, or `Error::Unsupported`. Useful to
  compare tiers against each other.

## Vulkan feature waterfall (`numerobis::vulkan`)

For a graphics host that wants the best Vulkan path the system allows, on
every OS (feature `vulkan`). Like `pci` it is data plus logic: no
Vulkan bindings, the host makes the calls with its own (ash, vulkanalia,
...), and the structs it hands over are plain C layouts.

- `vulkan::features`: per core version (1.0 to 1.4) its feature bits in
  declaration order, the struct that holds them, the extension each bit came
  from, the extensions the version promoted, which single struct a version
  struct absorbed when, and every extension's `depends` expression. Generated
  from `vk.xml` (see below).
- `vulkan::caps`: 94 `Capability`s, one module per area: `draw`, `mesh`
  (mesh and task shaders), `ray tracing` (acceleration structures,
  pipelines, ray queries, opacity micromaps, invocation reorder), `memory`,
  `shader`, `compute` (cooperative matrix, bfloat16, float8, memory model),
  `descriptor`, `pipeline` (dynamic state, pipeline libraries, shader
  objects, maintenance4-6), `sync` (timelines, queries, queue priority),
  `render` (texture compression, debug raster, shading rate, interlock,
  conservative rasterization), `present` (present id/wait), `interop` (fd, dma-buf and DRM
  modifiers on Linux; Win32 handles on Windows). Each has its routes in
  fallback order: newest core version first, then the extensions that
  offered it before, and may `need` others (`shader_object` needs
  `dynamic_rendering`). A route naming a member its struct doesn't have
  fails to compile.
- `Instance`: what `vkCreateInstance` needs: the `apiVersion`,
  `VK_KHR_portability_enumeration` and its flag (MoltenVK and other
  portability drivers stay hidden without), and on a 1.0 loader
  `VK_KHR_get_physical_device_properties2` and the other instance extensions
  1.1 made core.
- `Probe` / `Plan`: the waterfall. `vulkan::device` adds the device's
  identity (tied to `pci`: vendor, architecture, driver version), driver
  `Quirk`s and `rank` to pick among several devices; `vulkan::memory` the
  heaps, VRAM and how much of it the CPU can map.
- `vulkan::cache`: the plans kept for the process, like miraculix's
  `detect_features`: probe once at startup, ask `cache::plans()` /
  `cache::best()` anywhere later. `BudgetCache` reads the memory budget
  again once its copy is older than a set age (the one value that changes
  while the program runs).
- `vulkan::priority`: queue global priority (low, medium, high, realtime)
  for `vkCreateDevice`: the levels to try in order (only those the queue
  family allows when `global_priority_query` is granted), and
  `create_with_fallback`, which steps down on `VK_ERROR_NOT_PERMITTED`.
  It ranks this process's GPU queues against other processes on the same
  GPU; an NPU or another GPU has its own scheduler and doesn't see it.

```rust
use numerobis::vulkan::{ApiVersion, Capability as Cap, Chain, Instance, Offer, Probe, rank};

// 1. vkCreateInstance with instance.api.raw(), instance.flags and
//    instance.extensions() plus the host's own. `loader_version` is `None`
//    when the loader has no vkEnumerateInstanceVersion (1.0).
let instance = Instance::new(loader_version, ApiVersion::NEWEST, instance_extension_names);
// 2. Per physical device: its apiVersion and vkEnumerateDeviceExtensionProperties.
let device = ApiVersion::from_raw(props.api_version).expect("not Vulkan SC");
let offer = Offer::new(instance.api, device, extension_names)
    .with_instance(instance.extensions().iter().copied());
let mut probe = Probe::new(offer, &[Cap::DynamicRendering, Cap::Synchronization2, Cap::MeshShader]);
// 3. The queries (`probe.query()` says core or KHR entry points):
//    features2 for the bits, properties2 for subgroups, driver and PCI
//    address, memory_properties2 for heaps and budget.
unsafe {
    get_physical_device_features2(physical, probe.features2().cast());
    get_physical_device_properties2(physical, probe.properties2().cast());
    get_physical_device_memory_properties2(physical, probe.memory_properties2().cast());
}
let plan = probe.plan();
log::info!("{plan}");
// 4. With several devices: the best one that has everything required.
let best = rank(&plans, &[Cap::DynamicRendering, Cap::Synchronization2]).first().copied();
//    Or keep them for the process and ask later, without probing again:
let plans = numerobis::vulkan::cache::get_or_init(|| plans);
let (index, plan) = numerobis::vulkan::cache::best(&[Cap::DynamicRendering]).expect("a GPU");
//    Queue priority: the best level the driver allows, stepping down when refused.
let order = priority::candidates(plan, GlobalPriority::High, family.priorities.as_deref());
let (device, got) = priority::create_with_fallback(&order, |p| {
    // VkDeviceQueueCreateInfo.pNext = p.map(GlobalPriority::create_info).as_ptr()
    create_device(p)
})?;
//    The budget, fresh at most 100 ms old (per frame, before big allocations):
let mut budget = BudgetCache::for_plan(plan, Duration::from_millis(100));
let free = budget
    .get(|p| unsafe { get_physical_device_memory_properties2(physical, p.cast()) })
    .and_then(Memory::available_vram);
// 5. vkCreateDevice: pNext = chain.features2(), plus plan.extensions().
let mut chain = plan.device_chain();
if plan.supported(Chain::Core10, "fillModeNonSolid") == Some(true) {
    chain.enable(Chain::Core10, "fillModeNonSolid")?; // bits no capability covers
}
```

The device works at the lower of the instance's `apiVersion` and its own
version. On that version the first available route decides; if its bit is
off, the capability is missing. Later routes would read the same hardware
bit through another struct, and chaining a `VkPhysicalDeviceVulkan1xFeatures`
together with a single struct it absorbed is invalid. An extension route is
available when the extension and its `vk.xml` dependencies are met (core,
offered, or enabled on the instance), and `Plan::extensions` lists exactly
the device extensions that takes, `VK_KHR_swapchain` included for the
present capabilities. Bits the host adds go through `Probe::also_query` /
`Plan::supported` / `DeviceChain::enable`, which move a single struct into
its version struct for the same reason. `Plan::route` tells the host
whether a core or an extension entry point applies (core
`vkCmdDrawIndexedIndirectCount` vs. its `KHR` twin). A device that offers
`VK_KHR_portability_subset` gets it enabled, and the log names what the
subset leaves out. The log has one line per capability, and marks what was
never run on this OS:

```text
Vulkan 1.4 (instance asks 1.4, device has 1.4.359)
  device: Intel(R) Arc(TM) B580 Graphics, discrete, Intel Xe2-HPG [8086:e20b], driver 101.9175 (Intel Corporation)
  memory: 11.83 GiB VRAM, all CPU-visible, 15.97 GiB host
  + draw.draw_indirect_count: core 1.2 (VkPhysicalDeviceVulkan12Features.drawIndirectCount)
  - compute.float8: no route [skipped VK_EXT_shader_float8 (...): not offered]
  - interop.dma_buf: not on Windows
  + interop.external_memory_win32: VK_KHR_external_memory_win32
  extensions: VK_EXT_mesh_shader, VK_EXT_memory_priority, ...
```

`cargo run --example vulkan_probe` runs all of it against the installed
driver: every capability, the queries, `vkCreateDevice` with the plan's
chain and extensions under the validation layer (when installed), the
memory next to `pci::bar`, the budget read again, the cache, and the queue
priority (`NUMEROBIS_VK_PRIORITY=realtime|high|medium|low`, default `high`).
It fails on any validation error.
`NUMEROBIS_VK_API=1.1` caps the instance version, so the extension routes
and their dependencies are the ones exercised.

### Regenerating the tables

`src/vulkan/features/generated.rs` comes from the Vulkan registry
(`xml/vk.xml` of [Vulkan-Docs](https://github.com/KhronosGroup/Vulkan-Docs),
`registry/vk.xml` of Vulkan-Headers, or `share/vulkan/registry/vk.xml` of
the Vulkan SDK); its header records the version. To add a struct, list it in
`CHAINS` in the script; the dependencies of every extension a route names
are picked up by themselves.

```sh
python scripts/gen-vulkan.py path/to/vk.xml          # write
python scripts/gen-vulkan.py path/to/vk.xml --check  # fail if out of date
```

The script also fails on any `c"VK_..."` name under `src/vulkan` that the
registry doesn't know.

## DirectX (`numerobis::directx`)

Direct3D 8 to 12 for a graphics host (feature `directx`), on the same terms
as `vulkan`: data plus logic, no bindings, the host makes the COM calls and
hands over what they fill. The tables come from the Windows SDK headers
(`scripts/gen-directx.py`): every `CheckFeatureSupport` struct of D3D11 and
D3D12 with its id and members, the tier and shader model values,
`D3DCAPS9` and its bits, and the COM interface ids.

The two eras share nothing but the name, so they are kept apart:

- `directx::legacy`, **Direct3D 8 and 9**: the fixed-function pipeline next
  to shader models 1 to 3, read from one `D3DCAPS9` (`D3DCAPS8` is its first
  53 words, so one buffer holds either). 47 `Capability`s in `fixed
  function`, `shader`, `texture`, `output` and `vertex`. Direct3D had no
  feature levels then: `LegacyLevel` (fixed function, SM 1, 1.4, 2, 2.x, 3)
  is numerobis's own, with the nearest 9_x feature level where one fits.
- `directx::modern`, **Direct3D 10, 11 and 12** on DXGI and WDDM: no fixed
  function, geometry shaders from 10, Direct3D's own `FeatureLevel`s (9_1 to
  12_2, and 1_0_core / 1_0_generic for compute-only MCDM devices such as
  NPUs). 70 `Capability`s in twelve areas, with raytracing, mesh shaders,
  shading rate, sampler feedback and work graphs each their own. A
  capability's routes are alternatives: the feature level, a D3D12 member,
  a D3D11 member; D3D12 ones also need their HLSL shader model.

```rust
use numerobis::directx::{Api, FeatureLevel, Select};
use numerobis::directx::modern::{Capability as Cap, Probe, queue};

// Pin a version (`"11".parse()`) or detect the newest that works.
let select: Select = "auto".parse()?;
let (api, device) = select.first(|api| create_device(api)).expect("Direct3D");
// D3D12: the probe drives every CheckFeatureSupport, steps the shader model
// and root signature down where an older runtime refuses, and reads them.
let mut probe = Probe::new(Api::D3D12, level, &[Cap::Raytracing11, Cap::MeshShader])
    .with_adapter(adapter); // dxgi::Adapter::from_desc1(&GetDesc1)
probe.fill(|feature, data, size| unsafe { device.CheckFeatureSupport(feature, data, size) }.is_ok());
let plan = probe.plan();
// A queue at high priority, realtime only with SeIncreaseBasePriorityPrivilege.
let order = queue::candidates(queue::ListType::Direct, wanted, check, has_privilege);
```

Also there:

- `modern::dxgi`: the adapter (ids tied to `pci` as for Vulkan, `bar()` for
  resizable BAR, the LUID Vulkan reports too), the video memory budget with
  a `BudgetCache`, `rank`. `modern::cache` keeps the plans for the process.
- `modern::queue`: `D3D12_COMMAND_QUEUE_PRIORITY`. The device says which
  priorities a queue type takes, but not whether the process may: global
  realtime needs `SeIncreaseBasePriorityPrivilege`
  (`QueuePriority::privilege`), checked before trying.
- `directx::Runtime`: Microsoft, WARP, D3D9On12, the Xbox GDK, and the
  community layers DXVK, vkd3d-proton, wined3d and d3d8to9 (how Direct3D
  runs on Linux through Wine and Proton), with the interfaces only each
  answers (`QueryInterface` tells which one a device belongs to).
- `directx::vulkan` (with feature `vulkan`): conversions done in `const fn`s
  (queue priority, `DXGI_FORMAT` to `VkFormat` as DXVK maps them), the
  Vulkan capabilities DXVK / vkd3d-proton need before they report a Direct3D
  one, and the highest feature level a Vulkan device could back.

**Direct3D 8.** Microsoft ships `d3d8.dll` for 32-bit only. d3d8to9
(BSD-2-Clause) turns it into Direct3D 9 calls but builds for x86 only and
keeps shader handles in 32-bit `DWORD`s; DXVK's d3d8 (zlib) does the same on
its Direct3D 9, 64-bit. `Caps9::as_d3d8` is their caps conversion (DXVK's
bit list, d3d8to9's constant clamp, and three Direct3D 9 / 9Ex bits both
miss). Against Microsoft's real 32-bit runtime on the B580: 48 of 53 words
equal and every capability the same (a unit test keeps those caps).

**Older than 8** is not covered: DirectDraw, which Direct3D 7 lives on, is a
separate surface API, too much work for something nobody targets any more.
Direct3D 7-class hardware still shows up through the D3D8/9 caps as
`LegacyLevel::FixedFunction`. Not built yet either: DirectStorage (its own
redistributable and queues) and DirectSR (TODOs in `directx`).

`cargo run --example directx_probe --features directx` runs it against the
installed drivers: every adapter and every selected version, D3D12 and D3D11
under their debug layers, a command queue at the asked priority, the D3D9
caps with the D3D8 conversion, and in a 32-bit build the real D3D8 runtime
next to the conversion (`NUMEROBIS_DX=12|11|10|9|8|auto|auto-11`,
`NUMEROBIS_DX_PRIORITY=realtime|high|normal`). It fails on a debug layer
error or a pinned version that could not be created.

```sh
python scripts/gen-directx.py                # newest Windows SDK under Windows Kits
python scripts/gen-directx.py path/to/Include/10.0.26100.0 --check
```

## PCI BARs and resizable BAR (`numerobis::pci::bar`)

`pci::bar::scan()` lists every display-class PCI device from the OS, without
a driver API: whether the card can do resizable BAR, how wide and fast its
PCIe link is, and how much VRAM the CPU can map through the aperture BAR and
whether that BAR lies above 4 GiB ("Above 4G Decoding"). Linux reads sysfs
(`resource`, `resource<N>_resize`, `*_link_*`); Windows the configuration
manager (`cfgmgr32`: the assigned memory ranges, `DEVPKEY_PciDevice_*Link*`)
and the display driver's registry key for the VRAM size.
`vulkan::Memory::cpu_visible_vram` tells the same from Vulkan's side.

`scan()` reads the OS on its first call and keeps the result for the process
(BARs and links only change with a reboot, a firmware setting or a hot
plug); a failed read is tried again next time. `scan_fresh()` always reads,
`warm_up()` fills the cache at startup, and `find(vendor, device, address)` /
`vulkan::DeviceInfo::bar()` pick a Vulkan device's entry from it.

```text
$ cargo run --example bars
Intel(R) Arc(TM) B580 Graphics [8086:e20b] Intel Xe2-HPG at 0000:03:00.0
  link: behind the on-board switch, not reported by the OS (GPU-internal PCIe 1.0 x1 (0.25 GB/s))
  VRAM 12 GiB, CPU-visible 16 GiB at 0x2800000000 (above 4 GiB)
  resizable BAR: active
```

Cards with a PCIe switch on board (Intel Arc, recent AMD) show a narrow
internal link on the GPU itself; `Gpu::link` is the switch's upstream link,
the one in the slot. Linux reports it; Windows reports links on endpoints
only, so there it stays unknown. Whether a card is resizable-BAR capable but
has it turned off is known on Linux (`ResizableBar::Inactive`); Windows only
shows the aperture size.

## Platforms

What depends on the OS says per OS whether it was run there against real
hardware (`numerobis::platform::Support`): `Tested`, `Untested` (compiled
and unit-tested only) or `Unavailable`. The plan log marks a granted
capability that is untested on its OS with `(untested on Linux)`.

| Part | Linux | Windows |
|---|---|---|
| raw NVIDIA backend | tested (RTX 3080) | unavailable |
| raw Tenstorrent backend | kernel code under qemu, on-card pending | unavailable |
| `dmabuf`, `sync` | Linux only | unavailable |
| `pci` tables, `vulkan` tables and waterfall, `directx` tables | unit-tested | unit-tested |
| Vulkan capabilities on a real driver | untested | 77 tested (Intel Arc B580, driver 101.9175, API 1.0 to 1.4 under validation) |
| Direct3D 10 to 12 capabilities on a real driver | untested (DXVK, vkd3d-proton) | 59 tested (B580: D3D12 at 12_2 with shader model 6.8, D3D11, 10.1, under the debug layers) |
| Direct3D 8 / 9 capabilities | untested (DXVK) | 45 tested (B580: D3D9, and D3D8 from Microsoft's 32-bit runtime) |
| `directx::modern::queue` | untested | tested: normal and high; global realtime refused without the privilege, with it untested |
| `vulkan::priority` | untested | tested: all four levels accepted on the B580, API 1.0 to 1.4; the `NOT_PERMITTED` fallback unit-tested only |
| `vulkan::interop` fd / dma-buf / DRM modifiers | untested | unavailable |
| `vulkan::interop` Win32 handles | unavailable | tested |
| `pci::bar` | untested (parsers unit-tested) | tested (Intel Arc B580); a 32-bit process gets no ranges (the OS hides them under WOW64, reported as such) |
| 32-bit x86 (`i686-pc-windows-msvc`) | untested | tested: lib tests, `vulkan_probe`, `directx_probe` (needs miraculix 0.6.5) |
| `examples/vulkan_probe` loader | untested (`libvulkan.so.1`) | tested (`vulkan-1.dll`) |
| `examples/directx_probe` | unavailable (a Windows program; under Wine untested) | tested |

`Capability::spec().support` has the per-capability flag; on Windows the
capabilities the B580 doesn't have (bfloat16, float8, shader objects, ASTC,
the present extensions) stay untested.

## Usage

```rust
use numerobis::{Choice, Device, GemmShape, Overflow, Placement, elem::*, ops};

let dev = Device::open(Placement::Fastest)?;
let mut s = dev.stream();

// Upload F32, convert to FP8 on the device.
let a = s.upload::<F32>(&[1.0, 2.0, 3.0, 4.0])?;      // 2x2, row-major
let b = s.upload::<F32>(&[1.0, 0.0, 0.0, 1.0])?;      // 2x2, stored N x K
let mut a8 = s.alloc::<E4M3Fn>(4)?;
let mut b8 = s.alloc::<E4M3Fn>(4)?;
ops::convert(&mut s, Choice::Up, &a, &mut a8, Overflow::Saturate)?.wait()?;
ops::convert(&mut s, Choice::Up, &b, &mut b8, Overflow::Saturate)?.wait()?;

// C = 1.0 * A * B^T + 0.0 * C, accumulated in F32.
let mut c = s.alloc::<F32>(4)?;
ops::gemm(&mut s, Choice::Up, GemmShape::new(2, 2, 2), 1.0, &a8, &b8, 0.0, &mut c)?.wait()?;
assert_eq!(s.download(&c)?.wait()?.to_vec(), [1.0, 2.0, 3.0, 4.0]);
```

Ops return `Pending<T>` (`#[must_use]`); `wait()` blocks with a timeout,
`elapsed()` reports device time where the backend measures it (NVIDIA: GPU
timestamps). Buffers used by in-flight work stay alive until it retires.
Invalid combinations fail to compile: converting into a block format,
quantizing to GGUF or packing `I4` from `U8` are type errors.

## What is offered

### Formats (`numerobis::elem`)

| Kind | Types |
|---|---|
| Float | `F32`, `Tf32`, `F16`, `Bf16`, `E4M3Fn`, `E4M3Fnuz`, `E5M2`, `E5M2Fnuz`, `E2M1`, `E2M3`, `E3M2` |
| Integer | `I32`, `I8`, `U8`, packed `I4`, `U4`, `I2`, `I1` |
| Block-scaled | `Mx<E2M1 / E2M3 / E3M2 / E4M3Fn / E5M2 / I8>` (OCP MX, 32-element blocks), `Nvfp4` (16-element blocks + tensor scale), `Bfp<M7 / M3 / M1>` (block floating point, 16 per shared exponent) |
| GGUF | `Q8_0`, `Q6K` (dequantize) |

Device layout: packed sub-byte formats are LSB-first; block formats store the
elements first, then one scale byte per block; GGUF blocks are stored natively.

### Ops (`numerobis::ops`)

| Op | What |
|---|---|
| `convert` | float <-> float, with `Overflow::Saturate` or `Overflow::Special` |
| `quantize` / `dequantize` | F32 <-> MX, NVFP4, BFP; GGUF -> F32 |
| `pack` / `unpack` | `I8`/`U8` <-> packed `I4`, `U4`, `I2`, `I1` (saturating) |
| `gemm` | `C = alpha*A*B^T + beta*C`, batched; float/block inputs with F32 or F16 accumulator, integer inputs with I32 |
| `absmax_block` | per-block max of abs(x), for choosing scales |

Conversions and quantization are bit-exact with miraculix on every target.
GEMM is bit-exact in the Emulated tier (one sequential FMA chain per output);
native matrix tiers will get per-tier tolerances.

## Requirements

- Linux. Access to the device nodes without root:
  - NVIDIA: `/dev/nvidiactl`, `/dev/nvidia0..`, `/dev/nvidia-uvm` (open kernel
    modules). The RM ABI changes between driver releases, so only versions whose
    headers the struct layouts were checked against are opened (`620.70.06`).
    `NUMEROBIS_NV_ANY_DRIVER=1` tries others at your own risk.
  - Tenstorrent: `/dev/tenstorrent/N` (tt-kmd 2.x, card firmware running).
- Building the kernels (only when changing them, binaries are checked in):
  clang with amdgcn/nvptx/spirv64/riscv32 targets and lld, `ptxas`,
  optionally `spirv-val`. `scripts/build-kernels.sh` builds,
  `scripts/build-kernels.sh --check` verifies `kernels/bin` is up to date.
- Regenerating the Vulkan tables (only when adding structs or versions):
  Python 3, rustfmt and a `vk.xml` ([see above](#regenerating-the-tables)).
- Regenerating the DirectX tables: Python 3, rustfmt and the Windows SDK
  headers (`um/` and `shared/` of `Include/<version>`).

### Tenstorrent notes

- The Emulated kernels run on the BRISC core of one Tensix tile, default
  `(1,2)`, override with `NUMEROBIS_TT_CORE=x,y`. A launch's buffers must fit
  the ~1.4 MiB L1 staging area; device buffers live in DRAM channel 0.
- The backend checks the ARC heartbeat (sysfs) before touching the card and
  never probes tiles: a NoC read to a harvested or non-Tensix tile can hang a
  Blackhole until it is reset (`tt-smi -r` or a cold boot).
- Do not run it while tt-metal uses the same card.

## Testing

```sh
cargo test                                # mock + kernels compiled for the host
cargo test --features nvidia,tt           # adds every device that opens
cargo test --features tt --test tt_memory -- --nocapture   # card GDDR6 and L1 check
```

The suite runs every op on every target and compares with the mock reference:

- `host-cl`: the kernel sources compiled as C for the CPU;
- `rv32-qemu`: the Tenstorrent build (RV32IM + `kernels/tt/softfloat.c`) under
  `qemu-riscv32`, when installed;
- every hardware device of the enabled features.

`tests/softfloat.rs` checks the soft-float against the host FPU on a million
random cases including subnormals and specials. `tests/binaries.rs` checks
that the checked-in kernel binaries export every kernel and need no linking.

The `host-cl` tests build a small driver with `clang`; without `clang` on the
`PATH` they fail with "cannot run clang". Against real drivers, without a
backend feature:

```sh
cargo run --example vulkan_probe    # every capability, vkCreateDevice under validation
NUMEROBIS_VK_API=1.0 cargo run --example vulkan_probe   # the extension routes instead
cargo run --example bars            # PCIe link, BAR aperture, resizable BAR
```

Lints: `Cargo.toml` turns on `clippy::pedantic`; exceptions carry a
`reason`. The Linux-only code is checked from any host with
`cargo clippy --target x86_64-unknown-linux-gnu --features nvidia,tt --all-targets`.

## Planned

- **NVIDIA:** `mma.sync` tensor-core tier (`nb_nv.h`, Ampere/Ada); Hopper and
  Blackwell (`wgmma`, `tcgen05`, native MX/NVFP4) through the `cuda` comfort
  backend first.
- **Tenstorrent:** spread work over all Tensix tiles, then the FPU matrix tier
  with native BFP8/4/2 and BF16; Wormhole.
- **Intel** (`xe`: COMPUTE_WALKER, XMX/DPAS), **AMD** (KFD: AQL queues,
  MFMA/WMMA), **NPUs** (Intel NPU, AMD XDNA, RK3588, Ethos-U) with dma-buf
  sharing between iGPU and NPU, **Raspberry Pi** (`v3d` QPU code generation).
- **Comfort `vulkan`** backend as the universal fallback (cooperative matrix).
- **DirectX:** DirectStorage (GPU decompression from NVMe into D3D12
  resources) and DirectSR (super resolution) probes; DirectDraw / Direct3D 7
  only if someone needs them.
- Per-target kernel crates, so the crate stays under the crates.io size limit.
- Upstreaming what the prototypes prove (see `docs/llvm-todo.md`): MiniFloat
  types and block-scaled intrinsics in LLVM, kernels in Rust.

## License

MIT OR Apache-2.0

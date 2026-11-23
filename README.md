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
| `intel`, `amd`, `xdna`, `v3d`, `accel`, `vulkan`, `cuda`, `hip` | reserved, no code yet (`numerobis::vulkan` needs no feature) |

Without any backend feature, `Device::all()` is empty and the crate still builds
(type layer, ops API, ELF loader, `pci`, `vulkan`).

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
every OS and without any backend feature (the `vulkan` Cargo feature is the
planned compute backend, not this). Like `pci` it is data plus logic: no
Vulkan bindings, the host makes the calls with its own (ash, vulkanalia,
...), and the structs it hands over are plain C layouts.

- `vulkan::features`: per core version (1.0 to 1.4) its feature bits in
  declaration order, the struct that holds them, the extension each bit came
  from, the extensions the version promoted, and which single struct a
  version struct absorbed when. Generated from `vk.xml` (see below).
- `vulkan::caps`: the `Capability`s, one module per area (`draw`, `memory`,
  `shader`, `descriptor`, `sync`, `render`). Each has its routes in fallback
  order: newest core version first, then the extensions that offered it
  before. Examples: `draw_indirect_count` is core 1.2 or
  `VK_KHR_draw_indirect_count`, `buffer_device_address` is core 1.2 or
  `VK_KHR_buffer_device_address`, `mesh_shader` is `VK_EXT_mesh_shader`
  (with its 1.2 dependencies as extensions on 1.1). A route naming a member
  its struct doesn't have fails to compile.
- `Instance`: what `vkCreateInstance` needs: the `apiVersion`,
  `VK_KHR_portability_enumeration` and its flag (MoltenVK and other
  portability drivers stay hidden without), and
  `VK_KHR_get_physical_device_properties2` on a 1.0 loader.
- `Probe` / `Plan`: the waterfall.

```rust
use numerobis::vulkan::{ApiVersion, Capability as Cap, Chain, Instance, Offer, Probe};

// 1. vkCreateInstance with instance.api.raw(), instance.flags and
//    instance.extensions() plus the host's own. `loader_version` is `None`
//    when the loader has no vkEnumerateInstanceVersion (1.0).
let instance = Instance::new(loader_version, ApiVersion::NEWEST, instance_extension_names);
// 2. The picked device: its apiVersion and vkEnumerateDeviceExtensionProperties.
let device = ApiVersion::from_raw(props.api_version).expect("not Vulkan SC");
let offer = Offer::new(instance.api, device, extension_names)
    .with_instance(instance.extensions().iter().copied());
let mut probe = Probe::new(offer, &[Cap::DynamicRendering, Cap::Synchronization2, Cap::MeshShader]);
// 3. One feature query; `features2()` is a ready VkPhysicalDeviceFeatures2
//    chain (`probe.query()` says which entry point). Subgroup capabilities
//    also want `properties2()` through vkGetPhysicalDeviceProperties2.
unsafe { get_physical_device_features2(physical, probe.features2().cast()) };
let plan = probe.plan();
log::info!("{plan}");
assert!(plan.lacking(&[Cap::DynamicRendering, Cap::Synchronization2]).is_empty());
// 4. vkCreateDevice: pNext = chain.features2(), plus plan.extensions().
let mut chain = plan.device_chain();
if plan.supported(Chain::Core10, "fillModeNonSolid") == Some(true) {
    chain.enable(Chain::Core10, "fillModeNonSolid")?; // bits no capability covers
}
```

The device works at the lower of the instance's `apiVersion` and its own
version. On that version the first available route decides; if its bit is
off, the capability is missing. Later routes would read the same hardware
bit through another struct, and chaining a `VkPhysicalDeviceVulkan1xFeatures`
together with a single struct it absorbed is invalid. Bits the host adds go
through `Probe::also_query` / `Plan::supported` / `DeviceChain::enable`,
which move a single struct into its version struct for the same reason.
`Plan::route` tells the host whether a core or an extension entry point
applies (core `vkCmdDrawIndexedIndirectCount` vs. its `KHR` twin). A device
that offers `VK_KHR_portability_subset` gets it enabled, and the log names
what the subset leaves out. The log has one line per capability:

```text
Vulkan 1.1 (instance asks 1.1, device has 1.3)
  + draw.draw_indirect_count: VK_KHR_draw_indirect_count [skipped core 1.2 (VkPhysicalDeviceVulkan12Features.drawIndirectCount): needs 1.2]
  - shader.float16: no route [skipped core 1.2 (...): needs 1.2; VK_KHR_shader_float16_int8 (...): not offered]
  extensions: VK_KHR_draw_indirect_count
```

### Regenerating the tables

`src/vulkan/features/generated.rs` comes from the Vulkan registry
(`xml/vk.xml` of [Vulkan-Docs](https://github.com/KhronosGroup/Vulkan-Docs),
or `registry/vk.xml` of Vulkan-Headers); its header records the version.
To add a struct, list it in `CHAINS` in the script.

```sh
python scripts/gen-vulkan.py path/to/vk.xml          # write
python scripts/gen-vulkan.py path/to/vk.xml --check  # fail if out of date
```

The script also fails on any `c"VK_..."` name under `src/vulkan` that the
registry doesn't know.

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
- Per-target kernel crates, so the crate stays under the crates.io size limit.
- Upstreaming what the prototypes prove (see `docs/llvm-todo.md`): MiniFloat
  types and block-scaled intrinsics in LLVM, kernels in Rust.

## License

MIT OR Apache-2.0

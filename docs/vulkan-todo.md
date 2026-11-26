# Vulkan TODOs

Two kinds: logic a graphics host (Melon) writes today that fits
`numerobis::vulkan` as it is, data plus logic with no bindings; and a call
layer on top of it that would make the host's Vulkan code safe without
costing it speed. Code marks each spot with `// TODO(vulkan): <short>` and
links here.

## Moved into numerobis (no bindings): done

| TODO | In the host today | Now in numerobis |
|---|---|---|
| `Pod` for `u64`, `i16`, `i64`, `f64`, arrays of `Pod`; a safe `as_bytes` | Melon `mesh_arena::as_bytes` (`T: Copy`, which lets padding through), `occlusion.rs`, `gfx/pipeline` | `elem::{Pod, as_bytes}` |
| Memory types per use with their fallbacks, block size per heap, first-fit ranges with alignment and `bufferImageGranularity`, the live count | Melon `mesh_arena::FreeList`, `vk_util::{memory_types, allocate_memory, MAPPED_FOR_GPU, DEVICE_OR_HOST, HOST_FIRST, READBACK}`, `Residency::mesh_memory` | `vulkan::alloc::{Usage, Memory::types_for, block_size, dedicated, Block, AllocationCount}` |
| Present policy | Melon `vk_util::recreate_after_present`, `surface_extent` | `vulkan::swapchain::{recreate, SurfaceCapabilities::current}` |
| Swapchain choices | Melon `swapchain.rs`, `vk_util::{composite_alpha, choose_surface_format}` | `vulkan::swapchain::{present_mode, surface_format, SurfaceCapabilities::{extent, image_count, composite_alpha}}` |
| Depth format waterfall | Melon `vk_util::find_depth_format` | `vulkan::format::depth` |

Where it differs from what the table asked for:

- **New modules, not `caps::present` / `caps::render`.** The `caps`
  modules hold only capability tables (`Spec`s); policy sits beside them
  as its own module, as `priority` does.
- **Surface format by `Encoding`, not "sRGB first".** Melon wants UNORM
  first (the shaders write sRGB-encoded colour, as vanilla's GL
  framebuffer stores it); a host rendering linear colour wants `_SRGB`.
- **Depth candidates.** D32, D32S8, X8D24, D24S8, D16 without stencil;
  D32S8, D24S8, D16S8 with; `sampled` also requires `SAMPLED_IMAGE`.
- **`types_for` is stricter than Melon's lists.** Each type once;
  never protected memory or AMD's device-coherent debug types; lazily
  allocated memory only for `Transient`; `Device` puts VRAM the CPU can't
  map before the BAR; "host first" past the budget leaves the BAR out
  (Melon's `HOST_FIRST` matches BAR types too when the driver lists them
  before the host ones).
- **`Block` counts bytes, with any alignment.** Melon's mesh `FreeList`
  counts elements; with `Block` the element size is the alignment (12-byte
  vertices work).

Melon takes numerobis from crates.io (0.5.4), so it moves to these after
the next release. It's all additive, so a 0.5.x release is enough.

## Colour: done, with keinbonus

`vulkan` and `directx` now depend on keinbonus (re-exported as
`numerobis::keinbonus`), so a swapchain, a video decoder and a CPU encoder
describe pixels with the same H.273 types.

| What | Where |
|---|---|
| Every `VkColorSpaceKHR` as primaries and curve, its extension, `for_signal`, scRGB's 80 cd/m² | `vulkan::color_space::ColorSpace` |
| `VkHdrMetadataEXT` from HDR10 metadata | `vulkan::color_space::HdrMetadata` |
| HDR10, HLG, scRGB, Display P3 or SDR, in the renderer's order | `vulkan::swapchain::{Output, output_format}` |
| `VK_EXT_swapchain_colorspace` on the instance, `VK_EXT_hdr_metadata` on the device | `Instance::with_color_spaces`, `Capability::{SwapchainColorSpace, HdrMetadata}` |
| DXGI swap chain colour spaces, `DXGI_HDR_METADATA_HDR10`, both to Vulkan | `directx::modern::color`, `directx::vulkan::color_space` |

Fixed on the way: a route on an instance extension (`VK_EXT_swapchain_colorspace`)
was checked against the device's extension list and never granted; it now
counts when the instance enabled it.

On the B580 both capabilities are granted and the instance and device come
up under validation without errors. What a real HDR swapchain does with
them is not tested yet: that needs a window on an HDR display.

Open:

- **An HDR swapchain on a display.** Present PQ and scRGB on the B580 with
  Windows HDR on; check `vkSetHdrMetadataEXT` reaches the display (the
  monitor's OSD or the HDMI/DP infoframe).
- **DXGI's YCbCr colour spaces** (video surfaces, `ID3D11VideoProcessor`),
  with chroma siting, and the "gamma 2.2" BT.2020 and gamma 2.4 RGB ones,
  which DXGI's text leaves open.
- **`VkSamplerYcbcrConversionCreateInfo` from a `Cicp`** (model, range, chroma
  offsets): decoded video sampled straight from its planes, checked against
  keinbonus's reference.
- **HDR10+ and Dolby Vision on the swapchain.** Vulkan has no extension for
  dynamic metadata (DXGI has `DXGI_HDR_METADATA_TYPE_HDR10PLUS`); a renderer
  tone maps with keinbonus itself.
- **AMD native HDR** (`VkSwapchainDisplayNativeHdrCreateInfoAMD`, local
  dimming) and HUAWEI's HDR Vivid metadata chain.

## A safe call layer: its own crate, `majestix-vk`

Melon's renderer has about 900 unsafe Vulkan calls, each justified by the
same three rules: every handle comes from the one device and outlives its
uses; nothing a frame in flight reads is changed or destroyed before that
frame's fence; a command buffer is recorded by one thread at a time. A layer
that enforces those rules by construction would take the `unsafe` out of the
host, where vulkano-style tracking would cost it per call.

- **Bindings stay out of numerobis.** The layer wraps `ash` in a crate of
  its own (see below), so a host on another binding still uses the
  waterfall alone.
- **Destruction only through a retire queue.** `Device::retire(handle)`
  files the handle under the frame being recorded and destroys it once that
  frame's fence has signalled. If nothing can be destroyed any other way,
  every handle a command buffer names is alive until its fence: recording
  becomes safe without per-command reference counting. `Drop` of the device
  waits idle, then empties the queue.
- **Handles are `Copy` and carry their device.** A small generation index
  per handle type instead of `Arc`, so a stale handle (destroyed, or from
  another device) is caught by a debug assertion, not by the driver.
- **Command buffers as typestate.** `Recording` → `InPass` (render pass or
  dynamic rendering, from the `Plan`) → back; draw calls exist only on
  `InPass`, copies and barriers only outside it. One owner (`&mut`), so
  recording stays on one thread without locks.
- **Mapped memory as slices.** A mapped range is handed out as
  `&mut [T]` for `T: Pod`, owned by its frame slot, so a write can neither
  run past the mapping nor reach a slot the GPU still reads.
- **Infos built from typed builders.** Create infos take the layer's
  handles, so they can only name live handles of the same device.
- **Generated from `vk.xml`.** `scripts/gen-vulkan.py` already reads the
  registry: the same pass can emit each command's parent handle and its
  `externsync` parameters, which is most of what the wrappers need to
  check.
- **Vulkan 1.0 floor.** The layer picks the route the `Plan` granted
  (render passes or dynamic rendering, `vkCmdPipelineBarrier` or `2`), so a
  host keeps one code path per feature, not per version.
- **What stays the host's job.** Image layouts and barriers between passes
  are correctness, not memory safety; a declarative pass list (which pass
  reads and writes what) could derive them later. Validation layers keep
  checking the rest in debug builds.

### Why a crate and not a feature

A `vulkan-ash` feature of numerobis was the first idea. A crate in the
same repository (a workspace beside the root package), depending on
`numerobis` with `vulkan`, is better:

- **ash's versions stay out of numerobis's.** The layer has to show `ash`
  types (`raw()` handles for what it doesn't wrap, the infos it builds),
  so every `ash` bump that breaks is a breaking numerobis release, for
  every compute user who never turns the feature on. In its own crate it
  is a `numerobis-vk` release only.
- **Its own pace.** numerobis moves with the raw backends (KFD, SDMA,
  RM); the layer moves with what Melon's renderer needs.
- **One boundary to audit.** "Every unsafe Vulkan call is in
  `numerobis-vk`" is a statement the crate boundary makes true. Inside
  numerobis the layer's unsafe would sit beside the ioctl code of the raw
  backends.
- **The public API is enough.** What the layer uses (`Plan`, `Chain`,
  `Pod`, `alloc`, `swapchain`, `format`) is all public; nothing internal has
  to be opened up. The one exception is the generated
  `s_type` table, which the generator can emit into the layer too.
- **Features unify.** A `vulkan-ash` feature turned on by any crate in a
  build compiles `ash` for all of them; a separate crate is only built by
  who depends on it.

What it costs:

- **No `numerobis::vulkan::call` path.** numerobis can't re-export it
  (that would be a cycle); hosts write `numerobis_vk::...`.
- **The comfort Vulkan compute backend can't use it.** numerobis would
  depend on the layer, which depends on numerobis. That backend is small
  (buffers, compute pipelines, one queue) and stays in numerobis with its
  own `unsafe`, as the raw backends do; or it lives in `numerobis-vk`,
  which needs `Device::all()` to take backends from outside, which it
  doesn't today.
- **Two releases for one change** that touches both (a new `Capability`
  the layer routes). The workspace keeps it to one PR.

Layout: `vk/` beside `src/`, package `numerobis-vk`, depending on
`numerobis = { path = "..", features = ["vulkan"] }` and `ash`.
`scripts/gen-vulkan.py` writes the wrappers into `vk/src/generated.rs`
from the same `vk.xml` pass.

Steps:

1. The `Pod` and memory-logic moves above, so the host's own unsafe code
   shrinks first. Done; Melon moves to them after the next release.
2. The retire queue and `Copy` handles: the core of the safety argument.
3. Typestate command buffers and mapped slices.
4. Generated wrappers for the remaining calls; Melon moves to them module by
   module (`gfx` first, then `renderer`) and drops its `unsafe` blocks as it
   goes.

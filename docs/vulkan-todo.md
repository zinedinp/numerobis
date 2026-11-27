# Vulkan TODOs

Two kinds: logic a graphics host (Melon) writes today that fits
`numerobis::vulkan` as it is, data plus logic with no bindings; and a call
layer on top of it that would make the host's Vulkan code safe without
costing it speed. Code marks each spot with `// TODO(vulkan): <short>` and
links here.

Melon is on 0.5.6: its byte views are `elem::as_bytes`, its buffers and
images take their memory types from `Memory::types_for`, its textures share
`alloc::Block`s (575 live allocations down to 63 in its bench), and its
swapchains take `swapchain`'s and `format::depth`'s choices.

## Colour

- **What the display makes of it.** `examples/hdr_probe` presents HDR10
  (A2B10G10R10 in PQ, with `vkSetHdrMetadataEXT`) and scRGB (FP16) on the
  B580 under validation without errors; Windows with HDR on offers those
  two and no HLG. Still to check by eye or instrument: the patches step up
  to the panel's peak, and the metadata reaches the display (the monitor's
  OSD or the HDMI/DP infoframe). Display P3 isn't offered there either.
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

## The safe call layer, `majestix-vk`

Its own repository (`majestix` beside `numerobis`), wrapping `ash` and
depending on `numerobis` with `vulkan`. Melon's renderer has about 900
unsafe Vulkan calls, each justified by the same three rules: every handle
comes from the one device and outlives its uses; nothing a frame in flight
reads is changed or destroyed before that frame's fence; a command buffer
is recorded by one thread at a time. The layer enforces them by
construction:

- **The liveness check runs in release builds too.** One atomic load per
  handle per command, from a generation table that never moves. As a debug
  assertion only, a stale handle would be undefined behaviour in safe code
  in release builds.
- **One generation table per device, not per handle type.** The Rust type
  keeps the kinds apart; a slot holds the raw handle and its memory, so the
  device's drop destroys what the host never retired.
- **Epochs, not fences, in the retire queue.** A handle is filed under the
  frame being recorded and destroyed once every frame up to it has
  finished. `run_now` (uploads outside the frames) pins its epoch until
  its fence. One frame ring per device, as the queue counts its frames.
- **Passes never transition layouts.** Attachments stay in
  `ATTACHMENT_OPTIMAL` (lowered without `synchronization2`) on both routes,
  and barriers move them, so render passes behave as dynamic rendering does.
- **Owning types retire on drop.** `Mapped`, `Pass` and `Target` hold the
  device and retire their handles when dropped; `Copy` handles go through
  `Device::retire`.

### Why a crate and not a feature

- **ash's versions stay out of numerobis's.** The layer shows `ash` types,
  so every breaking `ash` bump would be a breaking numerobis release, for
  every compute user who never turns the feature on.
- **Its own pace.** numerobis moves with the raw backends (KFD, SDMA, RM);
  the layer moves with what Melon's renderer needs.
- **One boundary to audit.** "Every unsafe Vulkan call is in `majestix-vk`"
  is a statement the crate boundary makes true.
- **The public API is enough.** `Plan`, `Chain`, `Pod`, `alloc`,
  `swapchain` and `format` are public; only the generated `s_type` table
  isn't, and the generator can emit it into the layer too.
- **Features unify.** A `vulkan-ash` feature turned on by any crate in a
  build would compile `ash` for all of them.

What it costs: no `numerobis::vulkan::call` path (that would be a cycle);
the comfort Vulkan compute backend can't use the layer and keeps its own
`unsafe` in numerobis; a change touching both takes two releases and two
PRs.

### Open

What majestix needs on its own side is in its `TODO.md`; here what
touches numerobis:

- **Generated wrappers for the remaining calls** (step 4):
  `scripts/gen-vulkan.py` writes them into majestix's `src/generated.rs`
  from the same `vk.xml` pass (each command's parent handle and its
  `externsync` parameters). Melon moves to them module by module (`gfx`
  first, then `renderer`) and drops its `unsafe` blocks as it goes.
- **Timeline semaphores**, routed by the plan like the barriers: majestix
  has binary semaphores and fences only.
- **Image layouts between passes** stay the host's job (correctness, not
  memory safety); a declarative pass list (which pass reads and writes
  what) could derive them later.

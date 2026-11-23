use std::ffi::{CStr, c_void};

use super::features::{self, Chain, VERSIONS};
use super::ops::{Op, Route};
use super::plan::BITS;
use super::subgroup::{Subgroup, op as sg, stage};
use super::{ApiVersion as V, Offer, Probe, Skip, Verdict, app_version};

/// The C view a driver has of a feature struct.
#[repr(C)]
struct Raw {
    s_type: i32,
    p_next: *mut c_void,
    bits: [u32; 1],
}

const ALL_CHAINS: &[Chain] = &[
    Chain::Core10,
    Chain::Vulkan11,
    Chain::Vulkan12,
    Chain::Vulkan13,
    Chain::Vulkan14,
    Chain::Storage16Bit,
    Chain::Multiview,
    Chain::VariablePointers,
    Chain::ProtectedMemory,
    Chain::SamplerYcbcrConversion,
    Chain::ShaderDrawParameters,
    Chain::Storage8Bit,
    Chain::Float16Int8,
    Chain::BufferDeviceAddress,
    Chain::DescriptorIndexing,
    Chain::ScalarBlockLayout,
    Chain::TimelineSemaphore,
    Chain::HostQueryReset,
    Chain::Synchronization2,
    Chain::SubgroupSizeControl,
    Chain::DemoteToHelperInvocation,
    Chain::IntegerDotProduct,
    Chain::DynamicRendering,
    Chain::IndexTypeUint8,
    Chain::MeshShader,
    Chain::MemoryPriority,
    Chain::PageableDeviceLocalMemory,
    Chain::FifoLatestReady,
];

fn chain_of(s_type: i32) -> Chain {
    *ALL_CHAINS
        .iter()
        .find(|c| c.s_type() == s_type)
        .unwrap_or_else(|| panic!("unknown sType {s_type}"))
}

/// Walks a chain the way a driver does: every struct by its `sType`.
unsafe fn walk(head: *mut c_void, mut visit: impl FnMut(Chain, *mut u32)) {
    let mut at = head.cast::<Raw>();
    while !at.is_null() {
        // SAFETY: the chain's blocks are `Raw`-prefixed and alive.
        unsafe {
            let bits = std::ptr::addr_of_mut!((*at).bits).cast::<u32>();
            visit(chain_of((*at).s_type), bits);
            at = (*at).p_next.cast();
        }
    }
}

/// A driver that supports `bits` (struct, member) and nothing else.
fn fill(probe: &mut Probe, bits: &[(Chain, &str)]) -> Vec<Chain> {
    let mut seen = Vec::new();
    // SAFETY: the probe owns the chain until it is used again.
    unsafe {
        walk(probe.features2(), |chain, out| {
            seen.push(chain);
            for &(c, name) in bits.iter().filter(|(c, _)| *c == chain) {
                *out.add(c.index(name).expect("known bit")) = 1;
            }
        });
    }
    seen
}

fn enabled_chain(plan: &super::Plan) -> Vec<(Chain, Vec<&'static str>)> {
    let mut device = plan.device_chain();
    let mut out = Vec::new();
    // SAFETY: `device` owns the chain.
    unsafe {
        walk(device.features2(), |chain, bits| {
            let on = (0..chain.fields().len())
                .filter(|&i| *bits.add(i) != 0)
                .map(|i| chain.fields()[i].name)
                .collect();
            out.push((chain, on));
        });
    }
    out
}

#[test]
fn tables_have_the_c_member_counts() {
    let count = |chain: Chain| chain.fields().len();
    assert_eq!(count(Chain::Core10), 55);
    assert_eq!(count(Chain::Vulkan11), 12);
    assert_eq!(count(Chain::Vulkan12), 47);
    assert_eq!(count(Chain::Vulkan13), 15);
    assert_eq!(count(Chain::Vulkan14), 21);
    assert_eq!(count(Chain::DescriptorIndexing), 20);
    assert_eq!(count(Chain::MeshShader), 5);
    assert!(ALL_CHAINS.iter().all(|&c| count(c) <= BITS));
    // The single structs are runs of their version's struct, in order.
    let run = |single: Chain, core: Chain| {
        let names: Vec<_> = core.fields().iter().map(|f| f.name).collect();
        let first = core.index(single.fields()[0].name).expect("in core");
        for (i, field) in single.fields().iter().enumerate() {
            assert_eq!(field.name, names[first + i], "{single:?}");
        }
    };
    run(Chain::Storage16Bit, Chain::Vulkan11);
    run(Chain::Multiview, Chain::Vulkan11);
    run(Chain::VariablePointers, Chain::Vulkan11);
    run(Chain::Storage8Bit, Chain::Vulkan12);
    run(Chain::DescriptorIndexing, Chain::Vulkan12);
    run(Chain::BufferDeviceAddress, Chain::Vulkan12);
    run(Chain::SubgroupSizeControl, Chain::Vulkan13);
}

#[test]
fn struct_types_are_distinct() {
    for (i, a) in ALL_CHAINS.iter().enumerate() {
        for b in &ALL_CHAINS[i + 1..] {
            assert_ne!(a.s_type(), b.s_type(), "{a:?} {b:?}");
        }
    }
}

#[test]
fn each_core_bit_comes_from_an_extension_promoted_in_its_version() {
    for version in &VERSIONS {
        assert_eq!(version.chain.fields(), version.fields);
        for field in version.fields {
            if let Some(from) = field.from {
                assert_eq!(
                    features::promoted_in(from),
                    Some(version.version),
                    "{}",
                    field.name
                );
            }
        }
    }
    let all: Vec<&CStr> = VERSIONS
        .iter()
        .flat_map(|v| v.promoted.iter().copied())
        .collect();
    for (i, name) in all.iter().enumerate() {
        assert!(!all[i + 1..].contains(name), "{name:?} twice");
    }
}

#[test]
fn every_op_has_one_spec_with_valid_routes() {
    for &op in Op::ALL {
        let spec = op.spec();
        assert_eq!(spec.op, op);
        assert!(!spec.routes.is_empty(), "{}", spec.name);
        let names = Op::ALL.iter().filter(|o| o.name() == spec.name).count();
        assert_eq!(names, 1, "{}", spec.name);
        for (i, route) in spec.routes.iter().enumerate() {
            match *route {
                Route::Core { chain, fields, .. }
                | Route::Extension {
                    chain: Some(chain),
                    fields,
                    ..
                } => {
                    assert!(!fields.is_empty(), "{}", spec.name);
                    for field in fields {
                        assert!(chain.index(field).is_some(), "{} {field}", spec.name);
                    }
                }
                Route::Extension { fields, .. } => assert!(fields.is_empty()),
                Route::Subgroup { .. } => {}
            }
            // A promoted extension needs a core route before it that
            // covers the versions where the extension is skipped.
            if let Route::Extension { name, .. } = *route
                && let Some(core) = features::promoted_in(name)
            {
                let covered = spec.routes[..i]
                    .iter()
                    .any(|r| matches!(*r, Route::Core { since, .. } if since <= core));
                assert!(covered, "{} {name:?}", spec.name);
            }
        }
    }
}

#[test]
fn versions_pack_like_vulkan() {
    let v13 = (1 << 22) | (3 << 12);
    assert_eq!(V::V1_3.raw(), v13);
    assert_eq!(
        V::from_raw(v13 | 280),
        V {
            patch: 280,
            ..V::V1_3
        }
    );
    assert!(V::from_raw(v13 | 280) >= V::V1_3);
    assert!(V::from_raw(v13 | 280) < V::V1_4);
    assert_eq!(V::from_raw(v13 | 280).to_string(), "1.3.280");
}

#[test]
fn a_1_0_loader_gets_1_0() {
    assert_eq!(app_version(None, V::V1_4), V::V1_0);
    assert_eq!(app_version(Some(V::V1_0), V::V1_4), V::V1_0);
    assert_eq!(app_version(Some(V::V1_1), V::V1_4), V::V1_4);
    assert_eq!(app_version(Some(V::V1_4), V::V1_2), V::V1_2);
}

const ARC_EXTENSIONS: &[&CStr] = &[
    c"VK_KHR_swapchain",
    c"VK_KHR_draw_indirect_count",
    c"VK_KHR_buffer_device_address",
    c"VK_KHR_timeline_semaphore",
    c"VK_KHR_synchronization2",
    c"VK_KHR_dynamic_rendering",
    c"VK_KHR_depth_stencil_resolve",
    c"VK_KHR_create_renderpass2",
    c"VK_EXT_mesh_shader",
    c"VK_KHR_spirv_1_4",
    c"VK_KHR_shader_float_controls",
    c"VK_EXT_memory_priority",
    c"VK_EXT_pageable_device_local_memory",
];

const ARC_BITS: &[(Chain, &str)] = &[
    (Chain::Core10, "multiDrawIndirect"),
    (Chain::Core10, "drawIndirectFirstInstance"),
    (Chain::Vulkan11, "shaderDrawParameters"),
    (Chain::Vulkan11, "storageBuffer16BitAccess"),
    (Chain::Vulkan12, "drawIndirectCount"),
    (Chain::Vulkan12, "bufferDeviceAddress"),
    (Chain::Vulkan12, "timelineSemaphore"),
    (Chain::Vulkan12, "shaderFloat16"),
    (Chain::Vulkan13, "synchronization2"),
    (Chain::Vulkan13, "dynamicRendering"),
    (Chain::ShaderDrawParameters, "shaderDrawParameters"),
    (Chain::Storage16Bit, "storageBuffer16BitAccess"),
    (Chain::BufferDeviceAddress, "bufferDeviceAddress"),
    (Chain::TimelineSemaphore, "timelineSemaphore"),
    (Chain::Synchronization2, "synchronization2"),
    (Chain::DynamicRendering, "dynamicRendering"),
    (Chain::Float16Int8, "shaderFloat16"),
    (Chain::MeshShader, "meshShader"),
    (
        Chain::PageableDeviceLocalMemory,
        "pageableDeviceLocalMemory",
    ),
];

const WANTED: &[Op] = &[
    Op::MultiDrawIndirect,
    Op::ShaderDrawParameters,
    Op::DrawIndirectCount,
    Op::Storage16Bit,
    Op::BufferDeviceAddress,
    Op::TimelineSemaphore,
    Op::Synchronization2,
    Op::DynamicRendering,
    Op::Float16,
    Op::MeshShader,
    Op::PageableDeviceLocalMemory,
    Op::SubgroupArithmetic,
];

#[test]
fn at_1_3_the_core_structs_carry_everything_promoted() {
    let offer = Offer::new(
        V::V1_4,
        V::from_raw(V::V1_3.raw() | 280),
        ARC_EXTENSIONS.iter().copied(),
    );
    let mut probe = Probe::new(offer, WANTED);
    assert!(probe.wants_subgroup());
    let queried = fill(&mut probe, ARC_BITS);
    assert_eq!(
        queried,
        [
            Chain::Core10,
            Chain::Vulkan11,
            Chain::Vulkan12,
            Chain::Vulkan13,
            Chain::MeshShader,
            Chain::PageableDeviceLocalMemory,
        ]
    );
    probe.set_subgroup(Subgroup {
        size: 16,
        stages: stage::COMPUTE | stage::FRAGMENT,
        operations: sg::BASIC | sg::VOTE | sg::ARITHMETIC | sg::BALLOT,
    });
    let plan = probe.plan();
    assert_eq!(plan.api, V::from_raw(V::V1_3.raw() | 280));
    for &op in WANTED {
        assert!(plan.has(op), "{op:?}\n{plan}");
    }
    assert!(matches!(
        plan.route(Op::DrawIndirectCount),
        Some(Route::Core { since: V::V1_2, .. })
    ));
    // Promoted extensions stay off; their dependencies too.
    assert_eq!(
        plan.extensions(),
        [
            c"VK_EXT_mesh_shader",
            c"VK_EXT_pageable_device_local_memory",
            c"VK_EXT_memory_priority",
        ]
    );
    let enabled = enabled_chain(&plan);
    let on = |chain| {
        enabled
            .iter()
            .find(|(c, _)| *c == chain)
            .map(|(_, on)| on.clone())
            .unwrap_or_default()
    };
    assert_eq!(
        on(Chain::Core10),
        ["multiDrawIndirect", "drawIndirectFirstInstance"]
    );
    assert_eq!(
        on(Chain::Vulkan11),
        ["storageBuffer16BitAccess", "shaderDrawParameters"]
    );
    assert_eq!(
        on(Chain::Vulkan12),
        [
            "drawIndirectCount",
            "shaderFloat16",
            "timelineSemaphore",
            "bufferDeviceAddress"
        ]
    );
    assert_eq!(
        on(Chain::Vulkan13),
        ["synchronization2", "dynamicRendering"]
    );
    assert_eq!(on(Chain::MeshShader), ["meshShader"]);
    // No single struct beside the version struct that absorbed it.
    assert!(enabled.iter().all(|(c, _)| !matches!(
        c,
        Chain::TimelineSemaphore | Chain::BufferDeviceAddress | Chain::Synchronization2
    )));
}

#[test]
fn at_1_1_the_extensions_take_over() {
    let offer = Offer::new(V::V1_1, V::V1_3, ARC_EXTENSIONS.iter().copied());
    assert_eq!(offer.api(), V::V1_1);
    let mut probe = Probe::new(offer, WANTED);
    let queried = fill(&mut probe, ARC_BITS);
    // Nothing a 1.1 device can't take: no VkPhysicalDeviceVulkan1xFeatures.
    assert!(queried.iter().all(|c| !matches!(
        c,
        Chain::Vulkan11 | Chain::Vulkan12 | Chain::Vulkan13 | Chain::Vulkan14
    )));
    let plan = probe.plan();
    assert!(plan.has(Op::ShaderDrawParameters));
    assert!(matches!(
        plan.route(Op::ShaderDrawParameters),
        Some(Route::Core {
            since: V::V1_1,
            chain: Chain::ShaderDrawParameters,
            ..
        })
    ));
    assert!(matches!(
        plan.route(Op::DrawIndirectCount),
        Some(Route::Extension { .. })
    ));
    assert!(plan.has(Op::Synchronization2));
    assert!(!plan.has(Op::Float16), "not offered as an extension");
    // Dynamic rendering pulls its two 1.2 dependencies in.
    let extensions = plan.extensions();
    for name in [
        c"VK_KHR_draw_indirect_count",
        c"VK_KHR_buffer_device_address",
        c"VK_KHR_dynamic_rendering",
        c"VK_KHR_depth_stencil_resolve",
        c"VK_KHR_create_renderpass2",
        c"VK_KHR_spirv_1_4",
        c"VK_KHR_shader_float_controls",
    ] {
        assert!(extensions.contains(&name), "{name:?}");
    }
    // VK_KHR_device_group (buffer device address) is core in 1.1.
    assert!(!extensions.contains(&c"VK_KHR_device_group"));
    let report = plan.to_string();
    assert!(
        report.starts_with("Vulkan 1.1 (instance asks 1.1, device has 1.3)"),
        "{report}"
    );
    assert!(
        report.contains("- shader.float16: no route [skipped core 1.2"),
        "{report}"
    );
}

#[test]
fn an_available_route_that_says_no_ends_the_waterfall() {
    let offer = Offer::new(
        V::V1_3,
        V::V1_3,
        [
            c"VK_KHR_present_mode_fifo_latest_ready",
            c"VK_EXT_present_mode_fifo_latest_ready",
        ],
    );
    let mut probe = Probe::new(offer, &[Op::FifoLatestReady, Op::DrawIndirectCount]);
    fill(&mut probe, &[]);
    let plan = probe.plan();
    let latest = plan.outcome(Op::FifoLatestReady).expect("asked");
    assert_eq!(latest.verdict, Verdict::Off("presentModeFifoLatestReady"));
    assert!(latest.skipped.is_empty());
    assert!(plan.extensions().is_empty());
    let count = plan.outcome(Op::DrawIndirectCount).expect("asked");
    assert_eq!(count.verdict, Verdict::Off("drawIndirectCount"));
    assert!(plan.to_string().contains(
        "- draw.draw_indirect_count: drawIndirectCount is off (core 1.2 (VkPhysicalDeviceVulkan12Features.drawIndirectCount))"
    ));
}

#[test]
fn missing_dependencies_and_versions_are_logged() {
    let offer = Offer::new(V::V1_1, V::V1_1, [c"VK_EXT_mesh_shader"]);
    let mut probe = Probe::new(offer, &[Op::MeshShader, Op::PushDescriptor]);
    fill(&mut probe, &[(Chain::MeshShader, "meshShader")]);
    let plan = probe.plan();
    let mesh = plan.outcome(Op::MeshShader).expect("asked");
    assert_eq!(mesh.verdict, Verdict::NoRoute);
    assert_eq!(mesh.skipped[0].1, Skip::Dependency(c"VK_KHR_spirv_1_4"));
    let push = plan.outcome(Op::PushDescriptor).expect("asked");
    assert_eq!(push.skipped[0].1, Skip::Version(V::V1_4));
    assert_eq!(
        push.skipped[1].1,
        Skip::NotOffered(c"VK_KHR_push_descriptor")
    );
    assert!(
        plan.to_string()
            .contains("- draw.mesh_shader: no route [skipped VK_EXT_mesh_shader (VkPhysicalDeviceMeshShaderFeaturesEXT.meshShader): needs VK_KHR_spirv_1_4]"),
        "{plan}"
    );
}

#[test]
fn subgroup_ops_need_the_properties() {
    let offer = Offer::new(V::V1_2, V::V1_2, []);
    let mut probe = Probe::new(offer.clone(), &[Op::SubgroupArithmetic]);
    fill(&mut probe, &[]);
    assert_eq!(probe.plan().outcomes()[0].verdict, Verdict::SubgroupUnread);
    let mut probe = Probe::new(offer, &[Op::SubgroupArithmetic, Op::SubgroupShuffle]);
    probe.set_subgroup(Subgroup {
        size: 32,
        stages: stage::FRAGMENT | stage::COMPUTE,
        operations: sg::BASIC | sg::ARITHMETIC,
    });
    let plan = probe.plan();
    assert!(plan.has(Op::SubgroupArithmetic));
    assert_eq!(
        plan.outcome(Op::SubgroupShuffle).map(|o| o.verdict),
        Some(Verdict::SubgroupLacks)
    );
}

#[test]
fn a_1_0_device_reads_the_core_bits_only() {
    let offer = Offer::new(V::V1_0, V::V1_3, [c"VK_KHR_timeline_semaphore"]);
    let mut probe = Probe::new(offer, &[Op::MultiDrawIndirect, Op::TimelineSemaphore]);
    // vkGetPhysicalDeviceFeatures writes the bare 1.0 struct.
    // SAFETY: the probe owns 55 bits there.
    unsafe {
        let bits = probe.core_features().cast::<u32>();
        *bits.add(Chain::Core10.index("multiDrawIndirect").expect("bit")) = 1;
        *bits.add(
            Chain::Core10
                .index("drawIndirectFirstInstance")
                .expect("bit"),
        ) = 1;
    }
    let plan = probe.plan();
    assert!(plan.has(Op::MultiDrawIndirect));
    let timeline = plan.outcome(Op::TimelineSemaphore).expect("asked");
    assert_eq!(
        timeline.skipped.last().map(|s| s.1),
        Some(Skip::Version(V::V1_1))
    );
}

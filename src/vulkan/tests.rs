use std::ffi::{CStr, c_void};

use super::caps::{Capability, Route};
use super::features::{self, Chain, VERSIONS};
use super::plan::BITS;
use super::subgroup::{Subgroup, op as sg, stage};
use super::{
    ApiVersion as V, BitError, FeatureQuery, Instance, Offer, Probe, Skip, Verdict, app_version,
};

fn raw(raw: u32) -> V {
    V::from_raw(raw).expect("variant 0")
}

/// The C view a driver has of a feature struct.
#[repr(C)]
struct Raw {
    s_type: i32,
    p_next: *mut c_void,
    bits: [u32; 1],
}

fn chain_of(s_type: i32) -> Chain {
    *Chain::ALL
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
pub(super) fn fill(probe: &mut Probe, bits: &[(Chain, &str)]) -> Vec<Chain> {
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
    assert_eq!(count(Chain::PortabilitySubset), 15);
    assert!(Chain::ALL.iter().all(|&c| count(c) <= BITS));
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
    for (i, a) in Chain::ALL.iter().enumerate() {
        for b in &Chain::ALL[i + 1..] {
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
fn every_capability_has_a_unique_name_and_valid_routes() {
    for &cap in Capability::ALL {
        let spec = cap.spec();
        assert!(!spec.routes.is_empty(), "{}", spec.name);
        let names = Capability::ALL
            .iter()
            .filter(|o| o.name() == spec.name)
            .count();
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
                Route::Extension { fields, .. } => assert_eq!(fields, [] as [&str; 0]),
                Route::Version { .. } | Route::Subgroup { .. } => {}
            }
            // A promoted extension needs a core route before it that
            // covers the versions where the extension is skipped.
            if let Route::Extension { name, .. } = *route
                && let Some(core) = features::promoted_in(name)
            {
                let covered = spec.routes[..i]
                    .iter()
                    .any(|r| matches!(*r, Route::Core { since, .. } | Route::Version { since } if since <= core));
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
        raw(v13 + 280),
        V {
            patch: 280,
            ..V::V1_3
        }
    );
    assert!(raw(v13 + 280) >= V::V1_3);
    assert!(raw(v13 + 280) < V::V1_4);
    assert_eq!(raw(v13 + 280).to_string(), "1.3.280");
    // Vulkan SC 1.0 is variant 1.
    assert_eq!(V::from_raw((1 << 29) | (1 << 22)), None);
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

const WANTED: &[Capability] = &[
    Capability::MultiDrawIndirect,
    Capability::ShaderDrawParameters,
    Capability::DrawIndirectCount,
    Capability::Storage16Bit,
    Capability::BufferDeviceAddress,
    Capability::TimelineSemaphore,
    Capability::Synchronization2,
    Capability::DynamicRendering,
    Capability::Float16,
    Capability::MeshShader,
    Capability::PageableDeviceLocalMemory,
    Capability::SubgroupArithmetic,
];

#[test]
fn at_1_3_the_core_structs_carry_everything_promoted() {
    let offer = Offer::new(
        V::V1_4,
        raw(V::V1_3.raw() + 280),
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
    assert_eq!(plan.api, raw(V::V1_3.raw() + 280));
    for &cap in WANTED {
        assert!(plan.has(cap), "{cap:?}\n{plan}");
    }
    assert!(matches!(
        plan.route(Capability::DrawIndirectCount),
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
            .map_or_else(Vec::new, |(_, on)| on.clone())
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
    assert!(plan.has(Capability::ShaderDrawParameters));
    assert!(matches!(
        plan.route(Capability::ShaderDrawParameters),
        Some(Route::Core {
            since: V::V1_1,
            chain: Chain::ShaderDrawParameters,
            ..
        })
    ));
    assert!(matches!(
        plan.route(Capability::DrawIndirectCount),
        Some(Route::Extension { .. })
    ));
    assert!(plan.has(Capability::Synchronization2));
    assert!(
        !plan.has(Capability::Float16),
        "not offered as an extension"
    );
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
            c"VK_KHR_swapchain",
        ],
    )
    .with_instance([c"VK_KHR_surface"]);
    let mut probe = Probe::new(
        offer,
        &[Capability::FifoLatestReady, Capability::DrawIndirectCount],
    );
    fill(&mut probe, &[]);
    let plan = probe.plan();
    let latest = plan.outcome(Capability::FifoLatestReady).expect("asked");
    assert_eq!(latest.verdict, Verdict::Off("presentModeFifoLatestReady"));
    assert_eq!(latest.skipped.len(), 0, "{:?}", latest.skipped);
    assert_eq!(plan.extensions(), [] as [&CStr; 0]);
    let count = plan.outcome(Capability::DrawIndirectCount).expect("asked");
    assert_eq!(count.verdict, Verdict::Off("drawIndirectCount"));
    assert!(plan.to_string().contains(
        "- draw.draw_indirect_count: drawIndirectCount is off (core 1.2 (VkPhysicalDeviceVulkan12Features.drawIndirectCount))"
    ));
}

#[test]
fn missing_dependencies_and_versions_are_logged() {
    let offer = Offer::new(V::V1_1, V::V1_1, [c"VK_EXT_mesh_shader"]);
    let mut probe = Probe::new(offer, &[Capability::MeshShader, Capability::PushDescriptor]);
    fill(&mut probe, &[(Chain::MeshShader, "meshShader")]);
    let plan = probe.plan();
    let mesh = plan.outcome(Capability::MeshShader).expect("asked");
    assert_eq!(mesh.verdict, Verdict::NoRoute);
    assert_eq!(mesh.skipped[0].1, Skip::Dependency(c"VK_KHR_spirv_1_4"));
    let push = plan.outcome(Capability::PushDescriptor).expect("asked");
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
    let mut probe = Probe::new(offer.clone(), &[Capability::SubgroupArithmetic]);
    fill(&mut probe, &[]);
    assert_eq!(probe.plan().outcomes()[0].verdict, Verdict::SubgroupUnread);
    let mut probe = Probe::new(
        offer,
        &[Capability::SubgroupArithmetic, Capability::SubgroupShuffle],
    );
    probe.set_subgroup(Subgroup {
        size: 32,
        stages: stage::FRAGMENT | stage::COMPUTE,
        operations: sg::BASIC | sg::ARITHMETIC,
    });
    let plan = probe.plan();
    assert!(plan.has(Capability::SubgroupArithmetic));
    assert_eq!(
        plan.outcome(Capability::SubgroupShuffle).map(|o| o.verdict),
        Some(Verdict::SubgroupLacks)
    );
}

#[test]
fn a_1_0_device_reads_the_core_bits_only() {
    let offer = Offer::new(V::V1_0, V::V1_3, [c"VK_KHR_timeline_semaphore"]);
    let mut probe = Probe::new(
        offer,
        &[Capability::MultiDrawIndirect, Capability::TimelineSemaphore],
    );
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
    assert!(plan.has(Capability::MultiDrawIndirect));
    let timeline = plan.outcome(Capability::TimelineSemaphore).expect("asked");
    assert_eq!(
        timeline.skipped.last().map(|s| s.1),
        Some(Skip::Dependency(features::GET_PHYSICAL_DEVICE_PROPERTIES2))
    );
    assert_eq!(plan.query, FeatureQuery::Features);
}

#[test]
fn every_chain_route_comes_after_the_core_route_that_absorbs_it() {
    for &cap in Capability::ALL {
        let spec = cap.spec();
        for (i, route) in spec.routes.iter().enumerate() {
            let (Route::Core { chain, .. }
            | Route::Extension {
                chain: Some(chain), ..
            }) = *route
            else {
                continue;
            };
            if let Some((since, by)) = chain.absorbed_by() {
                let covered = spec.routes[..i].iter().any(|r| {
                    matches!(*r, Route::Core { since: s, chain: c, .. } if s <= since && c == by)
                });
                assert!(covered, "{}: {chain:?} without {by:?} before it", spec.name);
            }
        }
    }
}

#[test]
fn absorbed_structs_are_runs_of_their_version_struct() {
    for &chain in Chain::ALL {
        let Some((since, by)) = chain.absorbed_by() else {
            continue;
        };
        assert_eq!(by.version_struct_since().map(|v| v <= since), Some(true));
        for field in chain.fields() {
            assert!(by.index(field.name).is_some(), "{chain:?}.{}", field.name);
        }
    }
}

fn met(dep: &features::Depends, api: V, enabled: &[&CStr], instance: &[&CStr]) -> bool {
    use features::Depends;
    match *dep {
        Depends::Version(v) => api >= v,
        Depends::Extension(n) => {
            features::promoted_in(n).is_some_and(|v| v <= api)
                || enabled.contains(&n)
                || instance.contains(&n)
        }
        Depends::All(all) => all.iter().all(|d| met(d, api, enabled, instance)),
        Depends::Any(any) => any.iter().any(|d| met(d, api, enabled, instance)),
        Depends::Never => false,
    }
}

/// Every route offered, every bit on: whatever a device works at, no chain
/// holds a single struct beside the version struct that absorbed it, and no
/// version struct the device can't take.
#[test]
fn no_version_makes_an_invalid_chain() {
    // Every extension the routes name and everything they depend on.
    let all_extensions: Vec<&CStr> = features::known_extensions()
        .filter(|&n| !features::is_instance_extension(n))
        .chain([features::PORTABILITY_SUBSET_EXTENSION])
        .collect();
    let every_bit: Vec<(Chain, &str)> = Chain::ALL
        .iter()
        .flat_map(|&c| c.fields().iter().map(move |f| (c, f.name)))
        .collect();
    let every_instance: Vec<&CStr> = features::known_extensions()
        .filter(|&n| features::is_instance_extension(n))
        .collect();
    for api in [V::V1_0, V::V1_1, V::V1_2, V::V1_3, V::V1_4] {
        for instance in [
            &[][..],
            &[features::GET_PHYSICAL_DEVICE_PROPERTIES2][..],
            &every_instance[..],
        ] {
            let offer = Offer::new(api, V::V1_4, all_extensions.iter().copied())
                .with_instance(instance.iter().copied());
            let mut probe = Probe::new(offer, Capability::ALL);
            let queried = if probe.query().has_chain() {
                fill(&mut probe, &every_bit)
            } else {
                vec![Chain::Core10]
            };
            let plan = probe.plan();
            let enabled: Vec<Chain> = enabled_chain(&plan).into_iter().map(|(c, _)| c).collect();
            for chains in [&queried, &enabled] {
                for &chain in chains {
                    assert!(chain.at(api) == Ok(chain), "{api}: {chain:?} in {chains:?}");
                }
            }
            for name in plan.extensions() {
                assert!(!features::is_instance_extension(name), "{name:?}");
                // VUID-vkCreateDevice-ppEnabledExtensionNames-01387: what an
                // enabled extension depends on is enabled, core or on the
                // instance.
                if let Some(dep) = features::depends(name) {
                    assert!(
                        met(dep, api, plan.extensions(), instance),
                        "{api}: {name:?} without its dependencies {dep:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn a_1_0_instance_needs_get_physical_device_properties2() {
    let device = [
        c"VK_KHR_push_descriptor",
        c"VK_EXT_sampler_filter_minmax",
        c"VK_KHR_draw_indirect_count",
        c"VK_KHR_timeline_semaphore",
    ];
    let wanted = [
        Capability::PushDescriptor,
        Capability::SamplerFilterMinmax,
        Capability::DrawIndirectCount,
        Capability::TimelineSemaphore,
    ];
    // Without it only what needs neither a feature query nor properties2 is left.
    let bare = Probe::new(Offer::new(V::V1_0, V::V1_3, device), &wanted).plan();
    assert_eq!(bare.query, FeatureQuery::Features);
    assert_eq!(
        bare.lacking(&wanted),
        [
            Capability::PushDescriptor,
            Capability::SamplerFilterMinmax,
            Capability::TimelineSemaphore
        ]
    );
    let push = bare.outcome(Capability::PushDescriptor).expect("asked");
    assert_eq!(
        push.skipped.last().map(|s| s.1),
        Some(Skip::Dependency(features::GET_PHYSICAL_DEVICE_PROPERTIES2))
    );
    // With it, the extension structs are read through the KHR entry point.
    let instance = Instance::new(
        None,
        V::NEWEST,
        [
            features::GET_PHYSICAL_DEVICE_PROPERTIES2,
            c"VK_EXT_debug_utils",
        ],
    );
    assert_eq!(instance.api, V::V1_0);
    assert_eq!(
        instance.extensions(),
        [features::GET_PHYSICAL_DEVICE_PROPERTIES2]
    );
    let offer = Offer::new(instance.api, V::V1_3, device)
        .with_instance(instance.extensions().iter().copied());
    let mut probe = Probe::new(offer, &wanted);
    assert_eq!(probe.query(), FeatureQuery::Features2Khr);
    let queried = fill(
        &mut probe,
        &[(Chain::TimelineSemaphore, "timelineSemaphore")],
    );
    assert_eq!(queried, [Chain::Core10, Chain::TimelineSemaphore]);
    let plan = probe.plan();
    assert!(plan.lacking(&wanted).is_empty(), "{plan}");
    // The instance extension is a dependency, never a device extension.
    assert!(
        !plan
            .extensions()
            .contains(&features::GET_PHYSICAL_DEVICE_PROPERTIES2)
    );
    assert!(
        plan.to_string()
            .contains("through VK_KHR_get_physical_device_properties2")
    );
    // A newer loader needs none of it.
    let modern = Instance::new(
        Some(V::V1_3),
        V::NEWEST,
        [features::GET_PHYSICAL_DEVICE_PROPERTIES2],
    );
    assert_eq!(modern.extensions(), [] as [&CStr; 0]);
}

#[test]
fn portability_drivers_are_enumerated_and_their_subset_enabled() {
    let instance = Instance::new(
        Some(V::V1_3),
        V::NEWEST,
        [features::PORTABILITY_ENUMERATION],
    );
    assert_eq!(instance.extensions(), [features::PORTABILITY_ENUMERATION]);
    assert_eq!(instance.flags, features::ENUMERATE_PORTABILITY_BIT);
    // MoltenVK: 1.2, portability subset, no triangle fans.
    let offer = Offer::new(
        instance.api,
        V::V1_2,
        [
            features::PORTABILITY_SUBSET_EXTENSION,
            c"VK_KHR_synchronization2",
        ],
    )
    .with_instance(instance.extensions().iter().copied());
    let mut probe = Probe::new(offer, &[Capability::Synchronization2]);
    let has: Vec<(Chain, &str)> = Chain::PortabilitySubset
        .fields()
        .iter()
        .filter(|f| f.name != "triangleFans" && f.name != "pointPolygons")
        .map(|f| (Chain::PortabilitySubset, f.name))
        .chain([(Chain::Synchronization2, "synchronization2")])
        .collect();
    let queried = fill(&mut probe, &has);
    assert!(queried.contains(&Chain::PortabilitySubset));
    let plan = probe.plan();
    assert!(plan.is_portability_subset());
    assert_eq!(
        plan.extensions(),
        [
            features::PORTABILITY_SUBSET_EXTENSION,
            c"VK_KHR_synchronization2"
        ]
    );
    assert_eq!(plan.portability_lacks(), ["pointPolygons", "triangleFans"]);
    assert_eq!(
        plan.supported(Chain::PortabilitySubset, "events"),
        Some(true)
    );
    assert!(
        plan.to_string()
            .contains("portability subset, lacks pointPolygons, triangleFans"),
        "{plan}"
    );
    // A regular driver: nothing extra.
    let regular = Probe::new(Offer::new(V::V1_3, V::V1_3, []), &[]).plan();
    assert!(!regular.is_portability_subset());
    assert_eq!(regular.extensions(), [] as [&CStr; 0]);
}

fn chain_bits(device: &mut super::DeviceChain) -> Vec<(Chain, Vec<&'static str>)> {
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
fn host_bits_go_through_the_struct_that_absorbed_them() {
    let offer = Offer::new(V::V1_3, V::V1_2, []);
    let mut probe = Probe::new(offer, &[Capability::TimelineSemaphore]);
    // At 1.2 the timeline struct is part of Vulkan12, which the cap reads anyway.
    assert_eq!(probe.also_query(Chain::TimelineSemaphore), Ok(()));
    assert_eq!(
        probe.also_query(Chain::Vulkan13),
        Err(BitError::NeedsVersion(V::V1_3))
    );
    assert_eq!(probe.also_query(Chain::MeshShader), Ok(()));
    let queried = fill(
        &mut probe,
        &[
            (Chain::Vulkan12, "timelineSemaphore"),
            (Chain::Vulkan12, "descriptorIndexing"),
            (Chain::Core10, "fillModeNonSolid"),
        ],
    );
    assert_eq!(queried, [Chain::Core10, Chain::Vulkan12, Chain::MeshShader]);
    let plan = probe.plan();
    assert_eq!(
        plan.supported(Chain::Core10, "fillModeNonSolid"),
        Some(true)
    );
    assert_eq!(plan.supported(Chain::Core10, "wideLines"), Some(false));
    assert_eq!(
        plan.supported(Chain::Vulkan12, "descriptorIndexing"),
        Some(true)
    );
    assert_eq!(
        plan.supported(Chain::TimelineSemaphore, "timelineSemaphore"),
        Some(true)
    );
    assert_eq!(plan.supported(Chain::MeshShader, "taskShader"), Some(false));
    assert_eq!(plan.supported(Chain::Vulkan13, "dynamicRendering"), None);
    assert_eq!(plan.supported(Chain::Core10, "noSuchBit"), None);

    let mut device = plan.device_chain();
    assert_eq!(device.enable(Chain::Core10, "fillModeNonSolid"), Ok(()));
    assert_eq!(
        device.enable(Chain::DescriptorIndexing, "runtimeDescriptorArray"),
        Ok(())
    );
    assert_eq!(
        device.enable(Chain::Core10, "noSuchBit"),
        Err(BitError::UnknownField)
    );
    assert_eq!(
        device.enable(Chain::Vulkan14, "pushDescriptor"),
        Err(BitError::NeedsVersion(V::V1_4))
    );
    assert_eq!(
        chain_bits(&mut device),
        [
            (Chain::Core10, vec!["fillModeNonSolid"]),
            (
                Chain::Vulkan12,
                vec!["runtimeDescriptorArray", "timelineSemaphore"]
            ),
        ]
    );

    // At 1.1 the single struct is the right one.
    let plan = Probe::new(Offer::new(V::V1_1, V::V1_1, []), &[]).plan();
    let mut device = plan.device_chain();
    assert_eq!(
        device.enable(Chain::DescriptorIndexing, "runtimeDescriptorArray"),
        Ok(())
    );
    let seen: Vec<Chain> = chain_bits(&mut device)
        .into_iter()
        .map(|(c, _)| c)
        .collect();
    assert_eq!(seen, [Chain::Core10, Chain::DescriptorIndexing]);

    // On a bare 1.0 device only the 1.0 bits exist.
    let plan = Probe::new(Offer::new(V::V1_0, V::V1_0, []), &[]).plan();
    let mut device = plan.device_chain();
    assert_eq!(device.enable(Chain::Core10, "wideLines"), Ok(()));
    assert_eq!(
        device.enable(Chain::Multiview, "multiview"),
        Err(BitError::NeedsFeatures2)
    );
}

/// `VkPhysicalDeviceSubgroupProperties` as a driver sees it.
#[repr(C)]
struct RawSubgroup {
    s_type: i32,
    p_next: *mut c_void,
    size: u32,
    stages: u32,
    operations: u32,
    quad: u32,
}

#[test]
fn the_properties_query_carries_the_subgroup_properties() {
    let offer = Offer::new(V::V1_3, V::V1_3, []);
    let mut probe = Probe::new(
        offer,
        &[Capability::SubgroupBallot, Capability::SubgroupShuffle],
    );
    assert!(probe.wants_subgroup());
    // SAFETY: the probe owns the chain; a driver writes the structs it knows.
    unsafe {
        let head = probe.properties2().cast::<Raw>();
        assert_eq!((*head).s_type, 1_000_059_001);
        let sub = (*head).p_next.cast::<RawSubgroup>();
        assert_eq!((*sub).s_type, 1_000_094_000);
        // At 1.3 the driver properties follow, and nothing after them.
        let driver = (*sub).p_next.cast::<Raw>();
        assert_eq!((*driver).s_type, 1_000_196_000);
        assert!((*driver).p_next.is_null());
        (*sub).size = 64;
        (*sub).stages = stage::COMPUTE;
        (*sub).operations = sg::BASIC | sg::BALLOT;
        (*sub).quad = 1;
    }
    let plan = probe.plan();
    assert!(plan.has(Capability::SubgroupBallot));
    assert_eq!(
        plan.outcome(Capability::SubgroupShuffle).map(|o| o.verdict),
        Some(Verdict::SubgroupLacks)
    );
    assert_eq!(plan.subgroup().map(|s| s.size), Some(64));
    // Handed out but never filled: unread, as without the query.
    let mut probe = Probe::new(
        Offer::new(V::V1_3, V::V1_3, []),
        &[Capability::SubgroupVote],
    );
    let _ = probe.properties2();
    assert_eq!(probe.plan().outcomes()[0].verdict, Verdict::SubgroupUnread);
}

use super::{DeviceInfo, DeviceType, Os, Quirk, Status, rank};

#[test]
fn version_routes_need_no_bits() {
    let cap = [Capability::ExtendedDynamicState];
    let mut probe = Probe::new(Offer::new(V::V1_3, V::V1_3, []), &cap);
    let queried = fill(&mut probe, &[]);
    assert_eq!(queried, [Chain::Core10]);
    let plan = probe.plan();
    assert!(matches!(
        plan.route(Capability::ExtendedDynamicState),
        Some(Route::Version { since: V::V1_3 })
    ));
    assert_eq!(plan.extensions(), [] as [&CStr; 0]);
    // On 1.2 the extension and its bit decide.
    let offer = Offer::new(V::V1_2, V::V1_2, [c"VK_EXT_extended_dynamic_state"]);
    let mut probe = Probe::new(offer, &cap);
    fill(
        &mut probe,
        &[(Chain::ExtendedDynamicState, "extendedDynamicState")],
    );
    let plan = probe.plan();
    assert!(plan.has(Capability::ExtendedDynamicState));
    assert_eq!(plan.extensions(), [c"VK_EXT_extended_dynamic_state"]);
}

#[test]
fn os_specific_capabilities_are_skipped_elsewhere() {
    let offer = || {
        Offer::new(
            V::V1_3,
            V::V1_3,
            [
                c"VK_KHR_external_memory_win32",
                c"VK_KHR_external_memory_fd",
                c"VK_EXT_external_memory_dma_buf",
            ],
        )
    };
    let caps = [
        Capability::ExternalMemoryWin32,
        Capability::ExternalMemoryFd,
        Capability::DmaBuf,
    ];
    let windows = Probe::new(offer().with_os(Os::Windows), &caps).plan();
    assert!(windows.has(Capability::ExternalMemoryWin32));
    assert_eq!(
        windows.outcome(Capability::DmaBuf).map(|o| o.verdict),
        Some(Verdict::Unavailable(Os::Windows))
    );
    assert_eq!(windows.extensions(), [c"VK_KHR_external_memory_win32"]);
    assert!(
        windows
            .to_string()
            .contains("- interop.dma_buf: not on Windows"),
        "{windows}"
    );
    let linux = Probe::new(offer().with_os(Os::Linux), &caps).plan();
    assert_eq!(
        linux.lacking(&caps),
        [Capability::ExternalMemoryWin32],
        "{linux}"
    );
    assert_eq!(
        linux.extensions(),
        [
            c"VK_KHR_external_memory_fd",
            c"VK_EXT_external_memory_dma_buf"
        ]
    );
    // Neither exists on another OS.
    let other = Probe::new(offer().with_os(Os::Other), &caps).plan();
    assert_eq!(other.lacking(&caps).len(), 3);
}

#[test]
fn needs_are_asked_for_and_gate_the_capability() {
    let offer = || {
        Offer::new(
            V::V1_2,
            V::V1_2,
            [
                c"VK_EXT_shader_object",
                c"VK_KHR_dynamic_rendering",
                c"VK_KHR_depth_stencil_resolve",
                c"VK_KHR_create_renderpass2",
            ],
        )
    };
    let mut probe = Probe::new(offer(), &[Capability::ShaderObject]);
    // Dynamic rendering is off: shader objects fall with it.
    fill(&mut probe, &[(Chain::ShaderObject, "shaderObject")]);
    let plan = probe.plan();
    let shader_object = plan.outcome(Capability::ShaderObject).expect("asked");
    assert_eq!(
        shader_object.verdict,
        Verdict::Needs(Capability::DynamicRendering)
    );
    let dynamic = plan.outcome(Capability::DynamicRendering).expect("implied");
    assert!(dynamic.implied && !dynamic.granted());
    assert_eq!(plan.extensions(), [] as [&CStr; 0], "{plan}");
    assert!(
        plan.to_string()
            .contains("- pipeline.shader_object: needs dynamic_rendering"),
        "{plan}"
    );
    // With it both are granted, and the chain carries both bits.
    let mut probe = Probe::new(offer(), &[Capability::ShaderObject]);
    fill(
        &mut probe,
        &[
            (Chain::ShaderObject, "shaderObject"),
            (Chain::Vulkan12, "drawIndirectCount"),
            (Chain::DynamicRendering, "dynamicRendering"),
        ],
    );
    let plan = probe.plan();
    assert_eq!(plan.lacking(&[Capability::ShaderObject]), []);
    let enabled: Vec<Chain> = enabled_chain(&plan).into_iter().map(|(c, _)| c).collect();
    assert_eq!(
        enabled,
        [Chain::Core10, Chain::ShaderObject, Chain::DynamicRendering]
    );
    assert!(plan.extensions().contains(&c"VK_EXT_shader_object"));
}

#[test]
fn quirks_turn_a_granted_capability_off() {
    let offer = Offer::new(V::V1_3, V::V1_3, [c"VK_EXT_mesh_shader"]);
    let quirk = Quirk {
        capability: Capability::MeshShader,
        vendor: None,
        os: None,
        driver_below: None,
        reason: "hangs in this test",
    };
    let mut probe = Probe::new(offer, &[Capability::MeshShader]).with_quirks(&[quirk]);
    fill(&mut probe, &[(Chain::MeshShader, "meshShader")]);
    let plan = probe.plan();
    assert_eq!(
        plan.outcome(Capability::MeshShader).map(|o| o.verdict),
        Some(Verdict::Quirk("hangs in this test"))
    );
    assert_eq!(plan.extensions(), [] as [&CStr; 0]);
}

/// A `VkPhysicalDeviceProperties` header as a driver writes it.
unsafe fn write_properties(probe: &mut Probe, kind: u32, vendor: u32, device: u32, name: &[u8]) {
    // SAFETY: the probe owns the chain; offsets of VkPhysicalDeviceProperties.
    unsafe {
        let props = probe.properties2().cast::<u8>().add(16);
        for (at, v) in [
            (0, V::V1_3.raw()),
            (4, (101 << 14) + 9175),
            (8, vendor),
            (12, device),
            (16, kind),
        ] {
            props.add(at).cast::<u32>().write_unaligned(v);
        }
        props.add(20).copy_from(name.as_ptr(), name.len());
    }
}

/// A `VkPhysicalDeviceMemoryProperties`: one VRAM heap of `vram` bytes, a
/// host heap, and a `DEVICE_LOCAL|HOST_VISIBLE` type on a heap of `visible`.
unsafe fn write_memory(probe: &mut Probe, vram: u64, visible: u64) {
    // SAFETY: the probe owns the struct; offsets of VkPhysicalDeviceMemoryProperties.
    unsafe {
        let m = probe.memory_properties().cast::<u8>();
        let types: [(u32, u32); 3] = [(1, 0), (6, 1), (7, if visible == vram { 0 } else { 2 })];
        m.cast::<u32>().write_unaligned(3);
        for (i, (flags, heap)) in types.iter().enumerate() {
            m.add(4 + 8 * i).cast::<u32>().write_unaligned(*flags);
            m.add(8 + 8 * i).cast::<u32>().write_unaligned(*heap);
        }
        let heaps: [(u64, u32); 3] = [(vram, 1), (32 << 30, 0), (visible, 1)];
        m.add(260).cast::<u32>().write_unaligned(3);
        for (i, (size, flags)) in heaps.iter().enumerate() {
            m.add(264 + 16 * i).cast::<u64>().write_unaligned(*size);
            m.add(272 + 16 * i).cast::<u32>().write_unaligned(*flags);
        }
    }
}

#[test]
fn device_and_memory_flow_into_the_plan_and_the_ranking() {
    let caps = [Capability::DynamicRendering, Capability::MultiDrawIndirect];
    // A discrete GPU without resizable BAR, granting dynamic rendering only.
    let mut discrete = Probe::new(Offer::new(V::V1_3, V::V1_3, []).with_os(Os::Windows), &caps);
    fill(&mut discrete, &[(Chain::Vulkan13, "dynamicRendering")]);
    // SAFETY: as a driver writes them.
    unsafe {
        write_properties(&mut discrete, 2, 0x8086, 0xE20B, b"Arc B580\0");
        write_memory(&mut discrete, 12 << 30, 256 << 20);
    }
    let discrete = discrete.plan();
    let info = discrete.device_info().expect("read");
    assert_eq!(
        (info.device_type, info.name.as_str()),
        (DeviceType::DiscreteGpu, "Arc B580")
    );
    assert_eq!(
        discrete.memory().map(super::Memory::cpu_visible_vram),
        Some(256 << 20)
    );
    let log = discrete.to_string();
    assert!(
        log.contains("  device: Arc B580, discrete, Intel Xe2-HPG [8086:e20b], driver 101.9175"),
        "{log}"
    );
    assert!(
        log.contains("  memory: 12 GiB VRAM, 256 MiB CPU-visible, 32 GiB host"),
        "{log}"
    );
    // An integrated GPU granting both.
    let mut integrated = Probe::new(Offer::new(V::V1_3, V::V1_3, []).with_os(Os::Windows), &caps);
    fill(
        &mut integrated,
        &[
            (Chain::Vulkan13, "dynamicRendering"),
            (Chain::Core10, "multiDrawIndirect"),
            (Chain::Core10, "drawIndirectFirstInstance"),
        ],
    );
    // SAFETY: as above.
    unsafe { write_properties(&mut integrated, 1, 0x8086, 0xA780, b"UHD 770\0") };
    let integrated = integrated.plan();
    let unknown = Probe::new(Offer::new(V::V1_3, V::V1_3, []), &caps).plan();
    let plans = [integrated, discrete, unknown];
    // Discrete first although it grants less; the unknown device has nothing.
    assert_eq!(rank(&plans, &[Capability::DynamicRendering]), [1, 0]);
    assert_eq!(rank(&plans, &caps), [0]);
    // A host that read the properties itself.
    let named = DeviceInfo {
        name: "given".into(),
        ..plans[1].device_info().expect("read").clone()
    };
    let plan = Probe::new(Offer::new(V::V1_3, V::V1_3, []).with_device(named), &[]).plan();
    assert_eq!(plan.device_info().map(|d| d.name.as_str()), Some("given"));
}

/// The only test that touches the process-wide cache.
#[test]
fn the_cache_keeps_the_first_plans_and_ranks_over_them() {
    let caps = [Capability::DynamicRendering];
    let mut granted = Probe::new(Offer::new(V::V1_3, V::V1_3, []), &caps);
    fill(&mut granted, &[(Chain::Vulkan13, "dynamicRendering")]);
    let plans = vec![
        Probe::new(Offer::new(V::V1_3, V::V1_3, []), &caps).plan(),
        granted.plan(),
    ];
    let mut ran = 0;
    let kept = super::cache::get_or_init(|| {
        ran += 1;
        plans
    });
    assert_eq!(kept.len(), 2);
    // Kept: init doesn't run again, publish hands its plans back.
    assert!(std::ptr::eq(super::cache::get_or_init(Vec::new), kept));
    assert_eq!(ran, 1);
    assert!(super::cache::publish(Vec::new()).is_err());
    assert!(std::ptr::eq(super::cache::plans(), kept));
    assert_eq!(super::cache::best(&caps).map(|(i, _)| i), Some(1));
    assert_eq!(
        super::cache::best(&[Capability::MeshShader]).map(|(i, _)| i),
        None
    );
    // The budget cache starts from the plan's copy; no budget was read.
    let budget = super::BudgetCache::for_plan(&kept[1], std::time::Duration::from_secs(1));
    assert!(budget.cached().is_none());
}

#[test]
fn untested_platforms_are_marked_in_the_log() {
    let cap = [Capability::SamplerAnisotropy];
    for os in [Os::Linux, Os::Windows] {
        let mut probe = Probe::new(Offer::new(V::V1_3, V::V1_3, []).with_os(os), &cap);
        fill(&mut probe, &[(Chain::Core10, "samplerAnisotropy")]);
        let plan = probe.plan();
        let outcome = plan.outcome(Capability::SamplerAnisotropy).expect("asked");
        let untested = Capability::SamplerAnisotropy.spec().support.on(os) == Status::Untested;
        assert_eq!(outcome.untested_on, untested.then_some(os));
        assert_eq!(
            plan.to_string().contains(&format!("(untested on {os})")),
            untested,
            "{plan}"
        );
    }
}

#[test]
fn support_flags_are_consistent() {
    for &cap in Capability::ALL {
        let spec = cap.spec();
        // Win32 routes exist on Windows only, fd and dma-buf routes on Linux only.
        let names: Vec<&CStr> = spec
            .routes
            .iter()
            .filter_map(|r| match *r {
                Route::Extension { name, .. } => Some(name),
                _ => None,
            })
            .collect();
        let win32 = names.iter().any(|n| n.to_bytes().ends_with(b"_win32"));
        let posix = names.iter().any(|n| {
            let n = n.to_bytes();
            n.ends_with(b"_fd") || n.ends_with(b"_dma_buf") || n.ends_with(b"drm_format_modifier")
        });
        assert_eq!(
            spec.support.linux == Status::Unavailable,
            win32,
            "{}",
            spec.name
        );
        assert_eq!(
            spec.support.windows == Status::Unavailable,
            posix,
            "{}",
            spec.name
        );
        // Needs resolve to other capabilities, without cycles.
        let mut seen = vec![cap];
        let mut i = 0;
        while i < seen.len() {
            for &need in seen[i].spec().needs {
                assert_ne!(need, cap, "{} needs itself", spec.name);
                if !seen.contains(&need) {
                    seen.push(need);
                }
            }
            i += 1;
        }
    }
}

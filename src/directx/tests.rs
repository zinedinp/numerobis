use std::ffi::c_void;
use std::time::Duration;

use super::generated::{D3d11Feature, D3d12Feature, iid, value};
use super::legacy::{self, Caps9, bits};
use super::modern::queue::{self, ListType, QueuePriority};
use super::modern::{
    self, Adapter, BudgetCache, Capability as Cap, Probe, Route, Skip, Test, dxgi,
};
use super::{Api, Era, FeatureLevel, Guid, LegacyLevel, Runtime, Select, ShaderModel};

/// A `CheckFeatureSupport` that answers from `known`: per feature id, the
/// words to write; anything else is refused like an old runtime would.
fn device<'a>(known: &'a [(u32, &'a [u32])]) -> impl FnMut(u32, *mut c_void, u32) -> bool + 'a {
    move |feature, data, size| {
        let Some((_, words)) = known.iter().find(|(f, _)| *f == feature) else {
            return false;
        };
        let n = (size as usize / 4).min(words.len());
        // SAFETY: the probe hands a buffer of `size` bytes.
        unsafe { std::ptr::copy_nonoverlapping(words.as_ptr(), data.cast::<u32>(), n) };
        true
    }
}

fn words_with(feature: D3d12Feature, set: &[(&str, u32)]) -> Vec<u32> {
    let mut w = vec![0; feature.fields().len()];
    for (name, v) in set {
        w[feature.index(name).expect("member")] = *v;
    }
    w
}

#[test]
fn versions_parse_pin_and_detect() {
    assert_eq!("12".parse::<Api>(), Ok(Api::D3D12));
    assert_eq!("D3D9".parse::<Api>(), Ok(Api::D3D9));
    assert_eq!("dx11".parse::<Api>(), Ok(Api::D3D11));
    assert_eq!("Direct3D 8".parse::<Api>(), Ok(Api::D3D8));
    assert!("7".parse::<Api>().is_err(), "nothing before 8 (DirectDraw)");
    assert_eq!("10".parse::<Select>(), Ok(Select::Pin(Api::D3D10)));
    assert_eq!("auto".parse::<Select>(), Ok(Select::DETECT));
    assert_eq!(
        Select::DETECT.order(),
        [Api::D3D12, Api::D3D11, Api::D3D10, Api::D3D9, Api::D3D8]
    );
    assert_eq!(
        "auto-10".parse::<Select>().map(Select::order),
        Ok(vec![Api::D3D12, Api::D3D11, Api::D3D10])
    );
    // The first that works: 12 fails here, 11 is taken.
    let got = Select::DETECT.first(|api| (api <= Api::D3D11).then_some(api.dll()));
    assert_eq!(got, Some((Api::D3D11, "d3d11.dll")));
    assert_eq!(Select::Pin(Api::D3D12).first(|_| None::<()>), None);
}

#[test]
fn the_eras_split_at_ten() {
    for api in Api::ALL {
        let legacy = api <= Api::D3D9;
        assert_eq!(api.era() == Era::Legacy, legacy, "{api}");
        assert_eq!(api.levels().is_empty(), legacy, "{api}");
    }
    assert!(!Api::D3D8.has_64bit_runtime());
    assert_eq!(Api::D3D8.sdk_version(), Some(220));
    assert_eq!(Api::D3D11.sdk_version(), Some(7));
}

#[test]
fn guids_convert_while_compiling_and_print_back() {
    const ID: Guid = Guid::parse("189819F1-1DB6-4B57-BE54-1821339B85F7");
    assert_eq!(ID, iid::ID3D12Device);
    assert_eq!(ID.data1, 0x1898_19f1);
    assert_eq!(ID.data4, [0xbe, 0x54, 0x18, 0x21, 0x33, 0x9b, 0x85, 0xf7]);
    assert_eq!(ID.to_string(), "189819f1-1db6-4b57-be54-1821339b85f7");
    assert_eq!(size_of::<Guid>(), 16);
}

#[test]
#[should_panic(expected = "8-4-4-4-12")]
fn a_malformed_guid_panics() {
    let _ = Guid::parse(std::hint::black_box("189819f1-1db6-4b57-be54"));
}

#[test]
fn feature_levels_round_trip_and_order() {
    for l in FeatureLevel::ALL {
        assert_eq!(FeatureLevel::from_raw(l.raw()), Some(l));
    }
    assert!(
        FeatureLevel::ALL
            .windows(2)
            .all(|w| w[0] < w[1] && w[0].raw() < w[1].raw())
    );
    assert_eq!(FeatureLevel::L12_2.raw(), 0xc200);
    assert_eq!(FeatureLevel::from_raw(0x1234), None);
    assert_eq!(
        FeatureLevel::down_from(FeatureLevel::L11_1, FeatureLevel::L10_0),
        [0xb100, 0xb000, 0xa100, 0xa000]
    );
    assert!(FeatureLevel::L1_0Core.is_compute_only());
    assert_eq!(FeatureLevel::L1_0Core.shader_model(), None);
    assert_eq!(
        FeatureLevel::L12_2.shader_model(),
        Some(ShaderModel::new(6, 5))
    );
    assert_eq!(modern::D3D12_LEVELS[0], FeatureLevel::L11_0.raw());
}

#[test]
fn shader_models_in_both_encodings() {
    assert_eq!(ShaderModel::new(6, 5).raw(), value::D3D_SHADER_MODEL_6_5);
    assert_eq!(ShaderModel::from_raw(0x69), ShaderModel::NEWEST);
    let order = ShaderModel::d3d12_query_order();
    assert_eq!(order.first(), Some(&value::D3D_SHADER_MODEL_6_9));
    assert_eq!(order.last(), Some(&value::D3D_SHADER_MODEL_5_1));
    // D3DCAPS: vs_1_1 = 0xFFFE0101, ps_1_4 = 0xFFFF0104.
    assert_eq!(
        ShaderModel::from_vs(0xFFFE_0101),
        Some(ShaderModel::new(1, 1))
    );
    assert_eq!(
        ShaderModel::from_ps(0xFFFF_0104),
        Some(ShaderModel::new(1, 4))
    );
    assert_eq!(ShaderModel::from_ps(0xFFFE_0104), None, "a vertex version");
    assert_eq!(ShaderModel::from_vs(0), None);
    assert_eq!(ShaderModel::new(3, 0).ps_caps(), 0xFFFF_0300);
}

/// Caps of a shader model 3 card like the B580 reports (the parts the
/// tests read).
fn sm3_caps() -> Caps9 {
    let mut c = Caps9 {
        dev_caps: bits::D3DDEVCAPS_HWTRANSFORMANDLIGHT | bits::D3DDEVCAPS_PUREDEVICE,
        caps2: bits::D3DCAPS2_CANAUTOGENMIPMAP | bits::D3DCAPS2_DYNAMICTEXTURES,
        primitive_misc_caps: bits::D3DPMISCCAPS_SEPARATEALPHABLEND | bits::D3DPMISCCAPS_BLENDOP,
        raster_caps: bits::D3DPRASTERCAPS_SCISSORTEST | bits::D3DPRASTERCAPS_DEPTHBIAS,
        stencil_caps: bits::D3DSTENCILCAPS_TWOSIDED,
        texture_caps: bits::D3DPTEXTURECAPS_CUBEMAP | bits::D3DPTEXTURECAPS_MIPMAP,
        texture_address_caps: bits::D3DPTADDRESSCAPS_MIRRORONCE,
        max_texture_width: 16384,
        max_texture_height: 16384,
        max_volume_extent: 2048,
        max_anisotropy: 16,
        max_primitive_count: 0xFF_FFFF,
        max_vertex_index: 0xFF_FFFF,
        max_vertex_shader_const: 1024,
        num_simultaneous_rts: 4,
        ps20_caps_num_instruction_slots: 512,
        vertex_shader_version: ShaderModel::new(3, 0).vs_caps(),
        pixel_shader_version: ShaderModel::new(3, 0).ps_caps(),
        ..Caps9::default()
    };
    c.max_point_size = 256.0f32.to_bits();
    c
}

#[test]
fn the_caps_words_are_the_struct() {
    let caps = sm3_caps();
    let words = *caps.words();
    assert_eq!(words[legacy::word("DevCaps")], caps.dev_caps);
    assert_eq!(words[legacy::word("PS20Caps.NumInstructionSlots")], 512);
    assert_eq!(Caps9::from_words(words), caps);
    assert_eq!(
        legacy::CAPS9_MEMBERS[legacy::CAPS8_WORDS - 1],
        "PixelShader1xMaxValue"
    );
    assert!((caps.max_point_size() - 256.0).abs() < f32::EPSILON);
}

#[test]
fn direct3d8_caps_are_converted_like_the_wrappers() {
    let caps = sm3_caps();
    let c8 = caps.as_d3d8();
    // Shader versions capped, constants clamped (d3d8to9).
    assert_eq!(c8.vertex_shader(), Some(ShaderModel::new(1, 1)));
    assert_eq!(c8.pixel_shader(), Some(ShaderModel::new(1, 4)));
    assert_eq!(c8.max_vertex_shader_const, 256);
    // D3D9-only bits cleared, D3D8 bits set (DXVK's ConvertCaps8).
    assert_eq!(c8.caps2 & bits::D3DCAPS2_CANAUTOGENMIPMAP, 0);
    assert_ne!(c8.caps2 & legacy::bits8::D3DCAPS2_CANRENDERWINDOWED, 0);
    assert_eq!(c8.stencil_caps & bits::D3DSTENCILCAPS_TWOSIDED, 0);
    assert_eq!(c8.raster_caps & bits::D3DPRASTERCAPS_SCISSORTEST, 0);
    assert_ne!(c8.raster_caps & legacy::bits8::D3DPRASTERCAPS_ZBIAS, 0);
    assert_eq!(c8.primitive_misc_caps, bits::D3DPMISCCAPS_BLENDOP);
    // Words past D3DCAPS8 are gone; the rest is kept.
    assert!(c8.words()[legacy::CAPS8_WORDS..].iter().all(|&w| w == 0));
    assert_eq!(c8.max_texture_width, 16384);
    // No shaders stay no shaders.
    let ff = Caps9 {
        dev_caps: bits::D3DDEVCAPS_HWTRANSFORMANDLIGHT,
        ..Caps9::default()
    };
    assert_eq!(ff.as_d3d8().vertex_shader(), None);
}

#[test]
fn legacy_levels_from_the_shader_versions() {
    let mut caps = sm3_caps();
    assert_eq!(LegacyLevel::of(&caps), LegacyLevel::Sm3);
    assert_eq!(LegacyLevel::level9(&caps), Some(FeatureLevel::L9_3));
    caps.vertex_shader_version = ShaderModel::new(2, 0).vs_caps();
    caps.pixel_shader_version = ShaderModel::new(2, 0).ps_caps();
    assert_eq!(LegacyLevel::of(&caps), LegacyLevel::Sm2x);
    caps.ps20_caps_num_instruction_slots = 96;
    assert_eq!(LegacyLevel::of(&caps), LegacyLevel::Sm2);
    assert_eq!(
        LegacyLevel::level9(&caps),
        Some(FeatureLevel::L9_2),
        "no 512 slots: not 9_3"
    );
    caps.primitive_misc_caps = 0;
    assert_eq!(
        LegacyLevel::level9(&caps),
        Some(FeatureLevel::L9_1),
        "no separate alpha: 9_1"
    );
    let c8 = sm3_caps().as_d3d8();
    assert_eq!(LegacyLevel::of(&c8), LegacyLevel::Sm1_4);
    assert_eq!(LegacyLevel::level9(&c8), None);
    assert_eq!(
        LegacyLevel::of(&Caps9::default()),
        LegacyLevel::FixedFunction
    );
}

/// `GetDeviceCaps` of the B580 (driver 101.9175, Windows 11): Direct3D 9,
/// and Direct3D 8 from Microsoft's 32-bit `d3d8.dll`, read by a 32-bit
/// helper (numerobis itself doesn't build for 32-bit x86: miraculix).
const B580_CAPS9: [u32; 76] = [
    0x0000_0001,
    0x0000_0000,
    0x0002_0000,
    0xe002_0000,
    0x0000_0320,
    0x8000_000f,
    0x0000_0001,
    0x0019_ae50,
    0x002f_ccf2,
    0x0773_6191,
    0x0000_00ff,
    0x0000_3fff,
    0x0000_27ff,
    0x0000_00ff,
    0x0008_4208,
    0x0021_ec45,
    0x0703_0700,
    0x0703_0700,
    0x0703_0700,
    0x0000_003f,
    0x0000_003f,
    0x0000_001f,
    0x0000_4000,
    0x0000_4000,
    0x0000_0800,
    0x0000_4000,
    0x0000_4000,
    0x0000_0010,
    0x5015_02f9,
    0xc700_0000,
    0xc700_0000,
    0x4700_0000,
    0x4700_0000,
    0x0000_0000,
    0x0000_01ff,
    0x0018_0008,
    0x03ff_ffff,
    0x0000_0008,
    0x0000_0008,
    0x0000_017b,
    0x0000_0008,
    0x0000_0006,
    0x0000_0004,
    0x0000_0000,
    0x4380_0000,
    0x007f_ffff,
    0x00ff_ffff,
    0x0000_0010,
    0x0000_00ff,
    0xfffe_0300,
    0x0000_0100,
    0xffff_0300,
    0x7f7f_ffff,
    0x0000_0051,
    0x0000_0000,
    0x0000_0000,
    0x0000_0000,
    0x0000_0000,
    0x0000_0001,
    0x0000_037f,
    0x0000_0004,
    0x0300_0300,
    0x0000_0001,
    0x0000_0018,
    0x0000_0020,
    0x0000_0004,
    0x0000_001f,
    0x0000_0018,
    0x0000_0020,
    0x0000_0004,
    0x0000_0200,
    0x0303_0300,
    0xffff_ffff,
    0xffff_ffff,
    0x0000_8000,
    0x0000_8000,
];
const B580_CAPS8: [u32; 53] = [
    0x0000_0001,
    0x0000_0000,
    0x0002_0000,
    0x200a_0000,
    0x0000_00a0,
    0x8000_000f,
    0x0000_0001,
    0x0819_ae50,
    0x0000_0cf2,
    0x0073_6191,
    0x0000_00ff,
    0x0000_1fff,
    0x0000_07ff,
    0x0000_00ff,
    0x0008_4208,
    0x0001_ec45,
    0x0703_0700,
    0x0703_0700,
    0x0703_0700,
    0x0000_003f,
    0x0000_003f,
    0x0000_001f,
    0x0000_2000,
    0x0000_2000,
    0x0000_0800,
    0x0000_2000,
    0x0000_4000,
    0x0000_0010,
    0x5015_02f9,
    0xc700_0000,
    0xc700_0000,
    0x4700_0000,
    0x4700_0000,
    0x0000_0000,
    0x0000_00ff,
    0x0018_0008,
    0x03ff_ffff,
    0x0000_0008,
    0x0000_0008,
    0x0000_007b,
    0x0000_0008,
    0x0000_0006,
    0x0000_0004,
    0x0000_0000,
    0x4380_0000,
    0x007f_ffff,
    0x00ff_ffff,
    0x0000_0010,
    0x0000_00ff,
    0xfffe_0101,
    0x0000_0100,
    0xffff_0104,
    0x7f7f_ffff,
];

#[test]
fn the_conversion_matches_microsofts_direct3d8_on_the_b580() {
    use legacy::Capability as L;
    let caps9 = Caps9::from_words(B580_CAPS9);
    let mut words = [0u32; legacy::CAPS9_WORDS];
    words[..legacy::CAPS8_WORDS].copy_from_slice(&B580_CAPS8);
    let real8 = Caps9::from_words(words);
    let converted = caps9.as_d3d8();
    let differ: Vec<&str> = (0..legacy::CAPS8_WORDS)
        .filter(|&i| real8.words()[i] != converted.words()[i])
        .map(|i| legacy::CAPS9_MEMBERS[i])
        .collect();
    assert_eq!(
        differ,
        [
            "Caps3",
            "DevCaps",
            "MaxTextureWidth",
            "MaxTextureHeight",
            "MaxTextureRepeat"
        ]
    );
    let real = legacy::plan(Api::D3D8, &real8, L::ALL);
    let ours = legacy::plan(Api::D3D8, &converted, L::ALL);
    assert_eq!(real.outcomes(), ours.outcomes());
    assert_eq!(real.level, LegacyLevel::Sm1_4);
    let nine = legacy::plan(Api::D3D9, &caps9, L::ALL);
    assert_eq!(
        (nine.level, nine.level9),
        (LegacyLevel::Sm3, Some(FeatureLevel::L9_3))
    );
    assert_eq!(nine.outcomes().iter().filter(|o| o.granted()).count(), 45);
}

#[test]
fn a_direct3d8_plan_lacks_what_came_with_9() {
    use legacy::{Capability as L, Why};
    let caps = sm3_caps();
    let plan9 = legacy::plan(
        Api::D3D9,
        &caps,
        &[L::TwoSidedStencil, L::PureDevice, L::Instancing],
    );
    assert!(plan9.has(L::TwoSidedStencil) && plan9.has(L::PureDevice) && plan9.has(L::Instancing));
    assert!(
        plan9.has(L::HardwareTnl),
        "PureDevice needs it, so it is planned too"
    );
    let plan8 = legacy::plan(
        Api::D3D8,
        &caps.as_d3d8(),
        &[L::TwoSidedStencil, L::PixelShader14, L::Index32],
    );
    assert_eq!(
        plan8.outcome(L::TwoSidedStencil).map(|o| o.why),
        Some(Some(Why::Since(Api::D3D9)))
    );
    assert!(plan8.has(L::PixelShader14));
    assert!(plan8.has(L::Index32));
    assert_eq!(plan8.level9, None);
    // Full NPOT (POW2 clear) is conditional NPOT too.
    let plan = legacy::plan(Api::D3D9, &caps, &[L::NonPow2, L::NonPow2Conditional]);
    assert!(plan.has(L::NonPow2) && plan.has(L::NonPow2Conditional));
    let pow2 = Caps9 {
        texture_caps: bits::D3DPTEXTURECAPS_POW2,
        ..caps
    };
    let plan = legacy::plan(Api::D3D9, &pow2, &[L::NonPow2, L::NonPow2Conditional]);
    assert!(!plan.has(L::NonPow2) && !plan.has(L::NonPow2Conditional));
    assert!(plan.to_string().contains("not in TextureCaps"));
}

#[test]
#[should_panic(expected = "is not Direct3D 8 or 9")]
fn the_legacy_plan_refuses_modern_versions() {
    let _ = legacy::plan(Api::D3D11, &Caps9::default(), &[]);
}

#[test]
fn d3d12_reads_the_tiers_and_the_shader_model() {
    let opt5 = words_with(
        D3d12Feature::Options5,
        &[("RaytracingTier", value::D3D12_RAYTRACING_TIER_1_1)],
    );
    let opt7 = words_with(
        D3d12Feature::Options7,
        &[("MeshShaderTier", value::D3D12_MESH_SHADER_TIER_1)],
    );
    let sm = [value::D3D_SHADER_MODEL_6_6];
    let known: [(u32, &[u32]); 3] = [
        (D3d12Feature::Options5.id(), &opt5),
        (D3d12Feature::Options7.id(), &opt7),
        (D3d12Feature::ShaderModel.id(), &sm),
    ];
    let mut probe = Probe::new(
        Api::D3D12,
        FeatureLevel::L12_1,
        &[Cap::Raytracing12, Cap::MeshShader, Cap::WorkGraphs],
    );
    assert!(
        probe
            .structs()
            .contains(&"D3D12_FEATURE_DATA_D3D12_OPTIONS5")
    );
    probe.fill(device(&known));
    let plan = probe.plan();
    assert_eq!(plan.shader_model, Some(ShaderModel::new(6, 6)));
    assert!(plan.has(Cap::Raytracing), "pulled in as a need");
    assert!(plan.has(Cap::Raytracing11));
    assert!(plan.has(Cap::MeshShader));
    let rt12 = plan.outcome(Cap::Raytracing12).expect("planned");
    assert_eq!(
        rt12.skips,
        [Skip::Reported {
            member: "RaytracingTier",
            value: 11
        }]
    );
    // OPTIONS21 is refused: the runtime predates it.
    let wg = plan.outcome(Cap::WorkGraphs).expect("planned");
    assert_eq!(
        wg.skips,
        [Skip::Unknown("D3D12_FEATURE_DATA_D3D12_OPTIONS21")]
    );
}

#[test]
fn the_shader_model_query_steps_down_and_gates_capabilities() {
    // A runtime that knows shader model 6.5 at most and fails newer ones.
    let mut asked = Vec::new();
    let opt1 = vec![1u32, 32, 32, 0, 0, 1]; // WaveOps, lanes, Int64ShaderOps
    let mut probe = Probe::new(
        Api::D3D12,
        FeatureLevel::L12_0,
        &[Cap::WaveOps, Cap::Int64, Cap::Native16Bit],
    );
    probe.fill(|feature, data, size| {
        let words = data.cast::<u32>();
        if feature == D3d12Feature::ShaderModel.id() {
            // SAFETY: the probe hands one word.
            let want = unsafe { *words };
            asked.push(want);
            return want <= value::D3D_SHADER_MODEL_6_5;
        }
        if feature == D3d12Feature::Options1.id() {
            // SAFETY: OPTIONS1 is six words.
            unsafe { std::ptr::copy_nonoverlapping(opt1.as_ptr(), words, (size / 4) as usize) };
            return true;
        }
        if feature == D3d12Feature::Options4.id() {
            // SAFETY: OPTIONS4's third word is Native16BitShaderOpsSupported.
            unsafe { *words.add(2) = 1 };
            return true;
        }
        false
    });
    assert_eq!(&asked[..5], [0x69, 0x68, 0x67, 0x66, 0x65]);
    let plan = probe.plan();
    assert_eq!(plan.shader_model, Some(ShaderModel::new(6, 5)));
    assert!(plan.has(Cap::WaveOps) && plan.has(Cap::Int64) && plan.has(Cap::Native16Bit));

    // The same bits on a 5.1 device: wave ops need 6.0.
    let mut probe = Probe::new(Api::D3D12, FeatureLevel::L12_0, &[Cap::WaveOps]);
    let sm51 = [value::D3D_SHADER_MODEL_5_1];
    probe.fill(device(&[
        (D3d12Feature::Options1.id(), &opt1),
        (D3d12Feature::ShaderModel.id(), &sm51),
    ]));
    let plan = probe.plan();
    assert!(!plan.has(Cap::WaveOps));
    assert_eq!(
        plan.outcome(Cap::WaveOps)
            .and_then(|o| o.skips.last().copied()),
        Some(Skip::ShaderModel(ShaderModel::new(6, 0)))
    );
}

#[test]
fn d3d11_and_d3d10_take_their_own_routes() {
    let opt2 = {
        let f = D3d11Feature::Options2;
        let mut w = vec![0u32; f.fields().len()];
        w[f.index("ROVsSupported").expect("member")] = 1;
        w[f.index("ConservativeRasterizationTier").expect("member")] =
            value::D3D11_CONSERVATIVE_RASTERIZATION_TIER_1;
        w
    };
    let mut probe = Probe::new(
        Api::D3D11,
        FeatureLevel::L11_1,
        &[
            Cap::Rovs,
            Cap::ConservativeRasterization,
            Cap::Raytracing,
            Cap::Tessellation,
        ],
    );
    assert_eq!(probe.structs(), ["D3D11_FEATURE_DATA_D3D11_OPTIONS2"]);
    probe.fill(device(&[(D3d11Feature::Options2.id(), &opt2)]));
    let plan = probe.plan();
    assert!(
        plan.has(Cap::Rovs)
            && plan.has(Cap::ConservativeRasterization)
            && plan.has(Cap::Tessellation)
    );
    assert_eq!(
        plan.outcome(Cap::Raytracing).map(|o| o.skips.clone()),
        Some(vec![Skip::NotOn(Api::D3D11)])
    );
    assert_eq!(
        plan.shader_model,
        Some(ShaderModel::new(5, 0)),
        "the level's"
    );

    // Direct3D 10.1 has nothing to ask: the level decides.
    let plan = Probe::new(
        Api::D3D10,
        FeatureLevel::L10_1,
        &[Cap::GeometryShader, Cap::Tessellation, Cap::ComputeShader],
    )
    .plan();
    assert!(plan.has(Cap::GeometryShader));
    assert_eq!(
        plan.outcome(Cap::Tessellation).map(|o| o.skips.clone()),
        Some(vec![Skip::Level(FeatureLevel::L11_0)])
    );
    assert!(
        !plan.has(Cap::ComputeShader),
        "cs_4_x is asked through Direct3D 11 only"
    );
}

#[test]
fn root_signature_steps_down_too() {
    let mut probe = Probe::new(Api::D3D12, FeatureLevel::L12_0, &[Cap::RootSignature11]);
    probe.fill(|feature, data, _| {
        if feature != D3d12Feature::RootSignature.id() {
            return false;
        }
        // SAFETY: one word, HighestVersion.
        unsafe { *data.cast::<u32>() <= 2 }
    });
    assert!(probe.plan().has(Cap::RootSignature11));
}

#[test]
#[should_panic(expected = "is not Direct3D 10 to 12")]
fn the_modern_probe_refuses_legacy_versions() {
    let _ = Probe::new(Api::D3D9, FeatureLevel::L9_3, &[]);
}

#[test]
fn every_route_reads_a_member_of_the_right_kind() {
    for &cap in Cap::ALL {
        let spec = cap.spec();
        assert!(!spec.routes.is_empty(), "{}", spec.name);
        for route in spec.routes {
            let (kind, test) = match *route {
                Route::D3d12 {
                    feature,
                    field,
                    test,
                } => (feature.types()[field], test),
                Route::D3d11 {
                    feature,
                    field,
                    test,
                } => (feature.types()[field], test),
                Route::Level(_) | Route::Api(_) => continue,
            };
            if kind == "BOOL" {
                assert_eq!(test, Test::AtLeast(1), "{}: a BOOL is on or off", spec.name);
            }
            if let Test::Bits(_) = test {
                assert_ne!(kind, "BOOL", "{}", spec.name);
            }
        }
        for need in spec.needs {
            let i = Cap::ALL.iter().position(|c| c == need).expect("listed");
            let me = Cap::ALL.iter().position(|c| *c == cap).expect("listed");
            assert!(i != me, "{} needs itself", spec.name);
        }
    }
    let names: std::collections::HashSet<&str> = Cap::ALL.iter().map(|c| c.name()).collect();
    assert_eq!(names.len(), Cap::ALL.len(), "names are unique");
    let names: std::collections::HashSet<&str> =
        legacy::Capability::ALL.iter().map(|c| c.name()).collect();
    assert_eq!(names.len(), legacy::Capability::ALL.len());
}

#[test]
fn raytracing_and_mesh_shaders_have_areas_of_their_own() {
    use modern::Area;
    for &cap in Cap::ALL {
        let area = cap.spec().area;
        let name = cap.name();
        if name.starts_with("raytracing") {
            assert_eq!(area, Area::Raytracing, "{name}");
        }
        if name.starts_with("mesh_shader") {
            assert_eq!(area, Area::Mesh, "{name}");
        }
    }
}

#[test]
fn queue_priority_steps_down_when_refused() {
    assert_eq!(
        QueuePriority::from_raw(10000),
        Some(QueuePriority::GlobalRealtime)
    );
    assert_eq!(QueuePriority::High.raw(), 100);
    assert_eq!("realtime".parse(), Ok(QueuePriority::GlobalRealtime));
    // A device that takes high but not realtime.
    let check = |feature: u32, data: *mut c_void, _| {
        assert_eq!(feature, D3d12Feature::CommandQueuePriority.id());
        let d = data.cast::<u32>();
        // SAFETY: [type, priority, supported].
        unsafe {
            assert_eq!(*d, ListType::Compute.raw());
            *d.add(2) = u32::from(*d.add(1) <= 100);
        }
        true
    };
    assert!(!queue::supported(
        ListType::Compute,
        QueuePriority::GlobalRealtime,
        check
    ));
    let order = queue::candidates(
        ListType::Compute,
        QueuePriority::GlobalRealtime,
        check,
        |_| true,
    );
    assert_eq!(order, [QueuePriority::High, QueuePriority::Normal]);
    let all = [QueuePriority::High, QueuePriority::Normal];
    assert_eq!(
        queue::create_with_fallback(&all, |p| if p == QueuePriority::High {
            Err(-1)
        } else {
            Ok(1)
        }),
        Ok((1, QueuePriority::Normal))
    );
    assert_eq!(
        queue::create_with_fallback(&all, |_| Err::<(), _>(-7)),
        Err(-7)
    );
    assert_eq!(
        queue::create_with_fallback::<()>(&[], |_| Ok(())),
        Err(queue::E_ACCESSDENIED)
    );
}

fn adapter(name: &str, vendor: u32, vram: usize, flags: u32) -> Adapter {
    let mut desc = dxgi::AdapterDesc1::default();
    for (i, c) in name.encode_utf16().enumerate() {
        desc.description[i] = c;
    }
    desc.vendor_id = vendor;
    desc.device_id = 0xe20b;
    desc.dedicated_video_memory = vram;
    desc.adapter_luid = [0x1234, 0x1];
    desc.flags = flags;
    Adapter::from_desc1(&desc)
}

#[test]
fn adapters_read_from_dxgi() {
    // 2 GiB: `SIZE_T` is 32 bits in a 32-bit process (DXGI caps it there).
    let a = adapter("Intel(R) Arc(TM) B580 Graphics", 0x8086, 2 << 30, 0);
    assert_eq!(a.name, "Intel(R) Arc(TM) B580 Graphics");
    assert_eq!(a.luid, 0x1_0000_1234);
    assert!(!a.software);
    assert!(a.describe().contains("2 GiB VRAM"));
    assert!(adapter("WARP", 0x1414, 0, 0).software);
    assert!(adapter("x", 0x10de, 0, dxgi::ADAPTER_FLAG_SOFTWARE).software);
    assert!(Runtime::is_warp(0x1414, 0x8c));
}

#[test]
fn the_budget_is_read_again_when_old() {
    let memory = modern::VideoMemory {
        budget: 10,
        current_usage: 4,
        ..Default::default()
    };
    assert_eq!(memory.available(), 6);
    let mut cache = BudgetCache::new(Duration::from_secs(3600));
    let mut reads = 0;
    for _ in 0..3 {
        let got = cache.get(|m| {
            reads += 1;
            *m = memory;
            true
        });
        assert_eq!(got, Some(memory));
    }
    assert_eq!(reads, 1);
    cache.invalidate();
    assert_eq!(cache.get(|_| false), None, "a failed read is not kept");
    assert_eq!(cache.cached(), None);
}

#[test]
fn ranking_puts_hardware_and_newer_versions_first() {
    let plan = |api, level, a: Adapter| {
        Probe::new(api, level, &[Cap::GeometryShader])
            .with_adapter(a)
            .plan()
    };
    let plans = [
        plan(
            Api::D3D12,
            FeatureLevel::L12_1,
            adapter("WARP", 0x1414, 0, 0),
        ),
        plan(
            Api::D3D11,
            FeatureLevel::L12_1,
            adapter("B580", 0x8086, 2 << 30, 0),
        ),
        plan(
            Api::D3D12,
            FeatureLevel::L12_2,
            adapter("B580", 0x8086, 2 << 30, 0),
        ),
    ];
    assert_eq!(modern::rank(&plans, &[Cap::GeometryShader]), [2, 1, 0]);
    assert_eq!(
        modern::rank(&plans, &[Cap::Tessellation]),
        Vec::<usize>::new(),
        "not planned"
    );
}

#[test]
fn runtimes_answer_their_own_interfaces() {
    assert_eq!(
        Runtime::by_interface(&Guid::parse("39da4e09-bd1c-4198-9fae-86bbe3be41fd")),
        Some(Runtime::Vkd3dProton)
    );
    assert_eq!(
        Runtime::by_interface(&iid::IDirect3DDevice9On12),
        Some(Runtime::D3d9On12)
    );
    assert_eq!(Runtime::by_interface(&iid::ID3D12Device), None);
    assert_eq!(Runtime::Vkd3dProton.apis(), [Api::D3D12]);
    assert!(Runtime::Dxvk.apis().contains(&Api::D3D8));
    assert_eq!(Runtime::Dxvk.to_string(), "DXVK (on Vulkan)");
    assert!(!Runtime::Xbox.available_on(super::Os::Windows));
}

#[test]
fn the_cache_keeps_the_first_plans() {
    let plans = modern::cache::get_or_init(|| {
        vec![Probe::new(Api::D3D10, FeatureLevel::L10_0, &[Cap::GeometryShader]).plan()]
    });
    assert_eq!(plans.len(), 1);
    assert!(modern::cache::publish(Vec::new()).is_err());
    assert_eq!(
        modern::cache::best(&[Cap::GeometryShader]).map(|(i, _)| i),
        Some(0)
    );
}

#[cfg(feature = "vulkan")]
mod with_vulkan {
    use super::super::vulkan::{format, level_needs, needs};
    use super::*;
    use crate::vulkan::GlobalPriority;

    #[test]
    fn priorities_convert_both_ways() {
        for p in QueuePriority::ALL {
            assert_eq!(QueuePriority::from_vulkan(p.to_vulkan()), p);
        }
        assert_eq!(
            QueuePriority::from_vulkan(GlobalPriority::Low),
            QueuePriority::Normal
        );
    }

    #[test]
    fn formats_convert_one_to_one() {
        for &(dxgi, vk) in format::TABLE {
            assert_eq!(format::to_vk(dxgi), Some(vk));
            assert_eq!(format::from_vk(vk), Some(dxgi));
        }
        assert_eq!(format::to_vk(0), None);
    }

    #[test]
    fn translation_needs_are_vulkan_capabilities() {
        // Every D3D raytracing tier needs a raytracing pipeline on Vulkan;
        // mesh shaders need task shaders for amplification.
        for cap in [Cap::Raytracing, Cap::Raytracing11, Cap::Raytracing12] {
            assert!(
                needs(cap)
                    .is_some_and(|n| n.contains(&crate::vulkan::Capability::RayTracingPipeline))
            );
        }
        assert_eq!(needs(Cap::GpuUploadHeap), None, "a memory topology, no bit");
        assert_eq!(needs(Cap::ComputeShader), Some(&[][..]));
        assert_eq!(
            level_needs(FeatureLevel::L10_1),
            &[] as &[crate::vulkan::Capability]
        );
        assert!(
            level_needs(FeatureLevel::L12_1)
                .contains(&crate::vulkan::Capability::FragmentShaderInterlock)
        );
    }
}

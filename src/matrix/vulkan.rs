//! The matrix engines from the Vulkan side (feature `vulkan`):
//! `VK_KHR_cooperative_matrix`.
//!
//! The driver lists per device which `(A, B, C, Result)` types and `M×N×K`
//! tiles it takes. The host passes [`Properties`] to
//! `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` (count, then fill);
//! [`Properties::mma`] reads one entry in the engine model, and [`compare`]
//! sets the driver's list next to the engine's table: what matches, what
//! the hardware has but the driver does not expose, and what the table does
//! not explain (a table error, or a driver tile spanning formats it lacks).
//!
//! ```
//! use numerobis::matrix::vulkan::{Properties, compare, component};
//! use numerobis::matrix::{Engine, intel::Xmx};
//!
//! // What the Arc B580's driver (101.9175) reports.
//! let reported = [
//!     Properties::new(8, 16, 16, component::FLOAT16, component::FLOAT32),
//!     Properties::new(8, 16, 32, component::UINT8, component::UINT32),
//!     Properties::new(8, 16, 32, component::SINT8, component::SINT32),
//!     Properties::new(8, 16, 16, component::FLOAT16, component::FLOAT16),
//! ];
//! let mmas: Vec<_> = reported.iter().filter_map(|p| p.mma(16)).collect();
//! let c = compare(Engine::Xmx(Xmx::Xe2), &mmas);
//! assert_eq!((c.matched.len(), c.unexplained.len()), (4, 0));
//! // bf16, int4 and mixed signedness are in the hardware, not in the driver.
//! assert_eq!(c.hidden.len(), 5);
//! ```

use std::ffi::c_void;
use std::fmt;

use super::{Engine, Mma, Scope, Shape, f};
use crate::elem::Format;

/// `VkComponentTypeKHR` values.
pub mod component {
    pub const FLOAT16: i32 = 0;
    pub const FLOAT32: i32 = 1;
    pub const FLOAT64: i32 = 2;
    pub const SINT8: i32 = 3;
    pub const SINT16: i32 = 4;
    pub const SINT32: i32 = 5;
    pub const SINT64: i32 = 6;
    pub const UINT8: i32 = 7;
    pub const UINT16: i32 = 8;
    pub const UINT32: i32 = 9;
    pub const UINT64: i32 = 10;
    pub const BFLOAT16: i32 = 1_000_141_000;
    pub const FLOAT8_E4M3: i32 = 1_000_491_002;
    pub const FLOAT8_E5M2: i32 = 1_000_491_003;
    pub const FLOAT6_E2M3: i32 = 1_000_672_000;
    pub const FLOAT6_E3M2: i32 = 1_000_672_001;
    pub const FLOAT4_E2M1: i32 = 1_000_672_002;
}

/// `VkScopeKHR` values.
pub mod scope {
    pub const DEVICE: i32 = 1;
    pub const WORKGROUP: i32 = 2;
    pub const SUBGROUP: i32 = 3;
    pub const QUEUE_FAMILY: i32 = 5;
}

/// `VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR`.
pub const PROPERTIES_S_TYPE: i32 = 1_000_506_001;

/// The format of a component type. `UINT32` reads as [`Format::I32`]: a
/// wrapping 32-bit sum has the same bits either way. 16- and 64-bit
/// integers and `f64` have no format in [`elem`](crate::elem).
#[must_use]
pub const fn format(component: i32) -> Option<Format> {
    Some(match component {
        component::FLOAT16 => f::F16,
        component::FLOAT32 => f::F32,
        component::SINT8 => f::I8,
        component::UINT8 => f::U8,
        component::SINT32 | component::UINT32 => f::I32,
        component::BFLOAT16 => f::BF16,
        component::FLOAT8_E4M3 => f::E4M3,
        component::FLOAT8_E5M2 => f::E5M2,
        component::FLOAT6_E2M3 => f::E2M3,
        component::FLOAT6_E3M2 => f::E3M2,
        component::FLOAT4_E2M1 => f::E2M1,
        _ => return None,
    })
}

/// The component type for `x` as an A or B input; `I32` is `SINT32`.
#[must_use]
pub fn component(x: Format) -> Option<i32> {
    [
        component::FLOAT16,
        component::FLOAT32,
        component::SINT8,
        component::UINT8,
        component::SINT32,
        component::BFLOAT16,
        component::FLOAT8_E4M3,
        component::FLOAT8_E5M2,
        component::FLOAT6_E2M3,
        component::FLOAT6_E3M2,
        component::FLOAT4_E2M1,
    ]
    .into_iter()
    .find(|&c| format(c) == Some(x))
}

/// `VkCooperativeMatrixPropertiesKHR`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Properties {
    pub s_type: i32,
    pub p_next: *mut c_void,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub a: i32,
    pub b: i32,
    pub c: i32,
    pub result: i32,
    pub saturating_accumulation: u32,
    pub scope: i32,
}

// Nine 32-bit members after the header, padded to pointer alignment.
const _: () = assert!(
    size_of::<Properties>()
        == (crate::vulkan::HEADER + 36).next_multiple_of(align_of::<*const c_void>())
);

impl Properties {
    /// An entry for the driver to fill.
    #[must_use]
    pub const fn empty() -> Self {
        Self::new(0, 0, 0, 0, 0)
    }

    /// A subgroup entry with `a == b` and `c == result`, as most are.
    #[must_use]
    pub const fn new(m: u32, n: u32, k: u32, input: i32, acc: i32) -> Self {
        Self {
            s_type: PROPERTIES_S_TYPE,
            p_next: std::ptr::null_mut(),
            m,
            n,
            k,
            a: input,
            b: input,
            c: acc,
            result: acc,
            saturating_accumulation: 0,
            scope: scope::SUBGROUP,
        }
    }

    /// This entry as an [`Mma`] of a subgroup of `lanes` (the pipeline's
    /// subgroup size; Vulkan does not say). `None` for types without a
    /// format, a result type other than C's, saturating accumulation, or a
    /// device / queue family scope.
    #[must_use]
    pub fn mma(&self, lanes: u32) -> Option<Mma> {
        if self.c != self.result || self.saturating_accumulation != 0 {
            return None;
        }
        let scope = match self.scope {
            scope::SUBGROUP => Scope::Subgroup { lanes },
            scope::WORKGROUP => Scope::Cta { pair: false },
            _ => return None,
        };
        Some(Mma {
            a: format(self.a)?,
            b: format(self.b)?,
            acc: format(self.c)?,
            shape: Shape::new(self.m, self.n, self.k),
            scope,
            instr: "VK_KHR_cooperative_matrix",
        })
    }
}

/// A driver's cooperative matrices next to an engine's table ([`compare`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Comparison {
    pub engine: Engine,
    /// Each reported entry with the instruction it is made of.
    pub matched: Vec<(Mma, &'static Mma)>,
    /// Type combinations of the table the driver does not report at all.
    pub hidden: Vec<&'static Mma>,
    /// Reported entries no instruction of the table explains.
    pub unexplained: Vec<Mma>,
}

/// Sets `reported` (from [`Properties::mma`]) next to `engine`'s table. An
/// entry matches an instruction of the same formats whose tile it is made
/// of (a driver may tile several instructions into one cooperative matrix).
#[must_use]
pub fn compare(engine: Engine, reported: &[Mma]) -> Comparison {
    let mut matched = Vec::new();
    let mut unexplained = Vec::new();
    for r in reported {
        match engine
            .mmas()
            .find(|t| t.takes(r.a, r.b, r.acc) && r.shape.is_made_of(t.shape))
        {
            Some(t) => matched.push((*r, t)),
            None => unexplained.push(*r),
        }
    }
    let mut hidden: Vec<&'static Mma> = Vec::new();
    for t in engine.mmas() {
        let exposed = reported.iter().any(|r| r.takes(t.a, t.b, t.acc));
        if !exposed && !hidden.iter().any(|h| h.takes(t.a, t.b, t.acc)) {
            hidden.push(t);
        }
    }
    Comparison {
        engine,
        matched,
        hidden,
        unexplained,
    }
}

impl fmt::Display for Comparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let types = |m: &Mma| format!("{} x {} -> {}", m.a.name(), m.b.name(), m.acc.name());
        writeln!(
            f,
            "{}: {} reported cooperative matrices match its instructions",
            self.engine,
            self.matched.len()
        )?;
        for (r, t) in &self.matched {
            writeln!(f, "  {} {}: {}", types(r), r.shape, t.instr)?;
        }
        if !self.hidden.is_empty() {
            let names: Vec<String> = self.hidden.iter().map(|m| types(m)).collect();
            writeln!(
                f,
                "  in the hardware, not exposed by the driver: {}",
                names.join(", ")
            )?;
        }
        for r in &self.unexplained {
            writeln!(f, "  NOT IN THE TABLE: {} {}", types(r), r.shape)?;
        }
        Ok(())
    }
}

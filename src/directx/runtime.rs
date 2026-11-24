//! Who implements Direct3D for a process: Microsoft's runtime on Windows,
//! the Xbox GDK, or a community layer that translates it (DXVK and
//! vkd3d-proton on Vulkan, which is how Direct3D runs on Linux through Wine
//! and Proton, d3d8to9 on Direct3D 9).
//!
//! The layers answer `QueryInterface` for interfaces of their own; a host
//! asks with [`Runtime::interfaces`] to tell which one it got. The ids are
//! from their sources (DXVK `src/dxgi/dxgi_interfaces.h`,
//! `src/d3d11/d3d11_interfaces.h`, `src/d3d9/d3d9_interfaces.h`;
//! vkd3d-proton `include/vkd3d_device_vkd3d_ext.idl`), converted while
//! compiling.

use std::fmt;

use super::{Api, Guid, Os, Status, Support};

/// A Direct3D implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Runtime {
    /// The `d3d*.dll`s of Windows and the vendor drivers under them.
    Microsoft,
    /// WARP, Microsoft's software rasterizer (Direct3D 10 to 12).
    Warp,
    /// Direct3D 9 mapped onto Direct3D 12 by Microsoft (`D3D9On12`), which
    /// some drivers use instead of a native Direct3D 9 one.
    D3d9On12,
    /// The Xbox GDK: Direct3D 12 for the consoles (its own headers, under
    /// NDA, and no Rust target), and the PC side of the GDK, which is plain
    /// Windows.
    Xbox,
    /// DXVK: Direct3D 8 to 11 on Vulkan (zlib).
    Dxvk,
    /// vkd3d-proton: Direct3D 12 on Vulkan (LGPL-2.1).
    Vkd3dProton,
    /// Wine's own `wined3d`: Direct3D 1 to 11 on OpenGL or Vulkan.
    Wined3d,
    /// crosire's d3d8to9: a 32-bit `d3d8.dll` on Direct3D 9 (BSD-2-Clause).
    D3d8to9,
}

impl Runtime {
    pub const ALL: [Self; 8] = [
        Self::Microsoft,
        Self::Warp,
        Self::D3d9On12,
        Self::Xbox,
        Self::Dxvk,
        Self::Vkd3dProton,
        Self::Wined3d,
        Self::D3d8to9,
    ];

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Microsoft => "Microsoft",
            Self::Warp => "WARP",
            Self::D3d9On12 => "D3D9On12",
            Self::Xbox => "Xbox GDK",
            Self::Dxvk => "DXVK",
            Self::Vkd3dProton => "vkd3d-proton",
            Self::Wined3d => "wined3d",
            Self::D3d8to9 => "d3d8to9",
        }
    }

    /// The versions it implements.
    #[must_use]
    pub const fn apis(self) -> &'static [Api] {
        match self {
            Self::Microsoft => &Api::ALL,
            Self::Warp => &[Api::D3D10, Api::D3D11, Api::D3D12],
            Self::D3d9On12 => &[Api::D3D9],
            Self::Xbox | Self::Vkd3dProton => &[Api::D3D12],
            Self::Dxvk | Self::Wined3d => &[Api::D3D8, Api::D3D9, Api::D3D10, Api::D3D11],
            Self::D3d8to9 => &[Api::D3D8],
        }
    }

    /// What it runs on, for a layer.
    #[must_use]
    pub const fn over(self) -> Option<&'static str> {
        match self {
            Self::Microsoft | Self::Warp | Self::Xbox => None,
            Self::D3d9On12 => Some("Direct3D 12"),
            Self::Dxvk | Self::Vkd3dProton => Some("Vulkan"),
            Self::Wined3d => Some("OpenGL or Vulkan"),
            Self::D3d8to9 => Some("Direct3D 9"),
        }
    }

    /// Where it runs and whether numerobis was run against it. Linux means a
    /// Windows program under Wine or Proton, or a native build with
    /// dxvk-native.
    #[must_use]
    pub const fn support(self) -> Support {
        match self {
            Self::Microsoft => Support::WINDOWS_ONLY_TESTED,
            Self::Warp | Self::D3d9On12 | Self::D3d8to9 => Support::WINDOWS,
            // No OS this crate tells apart: the console is not Windows to
            // `cfg`, and nothing here ran there.
            Self::Xbox => Support::new(Status::Unavailable, Status::Unavailable),
            Self::Dxvk | Self::Vkd3dProton | Self::Wined3d => {
                Support::new(Status::Untested, Status::Untested)
            }
        }
    }

    /// Interfaces only this implementation answers, with the object to ask
    /// (`"IDXGIFactory"`, `"ID3D11Device"`, `"ID3D12Device"`,
    /// `"IDirect3D9"`, `"IDirect3DDevice9"`) and the interface's name.
    #[must_use]
    pub const fn interfaces(self) -> &'static [(&'static str, &'static str, Guid)] {
        match self {
            Self::Dxvk => DXVK,
            Self::Vkd3dProton => VKD3D_PROTON,
            Self::D3d9On12 => D3D9ON12,
            _ => &[],
        }
    }

    /// The runtime an object answering `iid` belongs to, if it is one of
    /// [`interfaces`](Self::interfaces).
    #[must_use]
    pub fn by_interface(iid: &Guid) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|r| r.interfaces().iter().any(|(_, _, g)| g == iid))
    }

    /// WARP by the adapter's ids (`0x1414:0x008c`, the Basic Render Driver).
    #[must_use]
    pub const fn is_warp(vendor_id: u32, device_id: u32) -> bool {
        vendor_id == 0x1414 && device_id == 0x008c
    }

    #[must_use]
    pub const fn available_on(self, os: Os) -> bool {
        self.support().available_on(os)
    }
}

impl fmt::Display for Runtime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.over() {
            Some(over) => write!(f, "{} (on {over})", self.name()),
            None => f.write_str(self.name()),
        }
    }
}

const DXVK: &[(&str, &str, Guid)] = &[
    (
        "IDXGIFactory",
        "IDXGIVkInteropFactory",
        Guid::parse("4c5e1b0d-b0c8-4131-bfd8-9b2476f7f408"),
    ),
    (
        "IDXGIAdapter",
        "IDXGIDXVKAdapter",
        Guid::parse("907bf281-ea3c-43b4-a8e4-9f231107b4ff"),
    ),
    (
        "ID3D11Device",
        "IDXGIDXVKDevice",
        Guid::parse("92a5d77b-b6e1-420a-b260-fdd701272827"),
    ),
    (
        "ID3D11Device",
        "ID3D11VkExtDevice",
        Guid::parse("8a6e3c42-f74c-45b7-8265-a231b677ca17"),
    ),
    (
        "IDirect3D9",
        "ID3D9VkInteropInterface",
        Guid::parse("3461a81b-ce41-485b-b6b5-fcf08ba6a6bd"),
    ),
    (
        "IDirect3DDevice9",
        "ID3D9VkInteropDevice",
        Guid::parse("2eaa4b89-0107-4bdb-87f7-0f541c493ce0"),
    ),
];

const VKD3D_PROTON: &[(&str, &str, Guid)] = &[
    (
        "ID3D12Device",
        "ID3D12DXVKInteropDevice",
        Guid::parse("39da4e09-bd1c-4198-9fae-86bbe3be41fd"),
    ),
    (
        "ID3D12Device",
        "ID3D12DeviceExt",
        Guid::parse("11ea7a1a-0f6a-49bf-b612-3e30f8e201dd"),
    ),
];

const D3D9ON12: &[(&str, &str, Guid)] = &[(
    "IDirect3DDevice9",
    "IDirect3DDevice9On12",
    super::generated::iid::IDirect3DDevice9On12,
)];

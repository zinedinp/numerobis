#!/usr/bin/env python3
"""Generates src/directx/generated.rs from the Windows SDK headers.

    python scripts/gen-directx.py                 # newest SDK under Windows Kits
    python scripts/gen-directx.py path/to/Include/10.0.26100.0
    python scripts/gen-directx.py ... --check     # fail if out of date

The directory is a Windows SDK `Include/<version>` (with `um/` and
`shared/`). From it come:

- every `CheckFeatureSupport` struct of D3D12 and D3D11 whose members are all
  4 bytes, with its `D3D12_FEATURE` / `D3D11_FEATURE` id and members in
  order (the hand-written routes name members, checked while compiling);
- the values of the enums those members have (tiers, shader models) and of
  `D3D_FEATURE_LEVEL`, `D3D12_COMMAND_QUEUE_PRIORITY`, ...;
- `D3DCAPS9` flattened to 4-byte words, `D3DCAPS8` as its prefix (the D3D8
  header is not in the SDK; its layout is D3DCAPS9 up to
  `PixelShader1xMaxValue`, which D3D8 calls `MaxPixelShaderValue`), and the
  `d3d9caps.h` bit constants.

Only the standard library is used; the output goes through rustfmt.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "src" / "directx" / "generated.rs"
KITS = Path(r"C:\Program Files (x86)\Windows Kits\10\Include")

# Members of these types are 4 bytes; any other type keeps a struct out.
WORD = {"BOOL", "UINT", "INT", "DWORD", "float", "D3DDEVTYPE", "DXGI_FORMAT"}

# d3d9caps.h prefixes whose constants are emitted.
CAPS9_PREFIXES = [
    "D3DCAPS_", "D3DCAPS2_", "D3DCAPS3_", "D3DCURSORCAPS_", "D3DDEVCAPS_",
    "D3DDEVCAPS2_", "D3DPMISCCAPS_", "D3DLINECAPS_", "D3DPRASTERCAPS_",
    "D3DPCMPCAPS_", "D3DPBLENDCAPS_", "D3DPSHADECAPS_", "D3DPTEXTURECAPS_",
    "D3DPTFILTERCAPS_", "D3DPTADDRESSCAPS_", "D3DSTENCILCAPS_", "D3DTEXOPCAPS_",
    "D3DFVFCAPS_", "D3DVTXPCAPS_", "D3DDTCAPS_", "D3DVS20CAPS_", "D3DPS20CAPS_",
    "D3DVS20_", "D3DPS20_", "D3DMIN30SHADERINSTRUCTIONS", "D3DMAX30SHADERINSTRUCTIONS",
]

# COM interfaces whose ids are emitted (`iid::*`), by header.
INTERFACES = [
    ("shared/dxgi.h", ["IDXGIFactory1", "IDXGIAdapter1"]),
    ("shared/dxgi1_4.h", ["IDXGIAdapter3"]),
    ("shared/dxgi1_6.h", ["IDXGIFactory6"]),
    ("um/d3d12.h", ["ID3D12Device", "ID3D12CommandQueue"]),
    ("um/d3d12sdklayers.h", ["ID3D12Debug", "ID3D12InfoQueue"]),
    ("um/d3d11.h", ["ID3D11Device"]),
    ("um/d3d11sdklayers.h", ["ID3D11InfoQueue"]),
    ("um/d3d10_1.h", ["ID3D10Device1"]),
    ("shared/d3d9.h", ["IDirect3D9", "IDirect3DDevice9"]),
    ("um/d3d9on12.h", ["IDirect3DDevice9On12"]),
]

# Extra enums wanted beyond those the feature structs use.
EXTRA_ENUMS = [
    "D3D_FEATURE_LEVEL", "D3D12_COMMAND_QUEUE_PRIORITY", "D3D12_COMMAND_LIST_TYPE",
    "D3D_SHADER_MODEL", "D3D_ROOT_SIGNATURE_VERSION",
]


def fail(msg):
    print(f"gen-directx: {msg}", file=sys.stderr)
    sys.exit(1)


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def evaluate(expr, known):
    """An enum value: a number, `-1`, `0x..`, `(1<<n)` or `( OTHER + 1 )`."""
    e = expr.strip()
    e = re.sub(r"\b(0x[0-9A-Fa-f]+|\d+)[uUlL]+\b", r"\1", e)
    word = r"(?<![0-9A-Za-z_])[A-Za-z_]\w*"
    for name in sorted(set(re.findall(word, e)), key=len, reverse=True):
        if name not in known:
            raise KeyError(name)
        e = re.sub(rf"(?<![0-9A-Za-z_]){name}\b", str(known[name]), e)
    if not re.fullmatch(r"[0-9xA-Fa-f()+\-<|\s]*", e):
        raise ValueError(expr)
    return int(eval(e, {"__builtins__": {}}))  # digits and operators only


class Header:
    def __init__(self, paths):
        self.text = "\n".join(strip_comments(p.read_text(encoding="utf-8", errors="replace")) for p in paths)
        self.enums = {}  # enum name -> [(member, value)]
        self.values = {}  # member -> value
        for m in re.finditer(r"typedef\s+enum\s+(\w+)\s*\{(.*?)\}\s*(\w+)\s*;", self.text, re.S):
            members = []
            for item in m.group(2).split(","):
                item = item.strip()
                if not item:
                    continue
                if "=" in item:
                    name, expr = (s.strip() for s in item.split("=", 1))
                    try:
                        value = evaluate(expr, self.values)
                    except (KeyError, ValueError, SyntaxError):
                        members = None
                        break
                else:
                    name = item
                    value = members[-1][1] + 1 if members else 0
                self.values[name] = value
                members.append((name, value))
            if members is not None:
                self.enums[m.group(3)] = members

    def structs(self, prefix):
        """(name, [(type, member)]) of each `typedef struct <prefix>...`."""
        out = []
        for m in re.finditer(rf"typedef\s+struct\s+({prefix}\w*)\s*\{{(.*?)\}}\s*(\w+)\s*;", self.text, re.S):
            fields = []
            for decl in m.group(2).split(";"):
                decl = re.sub(r"\b_\w+(\([^)]*\))?", " ", decl).strip()
                if not decl:
                    continue
                tokens = decl.replace(",", " , ").split()
                kind, rest = tokens[0], " ".join(tokens[1:])
                if kind == "const":
                    kind, rest = tokens[1], " ".join(tokens[2:])
                for member in (s.strip() for s in rest.split(",")):
                    fields.append((kind, member))
            out.append((m.group(3), fields))
        return out


def camel(name):
    return "".join(p[:1] + p[1:].lower() for p in name.split("_") if p)


def u32(value):
    return value & 0xFFFF_FFFF


def hexlit(value):
    """`0x10`, `0x0010_0000`: grouped by four digits past four (clippy)."""
    if value <= 0xFFFF:
        return f"{value:#x}"
    digits = f"{value:x}"
    digits = digits.zfill(-(-len(digits) // 4) * 4)
    return "0x" + "_".join(digits[i:i + 4] for i in range(0, len(digits), 4))


def feature_structs(h, api):
    """The 4-byte-only CheckFeatureSupport structs of `api` (D3D12 or D3D11)."""
    ids = dict(h.enums[f"{api}_FEATURE"])
    found = []
    for name, fields in h.structs(f"{api}_FEATURE_DATA_"):
        short = name[len(f"{api}_FEATURE_DATA_"):]
        feature = f"{api}_FEATURE_{short}"
        if feature not in ids:
            continue
        if not fields or any(
            k not in WORD and k not in h.enums or "*" in f or "[" in f or "*" in k
            for k, f in fields
        ):
            continue
        variant = camel(short[len(api) + 1:] if short.startswith(api + "_") else short)
        found.append((variant, name, feature, ids[feature], fields))
    found.sort(key=lambda s: s[3])
    return found


def emit_features(api, rust, structs, h):
    lines = [
        f"/// A `{'ID3D12Device' if api == 'D3D12' else 'ID3D11Device'}::CheckFeatureSupport` struct whose",
        "/// members are all 4 bytes (`BOOL`, `UINT`, enums), in declaration order.",
        "#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]",
        "#[non_exhaustive]",
        f"pub enum {rust} {{",
    ]
    for variant, name, feature, value, _ in structs:
        lines.append(f"    /// `{name}` (`{feature}` = {value}).")
        lines.append(f"    {variant},")
    lines.append("}")
    lines.append("")
    lines.append(f"impl {rust} {{")
    lines.append("    /// Every struct, by feature id.")
    lines.append("    pub const ALL: &[Self] = &[" + ", ".join(f"Self::{s[0]}" for s in structs) + "];")
    lines.append("")
    lines.append(f"    /// The `{api}_FEATURE` value.")
    lines.append("    #[must_use]")
    lines.append("    pub const fn id(self) -> u32 {")
    lines.append("        match self {")
    for variant, _, _, value, _ in structs:
        lines.append(f"            Self::{variant} => {value},")
    lines.append("        }")
    lines.append("    }")
    lines.append("")
    lines.append("    /// The C struct name.")
    lines.append("    #[must_use]")
    lines.append("    pub const fn c_name(self) -> &'static str {")
    lines.append("        match self {")
    for variant, name, _, _, _ in structs:
        lines.append(f'            Self::{variant} => "{name}",')
    lines.append("        }")
    lines.append("    }")
    lines.append("")
    lines.append("    /// Its members in order; member `i` is the `u32` at byte `4 * i`.")
    lines.append("    #[must_use]")
    lines.append("    pub const fn fields(self) -> &'static [&'static str] {")
    lines.append("        match self {")
    for variant, _, _, _, fields in structs:
        names = ", ".join(f'"{f}"' for _, f in fields)
        lines.append(f"            Self::{variant} => &[{names}],")
    lines.append("        }")
    lines.append("    }")
    lines.append("")
    lines.append("    /// The C type of each member, as [`fields`](Self::fields).")
    lines.append("    #[must_use]")
    lines.append("    pub const fn types(self) -> &'static [&'static str] {")
    lines.append("        match self {")
    for variant, _, _, _, fields in structs:
        kinds = ", ".join(f'"{k}"' for k, _ in fields)
        lines.append(f"            Self::{variant} => &[{kinds}],")
    lines.append("        }")
    lines.append("    }")
    lines.append("}")
    return lines


def emit_enums(names, h):
    lines = [
        "/// Enum values from the headers, under their C names. Negative values",
        "/// (`D3D12_TRI_STATE_UNKNOWN`) are their two's complement, as the `u32`",
        "/// a feature struct holds.",
        "pub mod value {",
    ]
    seen = set()
    for enum in names:
        if enum not in h.enums:
            fail(f"enum {enum} not in the headers")
        lines.append(f"    // {enum}.")
        for member, value in h.enums[enum]:
            if member in seen:
                continue
            seen.add(member)
            lines.append(f"    pub const {member}: u32 = {hexlit(u32(value))};")
    lines.append("}")
    return lines


def caps9(h):
    structs = dict(h.structs("_D3D"))
    caps = structs.get("D3DCAPS9")
    vs, ps = structs.get("D3DVSHADERCAPS2_0"), structs.get("D3DPSHADERCAPS2_0")
    if not caps or not vs or not ps:
        fail("D3DCAPS9 not found in d3d9caps.h")
    words = []
    for kind, member in caps:
        if kind == "D3DVSHADERCAPS2_0":
            words += [(k, f"{member}.{m}") for k, m in vs]
        elif kind == "D3DPSHADERCAPS2_0":
            words += [(k, f"{member}.{m}") for k, m in ps]
        elif kind in WORD:
            words.append((kind, member))
        else:
            fail(f"D3DCAPS9 member {member} has type {kind}")
    names = [m for _, m in words]
    if "PixelShader1xMaxValue" not in names:
        fail("D3DCAPS9 lacks PixelShader1xMaxValue")
    return words, names.index("PixelShader1xMaxValue") + 1


def snake(name):
    name = name.replace(".", "_").replace("RTs", "Rts")
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
    return s.lower()


def emit_caps9(words, caps8_len, defines):
    lines = [
        "/// `D3DCAPS9` (`IDirect3D9::GetDeviceCaps`), flattened: `VS20Caps` and",
        "/// `PS20Caps` are their members in order. Every member is 4 bytes; the",
        "/// `float`s are kept as their bits ([`f32::from_bits`]).",
        "#[repr(C)]",
        "#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]",
        "pub struct Caps9 {",
    ]
    for kind, member in words:
        note = " (`float` bits)" if kind == "float" else ""
        lines.append(f"    /// `{member}`{note}.")
        lines.append(f"    pub {snake(member)}: u32,")
    lines.append("}")
    lines.append("")
    lines.append(f"/// Words in [`Caps9`]; `D3DCAPS9` is {4 * len(words)} bytes.")
    lines.append(f"pub const CAPS9_WORDS: usize = {len(words)};")
    lines.append(f"/// `D3DCAPS8` is the first {caps8_len} words of `D3DCAPS9` ({4 * caps8_len} bytes).")
    lines.append(f"pub const CAPS8_WORDS: usize = {caps8_len};")
    lines.append("/// The C member names of [`Caps9`], in order.")
    lines.append("pub const CAPS9_MEMBERS: [&str; CAPS9_WORDS] = [" + ", ".join(f'"{m}"' for _, m in words) + "];")
    lines.append("/// Which of them are `float`s.")
    lines.append("pub const CAPS9_FLOATS: &[&str] = &[" + ", ".join(f'"{m}"' for k, m in words if k == "float") + "];")
    lines.append("")
    lines.append("/// The `d3d9caps.h` bit constants (the D3D8 ones are the same bits).")
    lines.append("pub mod caps9 {")
    for name, value in defines:
        lines.append(f"    pub const {name}: u32 = {hexlit(value)};")
    lines.append("}")
    return lines


def caps9_defines(path):
    text = strip_comments(path.read_text(encoding="utf-8", errors="replace"))
    out, known = [], {}
    for m in re.finditer(r"^\s*#define\s+(D3D\w+)\s+(.+?)\s*$", text, re.M):
        name, expr = m.group(1), m.group(2)
        if not any(name.startswith(p) for p in CAPS9_PREFIXES):
            continue
        try:
            value = evaluate(expr, known)
        except (KeyError, ValueError, SyntaxError):
            continue
        known[name] = value
        out.append((name, u32(value)))
    return out


def interface_ids(include):
    out = []
    for header, names in INTERFACES:
        path = include / header
        if not path.exists():
            path = include / ("um" if header.startswith("shared") else "shared") / Path(header).name
        text = path.read_text(encoding="utf-8", errors="replace")
        for name in names:
            m = (re.search(rf'MIDL_INTERFACE\("([0-9A-Fa-f-]{{36}})"\)\s*{name}\s*:', text)
                 or re.search(rf'DECLSPEC_UUID\("([0-9A-Fa-f-]{{36}})"\)\s*{name}\s*;', text))
            if m:
                out.append((name, m.group(1).lower()))
                continue
            m = re.search(rf"DEFINE_GUID\(\s*IID_{name}\s*,([^)]*)\)", text)
            if not m:
                fail(f"no interface id for {name} in {header}")
            v = [int(x, 0) for x in m.group(1).replace(" ", "").split(",")]
            out.append((name, f"{v[0]:08x}-{v[1]:04x}-{v[2]:04x}-{v[3]:02x}{v[4]:02x}-" + "".join(f"{b:02x}" for b in v[5:])))
    return out


def emit_iids(ids):
    lines = ["/// COM interface ids from the headers, for `QueryInterface` and the", "/// `riid` of the create functions.", "#[allow(non_upper_case_globals, reason = \"the C interface names\")]", "pub mod iid {", "    use super::Guid;"]
    for name, guid in ids:
        lines.append(f"    /// `IID_{name}`.")
        lines.append(f'    pub const {name}: Guid = Guid::parse("{guid}");')
    lines.append("}")
    return lines


def generate(include):
    um, shared = include / "um", include / "shared"
    h12 = Header([um / "d3dcommon.h", um / "d3d12.h"])
    h11 = Header([um / "d3dcommon.h", um / "d3d11.h", um / "d3d11_4.h"])
    d3d12 = feature_structs(h12, "D3D12")
    d3d11 = feature_structs(h11, "D3D11")
    enums = []
    for h, structs in ((h12, d3d12), (h11, d3d11)):
        for *_, fields in structs:
            for kind, _ in fields:
                if kind in h.enums and kind not in enums and kind != "DXGI_FORMAT":
                    enums.append(kind)
    words, caps8_len = caps9(Header([shared / "d3d9types.h", shared / "d3d9caps.h"]))
    defines = caps9_defines(shared / "d3d9caps.h")

    merged = Header.__new__(Header)
    merged.enums = {**h11.enums, **h12.enums}
    out = [
        f"//! Generated by `scripts/gen-directx.py` from the Windows SDK headers ({include.name}).",
        "//! Do not edit; change the script and run it again.",
        "",
        "#![allow(",
        "    clippy::too_many_lines,",
        "    clippy::match_same_arms,",
        "    reason = \"tables: one arm per struct, as the headers have them\"",
        ")]",
        "",
        "use super::Guid;",
        "",
    ]
    out += emit_features("D3D12", "D3d12Feature", d3d12, h12)
    out.append("")
    out += emit_features("D3D11", "D3d11Feature", d3d11, h11)
    out.append("")
    out += emit_enums(EXTRA_ENUMS + [e for e in enums if e not in EXTRA_ENUMS], merged)
    out.append("")
    out += emit_caps9(words, caps8_len, defines)
    out.append("")
    out += emit_iids(interface_ids(include))
    return "\n".join(out) + "\n"


def rustfmt(source):
    try:
        out = subprocess.run(
            ["rustfmt", "--edition", "2024", "--emit", "stdout"],
            input=source.encode(), capture_output=True, check=True,
        )
    except FileNotFoundError:
        fail("rustfmt not found (rustup component add rustfmt)")
    except subprocess.CalledProcessError as e:
        fail(f"rustfmt failed:\n{e.stderr.decode()}")
    return out.stdout.decode().replace("\r\n", "\n")


def newest_sdk():
    if not KITS.is_dir():
        fail(f"no Windows SDK at {KITS}; pass its Include/<version> directory")
    versions = sorted((p for p in KITS.iterdir() if (p / "um" / "d3d12.h").exists()),
                      key=lambda p: [int(x) for x in p.name.split(".") if x.isdigit()])
    if not versions:
        fail(f"no SDK with um/d3d12.h under {KITS}")
    return versions[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("include", type=Path, nargs="?", help="Windows SDK Include/<version>")
    ap.add_argument("--check", action="store_true", help="fail if the file is out of date")
    args = ap.parse_args()
    include = args.include or newest_sdk()
    source = rustfmt(generate(include))
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != source:
            fail(f"{OUT.relative_to(ROOT)} is out of date; run without --check")
        print(f"{OUT.relative_to(ROOT)} is up to date (SDK {include.name})")
        return
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(source, encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)} (SDK {include.name})")


if __name__ == "__main__":
    main()

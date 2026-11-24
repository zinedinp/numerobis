#!/usr/bin/env python3
"""Generates src/vulkan/features/generated.rs from the Vulkan registry.

    python scripts/gen-vulkan.py path/to/vk.xml          # write
    python scripts/gen-vulkan.py path/to/vk.xml --check  # fail if out of date

vk.xml is `xml/vk.xml` of KhronosGroup/Vulkan-Docs (or `registry/vk.xml` of
Vulkan-Headers). From it come, for the structs listed in CHAINS: their
`sType`, members in declaration order, which version struct absorbed them and
when, the extension each core bit came from, and per version the extensions
it promoted. The script also checks that every `c"VK_..."` name under
src/vulkan is an extension the registry knows.

Only the standard library is used; the output goes through rustfmt.
"""

import argparse
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "src" / "vulkan" / "features" / "generated.rs"
SCAN = ROOT / "src" / "vulkan"

VERSIONS = ["1_0", "1_1", "1_2", "1_3", "1_4"]

# (variant, struct). Core10 stands for VkPhysicalDeviceFeatures2 (its sType)
# around VkPhysicalDeviceFeatures (its members).
CHAINS = [
    ("Core10", "VkPhysicalDeviceFeatures"),
    ("Vulkan11", "VkPhysicalDeviceVulkan11Features"),
    ("Vulkan12", "VkPhysicalDeviceVulkan12Features"),
    ("Vulkan13", "VkPhysicalDeviceVulkan13Features"),
    ("Vulkan14", "VkPhysicalDeviceVulkan14Features"),
    ("Storage16Bit", "VkPhysicalDevice16BitStorageFeatures"),
    ("Multiview", "VkPhysicalDeviceMultiviewFeatures"),
    ("VariablePointers", "VkPhysicalDeviceVariablePointersFeatures"),
    ("ProtectedMemory", "VkPhysicalDeviceProtectedMemoryFeatures"),
    ("SamplerYcbcrConversion", "VkPhysicalDeviceSamplerYcbcrConversionFeatures"),
    ("ShaderDrawParameters", "VkPhysicalDeviceShaderDrawParametersFeatures"),
    ("Storage8Bit", "VkPhysicalDevice8BitStorageFeatures"),
    ("Float16Int8", "VkPhysicalDeviceShaderFloat16Int8Features"),
    ("BufferDeviceAddress", "VkPhysicalDeviceBufferDeviceAddressFeatures"),
    ("DescriptorIndexing", "VkPhysicalDeviceDescriptorIndexingFeatures"),
    ("ScalarBlockLayout", "VkPhysicalDeviceScalarBlockLayoutFeatures"),
    ("TimelineSemaphore", "VkPhysicalDeviceTimelineSemaphoreFeatures"),
    ("HostQueryReset", "VkPhysicalDeviceHostQueryResetFeatures"),
    ("Synchronization2", "VkPhysicalDeviceSynchronization2Features"),
    ("SubgroupSizeControl", "VkPhysicalDeviceSubgroupSizeControlFeatures"),
    ("DemoteToHelperInvocation", "VkPhysicalDeviceShaderDemoteToHelperInvocationFeatures"),
    ("IntegerDotProduct", "VkPhysicalDeviceShaderIntegerDotProductFeatures"),
    ("DynamicRendering", "VkPhysicalDeviceDynamicRenderingFeatures"),
    ("IndexTypeUint8", "VkPhysicalDeviceIndexTypeUint8Features"),
    ("MeshShader", "VkPhysicalDeviceMeshShaderFeaturesEXT"),
    ("MemoryPriority", "VkPhysicalDeviceMemoryPriorityFeaturesEXT"),
    ("PageableDeviceLocalMemory", "VkPhysicalDevicePageableDeviceLocalMemoryFeaturesEXT"),
    ("FifoLatestReady", "VkPhysicalDevicePresentModeFifoLatestReadyFeaturesKHR"),
    ("PortabilitySubset", "VkPhysicalDevicePortabilitySubsetFeaturesKHR"),
    ("VulkanMemoryModel", "VkPhysicalDeviceVulkanMemoryModelFeatures"),
    ("TextureCompressionAstcHdr", "VkPhysicalDeviceTextureCompressionASTCHDRFeatures"),
    ("Maintenance4", "VkPhysicalDeviceMaintenance4Features"),
    ("Maintenance5", "VkPhysicalDeviceMaintenance5Features"),
    ("Maintenance6", "VkPhysicalDeviceMaintenance6Features"),
    ("HostImageCopy", "VkPhysicalDeviceHostImageCopyFeatures"),
    ("VertexAttributeDivisor", "VkPhysicalDeviceVertexAttributeDivisorFeatures"),
    ("CooperativeMatrix", "VkPhysicalDeviceCooperativeMatrixFeaturesKHR"),
    ("ShaderBfloat16", "VkPhysicalDeviceShaderBfloat16FeaturesKHR"),
    ("ShaderFloat8", "VkPhysicalDeviceShaderFloat8FeaturesEXT"),
    ("Robustness2", "VkPhysicalDeviceRobustness2FeaturesKHR"),
    ("DescriptorBuffer", "VkPhysicalDeviceDescriptorBufferFeaturesEXT"),
    ("GraphicsPipelineLibrary", "VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT"),
    ("ShaderObject", "VkPhysicalDeviceShaderObjectFeaturesEXT"),
    ("ExtendedDynamicState", "VkPhysicalDeviceExtendedDynamicStateFeaturesEXT"),
    ("ExtendedDynamicState2", "VkPhysicalDeviceExtendedDynamicState2FeaturesEXT"),
    ("PresentId", "VkPhysicalDevicePresentIdFeaturesKHR"),
    ("PresentId2", "VkPhysicalDevicePresentId2FeaturesKHR"),
    ("PresentWait", "VkPhysicalDevicePresentWaitFeaturesKHR"),
    ("PresentWait2", "VkPhysicalDevicePresentWait2FeaturesKHR"),
]

# Property and query structs the probe reads: their sType values.
PROPERTIES = [
    ("PROPERTIES_2", "VkPhysicalDeviceProperties2"),
    ("SUBGROUP_PROPERTIES", "VkPhysicalDeviceSubgroupProperties"),
    ("DRIVER_PROPERTIES", "VkPhysicalDeviceDriverProperties"),
    ("PCI_BUS_INFO_PROPERTIES", "VkPhysicalDevicePCIBusInfoPropertiesEXT"),
    ("MEMORY_PROPERTIES_2", "VkPhysicalDeviceMemoryProperties2"),
    ("MEMORY_BUDGET_PROPERTIES", "VkPhysicalDeviceMemoryBudgetPropertiesEXT"),
]

# Core bits whose extension had no feature struct (being offered was
# support), so the registry can't tie them to it by member name.
FROM_OVERRIDES = {
    "shaderDrawParameters": "VK_KHR_shader_draw_parameters",
    "samplerMirrorClampToEdge": "VK_KHR_sampler_mirror_clamp_to_edge",
    "drawIndirectCount": "VK_KHR_draw_indirect_count",
    "samplerFilterMinmax": "VK_EXT_sampler_filter_minmax",
    "shaderOutputViewportIndex": "VK_EXT_shader_viewport_index_layer",
    "shaderOutputLayer": "VK_EXT_shader_viewport_index_layer",
    "pushDescriptor": "VK_KHR_push_descriptor",
}


def fail(msg):
    sys.exit(f"gen-vulkan: {msg}")


def for_vulkan(el, attr="api"):
    api = el.get(attr)
    return api is None or "vulkan" in api.split(",")


def text_of(member, tag):
    node = member.find(tag)
    return None if node is None else node.text


class Registry:
    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.header = None
        for t in root.iter("type"):
            if t.get("category") == "define" and for_vulkan(t):
                name = t.find("name")
                if name is not None and name.text == "VK_HEADER_VERSION":
                    self.header = name.tail.strip()
        self.structs = {}
        self.aliases = {}
        for t in root.find("types"):
            if t.get("category") != "struct" or not for_vulkan(t):
                continue
            if t.get("alias"):
                self.aliases[t.get("name")] = t.get("alias")
            else:
                self.structs[t.get("name")] = t
        self.s_types = {}
        self.s_type_aliases = {}
        for enums in root.findall("enums"):
            if enums.get("name") == "VkStructureType":
                for e in enums.findall("enum"):
                    self._enum(e, None, core=True)
        self.versions = {}  # "1_1" -> set of struct names it introduced
        for feat in root.findall("feature"):
            # VK_VERSION_1_x, and since 1.4.3xx also VK_{BASE,COMPUTE,GRAPHICS}_VERSION_1_x.
            if "VERSION_" not in feat.get("name", "") or not for_vulkan(feat):
                continue
            types = self.versions.setdefault(feat.get("number").replace(".", "_"), set())
            for req in feat.findall("require"):
                if not for_vulkan(req):
                    continue
                for e in req.findall("enum"):
                    self._enum(e, None)
                for t in req.findall("type"):
                    types.add(t.get("name"))
        self.extensions = {}
        for ext in root.find("extensions"):
            supported = ext.get("supported", "").split(",")
            number = int(ext.get("number"))
            types = set()
            for req in ext.findall("require"):
                if not for_vulkan(req):
                    continue
                for e in req.findall("enum"):
                    self._enum(e, number)
                for t in req.findall("type"):
                    types.add(t.get("name"))
            self.extensions[ext.get("name")] = {
                "vulkan": "vulkan" in supported,
                "type": ext.get("type"),
                "promotedto": ext.get("promotedto"),
                "depends": ext.get("depends"),
                "types": {self.resolve(t) for t in types},
            }

    def _enum(self, e, extnumber, core=False):
        if not core and e.get("extends") != "VkStructureType" or not for_vulkan(e):
            return
        name = e.get("name")
        if e.get("alias"):
            self.s_type_aliases[name] = e.get("alias")
        elif e.get("value") is not None:
            self.s_types[name] = int(e.get("value"), 0)
        elif e.get("offset") is not None:
            number = int(e.get("extnumber", extnumber))
            value = 1_000_000_000 + (number - 1) * 1000 + int(e.get("offset"))
            self.s_types[name] = -value if e.get("dir") == "-" else value

    def resolve(self, struct):
        while struct in self.aliases:
            struct = self.aliases[struct]
        return struct

    def s_type(self, struct):
        node = self.structs[self.resolve(struct)].find("member")
        name = node.get("values")
        while name in self.s_type_aliases:
            name = self.s_type_aliases[name]
        return self.s_types[name]

    def members(self, struct):
        out = []
        for m in self.structs[self.resolve(struct)].findall("member"):
            if not for_vulkan(m):
                continue
            name = text_of(m, "name")
            if name in ("sType", "pNext"):
                continue
            if text_of(m, "type") != "VkBool32":
                fail(f"{struct}.{name} is not a VkBool32")
            out.append(name)
        return out

    def introduced_in(self, struct):
        struct = self.resolve(struct)
        for version, types in self.versions.items():
            if struct in types or any(self.resolve(t) == struct for t in types):
                return version
        return None

    def promoted(self, version):
        return sorted(
            (n for n, e in self.extensions.items()
             if e["vulkan"] and e["promotedto"] == f"VK_VERSION_{version}"),
            key=ext_order,
        )

    def feature_structs(self, ext):
        return [
            t for t in self.extensions[ext]["types"]
            if t in self.structs
            and "VkPhysicalDeviceFeatures2" in (self.structs[t].get("structextends") or "")
        ]

    def bringers(self, struct):
        """The extensions that brought `struct`, in order."""
        struct = self.resolve(struct)
        return sorted(
            (n for n, e in self.extensions.items() if e["vulkan"] and struct in e["types"]),
            key=ext_order,
        )


def parse_depends(text):
    """`a+b`, `a,b`, parentheses; `+` and `,` never mix unparenthesised."""
    tokens = re.findall(r"[A-Za-z0-9_:]+|[+,()]", text)
    pos = 0

    def atom():
        nonlocal pos
        t = tokens[pos]
        pos += 1
        if t == "(":
            e = expr()
            if tokens[pos] != ")":
                fail(f"depends {text!r}: missing )")
            pos += 1
            return e
        m = re.fullmatch(r"VK_VERSION_(\d+)_(\d+)", t)
        if m:
            return ("Version", (int(m.group(1)), int(m.group(2))))
        if t.startswith("VKSC_"):
            return ("Never",)
        if "::" in t:
            fail(f"depends {text!r}: feature dependency {t}")
        return ("Extension", t)

    def expr():
        nonlocal pos
        items, op = [atom()], None
        while pos < len(tokens) and tokens[pos] in "+,":
            if op and tokens[pos] != op:
                fail(f"depends {text!r}: + and , mixed without parentheses")
            op = tokens[pos]
            pos += 1
            items.append(atom())
        if len(items) == 1:
            return items[0]
        return ("All" if op == "+" else "Any", items)

    e = expr()
    if pos != len(tokens):
        fail(f"depends {text!r}: trailing tokens")
    return e


def depends_rust(e):
    kind = e[0]
    if kind == "Version":
        major, minor = e[1]
        if major == 1 and minor <= 4:
            return f"Depends::Version(ApiVersion::V{major}_{minor})"
        return f"Depends::Version(ApiVersion::new({major}, {minor}))"
    if kind == "Never":
        return "Depends::Never"
    if kind == "Extension":
        return f'Depends::Extension(c"{e[1]}")'
    inner = ", ".join(depends_rust(x) for x in e[1])
    return f"Depends::{kind}(&[{inner}])"


def depends_names(e):
    if e[0] == "Extension":
        yield e[1]
    elif e[0] in ("All", "Any"):
        for x in e[1]:
            yield from depends_names(x)


def ext_order(name):
    vendor = name.split("_")[1]
    return ({"KHR": 0, "EXT": 1}.get(vendor, 2), name)


def grouped(pairs):
    """(value, key) pairs -> [(value, [keys])], first-seen order: one match arm per value."""
    out = {}
    for value, key in pairs:
        out.setdefault(value, []).append(key)
    return list(out.items())


def screaming(variant):
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", variant).upper() + "_FIELDS"


def version_tuple(v):
    major, minor = v.split("_")
    return f"ApiVersion::V{major}_{minor}"


def dotted(v):
    return v.replace("_", ".")


def generate(reg, used):
    variants = {v: s for v, s in CHAINS}
    version_chain = {v: CHAINS[i][0] for i, v in enumerate(VERSIONS)}
    chain_of_version = {c: v for v, c in version_chain.items()}
    members = {v: reg.members(s) for v, s in CHAINS}

    # When each version struct became chainable (Vulkan11 came with 1.2).
    since = {}
    for version, chain in version_chain.items():
        if chain == "Core10":
            continue
        at = reg.introduced_in(variants[chain])
        if at is None:
            fail(f"{variants[chain]} is in no VK_VERSION feature")
        since[chain] = at

    absorbed = {}
    for variant, struct in CHAINS[len(VERSIONS):]:
        at = reg.introduced_in(struct)
        if at is None or at == "1_0":
            continue
        by = version_chain[at]
        missing = [m for m in members[variant] if m not in members[by]]
        if missing:
            fail(f"{struct} is core {at} but {by} lacks {missing}")
        absorbed[variant] = (max(at, since[by]), by)

    lines = []
    w = lines.append
    w(f"//! Generated by `scripts/gen-vulkan.py` from `vk.xml` (header {reg.header}).")
    w("//! Do not edit; change the script or its tables and run it again.")
    w("")
    w("use std::ffi::CStr;")
    w("")
    w("use super::{Depends, Field, Version, e, f};")
    w("use crate::vulkan::ApiVersion;")
    w("")
    w("/// A feature struct that goes into a `pNext` chain.")
    w("#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]")
    w("#[non_exhaustive]")
    w("pub enum Chain {")
    for variant, struct in CHAINS:
        if variant == "Core10":
            doc = ("`VkPhysicalDeviceFeatures2`, whose `features` are the 1.0 bits. "
                   "Always the head of a chain.")
        elif variant in since:
            doc = f"`{struct}`, from {dotted(since[variant])}."
        else:
            core = reg.introduced_in(struct)
            exts = ", ".join(f"`{n}`" for n in reg.bringers(struct))
            if core:
                at, by = absorbed[variant]
                before = f", before that {exts}" if exts else ""
                doc = (f"`{struct}`: core {dotted(core)}{before}; "
                       f"part of `{by}` from {dotted(at)}.")
            else:
                doc = f"`{struct}` ({exts})."
        w(f"    /// {doc}")
        w(f"    {variant},")
    w("}")
    w("")
    w("impl Chain {")
    w("    /// Every struct these tables know.")
    w("    pub const ALL: &[Chain] = &[" + ", ".join(f"Self::{v}" for v, _ in CHAINS) + "];")
    w("")
    w("    /// The struct's `VkStructureType` value.")
    w("    #[must_use]")
    w("    pub const fn s_type(self) -> i32 {")
    w("        match self {")
    seen = {}
    for variant, struct in CHAINS:
        value = reg.s_type("VkPhysicalDeviceFeatures2" if variant == "Core10" else struct)
        if value in seen:
            fail(f"{variant} and {seen[value]} share sType {value}")
        seen[value] = variant
        w(f"            Self::{variant} => {value:_},")
    w("        }")
    w("    }")
    w("")
    w("    /// The C struct name, for logs.")
    w("    #[must_use]")
    w("    pub const fn name(self) -> &'static str {")
    w("        match self {")
    for variant, struct in CHAINS:
        w(f'            Self::{variant} => "{struct}",')
    w("        }")
    w("    }")
    w("")
    w("    /// The struct's members after `sType` and `pNext`, in declaration order.")
    w("    #[must_use]")
    w("    pub const fn fields(self) -> &'static [Field] {")
    w("        match self {")
    for variant, _ in CHAINS:
        table = f"CORE_{chain_of_version[variant]}" if variant in chain_of_version else screaming(variant)
        w(f"            Self::{variant} => {table},")
    w("        }")
    w("    }")
    w("")
    w("    /// For a `VkPhysicalDeviceVulkan1xFeatures` struct: the version a device")
    w("    /// needs to take it in a chain (`Vulkan11` came with 1.2).")
    w("    #[must_use]")
    w("    pub const fn version_struct_since(self) -> Option<ApiVersion> {")
    w("        match self {")
    for value, chains in grouped((f"Some({version_tuple(at)})", c) for c, at in since.items()):
        w(f"            {' | '.join(f'Self::{c}' for c in chains)} => {value},")
    w("            _ => None,")
    w("        }")
    w("    }")
    w("")
    w("    /// For a single struct a version struct took over: from which version on,")
    w("    /// and which struct. From then on the two must not share a chain")
    w("    /// (`VUID-VkDeviceCreateInfo-pNext-02829`, `-02830`, `-06532`, ...), and the")
    w("    /// bit is read and set through the version struct, by the same name.")
    w("    #[must_use]")
    w("    pub const fn absorbed_by(self) -> Option<(ApiVersion, Chain)> {")
    w("        match self {")
    for value, chains in grouped(
        (f"Some(({version_tuple(at)}, Self::{by}))", v) for v, (at, by) in absorbed.items()
    ):
        w(f"            {' | '.join(f'Self::{c}' for c in chains)} => {value},")
    w("            _ => None,")
    w("        }")
    w("    }")
    w("}")
    w("")
    w("/// Every core version these tables know, oldest first.")
    w(f"pub const VERSIONS: [Version; {len(VERSIONS)}] = [")
    for v in VERSIONS:
        promoted = f"PROMOTED_{v}" if v != "1_0" else "&[]"
        w(f"    Version {{ version: {version_tuple(v)}, chain: Chain::{version_chain[v]}, "
          f"fields: CORE_{v}, promoted: {promoted} }},")
    w("];")

    for v in VERSIONS:
        chain = version_chain[v]
        struct = "VkPhysicalDeviceFeatures" if v == "1_0" else variants[chain]
        promoted = reg.promoted(v) if v != "1_0" else []
        w("")
        w(f"/// `{struct}`.")
        w(f"pub const CORE_{v}: &[Field] = &[")
        for member in members[chain]:
            source = None
            if v != "1_0":
                found = [n for n in promoted
                         if any(member in reg.members(s) for s in reg.feature_structs(n))]
                if member in FROM_OVERRIDES:
                    source = FROM_OVERRIDES[member]
                    if source not in promoted:
                        fail(f"override {member}: {source} is not promoted to {dotted(v)}")
                elif len(found) == 1:
                    source = found[0]
                elif len(found) > 1:
                    fail(f"{member}: from {found}? add an override")
            w(f'    e("{member}", c"{source}"),' if source else f'    f("{member}"),')
        w("];")

    for variant, struct in CHAINS[len(VERSIONS):]:
        w("")
        w(f"const {screaming(variant)}: &[Field] = &[")
        for member in members[variant]:
            w(f'    f("{member}"),')
        w("];")

    for v in VERSIONS[1:]:
        w("")
        w(f"/// The extensions Vulkan {dotted(v)} made core.")
        w(f"const PROMOTED_{v}: &[&CStr] = &[")
        for name in reg.promoted(v):
            w(f'    c"{name}",')
        w("];")

    # The dependencies of every extension the hand-written code names, and
    # of theirs.
    todo, seen = sorted(used), set()
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        dep = reg.extensions[name]["depends"]
        if dep:
            todo.extend(n for n in depends_names(parse_depends(dep)) if n not in seen)
    w("")
    w("/// The `depends` expression of every extension the routes name, and of")
    w("/// their dependencies, as `vk.xml` states it.")
    w("pub(super) const DEPENDS: &[(&CStr, Depends)] = &[")
    for name in sorted(seen):
        dep = reg.extensions[name]["depends"]
        w(f'    (c"{name}", {depends_rust(parse_depends(dep)) if dep else "Depends::All(&[])"}),')
    w("];")
    w("")
    w("/// `sType` values of the property structs the probe reads.")
    w("pub(crate) mod s_type {")
    for const, struct in PROPERTIES:
        w(f"    /// `{struct}`.")
        w(f"    pub const {const}: i32 = {reg.s_type(struct):_};")
    w("}")
    w("")
    w("/// Every instance extension: enabled at `vkCreateInstance`, never at")
    w("/// `vkCreateDevice`.")
    w("pub(super) const INSTANCE_EXTENSIONS: &[&CStr] = &[")
    for name in sorted((n for n, e in reg.extensions.items()
                        if e["vulkan"] and e["type"] == "instance"), key=ext_order):
        w(f'    c"{name}",')
    w("];")
    return "\n".join(lines) + "\n"


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


def check_names(reg):
    """Every extension name the hand-written code spells must exist; returns them."""
    bad, used = [], set()
    for path in sorted(SCAN.rglob("*.rs")):
        if path == OUT:
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for name in re.findall(r'c"(VK_[A-Za-z0-9_]+)"', line):
                if name not in reg.extensions:
                    bad.append(f"{path.relative_to(ROOT)}:{n}: unknown extension {name}")
                else:
                    used.add(name)
    if bad:
        fail("\n".join(bad))
    return used


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("vk_xml", type=Path)
    ap.add_argument("--check", action="store_true", help="fail if the file is out of date")
    args = ap.parse_args()
    reg = Registry(args.vk_xml)
    used = check_names(reg)
    source = rustfmt(generate(reg, used))
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != source:
            fail(f"{OUT.relative_to(ROOT)} is out of date; run without --check")
        print(f"{OUT.relative_to(ROOT)} is up to date (vk.xml header {reg.header})")
        return
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(source, encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)} (vk.xml header {reg.header})")


if __name__ == "__main__":
    main()

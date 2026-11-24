//! Lists every GPU's PCIe link, BAR aperture and resizable-BAR state.
//!
//! `cargo run --example bars`. Windows: tested; Linux: untested
//! ([`numerobis::pci::bar::SUPPORT`]).

fn main() {
    println!(
        "pci::bar on {} ({})",
        numerobis::platform::Os::HOST,
        numerobis::pci::bar::SUPPORT
    );
    match numerobis::pci::bar::scan() {
        Ok(gpus) if gpus.is_empty() => println!("no display-class PCI devices"),
        Ok(gpus) => {
            for gpu in gpus {
                println!("{gpu}");
                for bar in &gpu.bars {
                    println!(
                        "    BAR{} {:>10} at {:#014x}{}{}",
                        bar.index.map_or_else(|| "?".into(), |i| i.to_string()),
                        numerobis::vulkan::memory::bytes(bar.size),
                        bar.base,
                        if bar.above_4g() { ", above 4 GiB" } else { "" },
                        if bar.resizable.is_empty() {
                            String::new()
                        } else {
                            format!(", resizable to {} sizes", bar.resizable.len())
                        },
                    );
                }
            }
        }
        Err(e) => println!("cannot read the PCI devices: {e}"),
    }
}

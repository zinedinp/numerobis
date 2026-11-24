//! Lists devices, engines, tiers and access layers.
//!
//! `cargo run --example probe` (add `--features mock` to include the host mock).

use numerobis::Device;

fn main() {
    #[allow(unused_mut)]
    let mut devices = Device::all();
    #[cfg(all(feature = "nvidia", target_os = "linux", target_pointer_width = "64"))]
    for r in numerobis::nvidia::backend::probe() {
        if let Err(e) = r {
            println!("nvidia: {e}");
        }
    }
    #[cfg(all(feature = "amd", target_os = "linux", target_pointer_width = "64"))]
    for r in numerobis::amd::backend::probe() {
        if let Err(e) = r {
            println!("amd: {e}");
        }
    }
    #[cfg(all(feature = "tt", target_os = "linux", target_pointer_width = "64"))]
    for r in numerobis::tt::backend::probe() {
        if let Err(e) = r {
            println!("tt: {e}");
        }
    }
    #[cfg(all(
        any(feature = "nvidia", feature = "amd", feature = "tt"),
        target_os = "linux",
        not(target_pointer_width = "64")
    ))]
    println!("nvidia, amd, tt: the raw backends need a 64-bit target, not built");
    #[cfg(feature = "mock")]
    devices.push(Device::mock());
    if devices.is_empty() {
        println!("no devices (enable a raw backend feature such as nvidia, or mock)");
    }
    for (i, d) in devices.iter().enumerate() {
        let f = d.features();
        println!("[{i}] {} ({:?}, driver {})", f.name, f.class, f.driver);
        for (e, a) in &f.engines {
            println!("    engine {e:?}: {a:?}");
        }
        let tiers: Vec<String> = f.tiers.iter().map(|t| t.name()).collect();
        println!("    tiers: {}", tiers.join(" < "));
    }
}

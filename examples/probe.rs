//! Lists devices, engines, tiers and access layers.
//!
//! `cargo run --example probe` (add `--features mock` to include the host mock).

use numerobis::Device;

fn main() {
    #[allow(unused_mut)]
    let mut devices = Device::all();
    #[cfg(feature = "nvidia")]
    for r in numerobis::nvidia::backend::probe() {
        if let Err(e) = r {
            println!("nvidia: {e}");
        }
    }
    #[cfg(feature = "amd")]
    for r in numerobis::amd::backend::probe() {
        if let Err(e) = r {
            println!("amd: {e}");
        }
    }
    #[cfg(feature = "tt")]
    for r in numerobis::tt::backend::probe() {
        if let Err(e) = r {
            println!("tt: {e}");
        }
    }
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

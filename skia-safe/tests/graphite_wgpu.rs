//! Smoke test for the experimental wgpu-backed proc table.
//!
//! Installs the wgpu proc table and then drives Graphite's standard Dawn
//! setup. Today this only exercises instance/adapter/device/queue creation;
//! any further calls (Context construction, Recorder, drawing) will likely
//! hit an unimplemented stub and abort.
//!
//! Run with:
//! `cargo test -p skia-safe --features graphite,wgpu --no-default-features --test graphite_wgpu`

#![cfg(all(feature = "graphite", feature = "wgpu"))]

use skia_safe::graphite::{
    dawn::{install_proc_table, DawnDevice},
    wgpu_backend::wgpu_proc_table,
};

#[test]
fn dawn_setup_through_wgpu_proc_table() {
    // Install the wgpu proc table before any wgpu* call. First-install-wins
    // semantics mean this runs before DawnDevice::new's internal default
    // install, so the wgpu thunks are used.
    static PROCS: std::sync::OnceLock<skia_bindings::DawnProcTable> = std::sync::OnceLock::new();
    let procs = PROCS.get_or_init(wgpu_proc_table);
    unsafe { install_proc_table(procs) };

    let Some(dawn) = DawnDevice::new() else {
        eprintln!("Skipping: no wgpu-compatible GPU available");
        return;
    };

    // We got an instance/adapter/device/queue back through the wgpu thunks.
    // That's a meaningful milestone: the dispatcher → wgpu Rust crate path
    // actually works end-to-end for the lifecycle subset.
    eprintln!(
        "DawnDevice::new() succeeded through the wgpu proc table; instance={:p}, device={:p}, queue={:p}",
        dawn.instance(),
        dawn.device(),
        dawn.queue(),
    );
}

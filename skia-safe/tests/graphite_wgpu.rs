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

use skia_safe::{
    graphite::{
        self,
        dawn::{install_proc_table, DawnDevice},
        wgpu_backend::wgpu_proc_table,
    },
    AlphaType, Color, ColorType, IRect, ImageInfo,
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

    let backend = dawn.backend_context();
    let opts = graphite::ContextOptions::default();
    let ctx = unsafe { graphite::Context::new_dawn(&backend, &opts) }
        .expect("Context::new_dawn through wgpu");
    assert_eq!(ctx.backend(), graphite::BackendApi::Dawn);
    eprintln!("Context::new_dawn succeeded through wgpu: {ctx:?}");
}

/// Draws red onto a Graphite-backed surface entirely through the
/// wgpu-routed proc table and verifies the pixel comes back. Closes the
/// loop: Skia's MakeDawn / Recorder / RenderPass / async read pixels all
/// run against the `wgpu` Rust crate via the proc-table dispatch.
#[test]
fn end_to_end_draw_red_via_wgpu() {
    static PROCS: std::sync::OnceLock<skia_bindings::DawnProcTable> = std::sync::OnceLock::new();
    let procs = PROCS.get_or_init(wgpu_proc_table);
    unsafe { install_proc_table(procs) };

    let Some(dawn) = DawnDevice::new() else {
        eprintln!("Skipping: no wgpu-compatible GPU available");
        return;
    };

    let backend = dawn.backend_context();
    let mut ctx = unsafe {
        graphite::Context::new_dawn(&backend, &graphite::ContextOptions::default())
    }
    .expect("Context::new_dawn");
    let mut recorder = ctx
        .make_recorder(&graphite::RecorderOptions::default())
        .expect("make_recorder");

    let info = ImageInfo::new((32, 32), ColorType::RGBA8888, AlphaType::Premul, None);
    let mut surface = graphite::surfaces::render_target(&mut recorder, &info, None, None)
        .expect("render_target");

    surface.canvas().clear(Color::RED);

    let mut recording = recorder.snap().expect("snap");
    let status = ctx.insert_recording(&mut recording, Some(&mut surface));
    assert_eq!(status, graphite::InsertStatus::Success);

    let mut pixels = vec![0u8; 32 * 32 * 4];
    let ok = ctx.read_pixels(&surface, &info, &mut pixels, 32 * 4, IRect::new(0, 0, 32, 32));
    assert!(ok, "read_pixels through wgpu");
    assert_eq!(&pixels[0..4], &[255, 0, 0, 255], "pixel(0,0) should be red");
}

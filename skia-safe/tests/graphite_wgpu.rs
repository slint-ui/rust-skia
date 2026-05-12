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
    AlphaType, Color, ColorType, IRect, ImageInfo, Paint, Rect,
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

/// Builds a Graphite Context from an externally-created wgpu Instance /
/// Adapter / Device / Queue (i.e. the integration pattern an application
/// that already drives its own rendering through `wgpu` would use).
/// Verifies that the wrapped device is usable by Graphite: clears the
/// surface to red and reads the pixel back.
#[test]
fn build_context_from_external_wgpu() {
    // Create wgpu objects ourselves, the way an application would.
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let Some(adapter) = pollster::block_on(instance.request_adapter(&Default::default())).ok()
    else {
        eprintln!("Skipping: no wgpu adapter available");
        return;
    };
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .expect("request_device");

    // Hand them to Graphite. This installs the proc table and manufactures
    // C handles backed by the same wgpu objects.
    let backend = graphite::wgpu_backend::install_and_wrap(instance, adapter, device, queue);

    let mut ctx = unsafe {
        graphite::Context::new_dawn(&backend, &graphite::ContextOptions::default())
    }
    .expect("Context::new_dawn from external wgpu");
    assert_eq!(ctx.backend(), graphite::BackendApi::Dawn);

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
    assert!(ok);
    assert_eq!(&pixels[0..4], &[255, 0, 0, 255], "pixel(0,0) should be red");
}

/// Like `end_to_end_draw_red_via_wgpu` but also draws a green rectangle —
/// exercises the full render-pipeline/bind-group/draw path.
#[test]
fn draw_geometry_via_wgpu() {
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

    let info = ImageInfo::new((64, 64), ColorType::RGBA8888, AlphaType::Premul, None);
    let mut surface = graphite::surfaces::render_target(&mut recorder, &info, None, None)
        .expect("render_target");

    let canvas = surface.canvas();
    canvas.clear(Color::BLACK);
    let mut paint = Paint::default();
    paint.set_anti_alias(true);
    paint.set_color(Color::GREEN);
    canvas.draw_rect(Rect::new(16.0, 16.0, 48.0, 48.0), &paint);

    let mut recording = recorder.snap().expect("snap");
    let status = ctx.insert_recording(&mut recording, Some(&mut surface));
    assert_eq!(status, graphite::InsertStatus::Success);

    let mut pixels = vec![0u8; 64 * 64 * 4];
    let ok = ctx.read_pixels(&surface, &info, &mut pixels, 64 * 4, IRect::new(0, 0, 64, 64));
    assert!(ok);
    // Pixel at the center of the rectangle.
    let center = (32 * 64 + 32) * 4;
    assert_eq!(
        &pixels[center..center + 4],
        &[0, 255, 0, 255],
        "center pixel should be green"
    );
}

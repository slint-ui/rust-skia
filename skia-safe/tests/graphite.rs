//! End-to-end smoke test for the Graphite backend.
//!
//! Runs the full pipeline (default Dawn setup → Context → Recorder →
//! Surface → draw → snap → insert → submit → readback) and verifies a known
//! pixel value. Skipped at runtime if no GPU is available.

#![cfg(feature = "graphite")]

use skia_safe::{
    graphite::{self, dawn::DawnDevice},
    AlphaType, Color, ColorType, IRect, ImageInfo,
};

#[test]
fn end_to_end_draw_red() {
    let Some(dawn) = DawnDevice::new() else {
        eprintln!("Skipping: no Dawn-compatible GPU available");
        return;
    };

    let backend = dawn.backend_context();
    let mut ctx = unsafe {
        graphite::Context::new_dawn(&backend, &graphite::ContextOptions::default())
    }
    .expect("Context::new_dawn returned None");

    let mut recorder = ctx
        .make_recorder(&graphite::RecorderOptions::default())
        .expect("make_recorder returned None");

    let info = ImageInfo::new((32, 32), ColorType::RGBA8888, AlphaType::Premul, None);
    let mut surface = graphite::surfaces::render_target(&mut recorder, &info, None, None)
        .expect("render_target returned None");

    surface.canvas().clear(Color::RED);

    let mut recording = recorder.snap().expect("snap returned None");
    let status = ctx.insert_recording(&mut recording, Some(&mut surface));
    assert_eq!(status, graphite::InsertStatus::Success, "insert_recording");

    // `Context::read_pixels` drives the async read + submit to completion
    // internally, so an explicit `submit` isn't needed before this call.
    let mut pixels = vec![0u8; 32 * 32 * 4];
    let ok = ctx.read_pixels(&surface, &info, &mut pixels, 32 * 4, IRect::new(0, 0, 32, 32));
    assert!(ok, "read_pixels");

    // Premultiplied red = (R=255, G=0, B=0, A=255).
    assert_eq!(&pixels[0..4], &[255, 0, 0, 255], "pixel(0,0) should be red");
}

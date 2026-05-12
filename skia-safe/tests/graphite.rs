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

/// Demonstrates wrapping an existing `WGPUTexture` (the pattern a caller
/// would use when rendering Skia into a swapchain image from their own
/// `wgpuSurfaceGetCurrentTexture` call).
#[test]
fn wrap_existing_wgpu_texture() {
    use core::ptr;
    use skia_bindings as sb;

    // WebGPU bitfield constants — these come from `webgpu.h` as `static const`
    // and aren't reachable through bindgen, but the values are stable per the
    // WebGPU spec.
    const WGPU_TEXTURE_USAGE_COPY_SRC: sb::WGPUTextureUsage = 0x0000_0000_0000_0001;
    const WGPU_TEXTURE_USAGE_TEXTURE_BINDING: sb::WGPUTextureUsage = 0x0000_0000_0000_0004;
    const WGPU_TEXTURE_USAGE_RENDER_ATTACHMENT: sb::WGPUTextureUsage = 0x0000_0000_0000_0010;

    let Some(dawn) = DawnDevice::new() else {
        eprintln!("Skipping: no Dawn-compatible GPU available");
        return;
    };

    // Create a Dawn-allocated 32x32 RGBA8 texture.
    let extent = sb::WGPUExtent3D {
        width: 32,
        height: 32,
        depthOrArrayLayers: 1,
    };
    let desc = sb::WGPUTextureDescriptor {
        nextInChain: ptr::null_mut(),
        label: sb::WGPUStringView {
            data: ptr::null(),
            length: 0,
        },
        usage: WGPU_TEXTURE_USAGE_COPY_SRC
            | WGPU_TEXTURE_USAGE_TEXTURE_BINDING
            | WGPU_TEXTURE_USAGE_RENDER_ATTACHMENT,
        dimension: sb::WGPUTextureDimension::WGPUTextureDimension_2D,
        size: extent,
        format: sb::WGPUTextureFormat::WGPUTextureFormat_RGBA8Unorm,
        mipLevelCount: 1,
        sampleCount: 1,
        viewFormatCount: 0,
        viewFormats: ptr::null(),
    };
    let wgpu_texture = unsafe { sb::wgpuDeviceCreateTexture(dawn.device(), &desc) };
    assert!(!wgpu_texture.is_null(), "wgpuDeviceCreateTexture");

    let backend = dawn.backend_context();
    let mut ctx = unsafe {
        graphite::Context::new_dawn(&backend, &graphite::ContextOptions::default())
    }
    .expect("Context::new_dawn");
    let mut recorder = ctx
        .make_recorder(&graphite::RecorderOptions::default())
        .expect("make_recorder");

    let backend_texture =
        unsafe { graphite::dawn::backend_texture_from_wgpu_texture(wgpu_texture) }
            .expect("backend_texture_from_wgpu_texture");
    assert!(backend_texture.is_valid(), "backend_texture is_valid");
    assert_eq!(backend_texture.dimensions(), (32, 32).into());

    let mut surface =
        graphite::surfaces::wrap_backend_texture(&mut recorder, &backend_texture, None, None)
            .expect("wrap_backend_texture");

    surface.canvas().clear(Color::GREEN);

    let mut recording = recorder.snap().expect("snap");
    let status = ctx.insert_recording(&mut recording, Some(&mut surface));
    assert_eq!(status, graphite::InsertStatus::Success);

    let info = ImageInfo::new((32, 32), ColorType::RGBA8888, AlphaType::Premul, None);
    let mut pixels = vec![0u8; 32 * 32 * 4];
    let ok = ctx.read_pixels(&surface, &info, &mut pixels, 32 * 4, IRect::new(0, 0, 32, 32));
    assert!(ok, "read_pixels");
    assert_eq!(&pixels[0..4], &[0, 255, 0, 255], "pixel(0,0) should be green");

    // Drop Skia handles before the underlying texture so the surface releases
    // any internal references it has on it.
    drop(surface);
    drop(recording);
    drop(backend_texture);
    drop(recorder);
    drop(ctx);
    unsafe { sb::wgpuTextureRelease(wgpu_texture) };
}

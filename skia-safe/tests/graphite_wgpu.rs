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

// WebGPU bitflag values — the bindgen run doesn't capture these C #defines,
// so we hard-code them here. From dawn.json / webgpu.h.
const WGPU_BUFFER_USAGE_MAP_READ: u64 = 0x0001;
const WGPU_BUFFER_USAGE_COPY_SRC: u64 = 0x0004;
const WGPU_BUFFER_USAGE_COPY_DST: u64 = 0x0008;
const WGPU_BUFFER_USAGE_STORAGE: u64 = 0x0080;
const WGPU_MAP_MODE_READ: u64 = 0x0001;
const WGPU_SHADER_STAGE_COMPUTE: u64 = 0x0004;
const WGPU_TEXTURE_USAGE_COPY_SRC: u64 = 0x0001;
const WGPU_TEXTURE_USAGE_COPY_DST: u64 = 0x0002;

fn sv(s: &str) -> skia_bindings::WGPUStringView {
    skia_bindings::WGPUStringView {
        data: s.as_ptr() as *const _,
        length: s.len(),
    }
}

unsafe fn poll_for_map(
    procs: &skia_bindings::DawnProcTable,
    device: skia_bindings::WGPUDevice,
    flag: &std::sync::atomic::AtomicBool,
) -> bool {
    for _ in 0..1000 {
        if flag.load(std::sync::atomic::Ordering::SeqCst) {
            return true;
        }
        (procs.deviceTick.unwrap())(device);
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    flag.load(std::sync::atomic::Ordering::SeqCst)
}

extern "C" fn map_done(
    _status: skia_bindings::WGPUMapAsyncStatus,
    _msg: skia_bindings::WGPUStringView,
    userdata1: *mut core::ffi::c_void,
    _userdata2: *mut core::ffi::c_void,
) {
    let flag = unsafe { &*(userdata1 as *const std::sync::atomic::AtomicBool) };
    flag.store(true, std::sync::atomic::Ordering::SeqCst);
}

/// Builds + dispatches a compute pipeline through the wgpu proc table.
/// The shader writes 42 into a storage buffer; we copy that into a
/// readback buffer and verify the value. Exercises shader module,
/// bind-group layout/group, pipeline layout, compute pipeline, compute
/// pass encoder, and the buffer map path through the proc table.
#[test]
fn compute_pipeline_via_wgpu_proc_table() {
    use skia_bindings as sb;
    static PROCS: std::sync::OnceLock<sb::DawnProcTable> = std::sync::OnceLock::new();
    let procs = PROCS.get_or_init(wgpu_proc_table);
    unsafe { install_proc_table(procs) };

    let Some(dawn) = DawnDevice::new() else {
        eprintln!("Skipping: no wgpu-compatible GPU available");
        return;
    };
    let device = dawn.device();
    let queue = dawn.queue();

    let wgsl_source = "@group(0) @binding(0) var<storage, read_write> out_buf: array<u32>;\n\
                       @compute @workgroup_size(1) fn main() { out_buf[0] = 42u; }\n";

    unsafe {
        let mut wgsl_desc: sb::WGPUShaderSourceWGSL = std::mem::MaybeUninit::zeroed().assume_init();
        wgsl_desc.chain.sType = sb::WGPUSType::WGPUSType_ShaderSourceWGSL;
        wgsl_desc.code = sv(wgsl_source);

        let mut shader_desc: sb::WGPUShaderModuleDescriptor = std::mem::MaybeUninit::zeroed().assume_init();
        shader_desc.nextInChain = &mut wgsl_desc.chain as *mut _;
        shader_desc.label = sv("compute_test");
        let shader = (procs.deviceCreateShaderModule.unwrap())(device, &shader_desc);
        assert!(!shader.is_null(), "shader module");

        let mut bgl_entry: sb::WGPUBindGroupLayoutEntry = std::mem::MaybeUninit::zeroed().assume_init();
        bgl_entry.binding = 0;
        bgl_entry.visibility = WGPU_SHADER_STAGE_COMPUTE;
        bgl_entry.buffer.type_ = sb::WGPUBufferBindingType::WGPUBufferBindingType_Storage;
        let mut bgl_desc: sb::WGPUBindGroupLayoutDescriptor = std::mem::MaybeUninit::zeroed().assume_init();
        bgl_desc.label = sv("bgl");
        bgl_desc.entryCount = 1;
        bgl_desc.entries = &bgl_entry;
        let bgl = (procs.deviceCreateBindGroupLayout.unwrap())(device, &bgl_desc);
        assert!(!bgl.is_null());

        let mut pl_desc: sb::WGPUPipelineLayoutDescriptor = std::mem::MaybeUninit::zeroed().assume_init();
        pl_desc.label = sv("pl");
        pl_desc.bindGroupLayoutCount = 1;
        pl_desc.bindGroupLayouts = &bgl;
        let pl = (procs.deviceCreatePipelineLayout.unwrap())(device, &pl_desc);
        assert!(!pl.is_null());

        let mut cp_desc: sb::WGPUComputePipelineDescriptor = std::mem::MaybeUninit::zeroed().assume_init();
        cp_desc.label = sv("cp");
        cp_desc.layout = pl;
        cp_desc.compute.module = shader;
        cp_desc.compute.entryPoint = sv("main");
        let cp = (procs.deviceCreateComputePipeline.unwrap())(device, &cp_desc);
        assert!(!cp.is_null(), "compute pipeline");

        let mut buf_desc: sb::WGPUBufferDescriptor = std::mem::MaybeUninit::zeroed().assume_init();
        buf_desc.label = sv("out");
        buf_desc.usage = WGPU_BUFFER_USAGE_STORAGE | WGPU_BUFFER_USAGE_COPY_SRC;
        buf_desc.size = 4;
        let buf = (procs.deviceCreateBuffer.unwrap())(device, &buf_desc);
        assert!(!buf.is_null());

        let mut readback_desc: sb::WGPUBufferDescriptor = std::mem::MaybeUninit::zeroed().assume_init();
        readback_desc.label = sv("rb");
        readback_desc.usage = WGPU_BUFFER_USAGE_MAP_READ | WGPU_BUFFER_USAGE_COPY_DST;
        readback_desc.size = 4;
        let readback = (procs.deviceCreateBuffer.unwrap())(device, &readback_desc);
        assert!(!readback.is_null());

        let mut bg_entry: sb::WGPUBindGroupEntry = std::mem::MaybeUninit::zeroed().assume_init();
        bg_entry.binding = 0;
        bg_entry.buffer = buf;
        bg_entry.offset = 0;
        bg_entry.size = 4;
        let mut bg_desc: sb::WGPUBindGroupDescriptor = std::mem::MaybeUninit::zeroed().assume_init();
        bg_desc.label = sv("bg");
        bg_desc.layout = bgl;
        bg_desc.entryCount = 1;
        bg_desc.entries = &bg_entry;
        let bg = (procs.deviceCreateBindGroup.unwrap())(device, &bg_desc);
        assert!(!bg.is_null());

        let mut enc_desc: sb::WGPUCommandEncoderDescriptor = std::mem::MaybeUninit::zeroed().assume_init();
        enc_desc.label = sv("enc");
        let enc = (procs.deviceCreateCommandEncoder.unwrap())(device, &enc_desc);
        assert!(!enc.is_null());

        let mut pass_desc: sb::WGPUComputePassDescriptor = std::mem::MaybeUninit::zeroed().assume_init();
        pass_desc.label = sv("pass");
        let pass = (procs.commandEncoderBeginComputePass.unwrap())(enc, &pass_desc);
        assert!(!pass.is_null(), "compute pass");
        (procs.computePassEncoderSetPipeline.unwrap())(pass, cp);
        (procs.computePassEncoderSetBindGroup.unwrap())(pass, 0, bg, 0, std::ptr::null());
        (procs.computePassEncoderDispatchWorkgroups.unwrap())(pass, 1, 1, 1);
        (procs.computePassEncoderEnd.unwrap())(pass);
        (procs.computePassEncoderRelease.unwrap())(pass);

        (procs.commandEncoderCopyBufferToBuffer.unwrap())(enc, buf, 0, readback, 0, 4);

        let mut cb_desc: sb::WGPUCommandBufferDescriptor = std::mem::MaybeUninit::zeroed().assume_init();
        cb_desc.label = sv("cb");
        let cb = (procs.commandEncoderFinish.unwrap())(enc, &cb_desc);
        assert!(!cb.is_null());
        (procs.commandEncoderRelease.unwrap())(enc);
        (procs.queueSubmit.unwrap())(queue, 1, &cb);
        (procs.commandBufferRelease.unwrap())(cb);

        let mapped = Box::new(std::sync::atomic::AtomicBool::new(false));
        let mut cb_info: sb::WGPUBufferMapCallbackInfo = std::mem::MaybeUninit::zeroed().assume_init();
        cb_info.mode = sb::WGPUCallbackMode::WGPUCallbackMode_AllowProcessEvents;
        cb_info.callback = Some(map_done);
        cb_info.userdata1 = &*mapped as *const _ as *mut _;
        (procs.bufferMapAsync.unwrap())(readback, WGPU_MAP_MODE_READ, 0, 4, cb_info);
        assert!(poll_for_map(procs, device, &mapped), "map callback");

        let ptr = (procs.bufferGetConstMappedRange.unwrap())(readback, 0, 4) as *const u32;
        assert!(!ptr.is_null());
        let value = std::ptr::read_unaligned(ptr);
        assert_eq!(value, 42, "compute output");

        (procs.bufferUnmap.unwrap())(readback);
        (procs.bufferRelease.unwrap())(readback);
        (procs.bufferRelease.unwrap())(buf);
        (procs.bindGroupRelease.unwrap())(bg);
        (procs.computePipelineRelease.unwrap())(cp);
        (procs.pipelineLayoutRelease.unwrap())(pl);
        (procs.bindGroupLayoutRelease.unwrap())(bgl);
        (procs.shaderModuleRelease.unwrap())(shader);
    }
}

/// Exercises queueWriteTexture: writes a 4x4 RGBA pattern into a texture
/// and reads it back via copy_texture_to_buffer + map.
#[test]
fn queue_write_texture_via_wgpu_proc_table() {
    use skia_bindings as sb;
    static PROCS: std::sync::OnceLock<sb::DawnProcTable> = std::sync::OnceLock::new();
    let procs = PROCS.get_or_init(wgpu_proc_table);
    unsafe { install_proc_table(procs) };

    let Some(dawn) = DawnDevice::new() else {
        eprintln!("Skipping: no wgpu-compatible GPU available");
        return;
    };
    let device = dawn.device();
    let queue = dawn.queue();

    unsafe {
        let mut tex_desc: sb::WGPUTextureDescriptor = std::mem::MaybeUninit::zeroed().assume_init();
        tex_desc.label = sv("tex");
        tex_desc.usage = WGPU_TEXTURE_USAGE_COPY_DST | WGPU_TEXTURE_USAGE_COPY_SRC;
        tex_desc.dimension = sb::WGPUTextureDimension::WGPUTextureDimension_2D;
        tex_desc.size = sb::WGPUExtent3D {
            width: 4,
            height: 4,
            depthOrArrayLayers: 1,
        };
        tex_desc.format = sb::WGPUTextureFormat::WGPUTextureFormat_RGBA8Unorm;
        tex_desc.mipLevelCount = 1;
        tex_desc.sampleCount = 1;
        let tex = (procs.deviceCreateTexture.unwrap())(device, &tex_desc);
        assert!(!tex.is_null());

        let mut src = vec![0u8; 4 * 4 * 4];
        for chunk in src.chunks_mut(4) {
            chunk.copy_from_slice(&[0x10, 0x20, 0x30, 0xFF]);
        }

        let mut dst_info: sb::WGPUTexelCopyTextureInfo = std::mem::MaybeUninit::zeroed().assume_init();
        dst_info.texture = tex;
        dst_info.aspect = sb::WGPUTextureAspect::WGPUTextureAspect_All;
        let mut layout: sb::WGPUTexelCopyBufferLayout = std::mem::MaybeUninit::zeroed().assume_init();
        layout.bytesPerRow = 16;
        layout.rowsPerImage = 4;
        let write_size = sb::WGPUExtent3D {
            width: 4,
            height: 4,
            depthOrArrayLayers: 1,
        };
        (procs.queueWriteTexture.unwrap())(
            queue,
            &dst_info,
            src.as_ptr() as *const _,
            src.len(),
            &layout,
            &write_size,
        );

        let mut buf_desc: sb::WGPUBufferDescriptor = std::mem::MaybeUninit::zeroed().assume_init();
        buf_desc.label = sv("rb");
        buf_desc.usage = WGPU_BUFFER_USAGE_MAP_READ | WGPU_BUFFER_USAGE_COPY_DST;
        buf_desc.size = 256 * 4;
        let buf = (procs.deviceCreateBuffer.unwrap())(device, &buf_desc);

        let mut enc_desc: sb::WGPUCommandEncoderDescriptor = std::mem::MaybeUninit::zeroed().assume_init();
        enc_desc.label = sv("enc");
        let enc = (procs.deviceCreateCommandEncoder.unwrap())(device, &enc_desc);

        let mut src_info: sb::WGPUTexelCopyTextureInfo = std::mem::MaybeUninit::zeroed().assume_init();
        src_info.texture = tex;
        src_info.aspect = sb::WGPUTextureAspect::WGPUTextureAspect_All;
        let mut dst_buf_info: sb::WGPUTexelCopyBufferInfo = std::mem::MaybeUninit::zeroed().assume_init();
        dst_buf_info.layout.bytesPerRow = 256;
        dst_buf_info.layout.rowsPerImage = 4;
        dst_buf_info.buffer = buf;
        let copy_size = sb::WGPUExtent3D {
            width: 4,
            height: 4,
            depthOrArrayLayers: 1,
        };
        (procs.commandEncoderCopyTextureToBuffer.unwrap())(
            enc, &src_info, &dst_buf_info, &copy_size,
        );

        let mut cb_desc: sb::WGPUCommandBufferDescriptor = std::mem::MaybeUninit::zeroed().assume_init();
        cb_desc.label = sv("cb");
        let cb = (procs.commandEncoderFinish.unwrap())(enc, &cb_desc);
        (procs.commandEncoderRelease.unwrap())(enc);
        (procs.queueSubmit.unwrap())(queue, 1, &cb);
        (procs.commandBufferRelease.unwrap())(cb);

        let mapped = Box::new(std::sync::atomic::AtomicBool::new(false));
        let mut cb_info: sb::WGPUBufferMapCallbackInfo = std::mem::MaybeUninit::zeroed().assume_init();
        cb_info.mode = sb::WGPUCallbackMode::WGPUCallbackMode_AllowProcessEvents;
        cb_info.callback = Some(map_done);
        cb_info.userdata1 = &*mapped as *const _ as *mut _;
        (procs.bufferMapAsync.unwrap())(buf, WGPU_MAP_MODE_READ, 0, 256 * 4, cb_info);
        assert!(poll_for_map(procs, device, &mapped), "map");

        let ptr = (procs.bufferGetConstMappedRange.unwrap())(buf, 0, 256 * 4) as *const u8;
        let first = std::slice::from_raw_parts(ptr, 16);
        assert_eq!(
            first,
            &[
                0x10, 0x20, 0x30, 0xFF, 0x10, 0x20, 0x30, 0xFF, 0x10, 0x20, 0x30, 0xFF, 0x10, 0x20,
                0x30, 0xFF,
            ],
            "first row of texture readback"
        );
        (procs.bufferUnmap.unwrap())(buf);
        (procs.bufferRelease.unwrap())(buf);
        (procs.textureRelease.unwrap())(tex);
    }
}

/// Exercises the QuerySet lifecycle: create, query count/type, destroy.
#[test]
fn query_set_via_wgpu_proc_table() {
    use skia_bindings as sb;
    static PROCS: std::sync::OnceLock<sb::DawnProcTable> = std::sync::OnceLock::new();
    let procs = PROCS.get_or_init(wgpu_proc_table);
    unsafe { install_proc_table(procs) };

    let Some(dawn) = DawnDevice::new() else {
        eprintln!("Skipping: no wgpu-compatible GPU available");
        return;
    };
    let device = dawn.device();

    unsafe {
        let mut desc: sb::WGPUQuerySetDescriptor = std::mem::MaybeUninit::zeroed().assume_init();
        desc.label = sv("qs");
        desc.type_ = sb::WGPUQueryType::WGPUQueryType_Occlusion;
        desc.count = 8;
        let qs = (procs.deviceCreateQuerySet.unwrap())(device, &desc);
        assert!(!qs.is_null());
        assert_eq!((procs.querySetGetCount.unwrap())(qs), 8);
        assert!(matches!(
            (procs.querySetGetType.unwrap())(qs),
            sb::WGPUQueryType::WGPUQueryType_Occlusion
        ));
        (procs.querySetDestroy.unwrap())(qs);
        (procs.querySetRelease.unwrap())(qs);
    }
}

/// Builds a RenderBundleEncoder, finishes it into a RenderBundle, and
/// releases it. The encoder ran zero draws — what we're checking is that
/// the lifecycle plumbing works end-to-end.
#[test]
fn render_bundle_via_wgpu_proc_table() {
    use skia_bindings as sb;
    static PROCS: std::sync::OnceLock<sb::DawnProcTable> = std::sync::OnceLock::new();
    let procs = PROCS.get_or_init(wgpu_proc_table);
    unsafe { install_proc_table(procs) };

    let Some(dawn) = DawnDevice::new() else {
        eprintln!("Skipping: no wgpu-compatible GPU available");
        return;
    };
    let device = dawn.device();

    unsafe {
        let color_format = sb::WGPUTextureFormat::WGPUTextureFormat_RGBA8Unorm;
        let mut rbe_desc: sb::WGPURenderBundleEncoderDescriptor = std::mem::MaybeUninit::zeroed().assume_init();
        rbe_desc.label = sv("rbe");
        rbe_desc.colorFormatCount = 1;
        rbe_desc.colorFormats = &color_format;
        rbe_desc.depthStencilFormat = sb::WGPUTextureFormat::WGPUTextureFormat_Undefined;
        rbe_desc.sampleCount = 1;
        let rbe = (procs.deviceCreateRenderBundleEncoder.unwrap())(device, &rbe_desc);
        assert!(!rbe.is_null(), "render bundle encoder");

        let mut rb_desc: sb::WGPURenderBundleDescriptor = std::mem::MaybeUninit::zeroed().assume_init();
        rb_desc.label = sv("rb");
        let rb = (procs.renderBundleEncoderFinish.unwrap())(rbe, &rb_desc);
        assert!(!rb.is_null(), "render bundle");
        (procs.renderBundleEncoderRelease.unwrap())(rbe);
        (procs.renderBundleRelease.unwrap())(rb);
    }
}

/// Drives a render bundle through executeBundles inside a render pass.
/// Records `set_pipeline + draw(3 verts, 1 inst)` into a bundle that
/// targets RGBA8 + no vertex buffers; the shader produces a green triangle
/// covering the whole framebuffer. Then executes the bundle from a real
/// render pass on a small color attachment and verifies the pixel.
///
/// Exercises: deviceCreateRenderBundleEncoder, renderBundleEncoderSetPipeline,
/// renderBundleEncoderDraw, renderBundleEncoderFinish,
/// renderPassEncoderExecuteBundles.
#[test]
fn render_bundle_draw_via_wgpu_proc_table() {
    use skia_bindings as sb;
    static PROCS: std::sync::OnceLock<sb::DawnProcTable> = std::sync::OnceLock::new();
    let procs = PROCS.get_or_init(wgpu_proc_table);
    unsafe { install_proc_table(procs) };

    let Some(dawn) = DawnDevice::new() else {
        eprintln!("Skipping: no wgpu-compatible GPU available");
        return;
    };
    let device = dawn.device();
    let queue = dawn.queue();

    let wgsl = "\
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {\n\
  // Big triangle covering the entire viewport.\n\
  let pos = array(vec2f(-1.0, -3.0), vec2f(-1.0, 1.0), vec2f(3.0, 1.0));\n\
  return vec4f(pos[i], 0.0, 1.0);\n\
}\n\
@fragment fn fs() -> @location(0) vec4f { return vec4f(0.0, 1.0, 0.0, 1.0); }\n";

    unsafe {
        // Shader module.
        let mut wgsl_desc: sb::WGPUShaderSourceWGSL =
            std::mem::MaybeUninit::zeroed().assume_init();
        wgsl_desc.chain.sType = sb::WGPUSType::WGPUSType_ShaderSourceWGSL;
        wgsl_desc.code = sv(wgsl);
        let mut sm_desc: sb::WGPUShaderModuleDescriptor =
            std::mem::MaybeUninit::zeroed().assume_init();
        sm_desc.nextInChain = &mut wgsl_desc.chain as *mut _;
        sm_desc.label = sv("rb_shader");
        let sm = (procs.deviceCreateShaderModule.unwrap())(device, &sm_desc);
        assert!(!sm.is_null(), "shader module");

        // Pipeline layout (no bind groups).
        let mut pl_desc: sb::WGPUPipelineLayoutDescriptor =
            std::mem::MaybeUninit::zeroed().assume_init();
        pl_desc.label = sv("pl");
        pl_desc.bindGroupLayoutCount = 0;
        pl_desc.bindGroupLayouts = std::ptr::null();
        let pl = (procs.deviceCreatePipelineLayout.unwrap())(device, &pl_desc);

        // Color target.
        let mut color_target: sb::WGPUColorTargetState =
            std::mem::MaybeUninit::zeroed().assume_init();
        color_target.format = sb::WGPUTextureFormat::WGPUTextureFormat_RGBA8Unorm;
        color_target.writeMask = 0xF;

        let mut frag: sb::WGPUFragmentState = std::mem::MaybeUninit::zeroed().assume_init();
        frag.module = sm;
        frag.entryPoint = sv("fs");
        frag.targetCount = 1;
        frag.targets = &color_target;

        let mut rp_desc: sb::WGPURenderPipelineDescriptor =
            std::mem::MaybeUninit::zeroed().assume_init();
        rp_desc.label = sv("rp");
        rp_desc.layout = pl;
        rp_desc.vertex.module = sm;
        rp_desc.vertex.entryPoint = sv("vs");
        rp_desc.vertex.bufferCount = 0;
        rp_desc.vertex.buffers = std::ptr::null();
        rp_desc.primitive.topology = sb::WGPUPrimitiveTopology::WGPUPrimitiveTopology_TriangleList;
        rp_desc.primitive.frontFace = sb::WGPUFrontFace::WGPUFrontFace_CCW;
        rp_desc.primitive.cullMode = sb::WGPUCullMode::WGPUCullMode_None;
        rp_desc.multisample.count = 1;
        rp_desc.multisample.mask = 0xFFFFFFFF;
        rp_desc.fragment = &frag;
        let pipeline = (procs.deviceCreateRenderPipeline.unwrap())(device, &rp_desc);
        assert!(!pipeline.is_null(), "render pipeline");

        // Create render bundle: set_pipeline + draw(3, 1, 0, 0); finish.
        let color_format = sb::WGPUTextureFormat::WGPUTextureFormat_RGBA8Unorm;
        let mut rbe_desc: sb::WGPURenderBundleEncoderDescriptor =
            std::mem::MaybeUninit::zeroed().assume_init();
        rbe_desc.label = sv("rbe");
        rbe_desc.colorFormatCount = 1;
        rbe_desc.colorFormats = &color_format;
        rbe_desc.depthStencilFormat = sb::WGPUTextureFormat::WGPUTextureFormat_Undefined;
        rbe_desc.sampleCount = 1;
        let rbe = (procs.deviceCreateRenderBundleEncoder.unwrap())(device, &rbe_desc);
        (procs.renderBundleEncoderSetPipeline.unwrap())(rbe, pipeline);
        (procs.renderBundleEncoderDraw.unwrap())(rbe, 3, 1, 0, 0);
        let mut rb_desc: sb::WGPURenderBundleDescriptor =
            std::mem::MaybeUninit::zeroed().assume_init();
        rb_desc.label = sv("rb");
        let bundle = (procs.renderBundleEncoderFinish.unwrap())(rbe, &rb_desc);
        assert!(!bundle.is_null(), "render bundle finish");
        (procs.renderBundleEncoderRelease.unwrap())(rbe);

        // Color attachment texture.
        let mut tex_desc: sb::WGPUTextureDescriptor =
            std::mem::MaybeUninit::zeroed().assume_init();
        tex_desc.label = sv("color");
        tex_desc.usage = (0x10 /* RenderAttachment */) | WGPU_TEXTURE_USAGE_COPY_SRC;
        tex_desc.dimension = sb::WGPUTextureDimension::WGPUTextureDimension_2D;
        tex_desc.size = sb::WGPUExtent3D {
            width: 4,
            height: 4,
            depthOrArrayLayers: 1,
        };
        tex_desc.format = sb::WGPUTextureFormat::WGPUTextureFormat_RGBA8Unorm;
        tex_desc.mipLevelCount = 1;
        tex_desc.sampleCount = 1;
        let tex = (procs.deviceCreateTexture.unwrap())(device, &tex_desc);
        let view = (procs.textureCreateView.unwrap())(tex, std::ptr::null());

        // Encoder + render pass.
        let mut enc_desc: sb::WGPUCommandEncoderDescriptor =
            std::mem::MaybeUninit::zeroed().assume_init();
        enc_desc.label = sv("enc");
        let enc = (procs.deviceCreateCommandEncoder.unwrap())(device, &enc_desc);

        let mut color_att: sb::WGPURenderPassColorAttachment =
            std::mem::MaybeUninit::zeroed().assume_init();
        color_att.view = view;
        color_att.depthSlice = u32::MAX;
        color_att.loadOp = sb::WGPULoadOp::WGPULoadOp_Clear;
        color_att.storeOp = sb::WGPUStoreOp::WGPUStoreOp_Store;
        color_att.clearValue = sb::WGPUColor {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        };

        let mut pass_desc: sb::WGPURenderPassDescriptor =
            std::mem::MaybeUninit::zeroed().assume_init();
        pass_desc.label = sv("pass");
        pass_desc.colorAttachmentCount = 1;
        pass_desc.colorAttachments = &color_att;
        let pass = (procs.commandEncoderBeginRenderPass.unwrap())(enc, &pass_desc);
        (procs.renderPassEncoderExecuteBundles.unwrap())(pass, 1, &bundle);
        (procs.renderPassEncoderEnd.unwrap())(pass);
        (procs.renderPassEncoderRelease.unwrap())(pass);

        // Read back.
        let mut buf_desc: sb::WGPUBufferDescriptor =
            std::mem::MaybeUninit::zeroed().assume_init();
        buf_desc.label = sv("rb_buf");
        buf_desc.usage = WGPU_BUFFER_USAGE_MAP_READ | WGPU_BUFFER_USAGE_COPY_DST;
        buf_desc.size = 256 * 4;
        let rb_buf = (procs.deviceCreateBuffer.unwrap())(device, &buf_desc);

        let mut src_info: sb::WGPUTexelCopyTextureInfo =
            std::mem::MaybeUninit::zeroed().assume_init();
        src_info.texture = tex;
        src_info.aspect = sb::WGPUTextureAspect::WGPUTextureAspect_All;
        let mut dst_buf_info: sb::WGPUTexelCopyBufferInfo =
            std::mem::MaybeUninit::zeroed().assume_init();
        dst_buf_info.layout.bytesPerRow = 256;
        dst_buf_info.layout.rowsPerImage = 4;
        dst_buf_info.buffer = rb_buf;
        let copy_size = sb::WGPUExtent3D {
            width: 4,
            height: 4,
            depthOrArrayLayers: 1,
        };
        (procs.commandEncoderCopyTextureToBuffer.unwrap())(
            enc, &src_info, &dst_buf_info, &copy_size,
        );

        let mut cb_desc: sb::WGPUCommandBufferDescriptor =
            std::mem::MaybeUninit::zeroed().assume_init();
        cb_desc.label = sv("cb");
        let cb = (procs.commandEncoderFinish.unwrap())(enc, &cb_desc);
        (procs.commandEncoderRelease.unwrap())(enc);
        (procs.queueSubmit.unwrap())(queue, 1, &cb);
        (procs.commandBufferRelease.unwrap())(cb);

        let mapped = Box::new(std::sync::atomic::AtomicBool::new(false));
        let mut cb_info: sb::WGPUBufferMapCallbackInfo =
            std::mem::MaybeUninit::zeroed().assume_init();
        cb_info.mode = sb::WGPUCallbackMode::WGPUCallbackMode_AllowProcessEvents;
        cb_info.callback = Some(map_done);
        cb_info.userdata1 = &*mapped as *const _ as *mut _;
        (procs.bufferMapAsync.unwrap())(rb_buf, WGPU_MAP_MODE_READ, 0, 256 * 4, cb_info);
        assert!(poll_for_map(procs, device, &mapped), "map");

        let ptr = (procs.bufferGetConstMappedRange.unwrap())(rb_buf, 0, 256 * 4) as *const u8;
        let first = std::slice::from_raw_parts(ptr, 4);
        assert_eq!(first, &[0, 255, 0, 255], "bundle rendered green");
        (procs.bufferUnmap.unwrap())(rb_buf);
        (procs.bufferRelease.unwrap())(rb_buf);
        (procs.textureViewRelease.unwrap())(view);
        (procs.textureRelease.unwrap())(tex);
        (procs.renderBundleRelease.unwrap())(bundle);
        (procs.renderPipelineRelease.unwrap())(pipeline);
        (procs.pipelineLayoutRelease.unwrap())(pl);
        (procs.shaderModuleRelease.unwrap())(sm);
    }
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

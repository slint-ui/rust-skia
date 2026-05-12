//! Experimental wgpu-backed proc table for Graphite.
//!
//! When the `wgpu` Cargo feature is enabled, this module exposes
//! [`wgpu_proc_table`], a [`DawnProcTable`](super::dawn::DawnProcTable) whose
//! entries route every `wgpu*` C call into the Rust `wgpu` crate. Combined
//! with [`super::dawn::install_proc_table`], this lets Skia/Graphite render
//! through the same WebGPU implementation as the rest of an application that
//! already uses `wgpu`.
//!
//! # Status
//!
//! **Incomplete.** The WebGPU C ABI surface is large (~270 functions). Only
//! the instance/adapter/device/queue lifecycle is implemented today;
//! everything else aborts the process with the function name on first call.
//! This module exists primarily as the scaffolding for filling out coverage
//! incrementally — landing a partial implementation lets Skia tell us, by
//! aborting, which functions it actually needs next.
//!
//! # Coexistence with Dawn
//!
//! Enabling `wgpu` does *not* unlink Dawn — `libdawn_combined.a` remains in
//! the static library. The proc-table mechanism is the only way Skia's
//! `wgpu*` calls are routed elsewhere. Users who want Dawn-native rendering
//! can simply not call `install_proc_table(&wgpu_proc_table())` and the
//! default Dawn path continues to work.

use core::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};

use skia_bindings as sb;

use crate::graphite::dawn::DawnProcTable;

/// Aborts the process with a message identifying the WGPU function that
/// hasn't been implemented yet. Used as the default proc-table entry until
/// real thunks land.
///
/// We deliberately call `std::process::abort()` rather than panicking: a
/// panic that crosses an FFI boundary back into Dawn's C dispatcher is UB.
pub(crate) fn unimplemented_stub(name: &'static str) -> ! {
    eprintln!("skia_safe::graphite::wgpu_backend: unimplemented WebGPU call `{name}`");
    std::process::abort();
}

//
// Resource handle infrastructure.
//
// Every `WGPU<Type>` C handle is the address of a `Resource<TypeData>`, a
// heap-allocated struct holding the wgpu Rust object plus an atomic reference
// counter. `add_ref`/`release` provide the WebGPU C ABI refcounting semantics
// that Dawn callers expect.
//

#[repr(C)]
struct Resource<T> {
    refcount: AtomicUsize,
    inner: T,
}

impl<T> Resource<T> {
    fn into_handle(inner: T) -> *mut Resource<T> {
        Box::into_raw(Box::new(Resource {
            refcount: AtomicUsize::new(1),
            inner,
        }))
    }

    unsafe fn inner(handle: *mut Resource<T>) -> &'static T {
        &(*handle).inner
    }

    unsafe fn add_ref(handle: *mut Resource<T>) {
        if handle.is_null() {
            return;
        }
        (*handle).refcount.fetch_add(1, Ordering::Relaxed);
    }

    unsafe fn release(handle: *mut Resource<T>) {
        if handle.is_null() {
            return;
        }
        if (*handle).refcount.fetch_sub(1, Ordering::Release) == 1 {
            std::sync::atomic::fence(Ordering::Acquire);
            drop(Box::from_raw(handle));
        }
    }
}

struct InstanceData {
    inner: wgpu::Instance,
}

struct AdapterData {
    inner: wgpu::Adapter,
    instance: wgpu::Instance,
}

struct DeviceData {
    inner: wgpu::Device,
    queue: wgpu::Queue,
    _adapter: wgpu::Adapter,
    /// A dummy buffer used to satisfy wgpu's validator for vertex-buffer
    /// slots that Skia declares as "unused" placeholders (`stepMode =
    /// Undefined`, stride 0, no attributes). The WebGPU spec considers
    /// such slots optional to bind, but wgpu requires *some* buffer to
    /// be set if a slot appears in the pipeline's vertex layout.
    dummy_vertex_buffer: std::sync::OnceLock<wgpu::Buffer>,
}

struct QueueData {
    inner: wgpu::Queue,
    _device: wgpu::Device,
}

struct ShaderModuleData {
    inner: wgpu::ShaderModule,
    _device: wgpu::Device,
}

struct BindGroupLayoutData {
    inner: wgpu::BindGroupLayout,
    _device: wgpu::Device,
}

struct SamplerData {
    inner: wgpu::Sampler,
    _device: wgpu::Device,
}

struct BufferData {
    inner: wgpu::Buffer,
    _device: wgpu::Device,
    /// Mapped views kept alive between get_mapped_range and unmap. wgpu
    /// records the mapped sub-range in `BufferViewMut`'s drop impl; if we
    /// drop the view immediately after extracting the raw pointer, wgpu
    /// stops considering that range as "mapped" and writes the caller
    /// performs through the raw pointer may not be flushed to the GPU on
    /// unmap. Storing the view here keeps the range tracked.
    active_views_mut: std::sync::Mutex<Vec<wgpu::BufferViewMut>>,
    active_views_const: std::sync::Mutex<Vec<wgpu::BufferView>>,
}

struct PipelineLayoutData {
    inner: wgpu::PipelineLayout,
    _device: wgpu::Device,
}

/// A CommandEncoder is consumed (moved) by `finish()`. wgpu's API takes
/// ownership of the encoder, returning a CommandBuffer. We model that by
/// stashing the encoder in a `Mutex<Option<...>>` so we can move it out
/// during `finish` regardless of which thread holds the C-side handle.
struct CommandEncoderData {
    inner: std::sync::Mutex<Option<wgpu::CommandEncoder>>,
    _device: wgpu::Device,
}

struct CommandBufferData {
    inner: std::sync::Mutex<Option<wgpu::CommandBuffer>>,
    _device: wgpu::Device,
}

struct TextureData {
    inner: wgpu::Texture,
    _device: wgpu::Device,
}

struct TextureViewData {
    inner: wgpu::TextureView,
    _texture: wgpu::Texture,
}

struct RenderPipelineData {
    inner: wgpu::RenderPipeline,
    /// Slot indices that the pipeline declared but Skia marked as
    /// "placeholder" (stepMode Undefined / VertexBufferNotUsed, no
    /// attributes). wgpu requires these slots to have a buffer bound
    /// before draws, so we auto-bind a per-device dummy on set_pipeline.
    placeholder_slots: Vec<u32>,
    /// Reference to the parent device so set_pipeline can fetch its
    /// dummy vertex buffer.
    device_handle: sb::WGPUDevice,
    _device: wgpu::Device,
}

struct BindGroupData {
    inner: wgpu::BindGroup,
    _device: wgpu::Device,
}

struct ComputePipelineData {
    inner: wgpu::ComputePipeline,
    _device: wgpu::Device,
}

struct ComputePassEncoderData {
    inner: std::sync::Mutex<Option<wgpu::ComputePass<'static>>>,
}

struct RenderBundleData {
    inner: wgpu::RenderBundle,
    _device: wgpu::Device,
}

struct RenderBundleEncoderData {
    inner: std::sync::Mutex<Option<wgpu::RenderBundleEncoder<'static>>>,
    _device: wgpu::Device,
}

struct QuerySetData {
    inner: wgpu::QuerySet,
    count: u32,
    ty: sb::WGPUQueryType,
    _device: wgpu::Device,
}

/// A live render pass. wgpu's `RenderPass` borrows from its parent
/// `CommandEncoder`, but we extend the lifetime to `'static` via
/// `forget_lifetime` so the C-side handle is independent of any Rust lock
/// guard. Callers must respect the WGPU contract that the encoder isn't
/// reused while a pass is active and that End is called before drop.
struct RenderPassEncoderData {
    inner: std::sync::Mutex<Option<wgpu::RenderPass<'static>>>,
}

/// Borrows a `WGPUStringView` as a Rust `&str`. Handles WebGPU's sentinel
/// values: `WGPU_STRLEN` (== `usize::MAX`) means "data is a C string, find
/// the NUL terminator yourself"; the empty view (null data, zero length) and
/// any other null-data case return `""`.
///
/// # Safety
/// The caller must ensure the underlying bytes outlive the returned reference
/// and contain valid UTF-8 (Skia/Dawn always produce UTF-8 strings here).
unsafe fn string_view_as_str<'a>(sv: sb::WGPUStringView) -> &'a str {
    if sv.data.is_null() {
        return "";
    }
    let len = if sv.length == usize::MAX {
        // WGPU_STRLEN sentinel — fall back to strlen.
        let mut n = 0;
        while *sv.data.add(n) != 0 {
            n += 1;
        }
        n
    } else {
        sv.length
    };
    if len == 0 {
        return "";
    }
    let slice = std::slice::from_raw_parts(sv.data as *const u8, len);
    std::str::from_utf8_unchecked(slice)
}

//
// Thunks
//

unsafe extern "C" fn create_instance(
    _descriptor: *const sb::WGPUInstanceDescriptor,
) -> sb::WGPUInstance {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    Resource::into_handle(InstanceData { inner: instance }) as sb::WGPUInstance
}

unsafe extern "C" fn instance_add_ref(handle: sb::WGPUInstance) {
    Resource::<InstanceData>::add_ref(handle as _);
}

unsafe extern "C" fn instance_release(handle: sb::WGPUInstance) {
    Resource::<InstanceData>::release(handle as _);
}

unsafe extern "C" fn instance_process_events(_handle: sb::WGPUInstance) {
    // wgpu drives event processing per-device, not per-instance. No-op here is
    // safe because the request_adapter / request_device thunks block on the
    // underlying futures synchronously, so there are no pending instance-level
    // events to drain.
}

unsafe extern "C" fn instance_wait_any(
    _instance: sb::WGPUInstance,
    future_count: usize,
    futures: *mut sb::WGPUFutureWaitInfo,
    _timeout_ns: u64,
) -> sb::WGPUWaitStatus {
    // Our request_adapter / request_device thunks invoke the C callbacks
    // synchronously before returning, so every WGPUFuture we hand out is
    // already complete. Mark them all done and return Success.
    if !futures.is_null() {
        let slice = std::slice::from_raw_parts_mut(futures, future_count);
        for wait in slice {
            wait.completed = 1;
        }
    }
    sb::WGPUWaitStatus::WGPUWaitStatus_Success
}

unsafe extern "C" fn instance_request_adapter(
    instance: sb::WGPUInstance,
    options: *const sb::WGPURequestAdapterOptions,
    callback_info: sb::WGPURequestAdapterCallbackInfo,
) -> sb::WGPUFuture {
    let instance_data = Resource::<InstanceData>::inner(instance as _);

    let power_preference = if options.is_null() {
        wgpu::PowerPreference::default()
    } else {
        match (*options).powerPreference {
            sb::WGPUPowerPreference::WGPUPowerPreference_LowPower => {
                wgpu::PowerPreference::LowPower
            }
            sb::WGPUPowerPreference::WGPUPowerPreference_HighPerformance => {
                wgpu::PowerPreference::HighPerformance
            }
            _ => wgpu::PowerPreference::None,
        }
    };
    let force_fallback = !options.is_null() && (*options).forceFallbackAdapter != 0;

    let request = instance_data.inner.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference,
        force_fallback_adapter: force_fallback,
        compatible_surface: None,
    });

    let (status, adapter_handle) = match pollster::block_on(request) {
        Ok(adapter) => {
            let handle = Resource::into_handle(AdapterData {
                inner: adapter,
                instance: instance_data.inner.clone(),
            }) as sb::WGPUAdapter;
            (
                sb::WGPURequestAdapterStatus::WGPURequestAdapterStatus_Success,
                handle,
            )
        }
        Err(_) => (
            sb::WGPURequestAdapterStatus::WGPURequestAdapterStatus_Unavailable,
            ptr::null_mut(),
        ),
    };

    if let Some(callback) = callback_info.callback {
        let message = sb::WGPUStringView {
            data: ptr::null(),
            length: 0,
        };
        callback(
            status,
            adapter_handle,
            message,
            callback_info.userdata1,
            callback_info.userdata2,
        );
    }

    sb::WGPUFuture { id: 0 }
}

unsafe extern "C" fn adapter_add_ref(handle: sb::WGPUAdapter) {
    Resource::<AdapterData>::add_ref(handle as _);
}

unsafe extern "C" fn adapter_release(handle: sb::WGPUAdapter) {
    Resource::<AdapterData>::release(handle as _);
}

unsafe extern "C" fn adapter_request_device(
    adapter: sb::WGPUAdapter,
    _descriptor: *const sb::WGPUDeviceDescriptor,
    callback_info: sb::WGPURequestDeviceCallbackInfo,
) -> sb::WGPUFuture {
    let adapter_data = Resource::<AdapterData>::inner(adapter as _);

    // For the initial Skia bring-up we ignore the caller's DeviceDescriptor
    // and request a device with wgpu's defaults. Mapping Dawn-flavoured
    // descriptors (features, limits, toggles) is a later step.
    let request = adapter_data
        .inner
        .request_device(&wgpu::DeviceDescriptor::default());

    let (status, device_handle) = match pollster::block_on(request) {
        Ok((device, queue)) => {
            let handle = Resource::into_handle(DeviceData {
                inner: device,
                queue,
                _adapter: adapter_data.inner.clone(),
                dummy_vertex_buffer: std::sync::OnceLock::new(),
            }) as sb::WGPUDevice;
            (
                sb::WGPURequestDeviceStatus::WGPURequestDeviceStatus_Success,
                handle,
            )
        }
        Err(_) => (
            sb::WGPURequestDeviceStatus::WGPURequestDeviceStatus_Error,
            ptr::null_mut(),
        ),
    };

    if let Some(callback) = callback_info.callback {
        let message = sb::WGPUStringView {
            data: ptr::null(),
            length: 0,
        };
        callback(
            status,
            device_handle,
            message,
            callback_info.userdata1,
            callback_info.userdata2,
        );
    }

    sb::WGPUFuture { id: 0 }
}

unsafe extern "C" fn device_add_ref(handle: sb::WGPUDevice) {
    Resource::<DeviceData>::add_ref(handle as _);
}

unsafe extern "C" fn device_release(handle: sb::WGPUDevice) {
    Resource::<DeviceData>::release(handle as _);
}

unsafe extern "C" fn device_get_queue(handle: sb::WGPUDevice) -> sb::WGPUQueue {
    let device_data = Resource::<DeviceData>::inner(handle as _);
    Resource::into_handle(QueueData {
        inner: device_data.queue.clone(),
        _device: device_data.inner.clone(),
    }) as sb::WGPUQueue
}

unsafe extern "C" fn queue_add_ref(handle: sb::WGPUQueue) {
    Resource::<QueueData>::add_ref(handle as _);
}

unsafe extern "C" fn queue_release(handle: sb::WGPUQueue) {
    Resource::<QueueData>::release(handle as _);
}

unsafe extern "C" fn device_create_shader_module(
    device: sb::WGPUDevice,
    descriptor: *const sb::WGPUShaderModuleDescriptor,
) -> sb::WGPUShaderModule {
    let device_data = Resource::<DeviceData>::inner(device as _);
    if descriptor.is_null() {
        return ptr::null_mut();
    }
    let desc = &*descriptor;

    // Walk the nextInChain looking for a WGSL source. We don't currently
    // support SPIR-V here because wgpu's SPIR-V feature is opt-in and Skia
    // typically generates WGSL.
    let mut wgsl: Option<&str> = None;
    let mut next = desc.nextInChain;
    while !next.is_null() {
        let chain = &*next;
        if chain.sType == sb::WGPUSType::WGPUSType_ShaderSourceWGSL {
            let wgsl_desc = next as *const sb::WGPUShaderSourceWGSL;
            wgsl = Some(string_view_as_str((*wgsl_desc).code));
            break;
        }
        next = chain.next;
    }
    let Some(wgsl) = wgsl else {
        eprintln!(
            "wgpu_backend: deviceCreateShaderModule received no recognised \
             source (only WGSL is supported); aborting"
        );
        std::process::abort();
    };

    let label = string_view_as_str(desc.label);
    let module = device_data.inner.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: if label.is_empty() { None } else { Some(label) },
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(wgsl)),
    });

    Resource::into_handle(ShaderModuleData {
        inner: module,
        _device: device_data.inner.clone(),
    }) as sb::WGPUShaderModule
}

unsafe extern "C" fn shader_module_add_ref(handle: sb::WGPUShaderModule) {
    Resource::<ShaderModuleData>::add_ref(handle as _);
}

unsafe extern "C" fn shader_module_release(handle: sb::WGPUShaderModule) {
    Resource::<ShaderModuleData>::release(handle as _);
}

unsafe extern "C" fn shader_module_set_label(
    _handle: sb::WGPUShaderModule,
    _label: sb::WGPUStringView,
) {
    // Labels are diagnostic and ignored.
}

//
// Render pipeline
//

fn convert_vertex_format(f: sb::WGPUVertexFormat) -> wgpu::VertexFormat {
    use sb::WGPUVertexFormat as W;
    use wgpu::VertexFormat as R;
    match f {
        W::WGPUVertexFormat_Uint8 => R::Uint8,
        W::WGPUVertexFormat_Uint8x2 => R::Uint8x2,
        W::WGPUVertexFormat_Uint8x4 => R::Uint8x4,
        W::WGPUVertexFormat_Sint8 => R::Sint8,
        W::WGPUVertexFormat_Sint8x2 => R::Sint8x2,
        W::WGPUVertexFormat_Sint8x4 => R::Sint8x4,
        W::WGPUVertexFormat_Unorm8 => R::Unorm8,
        W::WGPUVertexFormat_Unorm8x2 => R::Unorm8x2,
        W::WGPUVertexFormat_Unorm8x4 => R::Unorm8x4,
        W::WGPUVertexFormat_Snorm8 => R::Snorm8,
        W::WGPUVertexFormat_Snorm8x2 => R::Snorm8x2,
        W::WGPUVertexFormat_Snorm8x4 => R::Snorm8x4,
        W::WGPUVertexFormat_Uint16 => R::Uint16,
        W::WGPUVertexFormat_Uint16x2 => R::Uint16x2,
        W::WGPUVertexFormat_Uint16x4 => R::Uint16x4,
        W::WGPUVertexFormat_Sint16 => R::Sint16,
        W::WGPUVertexFormat_Sint16x2 => R::Sint16x2,
        W::WGPUVertexFormat_Sint16x4 => R::Sint16x4,
        W::WGPUVertexFormat_Unorm16 => R::Unorm16,
        W::WGPUVertexFormat_Unorm16x2 => R::Unorm16x2,
        W::WGPUVertexFormat_Unorm16x4 => R::Unorm16x4,
        W::WGPUVertexFormat_Snorm16 => R::Snorm16,
        W::WGPUVertexFormat_Snorm16x2 => R::Snorm16x2,
        W::WGPUVertexFormat_Snorm16x4 => R::Snorm16x4,
        W::WGPUVertexFormat_Float16 => R::Float16,
        W::WGPUVertexFormat_Float16x2 => R::Float16x2,
        W::WGPUVertexFormat_Float16x4 => R::Float16x4,
        W::WGPUVertexFormat_Float32 => R::Float32,
        W::WGPUVertexFormat_Float32x2 => R::Float32x2,
        W::WGPUVertexFormat_Float32x3 => R::Float32x3,
        W::WGPUVertexFormat_Float32x4 => R::Float32x4,
        W::WGPUVertexFormat_Uint32 => R::Uint32,
        W::WGPUVertexFormat_Uint32x2 => R::Uint32x2,
        W::WGPUVertexFormat_Uint32x3 => R::Uint32x3,
        W::WGPUVertexFormat_Uint32x4 => R::Uint32x4,
        W::WGPUVertexFormat_Sint32 => R::Sint32,
        W::WGPUVertexFormat_Sint32x2 => R::Sint32x2,
        W::WGPUVertexFormat_Sint32x3 => R::Sint32x3,
        W::WGPUVertexFormat_Sint32x4 => R::Sint32x4,
        _ => {
            eprintln!("wgpu_backend: unhandled WGPUVertexFormat {f:?}; falling back to Float32");
            R::Float32
        }
    }
}

fn convert_primitive_topology(t: sb::WGPUPrimitiveTopology) -> wgpu::PrimitiveTopology {
    use sb::WGPUPrimitiveTopology as W;
    match t {
        W::WGPUPrimitiveTopology_PointList => wgpu::PrimitiveTopology::PointList,
        W::WGPUPrimitiveTopology_LineList => wgpu::PrimitiveTopology::LineList,
        W::WGPUPrimitiveTopology_LineStrip => wgpu::PrimitiveTopology::LineStrip,
        W::WGPUPrimitiveTopology_TriangleStrip => wgpu::PrimitiveTopology::TriangleStrip,
        _ => wgpu::PrimitiveTopology::TriangleList,
    }
}

fn convert_index_format(f: sb::WGPUIndexFormat) -> Option<wgpu::IndexFormat> {
    match f {
        sb::WGPUIndexFormat::WGPUIndexFormat_Uint16 => Some(wgpu::IndexFormat::Uint16),
        sb::WGPUIndexFormat::WGPUIndexFormat_Uint32 => Some(wgpu::IndexFormat::Uint32),
        _ => None,
    }
}

fn convert_front_face(f: sb::WGPUFrontFace) -> wgpu::FrontFace {
    match f {
        sb::WGPUFrontFace::WGPUFrontFace_CW => wgpu::FrontFace::Cw,
        _ => wgpu::FrontFace::Ccw,
    }
}

fn convert_cull_mode(c: sb::WGPUCullMode) -> Option<wgpu::Face> {
    match c {
        sb::WGPUCullMode::WGPUCullMode_Front => Some(wgpu::Face::Front),
        sb::WGPUCullMode::WGPUCullMode_Back => Some(wgpu::Face::Back),
        _ => None,
    }
}

fn convert_step_mode(m: sb::WGPUVertexStepMode) -> wgpu::VertexStepMode {
    match m {
        sb::WGPUVertexStepMode::WGPUVertexStepMode_Instance => wgpu::VertexStepMode::Instance,
        _ => wgpu::VertexStepMode::Vertex,
    }
}

fn convert_blend_operation(o: sb::WGPUBlendOperation) -> wgpu::BlendOperation {
    use sb::WGPUBlendOperation as W;
    match o {
        W::WGPUBlendOperation_Subtract => wgpu::BlendOperation::Subtract,
        W::WGPUBlendOperation_ReverseSubtract => wgpu::BlendOperation::ReverseSubtract,
        W::WGPUBlendOperation_Min => wgpu::BlendOperation::Min,
        W::WGPUBlendOperation_Max => wgpu::BlendOperation::Max,
        _ => wgpu::BlendOperation::Add,
    }
}

fn convert_blend_factor(f: sb::WGPUBlendFactor) -> wgpu::BlendFactor {
    use sb::WGPUBlendFactor as W;
    use wgpu::BlendFactor as R;
    match f {
        W::WGPUBlendFactor_Zero => R::Zero,
        W::WGPUBlendFactor_One => R::One,
        W::WGPUBlendFactor_Src => R::Src,
        W::WGPUBlendFactor_OneMinusSrc => R::OneMinusSrc,
        W::WGPUBlendFactor_SrcAlpha => R::SrcAlpha,
        W::WGPUBlendFactor_OneMinusSrcAlpha => R::OneMinusSrcAlpha,
        W::WGPUBlendFactor_Dst => R::Dst,
        W::WGPUBlendFactor_OneMinusDst => R::OneMinusDst,
        W::WGPUBlendFactor_DstAlpha => R::DstAlpha,
        W::WGPUBlendFactor_OneMinusDstAlpha => R::OneMinusDstAlpha,
        W::WGPUBlendFactor_SrcAlphaSaturated => R::SrcAlphaSaturated,
        W::WGPUBlendFactor_Constant => R::Constant,
        W::WGPUBlendFactor_OneMinusConstant => R::OneMinusConstant,
        W::WGPUBlendFactor_Src1 => R::Src1,
        W::WGPUBlendFactor_OneMinusSrc1 => R::OneMinusSrc1,
        W::WGPUBlendFactor_Src1Alpha => R::Src1Alpha,
        W::WGPUBlendFactor_OneMinusSrc1Alpha => R::OneMinusSrc1Alpha,
        _ => R::One,
    }
}

fn convert_stencil_op(o: sb::WGPUStencilOperation) -> wgpu::StencilOperation {
    use sb::WGPUStencilOperation as W;
    use wgpu::StencilOperation as R;
    match o {
        W::WGPUStencilOperation_Zero => R::Zero,
        W::WGPUStencilOperation_Replace => R::Replace,
        W::WGPUStencilOperation_Invert => R::Invert,
        W::WGPUStencilOperation_IncrementClamp => R::IncrementClamp,
        W::WGPUStencilOperation_DecrementClamp => R::DecrementClamp,
        W::WGPUStencilOperation_IncrementWrap => R::IncrementWrap,
        W::WGPUStencilOperation_DecrementWrap => R::DecrementWrap,
        _ => R::Keep,
    }
}

fn convert_optional_bool(b: sb::WGPUOptionalBool) -> Option<bool> {
    match b {
        sb::WGPUOptionalBool::WGPUOptionalBool_True => Some(true),
        sb::WGPUOptionalBool::WGPUOptionalBool_False => Some(false),
        _ => None,
    }
}

fn convert_stencil_face(s: &sb::WGPUStencilFaceState) -> wgpu::StencilFaceState {
    wgpu::StencilFaceState {
        compare: convert_compare_function(s.compare).unwrap_or(wgpu::CompareFunction::Always),
        fail_op: convert_stencil_op(s.failOp),
        depth_fail_op: convert_stencil_op(s.depthFailOp),
        pass_op: convert_stencil_op(s.passOp),
    }
}

unsafe fn build_render_pipeline_descriptor<'a>(
    desc: &'a sb::WGPURenderPipelineDescriptor,
    label_str: &'a str,
    vertex_buffers: &'a [wgpu::VertexBufferLayout<'a>],
    color_targets: &'a [Option<wgpu::ColorTargetState>],
    fragment_entry: &'a str,
    vertex_entry: &'a str,
) -> wgpu::RenderPipelineDescriptor<'a> {
    let layout = if desc.layout.is_null() {
        None
    } else {
        Some(&Resource::<PipelineLayoutData>::inner(desc.layout as _).inner)
    };
    let vertex_module = &Resource::<ShaderModuleData>::inner(desc.vertex.module as _).inner;
    let fragment_module = if desc.fragment.is_null() {
        None
    } else {
        let f = &*desc.fragment;
        Some(&Resource::<ShaderModuleData>::inner(f.module as _).inner)
    };

    let depth_stencil = if desc.depthStencil.is_null() {
        None
    } else {
        let d = &*desc.depthStencil;
        Some(wgpu::DepthStencilState {
            format: convert_texture_format(d.format),
            depth_write_enabled: convert_optional_bool(d.depthWriteEnabled),
            depth_compare: convert_compare_function(d.depthCompare),
            stencil: wgpu::StencilState {
                front: convert_stencil_face(&d.stencilFront),
                back: convert_stencil_face(&d.stencilBack),
                read_mask: d.stencilReadMask,
                write_mask: d.stencilWriteMask,
            },
            bias: wgpu::DepthBiasState {
                constant: d.depthBias,
                slope_scale: d.depthBiasSlopeScale,
                clamp: d.depthBiasClamp,
            },
        })
    };

    wgpu::RenderPipelineDescriptor {
        label: if label_str.is_empty() { None } else { Some(label_str) },
        layout,
        vertex: wgpu::VertexState {
            module: vertex_module,
            entry_point: if vertex_entry.is_empty() {
                None
            } else {
                Some(vertex_entry)
            },
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: vertex_buffers,
        },
        primitive: wgpu::PrimitiveState {
            topology: convert_primitive_topology(desc.primitive.topology),
            strip_index_format: convert_index_format(desc.primitive.stripIndexFormat),
            front_face: convert_front_face(desc.primitive.frontFace),
            cull_mode: convert_cull_mode(desc.primitive.cullMode),
            unclipped_depth: desc.primitive.unclippedDepth != 0,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil,
        multisample: wgpu::MultisampleState {
            count: desc.multisample.count.max(1),
            mask: desc.multisample.mask as u64,
            alpha_to_coverage_enabled: desc.multisample.alphaToCoverageEnabled != 0,
        },
        fragment: fragment_module.map(|module| wgpu::FragmentState {
            module,
            entry_point: if fragment_entry.is_empty() {
                None
            } else {
                Some(fragment_entry)
            },
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: color_targets,
        }),
        cache: None,
        multiview_mask: None,
    }
}

unsafe fn create_render_pipeline_impl(
    device: sb::WGPUDevice,
    descriptor: *const sb::WGPURenderPipelineDescriptor,
) -> sb::WGPURenderPipeline {
    if descriptor.is_null() {
        return ptr::null_mut();
    }
    let device_data = Resource::<DeviceData>::inner(device as _);
    let desc = &*descriptor;
    let label_str = string_view_as_str(desc.label);
    let vertex_entry = string_view_as_str(desc.vertex.entryPoint);

    // Build owned vectors of attributes / buffer layouts so they live during
    // the create call.
    let buffer_slice = if desc.vertex.bufferCount == 0 || desc.vertex.buffers.is_null() {
        &[][..]
    } else {
        std::slice::from_raw_parts(desc.vertex.buffers, desc.vertex.bufferCount)
    };

    // Compute the largest shader_location actually used by Skia so our
    // dummy attribute (added below for empty layouts) doesn't collide with
    // a real one. wgpu's `max_vertex_attributes` limit is 16 on most
    // adapters and 8 on the WebGPU minimum — using `u32::MAX` is invalid,
    // so we pick a small number above what Skia uses.
    let mut max_real_location: u32 = 0;
    for b in buffer_slice {
        if b.attributeCount == 0 || b.attributes.is_null() {
            continue;
        }
        for a in std::slice::from_raw_parts(b.attributes, b.attributeCount) {
            if a.shaderLocation >= max_real_location {
                max_real_location = a.shaderLocation + 1;
            }
        }
    }

    let mut attribute_storage: Vec<Vec<wgpu::VertexAttribute>> =
        Vec::with_capacity(buffer_slice.len());
    let mut placeholder_slots = Vec::new();
    for (slot, b) in buffer_slice.iter().enumerate() {
        let attrs: Vec<wgpu::VertexAttribute> =
            if b.attributeCount == 0 || b.attributes.is_null() {
                Vec::new()
            } else {
                std::slice::from_raw_parts(b.attributes, b.attributeCount)
                    .iter()
                    .map(|a| wgpu::VertexAttribute {
                        format: convert_vertex_format(a.format),
                        offset: a.offset,
                        shader_location: a.shaderLocation,
                    })
                    .collect()
            };
        let is_placeholder = matches!(
            b.stepMode,
            sb::WGPUVertexStepMode::WGPUVertexStepMode_Undefined
        ) || (b.arrayStride == 0 && attrs.is_empty());
        if is_placeholder {
            placeholder_slots.push(slot as u32);
            // wgpu-core's create_render_pipeline drops VertexBufferLayouts
            // with no attributes. That renumbers every later binding and
            // makes Skia's set_vertex_buffer(slot=N) miss the pipeline's
            // binding N. Insert one dummy attribute so wgpu keeps the
            // slot. The dummy attribute reads from a u32 at offset 0; we
            // bind our zero-filled `dummy_vertex_buffer` to this slot, so
            // the data is a valid (but never read) zero.
            let mut attrs = attrs;
            attrs.push(wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Uint32,
                offset: 0,
                shader_location: max_real_location,
            });
            max_real_location += 1;
            attribute_storage.push(attrs);
        } else {
            attribute_storage.push(attrs);
        }
    }
    let vertex_buffers: Vec<wgpu::VertexBufferLayout> = buffer_slice
        .iter()
        .zip(attribute_storage.iter())
        .map(|(b, attrs)| {
            // For placeholder layouts give wgpu a non-zero stride so the
            // dummy attribute (Uint32 = 4 bytes) actually fits.
            let stride = if b.arrayStride == 0 && !attrs.is_empty() {
                4
            } else {
                b.arrayStride
            };
            wgpu::VertexBufferLayout {
                array_stride: stride,
                step_mode: convert_step_mode(b.stepMode),
                attributes: attrs.as_slice(),
            }
        })
        .collect();


    let (fragment_entry, color_targets) = if desc.fragment.is_null() {
        (String::new(), Vec::new())
    } else {
        let f = &*desc.fragment;
        let entry = string_view_as_str(f.entryPoint).to_string();
        let targets_slice = if f.targetCount == 0 || f.targets.is_null() {
            &[][..]
        } else {
            std::slice::from_raw_parts(f.targets, f.targetCount)
        };
        let targets: Vec<Option<wgpu::ColorTargetState>> = targets_slice
            .iter()
            .map(|t| {
                if t.format == sb::WGPUTextureFormat::WGPUTextureFormat_Undefined {
                    None
                } else {
                    let blend = if t.blend.is_null() {
                        None
                    } else {
                        let b = &*t.blend;
                        Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: convert_blend_factor(b.color.srcFactor),
                                dst_factor: convert_blend_factor(b.color.dstFactor),
                                operation: convert_blend_operation(b.color.operation),
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: convert_blend_factor(b.alpha.srcFactor),
                                dst_factor: convert_blend_factor(b.alpha.dstFactor),
                                operation: convert_blend_operation(b.alpha.operation),
                            },
                        })
                    };
                    Some(wgpu::ColorTargetState {
                        format: convert_texture_format(t.format),
                        blend,
                        write_mask: wgpu::ColorWrites::from_bits_truncate(t.writeMask as u32),
                    })
                }
            })
            .collect();
        (entry, targets)
    };

    let wgpu_desc = build_render_pipeline_descriptor(
        desc,
        label_str,
        &vertex_buffers,
        &color_targets,
        fragment_entry.as_str(),
        vertex_entry,
    );
    let pipeline = device_data.inner.create_render_pipeline(&wgpu_desc);
    Resource::into_handle(RenderPipelineData {
        inner: pipeline,
        placeholder_slots,
        device_handle: device,
        _device: device_data.inner.clone(),
    }) as sb::WGPURenderPipeline
}

unsafe extern "C" fn device_create_render_pipeline(
    device: sb::WGPUDevice,
    descriptor: *const sb::WGPURenderPipelineDescriptor,
) -> sb::WGPURenderPipeline {
    create_render_pipeline_impl(device, descriptor)
}

unsafe extern "C" fn device_create_render_pipeline_async(
    device: sb::WGPUDevice,
    descriptor: *const sb::WGPURenderPipelineDescriptor,
    callback_info: sb::WGPUCreateRenderPipelineAsyncCallbackInfo,
) -> sb::WGPUFuture {
    let pipeline = create_render_pipeline_impl(device, descriptor);
    let (status, msg) = if pipeline.is_null() {
        (
            sb::WGPUCreatePipelineAsyncStatus::WGPUCreatePipelineAsyncStatus_ValidationError,
            "create_render_pipeline returned null",
        )
    } else {
        (
            sb::WGPUCreatePipelineAsyncStatus::WGPUCreatePipelineAsyncStatus_Success,
            "",
        )
    };
    if let Some(callback) = callback_info.callback {
        let message_view = sb::WGPUStringView {
            data: if msg.is_empty() { ptr::null() } else { msg.as_ptr() as _ },
            length: msg.len(),
        };
        callback(
            status,
            pipeline,
            message_view,
            callback_info.userdata1,
            callback_info.userdata2,
        );
    }
    sb::WGPUFuture { id: 0 }
}

unsafe extern "C" fn render_pipeline_add_ref(handle: sb::WGPURenderPipeline) {
    Resource::<RenderPipelineData>::add_ref(handle as _);
}

unsafe extern "C" fn render_pipeline_release(handle: sb::WGPURenderPipeline) {
    Resource::<RenderPipelineData>::release(handle as _);
}

unsafe extern "C" fn render_pipeline_set_label(
    _handle: sb::WGPURenderPipeline,
    _label: sb::WGPUStringView,
) {
}

unsafe extern "C" fn device_create_bind_group(
    device: sb::WGPUDevice,
    descriptor: *const sb::WGPUBindGroupDescriptor,
) -> sb::WGPUBindGroup {
    if descriptor.is_null() {
        return ptr::null_mut();
    }
    let device_data = Resource::<DeviceData>::inner(device as _);
    let desc = &*descriptor;
    let label_str = string_view_as_str(desc.label);
    if desc.layout.is_null() {
        return ptr::null_mut();
    }
    let layout = &Resource::<BindGroupLayoutData>::inner(desc.layout as _).inner;

    let entries_slice = if desc.entryCount == 0 || desc.entries.is_null() {
        &[][..]
    } else {
        std::slice::from_raw_parts(desc.entries, desc.entryCount)
    };

    // Each binding has a tagged-union shape: at most one of buffer/sampler/
    // textureView is non-null.
    let mut rust_entries: Vec<wgpu::BindGroupEntry<'_>> = Vec::with_capacity(entries_slice.len());
    let mut buffer_bindings_storage: Vec<wgpu::BufferBinding<'_>> = Vec::new();
    // First pass: collect BufferBindings so we can borrow them in the entry vec.
    for entry in entries_slice {
        if !entry.buffer.is_null() {
            let buf = &Resource::<BufferData>::inner(entry.buffer as _).inner;
            // `WGPU_WHOLE_SIZE` == u64::MAX means "rest of buffer" — represent
            // that as None in wgpu's `BufferBinding::size`.
            let size_opt = if entry.size == 0 || entry.size == u64::MAX {
                None
            } else {
                core::num::NonZeroU64::new(entry.size)
            };
            buffer_bindings_storage.push(wgpu::BufferBinding {
                buffer: buf,
                offset: entry.offset,
                size: size_opt,
            });
        }
    }
    let mut buffer_iter = buffer_bindings_storage.iter();
    for entry in entries_slice {
        let resource = if !entry.buffer.is_null() {
            wgpu::BindingResource::Buffer(buffer_iter.next().unwrap().clone())
        } else if !entry.sampler.is_null() {
            wgpu::BindingResource::Sampler(
                &Resource::<SamplerData>::inner(entry.sampler as _).inner,
            )
        } else if !entry.textureView.is_null() {
            wgpu::BindingResource::TextureView(
                &Resource::<TextureViewData>::inner(entry.textureView as _).inner,
            )
        } else {
            continue;
        };
        rust_entries.push(wgpu::BindGroupEntry {
            binding: entry.binding,
            resource,
        });
    }

    let bind_group = device_data.inner.create_bind_group(&wgpu::BindGroupDescriptor {
        label: if label_str.is_empty() {
            None
        } else {
            Some(label_str)
        },
        layout,
        entries: &rust_entries,
    });
    Resource::into_handle(BindGroupData {
        inner: bind_group,
        _device: device_data.inner.clone(),
    }) as sb::WGPUBindGroup
}

unsafe extern "C" fn bind_group_add_ref(handle: sb::WGPUBindGroup) {
    Resource::<BindGroupData>::add_ref(handle as _);
}

unsafe extern "C" fn bind_group_release(handle: sb::WGPUBindGroup) {
    Resource::<BindGroupData>::release(handle as _);
}

unsafe extern "C" fn bind_group_set_label(_handle: sb::WGPUBindGroup, _label: sb::WGPUStringView) {}

//
// ComputePipeline
//

unsafe extern "C" fn device_create_compute_pipeline(
    device: sb::WGPUDevice,
    descriptor: *const sb::WGPUComputePipelineDescriptor,
) -> sb::WGPUComputePipeline {
    if descriptor.is_null() {
        return ptr::null_mut();
    }
    let device_data = Resource::<DeviceData>::inner(device as _);
    let desc = &*descriptor;
    let label_str = string_view_as_str(desc.label);
    let entry_str = string_view_as_str(desc.compute.entryPoint);

    let module = &Resource::<ShaderModuleData>::inner(desc.compute.module as _).inner;
    let layout = if desc.layout.is_null() {
        None
    } else {
        Some(&Resource::<PipelineLayoutData>::inner(desc.layout as _).inner)
    };

    let pipeline = device_data
        .inner
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: if label_str.is_empty() { None } else { Some(label_str) },
            layout,
            module,
            entry_point: if entry_str.is_empty() { None } else { Some(entry_str) },
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    Resource::into_handle(ComputePipelineData {
        inner: pipeline,
        _device: device_data.inner.clone(),
    }) as sb::WGPUComputePipeline
}

unsafe extern "C" fn device_create_compute_pipeline_async(
    device: sb::WGPUDevice,
    descriptor: *const sb::WGPUComputePipelineDescriptor,
    callback_info: sb::WGPUCreateComputePipelineAsyncCallbackInfo,
) -> sb::WGPUFuture {
    let pipeline = device_create_compute_pipeline(device, descriptor);
    if let Some(callback) = callback_info.callback {
        let status = if pipeline.is_null() {
            sb::WGPUCreatePipelineAsyncStatus::WGPUCreatePipelineAsyncStatus_ValidationError
        } else {
            sb::WGPUCreatePipelineAsyncStatus::WGPUCreatePipelineAsyncStatus_Success
        };
        let message = sb::WGPUStringView {
            data: ptr::null(),
            length: 0,
        };
        callback(
            status,
            pipeline,
            message,
            callback_info.userdata1,
            callback_info.userdata2,
        );
    }
    sb::WGPUFuture { id: 0 }
}

unsafe extern "C" fn compute_pipeline_add_ref(handle: sb::WGPUComputePipeline) {
    Resource::<ComputePipelineData>::add_ref(handle as _);
}

unsafe extern "C" fn compute_pipeline_release(handle: sb::WGPUComputePipeline) {
    Resource::<ComputePipelineData>::release(handle as _);
}

unsafe extern "C" fn compute_pipeline_set_label(
    _handle: sb::WGPUComputePipeline,
    _label: sb::WGPUStringView,
) {
}

unsafe extern "C" fn compute_pipeline_get_bind_group_layout(
    handle: sb::WGPUComputePipeline,
    group_index: u32,
) -> sb::WGPUBindGroupLayout {
    let pipeline_data = Resource::<ComputePipelineData>::inner(handle as _);
    let layout = pipeline_data.inner.get_bind_group_layout(group_index);
    Resource::into_handle(BindGroupLayoutData {
        inner: layout,
        _device: pipeline_data._device.clone(),
    }) as sb::WGPUBindGroupLayout
}

//
// ComputePassEncoder
//

unsafe extern "C" fn command_encoder_begin_compute_pass(
    encoder: sb::WGPUCommandEncoder,
    descriptor: *const sb::WGPUComputePassDescriptor,
) -> sb::WGPUComputePassEncoder {
    let encoder_data = Resource::<CommandEncoderData>::inner(encoder as _);
    let label_str = if descriptor.is_null() {
        ""
    } else {
        string_view_as_str((*descriptor).label)
    };
    let pass_static = {
        let mut guard = encoder_data.inner.lock().expect("encoder mutex poisoned");
        let Some(enc) = guard.as_mut() else {
            return ptr::null_mut();
        };
        let pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: if label_str.is_empty() { None } else { Some(label_str) },
            timestamp_writes: None,
        });
        pass.forget_lifetime()
    };
    Resource::into_handle(ComputePassEncoderData {
        inner: std::sync::Mutex::new(Some(pass_static)),
    }) as sb::WGPUComputePassEncoder
}

unsafe extern "C" fn compute_pass_encoder_add_ref(handle: sb::WGPUComputePassEncoder) {
    Resource::<ComputePassEncoderData>::add_ref(handle as _);
}

unsafe extern "C" fn compute_pass_encoder_release(handle: sb::WGPUComputePassEncoder) {
    Resource::<ComputePassEncoderData>::release(handle as _);
}

unsafe extern "C" fn compute_pass_encoder_set_label(
    _handle: sb::WGPUComputePassEncoder,
    _label: sb::WGPUStringView,
) {
}

fn with_compute_pass<F>(handle: sb::WGPUComputePassEncoder, f: F)
where
    F: FnOnce(&mut wgpu::ComputePass<'static>),
{
    if handle.is_null() {
        return;
    }
    let data = unsafe { Resource::<ComputePassEncoderData>::inner(handle as _) };
    if let Ok(mut guard) = data.inner.lock() {
        if let Some(pass) = guard.as_mut() {
            f(pass);
        }
    }
}

unsafe extern "C" fn compute_pass_encoder_end(handle: sb::WGPUComputePassEncoder) {
    if handle.is_null() {
        return;
    }
    let data = Resource::<ComputePassEncoderData>::inner(handle as _);
    if let Ok(mut guard) = data.inner.lock() {
        // Dropping the ComputePass encodes the End command into the parent
        // CommandEncoder.
        let _ = guard.take();
    }
}

unsafe extern "C" fn compute_pass_encoder_set_pipeline(
    handle: sb::WGPUComputePassEncoder,
    pipeline: sb::WGPUComputePipeline,
) {
    if pipeline.is_null() {
        return;
    }
    let pipeline_inner: *const wgpu::ComputePipeline =
        &Resource::<ComputePipelineData>::inner(pipeline as _).inner;
    with_compute_pass(handle, |pass| {
        pass.set_pipeline(&*pipeline_inner);
    });
}

unsafe extern "C" fn compute_pass_encoder_set_bind_group(
    handle: sb::WGPUComputePassEncoder,
    group_index: u32,
    bind_group: sb::WGPUBindGroup,
    offset_count: usize,
    offsets: *const u32,
) {
    let bind_group_ptr: Option<*const wgpu::BindGroup> = if bind_group.is_null() {
        None
    } else {
        Some(&Resource::<BindGroupData>::inner(bind_group as _).inner)
    };
    let offsets_slice = if offset_count == 0 || offsets.is_null() {
        &[][..]
    } else {
        std::slice::from_raw_parts(offsets, offset_count)
    };
    with_compute_pass(handle, |pass| match bind_group_ptr {
        Some(p) => pass.set_bind_group(group_index, &*p, offsets_slice),
        None => pass.set_bind_group(group_index, None, offsets_slice),
    });
}

unsafe extern "C" fn compute_pass_encoder_dispatch_workgroups(
    handle: sb::WGPUComputePassEncoder,
    x: u32,
    y: u32,
    z: u32,
) {
    with_compute_pass(handle, |pass| {
        pass.dispatch_workgroups(x, y, z);
    });
}

unsafe extern "C" fn compute_pass_encoder_dispatch_workgroups_indirect(
    handle: sb::WGPUComputePassEncoder,
    indirect_buffer: sb::WGPUBuffer,
    indirect_offset: u64,
) {
    if indirect_buffer.is_null() {
        return;
    }
    let buf_ptr: *const wgpu::Buffer =
        &Resource::<BufferData>::inner(indirect_buffer as _).inner;
    with_compute_pass(handle, |pass| {
        pass.dispatch_workgroups_indirect(&*buf_ptr, indirect_offset);
    });
}

unsafe extern "C" fn compute_pass_encoder_insert_debug_marker(
    _handle: sb::WGPUComputePassEncoder,
    _label: sb::WGPUStringView,
) {
}

unsafe extern "C" fn compute_pass_encoder_push_debug_group(
    _handle: sb::WGPUComputePassEncoder,
    _label: sb::WGPUStringView,
) {
}

unsafe extern "C" fn compute_pass_encoder_pop_debug_group(_handle: sb::WGPUComputePassEncoder) {}

//
// queueWriteTexture
//

unsafe extern "C" fn queue_write_texture(
    queue: sb::WGPUQueue,
    destination: *const sb::WGPUTexelCopyTextureInfo,
    data: *const core::ffi::c_void,
    data_size: usize,
    data_layout: *const sb::WGPUTexelCopyBufferLayout,
    write_size: *const sb::WGPUExtent3D,
) {
    if destination.is_null() || data.is_null() || data_layout.is_null() || write_size.is_null() {
        return;
    }
    let queue_data = Resource::<QueueData>::inner(queue as _);
    let dst = convert_texel_copy_texture(&*destination);
    let layout = &*data_layout;
    let size = *write_size;
    let slice = std::slice::from_raw_parts(data as *const u8, data_size);
    queue_data.inner.write_texture(
        dst,
        slice,
        wgpu::TexelCopyBufferLayout {
            offset: layout.offset,
            bytes_per_row: if layout.bytesPerRow == u32::MAX {
                None
            } else {
                Some(layout.bytesPerRow)
            },
            rows_per_image: if layout.rowsPerImage == u32::MAX {
                None
            } else {
                Some(layout.rowsPerImage)
            },
        },
        wgpu::Extent3d {
            width: size.width,
            height: size.height,
            depth_or_array_layers: size.depthOrArrayLayers,
        },
    );
}

//
// RenderBundle
//

unsafe extern "C" fn device_create_render_bundle_encoder(
    device: sb::WGPUDevice,
    descriptor: *const sb::WGPURenderBundleEncoderDescriptor,
) -> sb::WGPURenderBundleEncoder {
    if descriptor.is_null() {
        return ptr::null_mut();
    }
    let device_data = Resource::<DeviceData>::inner(device as _);
    let desc = &*descriptor;
    let label_str = string_view_as_str(desc.label);

    let color_formats: Vec<Option<wgpu::TextureFormat>> = if desc.colorFormatCount == 0
        || desc.colorFormats.is_null()
    {
        Vec::new()
    } else {
        std::slice::from_raw_parts(desc.colorFormats, desc.colorFormatCount)
            .iter()
            .map(|f| {
                if *f == sb::WGPUTextureFormat::WGPUTextureFormat_Undefined {
                    None
                } else {
                    Some(convert_texture_format(*f))
                }
            })
            .collect()
    };
    let depth_stencil = if desc.depthStencilFormat
        == sb::WGPUTextureFormat::WGPUTextureFormat_Undefined
    {
        None
    } else {
        Some(wgpu::RenderBundleDepthStencil {
            format: convert_texture_format(desc.depthStencilFormat),
            depth_read_only: desc.depthReadOnly != 0,
            stencil_read_only: desc.stencilReadOnly != 0,
        })
    };

    let encoder = device_data
        .inner
        .create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
            label: if label_str.is_empty() { None } else { Some(label_str) },
            color_formats: &color_formats,
            depth_stencil,
            sample_count: desc.sampleCount.max(1),
            multiview: None,
        });
    Resource::into_handle(RenderBundleEncoderData {
        inner: std::sync::Mutex::new(Some(encoder)),
        _device: device_data.inner.clone(),
    }) as sb::WGPURenderBundleEncoder
}

unsafe extern "C" fn render_bundle_encoder_add_ref(handle: sb::WGPURenderBundleEncoder) {
    Resource::<RenderBundleEncoderData>::add_ref(handle as _);
}

unsafe extern "C" fn render_bundle_encoder_release(handle: sb::WGPURenderBundleEncoder) {
    Resource::<RenderBundleEncoderData>::release(handle as _);
}

unsafe extern "C" fn render_bundle_encoder_set_label(
    _handle: sb::WGPURenderBundleEncoder,
    _label: sb::WGPUStringView,
) {
}

unsafe extern "C" fn render_bundle_encoder_finish(
    handle: sb::WGPURenderBundleEncoder,
    descriptor: *const sb::WGPURenderBundleDescriptor,
) -> sb::WGPURenderBundle {
    let encoder_data = Resource::<RenderBundleEncoderData>::inner(handle as _);
    let Some(encoder) = encoder_data.inner.lock().ok().and_then(|mut g| g.take()) else {
        return ptr::null_mut();
    };
    let label_str = if descriptor.is_null() {
        String::new()
    } else {
        string_view_as_str((*descriptor).label).to_string()
    };
    let bundle = encoder.finish(&wgpu::RenderBundleDescriptor {
        label: if label_str.is_empty() {
            None
        } else {
            Some(label_str.as_str())
        },
    });
    Resource::into_handle(RenderBundleData {
        inner: bundle,
        _device: encoder_data._device.clone(),
    }) as sb::WGPURenderBundle
}

unsafe extern "C" fn render_bundle_add_ref(handle: sb::WGPURenderBundle) {
    Resource::<RenderBundleData>::add_ref(handle as _);
}

unsafe extern "C" fn render_bundle_release(handle: sb::WGPURenderBundle) {
    Resource::<RenderBundleData>::release(handle as _);
}

unsafe extern "C" fn render_bundle_set_label(
    _handle: sb::WGPURenderBundle,
    _label: sb::WGPUStringView,
) {
}

//
// QuerySet
//

unsafe extern "C" fn device_create_query_set(
    device: sb::WGPUDevice,
    descriptor: *const sb::WGPUQuerySetDescriptor,
) -> sb::WGPUQuerySet {
    if descriptor.is_null() {
        return ptr::null_mut();
    }
    let device_data = Resource::<DeviceData>::inner(device as _);
    let desc = &*descriptor;
    let label_str = string_view_as_str(desc.label);
    let ty = match desc.type_ {
        sb::WGPUQueryType::WGPUQueryType_Timestamp => wgpu::QueryType::Timestamp,
        _ => wgpu::QueryType::Occlusion,
    };
    let query_set = device_data.inner.create_query_set(&wgpu::QuerySetDescriptor {
        label: if label_str.is_empty() { None } else { Some(label_str) },
        ty,
        count: desc.count,
    });
    Resource::into_handle(QuerySetData {
        inner: query_set,
        count: desc.count,
        ty: desc.type_,
        _device: device_data.inner.clone(),
    }) as sb::WGPUQuerySet
}

unsafe extern "C" fn query_set_add_ref(handle: sb::WGPUQuerySet) {
    Resource::<QuerySetData>::add_ref(handle as _);
}

unsafe extern "C" fn query_set_release(handle: sb::WGPUQuerySet) {
    Resource::<QuerySetData>::release(handle as _);
}

unsafe extern "C" fn query_set_destroy(_handle: sb::WGPUQuerySet) {}

unsafe extern "C" fn query_set_get_count(handle: sb::WGPUQuerySet) -> u32 {
    Resource::<QuerySetData>::inner(handle as _).count
}

unsafe extern "C" fn query_set_get_type(handle: sb::WGPUQuerySet) -> sb::WGPUQueryType {
    Resource::<QuerySetData>::inner(handle as _).ty
}

unsafe extern "C" fn query_set_set_label(_handle: sb::WGPUQuerySet, _label: sb::WGPUStringView) {}

unsafe extern "C" fn render_pipeline_get_bind_group_layout(
    handle: sb::WGPURenderPipeline,
    group_index: u32,
) -> sb::WGPUBindGroupLayout {
    let pipeline_data = Resource::<RenderPipelineData>::inner(handle as _);
    let layout = pipeline_data.inner.get_bind_group_layout(group_index);
    Resource::into_handle(BindGroupLayoutData {
        inner: layout,
        _device: pipeline_data._device.clone(),
    }) as sb::WGPUBindGroupLayout
}

unsafe extern "C" fn shader_module_get_compilation_info(
    _handle: sb::WGPUShaderModule,
    callback_info: sb::WGPUCompilationInfoCallbackInfo,
) -> sb::WGPUFuture {
    // wgpu reports shader compile errors via the result of `create_shader_module`
    // (or its async completion), so by the time Skia asks for compilation info
    // we've already accepted the shader. Report Success with no messages.
    if let Some(callback) = callback_info.callback {
        let info = sb::WGPUCompilationInfo {
            nextInChain: ptr::null_mut(),
            messageCount: 0,
            messages: ptr::null(),
        };
        callback(
            sb::WGPUCompilationInfoRequestStatus::WGPUCompilationInfoRequestStatus_Success,
            &info,
            callback_info.userdata1,
            callback_info.userdata2,
        );
    }
    sb::WGPUFuture { id: 0 }
}

//
// Adapter / Device introspection
//

unsafe extern "C" fn device_get_adapter(handle: sb::WGPUDevice) -> sb::WGPUAdapter {
    let device_data = Resource::<DeviceData>::inner(handle as _);
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    Resource::into_handle(AdapterData {
        inner: device_data._adapter.clone(),
        instance,
    }) as sb::WGPUAdapter
}

unsafe extern "C" fn adapter_get_instance(handle: sb::WGPUAdapter) -> sb::WGPUInstance {
    let adapter_data = Resource::<AdapterData>::inner(handle as _);
    Resource::into_handle(InstanceData {
        inner: adapter_data.instance.clone(),
    }) as sb::WGPUInstance
}

unsafe extern "C" fn adapter_get_info(
    handle: sb::WGPUAdapter,
    out: *mut sb::WGPUAdapterInfo,
) -> sb::WGPUStatus {
    if out.is_null() {
        return sb::WGPUStatus::WGPUStatus_Error;
    }
    let adapter = &Resource::<AdapterData>::inner(handle as _).inner;
    let info = adapter.get_info();

    let out = &mut *out;
    out.nextInChain = ptr::null_mut();
    out.vendor = sb::WGPUStringView {
        data: ptr::null(),
        length: 0,
    };
    out.architecture = sb::WGPUStringView {
        data: ptr::null(),
        length: 0,
    };
    out.device = sb::WGPUStringView {
        data: ptr::null(),
        length: 0,
    };
    out.description = sb::WGPUStringView {
        data: ptr::null(),
        length: 0,
    };
    out.backendType = match info.backend {
        wgpu::Backend::Vulkan => sb::WGPUBackendType::WGPUBackendType_Vulkan,
        wgpu::Backend::Metal => sb::WGPUBackendType::WGPUBackendType_Metal,
        wgpu::Backend::Dx12 => sb::WGPUBackendType::WGPUBackendType_D3D12,
        wgpu::Backend::Gl => sb::WGPUBackendType::WGPUBackendType_OpenGL,
        wgpu::Backend::BrowserWebGpu => sb::WGPUBackendType::WGPUBackendType_WebGPU,
        wgpu::Backend::Noop => sb::WGPUBackendType::WGPUBackendType_Null,
    };
    out.adapterType = match info.device_type {
        wgpu::DeviceType::DiscreteGpu => sb::WGPUAdapterType::WGPUAdapterType_DiscreteGPU,
        wgpu::DeviceType::IntegratedGpu => sb::WGPUAdapterType::WGPUAdapterType_IntegratedGPU,
        wgpu::DeviceType::Cpu => sb::WGPUAdapterType::WGPUAdapterType_CPU,
        wgpu::DeviceType::VirtualGpu | wgpu::DeviceType::Other => {
            sb::WGPUAdapterType::WGPUAdapterType_Unknown
        }
    };
    out.vendorID = info.vendor;
    out.deviceID = info.device;
    out.subgroupMinSize = 0;
    out.subgroupMaxSize = 0;
    sb::WGPUStatus::WGPUStatus_Success
}

unsafe extern "C" fn adapter_info_free_members(_info: sb::WGPUAdapterInfo) {
    // No-op: we hand out null/empty string views with no allocated storage.
}

fn write_limits(out: &mut sb::WGPULimits, limits: wgpu::Limits) {
    out.nextInChain = ptr::null_mut();
    out.maxTextureDimension1D = limits.max_texture_dimension_1d;
    out.maxTextureDimension2D = limits.max_texture_dimension_2d;
    out.maxTextureDimension3D = limits.max_texture_dimension_3d;
    out.maxTextureArrayLayers = limits.max_texture_array_layers;
    out.maxBindGroups = limits.max_bind_groups;
    out.maxBindGroupsPlusVertexBuffers =
        limits.max_bind_groups + limits.max_vertex_buffers;
    out.maxBindingsPerBindGroup = limits.max_bindings_per_bind_group;
    out.maxDynamicUniformBuffersPerPipelineLayout =
        limits.max_dynamic_uniform_buffers_per_pipeline_layout;
    out.maxDynamicStorageBuffersPerPipelineLayout =
        limits.max_dynamic_storage_buffers_per_pipeline_layout;
    out.maxSampledTexturesPerShaderStage = limits.max_sampled_textures_per_shader_stage;
    out.maxSamplersPerShaderStage = limits.max_samplers_per_shader_stage;
    out.maxStorageBuffersPerShaderStage = limits.max_storage_buffers_per_shader_stage;
    out.maxStorageTexturesPerShaderStage = limits.max_storage_textures_per_shader_stage;
    out.maxUniformBuffersPerShaderStage = limits.max_uniform_buffers_per_shader_stage;
    out.maxUniformBufferBindingSize = limits.max_uniform_buffer_binding_size as u64;
    out.maxStorageBufferBindingSize = limits.max_storage_buffer_binding_size as u64;
    out.minUniformBufferOffsetAlignment = limits.min_uniform_buffer_offset_alignment;
    out.minStorageBufferOffsetAlignment = limits.min_storage_buffer_offset_alignment;
    out.maxVertexBuffers = limits.max_vertex_buffers;
    out.maxBufferSize = limits.max_buffer_size;
    out.maxVertexAttributes = limits.max_vertex_attributes;
    out.maxVertexBufferArrayStride = limits.max_vertex_buffer_array_stride;
    out.maxInterStageShaderVariables = limits.max_inter_stage_shader_variables;
    out.maxColorAttachments = limits.max_color_attachments;
    out.maxColorAttachmentBytesPerSample = limits.max_color_attachment_bytes_per_sample;
    out.maxComputeWorkgroupStorageSize = limits.max_compute_workgroup_storage_size;
    out.maxComputeInvocationsPerWorkgroup = limits.max_compute_invocations_per_workgroup;
    out.maxComputeWorkgroupSizeX = limits.max_compute_workgroup_size_x;
    out.maxComputeWorkgroupSizeY = limits.max_compute_workgroup_size_y;
    out.maxComputeWorkgroupSizeZ = limits.max_compute_workgroup_size_z;
    out.maxComputeWorkgroupsPerDimension = limits.max_compute_workgroups_per_dimension;
    out.maxImmediateSize = 0;
}

unsafe extern "C" fn device_get_limits(
    handle: sb::WGPUDevice,
    out: *mut sb::WGPULimits,
) -> sb::WGPUStatus {
    if out.is_null() {
        return sb::WGPUStatus::WGPUStatus_Error;
    }
    let device = &Resource::<DeviceData>::inner(handle as _).inner;
    write_limits(&mut *out, device.limits());
    sb::WGPUStatus::WGPUStatus_Success
}

unsafe extern "C" fn adapter_get_limits(
    handle: sb::WGPUAdapter,
    out: *mut sb::WGPULimits,
) -> sb::WGPUStatus {
    if out.is_null() {
        return sb::WGPUStatus::WGPUStatus_Error;
    }
    let adapter = &Resource::<AdapterData>::inner(handle as _).inner;
    write_limits(&mut *out, adapter.limits());
    sb::WGPUStatus::WGPUStatus_Success
}

unsafe extern "C" fn device_has_feature(
    _handle: sb::WGPUDevice,
    _feature: sb::WGPUFeatureName,
) -> sb::WGPUBool {
    0
}

unsafe extern "C" fn adapter_has_feature(
    _handle: sb::WGPUAdapter,
    _feature: sb::WGPUFeatureName,
) -> sb::WGPUBool {
    0
}

unsafe extern "C" fn device_get_features(
    _handle: sb::WGPUDevice,
    out: *mut sb::WGPUSupportedFeatures,
) {
    if !out.is_null() {
        let out = &mut *out;
        out.featureCount = 0;
        out.features = ptr::null();
    }
}

unsafe extern "C" fn adapter_get_features(
    _handle: sb::WGPUAdapter,
    out: *mut sb::WGPUSupportedFeatures,
) {
    if !out.is_null() {
        let out = &mut *out;
        out.featureCount = 0;
        out.features = ptr::null();
    }
}

unsafe extern "C" fn supported_features_free_members(_features: sb::WGPUSupportedFeatures) {
    // No-op: we never allocated the feature list.
}

unsafe extern "C" fn device_tick(_handle: sb::WGPUDevice) {
    // wgpu drives polling through `Device::poll`; for now, accept ticks as no-ops.
}

unsafe extern "C" fn device_set_logging_callback(
    _handle: sb::WGPUDevice,
    _callback_info: sb::WGPULoggingCallbackInfo,
) {
    // Ignored — wgpu has its own logging story.
}

unsafe extern "C" fn device_set_label(_handle: sb::WGPUDevice, _label: sb::WGPUStringView) {}

unsafe extern "C" fn adapter_get_format_capabilities(
    _handle: sb::WGPUAdapter,
    _format: sb::WGPUTextureFormat,
    _capabilities: *mut sb::WGPUDawnFormatCapabilities,
) -> sb::WGPUStatus {
    // No supported capabilities advertised. Skia uses this to probe Dawn-specific
    // format features; returning Error makes it fall back to defaults.
    sb::WGPUStatus::WGPUStatus_Error
}

//
// Format / dimension helpers
//

fn convert_view_dimension(d: sb::WGPUTextureViewDimension) -> wgpu::TextureViewDimension {
    use sb::WGPUTextureViewDimension as W;
    match d {
        W::WGPUTextureViewDimension_1D => wgpu::TextureViewDimension::D1,
        W::WGPUTextureViewDimension_2D => wgpu::TextureViewDimension::D2,
        W::WGPUTextureViewDimension_2DArray => wgpu::TextureViewDimension::D2Array,
        W::WGPUTextureViewDimension_Cube => wgpu::TextureViewDimension::Cube,
        W::WGPUTextureViewDimension_CubeArray => wgpu::TextureViewDimension::CubeArray,
        W::WGPUTextureViewDimension_3D => wgpu::TextureViewDimension::D3,
        // Undefined / unknown: treat as D2 (most common default).
        _ => wgpu::TextureViewDimension::D2,
    }
}

/// Maps the WebGPU C `WGPUTextureFormat` enum to wgpu's `TextureFormat`. Covers
/// the formats Graphite/Skia is known to request; falls back to RGBA8 for
/// anything else with a stderr note, since most uncommon formats would lead to
/// pipeline-creation failures further down the line anyway.
fn convert_texture_format(f: sb::WGPUTextureFormat) -> wgpu::TextureFormat {
    use sb::WGPUTextureFormat as W;
    use wgpu::TextureFormat as R;
    match f {
        W::WGPUTextureFormat_R8Unorm => R::R8Unorm,
        W::WGPUTextureFormat_R8Snorm => R::R8Snorm,
        W::WGPUTextureFormat_R8Uint => R::R8Uint,
        W::WGPUTextureFormat_R8Sint => R::R8Sint,
        W::WGPUTextureFormat_R16Uint => R::R16Uint,
        W::WGPUTextureFormat_R16Sint => R::R16Sint,
        W::WGPUTextureFormat_R16Float => R::R16Float,
        W::WGPUTextureFormat_RG8Unorm => R::Rg8Unorm,
        W::WGPUTextureFormat_RG8Snorm => R::Rg8Snorm,
        W::WGPUTextureFormat_RG8Uint => R::Rg8Uint,
        W::WGPUTextureFormat_RG8Sint => R::Rg8Sint,
        W::WGPUTextureFormat_R32Float => R::R32Float,
        W::WGPUTextureFormat_R32Uint => R::R32Uint,
        W::WGPUTextureFormat_R32Sint => R::R32Sint,
        W::WGPUTextureFormat_RG16Uint => R::Rg16Uint,
        W::WGPUTextureFormat_RG16Sint => R::Rg16Sint,
        W::WGPUTextureFormat_RG16Float => R::Rg16Float,
        W::WGPUTextureFormat_RGBA8Unorm => R::Rgba8Unorm,
        W::WGPUTextureFormat_RGBA8UnormSrgb => R::Rgba8UnormSrgb,
        W::WGPUTextureFormat_RGBA8Snorm => R::Rgba8Snorm,
        W::WGPUTextureFormat_RGBA8Uint => R::Rgba8Uint,
        W::WGPUTextureFormat_RGBA8Sint => R::Rgba8Sint,
        W::WGPUTextureFormat_BGRA8Unorm => R::Bgra8Unorm,
        W::WGPUTextureFormat_BGRA8UnormSrgb => R::Bgra8UnormSrgb,
        W::WGPUTextureFormat_RG32Float => R::Rg32Float,
        W::WGPUTextureFormat_RG32Uint => R::Rg32Uint,
        W::WGPUTextureFormat_RG32Sint => R::Rg32Sint,
        W::WGPUTextureFormat_RGBA16Uint => R::Rgba16Uint,
        W::WGPUTextureFormat_RGBA16Sint => R::Rgba16Sint,
        W::WGPUTextureFormat_RGBA16Float => R::Rgba16Float,
        W::WGPUTextureFormat_RGBA32Float => R::Rgba32Float,
        W::WGPUTextureFormat_RGBA32Uint => R::Rgba32Uint,
        W::WGPUTextureFormat_RGBA32Sint => R::Rgba32Sint,
        W::WGPUTextureFormat_Depth16Unorm => R::Depth16Unorm,
        W::WGPUTextureFormat_Depth24Plus => R::Depth24Plus,
        W::WGPUTextureFormat_Depth24PlusStencil8 => R::Depth24PlusStencil8,
        W::WGPUTextureFormat_Depth32Float => R::Depth32Float,
        W::WGPUTextureFormat_Depth32FloatStencil8 => R::Depth32FloatStencil8,
        W::WGPUTextureFormat_Stencil8 => R::Stencil8,
        other => {
            eprintln!(
                "wgpu_backend: unhandled WGPUTextureFormat {other:?}; falling back to Rgba8Unorm"
            );
            R::Rgba8Unorm
        }
    }
}

//
// BindGroupLayout
//

unsafe extern "C" fn device_create_bind_group_layout(
    device: sb::WGPUDevice,
    descriptor: *const sb::WGPUBindGroupLayoutDescriptor,
) -> sb::WGPUBindGroupLayout {
    let device_data = Resource::<DeviceData>::inner(device as _);
    if descriptor.is_null() {
        return ptr::null_mut();
    }
    let desc = &*descriptor;

    let entries_slice = if desc.entryCount == 0 || desc.entries.is_null() {
        &[][..]
    } else {
        std::slice::from_raw_parts(desc.entries, desc.entryCount)
    };

    let mut rust_entries = Vec::with_capacity(entries_slice.len());
    for entry in entries_slice {
        let visibility = wgpu::ShaderStages::from_bits_truncate(entry.visibility as u32);
        let count = if entry.bindingArraySize > 1 {
            core::num::NonZeroU32::new(entry.bindingArraySize)
        } else {
            None
        };

        use sb::WGPUBufferBindingType as Buf;
        use sb::WGPUSamplerBindingType as Smp;
        use sb::WGPUStorageTextureAccess as Stg;
        use sb::WGPUTextureSampleType as Tex;

        let ty = if entry.buffer.type_ != Buf::WGPUBufferBindingType_BindingNotUsed
            && entry.buffer.type_ != Buf::WGPUBufferBindingType_Undefined
        {
            let buf_ty = match entry.buffer.type_ {
                Buf::WGPUBufferBindingType_Storage => {
                    wgpu::BufferBindingType::Storage { read_only: false }
                }
                Buf::WGPUBufferBindingType_ReadOnlyStorage => {
                    wgpu::BufferBindingType::Storage { read_only: true }
                }
                _ => wgpu::BufferBindingType::Uniform,
            };
            wgpu::BindingType::Buffer {
                ty: buf_ty,
                has_dynamic_offset: entry.buffer.hasDynamicOffset != 0,
                min_binding_size: core::num::NonZeroU64::new(entry.buffer.minBindingSize),
            }
        } else if entry.sampler.type_ != Smp::WGPUSamplerBindingType_BindingNotUsed
            && entry.sampler.type_ != Smp::WGPUSamplerBindingType_Undefined
        {
            wgpu::BindingType::Sampler(match entry.sampler.type_ {
                Smp::WGPUSamplerBindingType_NonFiltering => wgpu::SamplerBindingType::NonFiltering,
                Smp::WGPUSamplerBindingType_Comparison => wgpu::SamplerBindingType::Comparison,
                _ => wgpu::SamplerBindingType::Filtering,
            })
        } else if entry.texture.sampleType != Tex::WGPUTextureSampleType_BindingNotUsed
            && entry.texture.sampleType != Tex::WGPUTextureSampleType_Undefined
        {
            wgpu::BindingType::Texture {
                sample_type: match entry.texture.sampleType {
                    Tex::WGPUTextureSampleType_UnfilterableFloat => {
                        wgpu::TextureSampleType::Float { filterable: false }
                    }
                    Tex::WGPUTextureSampleType_Depth => wgpu::TextureSampleType::Depth,
                    Tex::WGPUTextureSampleType_Sint => wgpu::TextureSampleType::Sint,
                    Tex::WGPUTextureSampleType_Uint => wgpu::TextureSampleType::Uint,
                    _ => wgpu::TextureSampleType::Float { filterable: true },
                },
                view_dimension: convert_view_dimension(entry.texture.viewDimension),
                multisampled: entry.texture.multisampled != 0,
            }
        } else if entry.storageTexture.access != Stg::WGPUStorageTextureAccess_BindingNotUsed
            && entry.storageTexture.access != Stg::WGPUStorageTextureAccess_Undefined
        {
            wgpu::BindingType::StorageTexture {
                access: match entry.storageTexture.access {
                    Stg::WGPUStorageTextureAccess_ReadOnly => wgpu::StorageTextureAccess::ReadOnly,
                    Stg::WGPUStorageTextureAccess_ReadWrite => {
                        wgpu::StorageTextureAccess::ReadWrite
                    }
                    _ => wgpu::StorageTextureAccess::WriteOnly,
                },
                format: convert_texture_format(entry.storageTexture.format),
                view_dimension: convert_view_dimension(entry.storageTexture.viewDimension),
            }
        } else {
            // No active binding type — skip rather than poison the layout.
            continue;
        };

        rust_entries.push(wgpu::BindGroupLayoutEntry {
            binding: entry.binding,
            visibility,
            ty,
            count,
        });
    }

    let label = string_view_as_str(desc.label);
    let layout = device_data
        .inner
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: if label.is_empty() { None } else { Some(label) },
            entries: &rust_entries,
        });

    Resource::into_handle(BindGroupLayoutData {
        inner: layout,
        _device: device_data.inner.clone(),
    }) as sb::WGPUBindGroupLayout
}

unsafe extern "C" fn bind_group_layout_add_ref(handle: sb::WGPUBindGroupLayout) {
    Resource::<BindGroupLayoutData>::add_ref(handle as _);
}

unsafe extern "C" fn bind_group_layout_release(handle: sb::WGPUBindGroupLayout) {
    Resource::<BindGroupLayoutData>::release(handle as _);
}

unsafe extern "C" fn bind_group_layout_set_label(
    _handle: sb::WGPUBindGroupLayout,
    _label: sb::WGPUStringView,
) {
}

//
// Sampler
//

fn convert_address_mode(m: sb::WGPUAddressMode) -> wgpu::AddressMode {
    use sb::WGPUAddressMode as W;
    match m {
        W::WGPUAddressMode_Repeat => wgpu::AddressMode::Repeat,
        W::WGPUAddressMode_MirrorRepeat => wgpu::AddressMode::MirrorRepeat,
        _ => wgpu::AddressMode::ClampToEdge,
    }
}

fn convert_filter_mode(m: sb::WGPUFilterMode) -> wgpu::FilterMode {
    match m {
        sb::WGPUFilterMode::WGPUFilterMode_Linear => wgpu::FilterMode::Linear,
        _ => wgpu::FilterMode::Nearest,
    }
}

fn convert_mipmap_filter(m: sb::WGPUMipmapFilterMode) -> wgpu::MipmapFilterMode {
    match m {
        sb::WGPUMipmapFilterMode::WGPUMipmapFilterMode_Linear => wgpu::MipmapFilterMode::Linear,
        _ => wgpu::MipmapFilterMode::Nearest,
    }
}

fn convert_compare_function(c: sb::WGPUCompareFunction) -> Option<wgpu::CompareFunction> {
    use sb::WGPUCompareFunction as W;
    Some(match c {
        W::WGPUCompareFunction_Never => wgpu::CompareFunction::Never,
        W::WGPUCompareFunction_Less => wgpu::CompareFunction::Less,
        W::WGPUCompareFunction_Equal => wgpu::CompareFunction::Equal,
        W::WGPUCompareFunction_LessEqual => wgpu::CompareFunction::LessEqual,
        W::WGPUCompareFunction_Greater => wgpu::CompareFunction::Greater,
        W::WGPUCompareFunction_NotEqual => wgpu::CompareFunction::NotEqual,
        W::WGPUCompareFunction_GreaterEqual => wgpu::CompareFunction::GreaterEqual,
        W::WGPUCompareFunction_Always => wgpu::CompareFunction::Always,
        _ => return None,
    })
}

unsafe extern "C" fn device_create_sampler(
    device: sb::WGPUDevice,
    descriptor: *const sb::WGPUSamplerDescriptor,
) -> sb::WGPUSampler {
    let device_data = Resource::<DeviceData>::inner(device as _);
    let (label, desc) = if descriptor.is_null() {
        ("", None)
    } else {
        let d = &*descriptor;
        (string_view_as_str(d.label), Some(d))
    };

    let wgpu_desc = match desc {
        Some(d) => wgpu::SamplerDescriptor {
            label: if label.is_empty() { None } else { Some(label) },
            address_mode_u: convert_address_mode(d.addressModeU),
            address_mode_v: convert_address_mode(d.addressModeV),
            address_mode_w: convert_address_mode(d.addressModeW),
            mag_filter: convert_filter_mode(d.magFilter),
            min_filter: convert_filter_mode(d.minFilter),
            mipmap_filter: convert_mipmap_filter(d.mipmapFilter),
            lod_min_clamp: d.lodMinClamp,
            lod_max_clamp: d.lodMaxClamp,
            compare: convert_compare_function(d.compare),
            anisotropy_clamp: d.maxAnisotropy.max(1),
            border_color: None,
        },
        None => wgpu::SamplerDescriptor::default(),
    };

    let sampler = device_data.inner.create_sampler(&wgpu_desc);
    Resource::into_handle(SamplerData {
        inner: sampler,
        _device: device_data.inner.clone(),
    }) as sb::WGPUSampler
}

unsafe extern "C" fn sampler_add_ref(handle: sb::WGPUSampler) {
    Resource::<SamplerData>::add_ref(handle as _);
}

unsafe extern "C" fn sampler_release(handle: sb::WGPUSampler) {
    Resource::<SamplerData>::release(handle as _);
}

unsafe extern "C" fn sampler_set_label(_handle: sb::WGPUSampler, _label: sb::WGPUStringView) {}

//
// Buffer
//

unsafe extern "C" fn device_create_buffer(
    device: sb::WGPUDevice,
    descriptor: *const sb::WGPUBufferDescriptor,
) -> sb::WGPUBuffer {
    let device_data = Resource::<DeviceData>::inner(device as _);
    if descriptor.is_null() {
        return ptr::null_mut();
    }
    let d = &*descriptor;
    let label = string_view_as_str(d.label);
    let usage = wgpu::BufferUsages::from_bits_truncate(d.usage as u32);
    let buffer = device_data.inner.create_buffer(&wgpu::BufferDescriptor {
        label: if label.is_empty() { None } else { Some(label) },
        size: d.size,
        usage,
        mapped_at_creation: d.mappedAtCreation != 0,
    });
    Resource::into_handle(BufferData {
        inner: buffer,
        _device: device_data.inner.clone(),
        active_views_mut: std::sync::Mutex::new(Vec::new()),
        active_views_const: std::sync::Mutex::new(Vec::new()),
    }) as sb::WGPUBuffer
}

unsafe extern "C" fn buffer_add_ref(handle: sb::WGPUBuffer) {
    Resource::<BufferData>::add_ref(handle as _);
}

unsafe extern "C" fn buffer_release(handle: sb::WGPUBuffer) {
    Resource::<BufferData>::release(handle as _);
}

unsafe extern "C" fn buffer_destroy(handle: sb::WGPUBuffer) {
    if handle.is_null() {
        return;
    }
    Resource::<BufferData>::inner(handle as _).inner.destroy();
}

unsafe extern "C" fn buffer_get_size(handle: sb::WGPUBuffer) -> u64 {
    Resource::<BufferData>::inner(handle as _).inner.size()
}

unsafe extern "C" fn buffer_get_usage(handle: sb::WGPUBuffer) -> sb::WGPUBufferUsage {
    Resource::<BufferData>::inner(handle as _).inner.usage().bits() as sb::WGPUBufferUsage
}

unsafe extern "C" fn buffer_set_label(_handle: sb::WGPUBuffer, _label: sb::WGPUStringView) {}

fn buffer_slice_range(buffer: &wgpu::Buffer, offset: usize, size: usize) -> (u64, u64) {
    let start = offset as u64;
    let end = if size == usize::MAX || (offset == 0 && size == 0) {
        buffer.size()
    } else {
        start.saturating_add(size as u64).min(buffer.size())
    };
    (start, end)
}

unsafe extern "C" fn buffer_get_mapped_range(
    handle: sb::WGPUBuffer,
    offset: usize,
    size: usize,
) -> *mut core::ffi::c_void {
    let buffer_data = Resource::<BufferData>::inner(handle as _);
    let (start, end) = buffer_slice_range(&buffer_data.inner, offset, size);
    let mut view = buffer_data.inner.slice(start..end).get_mapped_range_mut();
    let ptr = view.slice(..).as_raw_element_ptr().as_ptr() as *mut core::ffi::c_void;
    // Keep the view alive until unmap — see BufferData docstring for why.
    if let Ok(mut active) = buffer_data.active_views_mut.lock() {
        active.push(view);
    }
    ptr
}

unsafe extern "C" fn buffer_get_const_mapped_range(
    handle: sb::WGPUBuffer,
    offset: usize,
    size: usize,
) -> *const core::ffi::c_void {
    let buffer_data = Resource::<BufferData>::inner(handle as _);
    let (start, end) = buffer_slice_range(&buffer_data.inner, offset, size);
    let view = buffer_data.inner.slice(start..end).get_mapped_range();
    let ptr = (*view).as_ptr() as *const core::ffi::c_void;
    if let Ok(mut active) = buffer_data.active_views_const.lock() {
        active.push(view);
    }
    ptr
}

unsafe extern "C" fn buffer_unmap(handle: sb::WGPUBuffer) {
    if handle.is_null() {
        return;
    }
    let buffer_data = Resource::<BufferData>::inner(handle as _);
    // Drop any active views first so wgpu's map_context tracking is clean
    // before we call unmap (which would otherwise reject "views still
    // accessible").
    if let Ok(mut active) = buffer_data.active_views_mut.lock() {
        active.clear();
    }
    if let Ok(mut active) = buffer_data.active_views_const.lock() {
        active.clear();
    }
    buffer_data.inner.unmap();
}

unsafe extern "C" fn buffer_get_map_state(_handle: sb::WGPUBuffer) -> sb::WGPUBufferMapState {
    // wgpu 29 doesn't yet expose Buffer::map_state(). Returning Unmapped is
    // conservative — Skia uses this to decide whether to call unmap during
    // cleanup; reporting "not mapped" just skips a no-op unmap.
    sb::WGPUBufferMapState::WGPUBufferMapState_Unmapped
}

unsafe extern "C" fn buffer_map_async(
    handle: sb::WGPUBuffer,
    mode: sb::WGPUMapMode,
    offset: usize,
    size: usize,
    callback_info: sb::WGPUBufferMapCallbackInfo,
) -> sb::WGPUFuture {
    let buffer_data = Resource::<BufferData>::inner(handle as _);
    let buffer = &buffer_data.inner;
    let (start, end) = buffer_slice_range(buffer, offset, size);
    // WGPUMapMode is a bitflag (Dawn's webgpu.h: Read = 0x1, Write = 0x2).
    let map_mode = if mode & 0x1 != 0 {
        wgpu::MapMode::Read
    } else {
        wgpu::MapMode::Write
    };

    // Bridge the async map to a synchronous block_on by stashing the result
    // in a one-shot channel that the callback fills from wgpu's thread.
    let (tx, rx) = std::sync::mpsc::channel();
    buffer
        .slice(start..end)
        .map_async(map_mode, move |result| {
            let _ = tx.send(result);
        });

    // Drive submitted GPU work to completion so the map callback can fire.
    buffer_data
        ._device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .ok();

    let status = match rx.recv().unwrap_or(Err(wgpu::BufferAsyncError)) {
        Ok(()) => sb::WGPUMapAsyncStatus::WGPUMapAsyncStatus_Success,
        Err(_) => sb::WGPUMapAsyncStatus::WGPUMapAsyncStatus_Error,
    };

    if let Some(callback) = callback_info.callback {
        let message = sb::WGPUStringView {
            data: ptr::null(),
            length: 0,
        };
        callback(
            status,
            message,
            callback_info.userdata1,
            callback_info.userdata2,
        );
    }

    sb::WGPUFuture { id: 0 }
}

//
// queueWriteBuffer
//

unsafe extern "C" fn queue_write_buffer(
    queue: sb::WGPUQueue,
    buffer: sb::WGPUBuffer,
    offset: u64,
    data: *const core::ffi::c_void,
    size: usize,
) {
    let queue_data = Resource::<QueueData>::inner(queue as _);
    let buffer_data = Resource::<BufferData>::inner(buffer as _);
    if data.is_null() || size == 0 {
        return;
    }
    let slice = std::slice::from_raw_parts(data as *const u8, size);
    queue_data.inner.write_buffer(&buffer_data.inner, offset, slice);
}

unsafe extern "C" fn queue_submit(
    queue: sb::WGPUQueue,
    command_count: usize,
    commands: *const sb::WGPUCommandBuffer,
) {
    let queue_data = Resource::<QueueData>::inner(queue as _);
    if command_count == 0 || commands.is_null() {
        let _ = queue_data.inner.submit(std::iter::empty());
        return;
    }
    let handles = std::slice::from_raw_parts(commands, command_count);
    let buffers: Vec<wgpu::CommandBuffer> = handles
        .iter()
        .filter_map(|h| {
            if h.is_null() {
                None
            } else {
                Resource::<CommandBufferData>::inner(*h as _)
                    .inner
                    .lock()
                    .ok()
                    .and_then(|mut g| g.take())
            }
        })
        .collect();
    let _ = queue_data.inner.submit(buffers);
}

//
// CommandEncoder
//

unsafe extern "C" fn device_create_command_encoder(
    device: sb::WGPUDevice,
    descriptor: *const sb::WGPUCommandEncoderDescriptor,
) -> sb::WGPUCommandEncoder {
    let device_data = Resource::<DeviceData>::inner(device as _);
    let label_str = if descriptor.is_null() {
        ""
    } else {
        string_view_as_str((*descriptor).label)
    };
    let encoder = device_data
        .inner
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: if label_str.is_empty() {
                None
            } else {
                Some(label_str)
            },
        });
    Resource::into_handle(CommandEncoderData {
        inner: std::sync::Mutex::new(Some(encoder)),
        _device: device_data.inner.clone(),
    }) as sb::WGPUCommandEncoder
}

unsafe extern "C" fn command_encoder_add_ref(handle: sb::WGPUCommandEncoder) {
    Resource::<CommandEncoderData>::add_ref(handle as _);
}

unsafe extern "C" fn command_encoder_release(handle: sb::WGPUCommandEncoder) {
    Resource::<CommandEncoderData>::release(handle as _);
}

unsafe extern "C" fn command_encoder_set_label(
    _handle: sb::WGPUCommandEncoder,
    _label: sb::WGPUStringView,
) {
}

unsafe extern "C" fn command_encoder_finish(
    handle: sb::WGPUCommandEncoder,
    _descriptor: *const sb::WGPUCommandBufferDescriptor,
) -> sb::WGPUCommandBuffer {
    let encoder_data = Resource::<CommandEncoderData>::inner(handle as _);
    let Some(encoder) = encoder_data.inner.lock().ok().and_then(|mut g| g.take()) else {
        return ptr::null_mut();
    };
    let buffer = encoder.finish();
    Resource::into_handle(CommandBufferData {
        inner: std::sync::Mutex::new(Some(buffer)),
        _device: encoder_data._device.clone(),
    }) as sb::WGPUCommandBuffer
}

unsafe extern "C" fn command_encoder_insert_debug_marker(
    _handle: sb::WGPUCommandEncoder,
    _label: sb::WGPUStringView,
) {
}

unsafe extern "C" fn command_encoder_push_debug_group(
    _handle: sb::WGPUCommandEncoder,
    _label: sb::WGPUStringView,
) {
}

unsafe extern "C" fn command_encoder_pop_debug_group(_handle: sb::WGPUCommandEncoder) {}

fn with_encoder<F>(encoder: sb::WGPUCommandEncoder, f: F)
where
    F: FnOnce(&mut wgpu::CommandEncoder),
{
    if encoder.is_null() {
        return;
    }
    let data = unsafe { Resource::<CommandEncoderData>::inner(encoder as _) };
    if let Ok(mut guard) = data.inner.lock() {
        if let Some(enc) = guard.as_mut() {
            f(enc);
        }
    }
}

unsafe extern "C" fn command_encoder_copy_buffer_to_buffer(
    encoder: sb::WGPUCommandEncoder,
    source: sb::WGPUBuffer,
    source_offset: u64,
    destination: sb::WGPUBuffer,
    destination_offset: u64,
    size: u64,
) {
    let src = &Resource::<BufferData>::inner(source as _).inner;
    let dst = &Resource::<BufferData>::inner(destination as _).inner;
    with_encoder(encoder, |enc| {
        enc.copy_buffer_to_buffer(src, source_offset, dst, destination_offset, Some(size));
    });
}

fn convert_texture_aspect(a: sb::WGPUTextureAspect) -> wgpu::TextureAspect {
    match a {
        sb::WGPUTextureAspect::WGPUTextureAspect_StencilOnly => wgpu::TextureAspect::StencilOnly,
        sb::WGPUTextureAspect::WGPUTextureAspect_DepthOnly => wgpu::TextureAspect::DepthOnly,
        _ => wgpu::TextureAspect::All,
    }
}

unsafe fn convert_texel_copy_texture(
    info: &sb::WGPUTexelCopyTextureInfo,
) -> wgpu::TexelCopyTextureInfo<'_> {
    let texture = &Resource::<TextureData>::inner(info.texture as _).inner;
    wgpu::TexelCopyTextureInfo {
        texture,
        mip_level: info.mipLevel,
        origin: wgpu::Origin3d {
            x: info.origin.x,
            y: info.origin.y,
            z: info.origin.z,
        },
        aspect: convert_texture_aspect(info.aspect),
    }
}

unsafe fn convert_texel_copy_buffer(
    info: &sb::WGPUTexelCopyBufferInfo,
) -> wgpu::TexelCopyBufferInfo<'_> {
    let buffer = &Resource::<BufferData>::inner(info.buffer as _).inner;
    wgpu::TexelCopyBufferInfo {
        buffer,
        layout: wgpu::TexelCopyBufferLayout {
            offset: info.layout.offset,
            bytes_per_row: if info.layout.bytesPerRow == u32::MAX {
                None
            } else {
                Some(info.layout.bytesPerRow)
            },
            rows_per_image: if info.layout.rowsPerImage == u32::MAX {
                None
            } else {
                Some(info.layout.rowsPerImage)
            },
        },
    }
}

unsafe extern "C" fn command_encoder_copy_texture_to_buffer(
    encoder: sb::WGPUCommandEncoder,
    source: *const sb::WGPUTexelCopyTextureInfo,
    destination: *const sb::WGPUTexelCopyBufferInfo,
    copy_size: *const sb::WGPUExtent3D,
) {
    if source.is_null() || destination.is_null() || copy_size.is_null() {
        return;
    }
    let src = convert_texel_copy_texture(&*source);
    let dst = convert_texel_copy_buffer(&*destination);
    let size = *copy_size;
    with_encoder(encoder, |enc| {
        enc.copy_texture_to_buffer(
            src,
            dst,
            wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: size.depthOrArrayLayers,
            },
        );
    });
}

unsafe extern "C" fn command_encoder_copy_buffer_to_texture(
    encoder: sb::WGPUCommandEncoder,
    source: *const sb::WGPUTexelCopyBufferInfo,
    destination: *const sb::WGPUTexelCopyTextureInfo,
    copy_size: *const sb::WGPUExtent3D,
) {
    if source.is_null() || destination.is_null() || copy_size.is_null() {
        return;
    }
    let src = convert_texel_copy_buffer(&*source);
    let dst = convert_texel_copy_texture(&*destination);
    let size = *copy_size;
    with_encoder(encoder, |enc| {
        enc.copy_buffer_to_texture(
            src,
            dst,
            wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: size.depthOrArrayLayers,
            },
        );
    });
}

unsafe extern "C" fn command_encoder_copy_texture_to_texture(
    encoder: sb::WGPUCommandEncoder,
    source: *const sb::WGPUTexelCopyTextureInfo,
    destination: *const sb::WGPUTexelCopyTextureInfo,
    copy_size: *const sb::WGPUExtent3D,
) {
    if source.is_null() || destination.is_null() || copy_size.is_null() {
        return;
    }
    let src = convert_texel_copy_texture(&*source);
    let dst = convert_texel_copy_texture(&*destination);
    let size = *copy_size;
    with_encoder(encoder, |enc| {
        enc.copy_texture_to_texture(
            src,
            dst,
            wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: size.depthOrArrayLayers,
            },
        );
    });
}

unsafe extern "C" fn command_encoder_clear_buffer(
    encoder: sb::WGPUCommandEncoder,
    buffer: sb::WGPUBuffer,
    offset: u64,
    size: u64,
) {
    let buf = &Resource::<BufferData>::inner(buffer as _).inner;
    with_encoder(encoder, |enc| {
        enc.clear_buffer(buf, offset, if size == u64::MAX { None } else { Some(size) });
    });
}

unsafe extern "C" fn command_buffer_add_ref(handle: sb::WGPUCommandBuffer) {
    Resource::<CommandBufferData>::add_ref(handle as _);
}

unsafe extern "C" fn command_buffer_release(handle: sb::WGPUCommandBuffer) {
    Resource::<CommandBufferData>::release(handle as _);
}

unsafe extern "C" fn command_buffer_set_label(
    _handle: sb::WGPUCommandBuffer,
    _label: sb::WGPUStringView,
) {
}

fn convert_texture_dimension(d: sb::WGPUTextureDimension) -> wgpu::TextureDimension {
    match d {
        sb::WGPUTextureDimension::WGPUTextureDimension_1D => wgpu::TextureDimension::D1,
        sb::WGPUTextureDimension::WGPUTextureDimension_3D => wgpu::TextureDimension::D3,
        _ => wgpu::TextureDimension::D2,
    }
}

unsafe extern "C" fn device_create_texture(
    device: sb::WGPUDevice,
    descriptor: *const sb::WGPUTextureDescriptor,
) -> sb::WGPUTexture {
    let device_data = Resource::<DeviceData>::inner(device as _);
    if descriptor.is_null() {
        return ptr::null_mut();
    }
    let d = &*descriptor;
    let label = string_view_as_str(d.label);

    let view_formats_slice: &[sb::WGPUTextureFormat] = if d.viewFormatCount == 0
        || d.viewFormats.is_null()
    {
        &[]
    } else {
        std::slice::from_raw_parts(d.viewFormats, d.viewFormatCount)
    };
    let view_formats: Vec<wgpu::TextureFormat> = view_formats_slice
        .iter()
        .map(|f| convert_texture_format(*f))
        .collect();

    let texture = device_data.inner.create_texture(&wgpu::TextureDescriptor {
        label: if label.is_empty() { None } else { Some(label) },
        size: wgpu::Extent3d {
            width: d.size.width,
            height: d.size.height,
            depth_or_array_layers: d.size.depthOrArrayLayers,
        },
        mip_level_count: d.mipLevelCount,
        sample_count: d.sampleCount,
        dimension: convert_texture_dimension(d.dimension),
        format: convert_texture_format(d.format),
        usage: wgpu::TextureUsages::from_bits_truncate(d.usage as u32),
        view_formats: &view_formats,
    });

    Resource::into_handle(TextureData {
        inner: texture,
        _device: device_data.inner.clone(),
    }) as sb::WGPUTexture
}

unsafe extern "C" fn texture_add_ref(handle: sb::WGPUTexture) {
    Resource::<TextureData>::add_ref(handle as _);
}

unsafe extern "C" fn texture_release(handle: sb::WGPUTexture) {
    Resource::<TextureData>::release(handle as _);
}

unsafe extern "C" fn texture_destroy(handle: sb::WGPUTexture) {
    if handle.is_null() {
        return;
    }
    Resource::<TextureData>::inner(handle as _).inner.destroy();
}

unsafe extern "C" fn texture_set_label(_handle: sb::WGPUTexture, _label: sb::WGPUStringView) {}

unsafe extern "C" fn texture_get_width(handle: sb::WGPUTexture) -> u32 {
    Resource::<TextureData>::inner(handle as _).inner.width()
}

unsafe extern "C" fn texture_get_height(handle: sb::WGPUTexture) -> u32 {
    Resource::<TextureData>::inner(handle as _).inner.height()
}

unsafe extern "C" fn texture_get_depth_or_array_layers(handle: sb::WGPUTexture) -> u32 {
    Resource::<TextureData>::inner(handle as _)
        .inner
        .depth_or_array_layers()
}

unsafe extern "C" fn texture_get_mip_level_count(handle: sb::WGPUTexture) -> u32 {
    Resource::<TextureData>::inner(handle as _).inner.mip_level_count()
}

unsafe extern "C" fn texture_get_sample_count(handle: sb::WGPUTexture) -> u32 {
    Resource::<TextureData>::inner(handle as _).inner.sample_count()
}

unsafe extern "C" fn texture_get_dimension(handle: sb::WGPUTexture) -> sb::WGPUTextureDimension {
    match Resource::<TextureData>::inner(handle as _).inner.dimension() {
        wgpu::TextureDimension::D1 => sb::WGPUTextureDimension::WGPUTextureDimension_1D,
        wgpu::TextureDimension::D2 => sb::WGPUTextureDimension::WGPUTextureDimension_2D,
        wgpu::TextureDimension::D3 => sb::WGPUTextureDimension::WGPUTextureDimension_3D,
    }
}

unsafe extern "C" fn texture_get_usage(handle: sb::WGPUTexture) -> sb::WGPUTextureUsage {
    Resource::<TextureData>::inner(handle as _).inner.usage().bits() as sb::WGPUTextureUsage
}

unsafe extern "C" fn texture_get_format(handle: sb::WGPUTexture) -> sb::WGPUTextureFormat {
    let format = Resource::<TextureData>::inner(handle as _).inner.format();
    // Inverse mapping for the common cases. We hand back Undefined for
    // anything we don't reverse-map; Skia's code paths that consume this
    // mostly compare against a small set of formats it cares about.
    convert_texture_format_back(format)
}

fn convert_texture_format_back(f: wgpu::TextureFormat) -> sb::WGPUTextureFormat {
    use wgpu::TextureFormat as R;
    use sb::WGPUTextureFormat as W;
    match f {
        R::R8Unorm => W::WGPUTextureFormat_R8Unorm,
        R::R8Snorm => W::WGPUTextureFormat_R8Snorm,
        R::R8Uint => W::WGPUTextureFormat_R8Uint,
        R::R8Sint => W::WGPUTextureFormat_R8Sint,
        R::R16Float => W::WGPUTextureFormat_R16Float,
        R::Rg8Unorm => W::WGPUTextureFormat_RG8Unorm,
        R::R32Float => W::WGPUTextureFormat_R32Float,
        R::Rg16Float => W::WGPUTextureFormat_RG16Float,
        R::Rgba8Unorm => W::WGPUTextureFormat_RGBA8Unorm,
        R::Rgba8UnormSrgb => W::WGPUTextureFormat_RGBA8UnormSrgb,
        R::Bgra8Unorm => W::WGPUTextureFormat_BGRA8Unorm,
        R::Bgra8UnormSrgb => W::WGPUTextureFormat_BGRA8UnormSrgb,
        R::Rg32Float => W::WGPUTextureFormat_RG32Float,
        R::Rgba16Float => W::WGPUTextureFormat_RGBA16Float,
        R::Rgba32Float => W::WGPUTextureFormat_RGBA32Float,
        R::Depth16Unorm => W::WGPUTextureFormat_Depth16Unorm,
        R::Depth24Plus => W::WGPUTextureFormat_Depth24Plus,
        R::Depth24PlusStencil8 => W::WGPUTextureFormat_Depth24PlusStencil8,
        R::Depth32Float => W::WGPUTextureFormat_Depth32Float,
        R::Depth32FloatStencil8 => W::WGPUTextureFormat_Depth32FloatStencil8,
        R::Stencil8 => W::WGPUTextureFormat_Stencil8,
        _ => W::WGPUTextureFormat_Undefined,
    }
}

unsafe extern "C" fn texture_create_view(
    handle: sb::WGPUTexture,
    descriptor: *const sb::WGPUTextureViewDescriptor,
) -> sb::WGPUTextureView {
    let texture_data = Resource::<TextureData>::inner(handle as _);
    let desc = if descriptor.is_null() {
        wgpu::TextureViewDescriptor::default()
    } else {
        let d = &*descriptor;
        let label = string_view_as_str(d.label);
        wgpu::TextureViewDescriptor {
            label: if label.is_empty() { None } else { Some(label) },
            format: if d.format == sb::WGPUTextureFormat::WGPUTextureFormat_Undefined {
                None
            } else {
                Some(convert_texture_format(d.format))
            },
            dimension: if d.dimension == sb::WGPUTextureViewDimension::WGPUTextureViewDimension_Undefined {
                None
            } else {
                Some(convert_view_dimension(d.dimension))
            },
            usage: None,
            aspect: wgpu::TextureAspect::All,
            base_mip_level: d.baseMipLevel,
            // WGPU_MIP_LEVEL_COUNT_UNDEFINED == u32::MAX -> "all remaining levels".
            mip_level_count: if d.mipLevelCount == 0 || d.mipLevelCount == u32::MAX {
                None
            } else {
                Some(d.mipLevelCount)
            },
            base_array_layer: d.baseArrayLayer,
            // WGPU_ARRAY_LAYER_COUNT_UNDEFINED == u32::MAX -> "all remaining layers".
            array_layer_count: if d.arrayLayerCount == 0 || d.arrayLayerCount == u32::MAX {
                None
            } else {
                Some(d.arrayLayerCount)
            },
        }
    };
    let view = texture_data.inner.create_view(&desc);
    Resource::into_handle(TextureViewData {
        inner: view,
        _texture: texture_data.inner.clone(),
    }) as sb::WGPUTextureView
}

unsafe extern "C" fn texture_view_add_ref(handle: sb::WGPUTextureView) {
    Resource::<TextureViewData>::add_ref(handle as _);
}

unsafe extern "C" fn texture_view_release(handle: sb::WGPUTextureView) {
    Resource::<TextureViewData>::release(handle as _);
}

unsafe extern "C" fn texture_view_set_label(
    _handle: sb::WGPUTextureView,
    _label: sb::WGPUStringView,
) {
}

//
// RenderPass
//

fn convert_load_op(op: sb::WGPULoadOp, clear: sb::WGPUColor) -> wgpu::LoadOp<wgpu::Color> {
    match op {
        sb::WGPULoadOp::WGPULoadOp_Load => wgpu::LoadOp::Load,
        // Clear and any unrecognised op fall through to Clear with the
        // caller-supplied value.
        _ => wgpu::LoadOp::Clear(wgpu::Color {
            r: clear.r,
            g: clear.g,
            b: clear.b,
            a: clear.a,
        }),
    }
}

fn convert_depth_load_op(op: sb::WGPULoadOp, clear: f32) -> wgpu::LoadOp<f32> {
    match op {
        sb::WGPULoadOp::WGPULoadOp_Load => wgpu::LoadOp::Load,
        _ => wgpu::LoadOp::Clear(clear),
    }
}

fn convert_stencil_load_op(op: sb::WGPULoadOp, clear: u32) -> wgpu::LoadOp<u32> {
    match op {
        sb::WGPULoadOp::WGPULoadOp_Load => wgpu::LoadOp::Load,
        _ => wgpu::LoadOp::Clear(clear),
    }
}

fn convert_store_op(op: sb::WGPUStoreOp) -> wgpu::StoreOp {
    match op {
        sb::WGPUStoreOp::WGPUStoreOp_Discard => wgpu::StoreOp::Discard,
        _ => wgpu::StoreOp::Store,
    }
}

unsafe extern "C" fn command_encoder_begin_render_pass(
    encoder: sb::WGPUCommandEncoder,
    descriptor: *const sb::WGPURenderPassDescriptor,
) -> sb::WGPURenderPassEncoder {
    let encoder_data = Resource::<CommandEncoderData>::inner(encoder as _);
    if descriptor.is_null() {
        return ptr::null_mut();
    }
    let desc = &*descriptor;
    let label = string_view_as_str(desc.label);

    let color_slice = if desc.colorAttachmentCount == 0 || desc.colorAttachments.is_null() {
        &[][..]
    } else {
        std::slice::from_raw_parts(desc.colorAttachments, desc.colorAttachmentCount)
    };
    let color_attachments: Vec<Option<wgpu::RenderPassColorAttachment>> = color_slice
        .iter()
        .map(|att| {
            if att.view.is_null() {
                None
            } else {
                let view = &Resource::<TextureViewData>::inner(att.view as _).inner;
                let resolve_target = if att.resolveTarget.is_null() {
                    None
                } else {
                    Some(&Resource::<TextureViewData>::inner(att.resolveTarget as _).inner)
                };
                Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: convert_load_op(att.loadOp, att.clearValue),
                        store: convert_store_op(att.storeOp),
                    },
                })
            }
        })
        .collect();

    let depth_stencil_attachment = if desc.depthStencilAttachment.is_null() {
        None
    } else {
        let ds = &*desc.depthStencilAttachment;
        if ds.view.is_null() {
            None
        } else {
            let view = &Resource::<TextureViewData>::inner(ds.view as _).inner;
            let depth_ops = if ds.depthLoadOp == sb::WGPULoadOp::WGPULoadOp_Undefined
                && ds.depthStoreOp == sb::WGPUStoreOp::WGPUStoreOp_Undefined
            {
                None
            } else {
                Some(wgpu::Operations {
                    load: convert_depth_load_op(ds.depthLoadOp, ds.depthClearValue),
                    store: convert_store_op(ds.depthStoreOp),
                })
            };
            let stencil_ops = if ds.stencilLoadOp == sb::WGPULoadOp::WGPULoadOp_Undefined
                && ds.stencilStoreOp == sb::WGPUStoreOp::WGPUStoreOp_Undefined
            {
                None
            } else {
                Some(wgpu::Operations {
                    load: convert_stencil_load_op(ds.stencilLoadOp, ds.stencilClearValue),
                    store: convert_store_op(ds.stencilStoreOp),
                })
            };
            Some(wgpu::RenderPassDepthStencilAttachment {
                view,
                depth_ops,
                stencil_ops,
            })
        }
    };

    let pass_static = {
        let mut guard = encoder_data
            .inner
            .lock()
            .expect("command encoder mutex poisoned");
        let Some(enc) = guard.as_mut() else {
            return ptr::null_mut();
        };
        let pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: if label.is_empty() { None } else { Some(label) },
            color_attachments: &color_attachments,
            depth_stencil_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.forget_lifetime()
    };

    Resource::into_handle(RenderPassEncoderData {
        inner: std::sync::Mutex::new(Some(pass_static)),
    }) as sb::WGPURenderPassEncoder
}

unsafe extern "C" fn render_pass_encoder_add_ref(handle: sb::WGPURenderPassEncoder) {
    Resource::<RenderPassEncoderData>::add_ref(handle as _);
}

unsafe extern "C" fn render_pass_encoder_release(handle: sb::WGPURenderPassEncoder) {
    Resource::<RenderPassEncoderData>::release(handle as _);
}

unsafe extern "C" fn render_pass_encoder_set_label(
    _handle: sb::WGPURenderPassEncoder,
    _label: sb::WGPUStringView,
) {
}

unsafe extern "C" fn render_pass_encoder_end(handle: sb::WGPURenderPassEncoder) {
    if handle.is_null() {
        return;
    }
    let data = Resource::<RenderPassEncoderData>::inner(handle as _);
    if let Ok(mut guard) = data.inner.lock() {
        // wgpu 29 ends the pass via Drop; taking the Option drops the pass
        // here, which encodes the End command into the parent CommandEncoder.
        let _ = guard.take();
    }
}

unsafe extern "C" fn render_pass_encoder_insert_debug_marker(
    _handle: sb::WGPURenderPassEncoder,
    _label: sb::WGPUStringView,
) {
}

unsafe extern "C" fn render_pass_encoder_push_debug_group(
    _handle: sb::WGPURenderPassEncoder,
    _label: sb::WGPUStringView,
) {
}

unsafe extern "C" fn render_pass_encoder_pop_debug_group(_handle: sb::WGPURenderPassEncoder) {}

fn with_render_pass<F>(handle: sb::WGPURenderPassEncoder, f: F)
where
    F: FnOnce(&mut wgpu::RenderPass<'static>),
{
    if handle.is_null() {
        return;
    }
    let data = unsafe { Resource::<RenderPassEncoderData>::inner(handle as _) };
    if let Ok(mut guard) = data.inner.lock() {
        if let Some(pass) = guard.as_mut() {
            f(pass);
        }
    }
}

unsafe extern "C" fn render_pass_encoder_set_viewport(
    handle: sb::WGPURenderPassEncoder,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    min_depth: f32,
    max_depth: f32,
) {
    with_render_pass(handle, |pass| {
        pass.set_viewport(x, y, width, height, min_depth, max_depth);
    });
}

unsafe extern "C" fn render_pass_encoder_set_scissor_rect(
    handle: sb::WGPURenderPassEncoder,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) {
    with_render_pass(handle, |pass| {
        pass.set_scissor_rect(x, y, width, height);
    });
}

unsafe extern "C" fn render_pass_encoder_set_blend_constant(
    handle: sb::WGPURenderPassEncoder,
    color: *const sb::WGPUColor,
) {
    if color.is_null() {
        return;
    }
    let c = *color;
    with_render_pass(handle, |pass| {
        pass.set_blend_constant(wgpu::Color {
            r: c.r,
            g: c.g,
            b: c.b,
            a: c.a,
        });
    });
}

unsafe extern "C" fn render_pass_encoder_set_stencil_reference(
    handle: sb::WGPURenderPassEncoder,
    reference: u32,
) {
    with_render_pass(handle, |pass| {
        pass.set_stencil_reference(reference);
    });
}

unsafe extern "C" fn render_pass_encoder_draw(
    handle: sb::WGPURenderPassEncoder,
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
) {
    with_render_pass(handle, |pass| {
        pass.draw(
            first_vertex..first_vertex + vertex_count,
            first_instance..first_instance + instance_count,
        );
    });
}

unsafe extern "C" fn render_pass_encoder_set_pipeline(
    handle: sb::WGPURenderPassEncoder,
    pipeline: sb::WGPURenderPipeline,
) {
    if pipeline.is_null() {
        return;
    }
    let pipeline_data = Resource::<RenderPipelineData>::inner(pipeline as _);
    let pipeline_inner: *const wgpu::RenderPipeline = &pipeline_data.inner;
    let placeholder_slots = pipeline_data.placeholder_slots.clone();
    let device_handle = pipeline_data.device_handle;

    with_render_pass(handle, |pass| {
        // SAFETY: the pipeline Resource is refcount-held by the caller, so the
        // wgpu::RenderPipeline reference outlives this set_pipeline call.
        pass.set_pipeline(&*pipeline_inner);

        // Auto-bind the device's dummy vertex buffer to any slots Skia
        // declared as placeholders. wgpu's validator demands a binding even
        // for slots whose stepMode is Undefined / VertexBufferNotUsed.
        if !placeholder_slots.is_empty() && !device_handle.is_null() {
            let device_data = Resource::<DeviceData>::inner(device_handle as _);
            let dummy = device_data.dummy_vertex_buffer.get_or_init(|| {
                device_data.inner.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("wgpu_backend dummy vertex placeholder"),
                    size: 16,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            });
            for slot in placeholder_slots {
                pass.set_vertex_buffer(slot, dummy.slice(..));
            }
        }
    });
}

unsafe extern "C" fn render_pass_encoder_set_bind_group(
    handle: sb::WGPURenderPassEncoder,
    group_index: u32,
    bind_group: sb::WGPUBindGroup,
    offset_count: usize,
    offsets: *const u32,
) {
    let bind_group_ptr: Option<*const wgpu::BindGroup> = if bind_group.is_null() {
        None
    } else {
        Some(&Resource::<BindGroupData>::inner(bind_group as _).inner)
    };
    let offsets_slice = if offset_count == 0 || offsets.is_null() {
        &[][..]
    } else {
        std::slice::from_raw_parts(offsets, offset_count)
    };
    with_render_pass(handle, |pass| match bind_group_ptr {
        Some(p) => pass.set_bind_group(group_index, &*p, offsets_slice),
        None => pass.set_bind_group(group_index, None, offsets_slice),
    });
}

unsafe extern "C" fn render_pass_encoder_set_vertex_buffer(
    handle: sb::WGPURenderPassEncoder,
    slot: u32,
    buffer: sb::WGPUBuffer,
    offset: u64,
    size: u64,
) {
    if buffer.is_null() {
        return;
    }
    let buffer_ptr: *const wgpu::Buffer = &Resource::<BufferData>::inner(buffer as _).inner;
    let end = if size == u64::MAX {
        (&*buffer_ptr).size()
    } else {
        offset + size
    };
    with_render_pass(handle, |pass| {
        pass.set_vertex_buffer(slot, (&*buffer_ptr).slice(offset..end));
    });
}

unsafe extern "C" fn render_pass_encoder_set_index_buffer(
    handle: sb::WGPURenderPassEncoder,
    buffer: sb::WGPUBuffer,
    format: sb::WGPUIndexFormat,
    offset: u64,
    size: u64,
) {
    if buffer.is_null() {
        return;
    }
    let buffer_ptr: *const wgpu::Buffer = &Resource::<BufferData>::inner(buffer as _).inner;
    let end = if size == u64::MAX {
        (&*buffer_ptr).size()
    } else {
        offset + size
    };
    let index_format = convert_index_format(format).unwrap_or(wgpu::IndexFormat::Uint32);
    with_render_pass(handle, |pass| {
        pass.set_index_buffer((&*buffer_ptr).slice(offset..end), index_format);
    });
}

unsafe extern "C" fn render_pass_encoder_draw_indexed(
    handle: sb::WGPURenderPassEncoder,
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
) {
    with_render_pass(handle, |pass| {
        pass.draw_indexed(
            first_index..first_index + index_count,
            base_vertex,
            first_instance..first_instance + instance_count,
        );
    });
}

unsafe extern "C" fn queue_on_submitted_work_done(
    queue: sb::WGPUQueue,
    callback_info: sb::WGPUQueueWorkDoneCallbackInfo,
) -> sb::WGPUFuture {
    let queue_data = Resource::<QueueData>::inner(queue as _);
    // Drive any pending GPU work to completion synchronously so the callback
    // fires from this thread before we return.
    queue_data
        ._device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .ok();

    if let Some(callback) = callback_info.callback {
        let message = sb::WGPUStringView {
            data: ptr::null(),
            length: 0,
        };
        callback(
            sb::WGPUQueueWorkDoneStatus::WGPUQueueWorkDoneStatus_Success,
            message,
            callback_info.userdata1,
            callback_info.userdata2,
        );
    }

    sb::WGPUFuture { id: 0 }
}

//
// PipelineLayout
//

unsafe extern "C" fn device_create_pipeline_layout(
    device: sb::WGPUDevice,
    descriptor: *const sb::WGPUPipelineLayoutDescriptor,
) -> sb::WGPUPipelineLayout {
    let device_data = Resource::<DeviceData>::inner(device as _);
    if descriptor.is_null() {
        return ptr::null_mut();
    }
    let d = &*descriptor;

    let bg_handles = if d.bindGroupLayoutCount == 0 || d.bindGroupLayouts.is_null() {
        &[][..]
    } else {
        std::slice::from_raw_parts(d.bindGroupLayouts, d.bindGroupLayoutCount)
    };
    let bg_layouts: Vec<Option<&wgpu::BindGroupLayout>> = bg_handles
        .iter()
        .map(|h| {
            if h.is_null() {
                None
            } else {
                Some(&Resource::<BindGroupLayoutData>::inner(*h as _).inner)
            }
        })
        .collect();

    let label = string_view_as_str(d.label);
    let layout = device_data
        .inner
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: if label.is_empty() { None } else { Some(label) },
            bind_group_layouts: &bg_layouts,
            immediate_size: d.immediateSize,
        });

    Resource::into_handle(PipelineLayoutData {
        inner: layout,
        _device: device_data.inner.clone(),
    }) as sb::WGPUPipelineLayout
}

unsafe extern "C" fn pipeline_layout_add_ref(handle: sb::WGPUPipelineLayout) {
    Resource::<PipelineLayoutData>::add_ref(handle as _);
}

unsafe extern "C" fn pipeline_layout_release(handle: sb::WGPUPipelineLayout) {
    Resource::<PipelineLayoutData>::release(handle as _);
}

unsafe extern "C" fn pipeline_layout_set_label(
    _handle: sb::WGPUPipelineLayout,
    _label: sb::WGPUStringView,
) {
}

//
// Proc table assembly
//

/// Installs an abort-on-call stub on each named field. Each stub aborts the
/// process with the field name, so a missing thunk surfaces as a clean error
/// message rather than a null-pointer crash.
///
/// The stub has signature `unsafe extern "C" fn() -> !` and is transmuted to
/// the field's expected signature. This is technically calling-convention
/// abuse — the caller pushes args we ignore — but the stub aborts before
/// returning, so no return value is ever observed and any ABI mismatch on
/// args is harmless in practice on the platforms wgpu/Dawn target.
macro_rules! install_abort_stubs {
    ($table:expr, $($field:ident),* $(,)?) => {
        $(
            $table.$field = Some({
                #[allow(non_snake_case)]
                unsafe extern "C" fn stub() -> ! {
                    $crate::graphite::wgpu_backend::unimplemented_stub(stringify!($field))
                }
                // SAFETY: stub diverges before any return-value or arg-handling
                // matters. Function-pointer types are all the same size, so the
                // transmute itself is well-formed.
                unsafe { core::mem::transmute::<unsafe extern "C" fn() -> !, _>(stub) }
            });
        )*
    };
}

/// Wraps an existing [`wgpu::Instance`] / [`wgpu::Adapter`] / [`wgpu::Device`]
/// / [`wgpu::Queue`] in WGPU C handles compatible with Graphite. The proc
/// table is installed automatically (first-caller-wins, so this is safe to
/// call after another install).
///
/// The caller keeps its original wgpu objects — we clone them, and wgpu's
/// types are Arc-internally, so the clone shares the underlying GPU device
/// with the caller's renderer. After this call:
///
/// - Pass the returned [`BackendContext`](super::dawn::BackendContext) to
///   [`Context::new_dawn`](super::Context::new_dawn) to build a Graphite
///   Context on the user-provided device.
/// - The user's `wgpu::Device` and `wgpu::Queue` remain fully usable for
///   other rendering work. Skia's Context and the user's wgpu work share
///   the same queue and can interoperate via textures/buffers extracted
///   from one and wrapped into the other.
///
/// This is the entry point for "use my existing wgpu device" interop, e.g.
/// when integrating Skia/Graphite into an application that already drives
/// its rendering through the [`wgpu`] crate.
pub fn install_and_wrap(
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
) -> crate::graphite::dawn::BackendContext {
    static PROCS: std::sync::OnceLock<DawnProcTable> = std::sync::OnceLock::new();
    let procs = PROCS.get_or_init(wgpu_proc_table);
    // SAFETY: `procs` is `'static` (held by `PROCS`).
    unsafe { crate::graphite::dawn::install_proc_table(procs) };

    let instance_handle = Resource::into_handle(InstanceData {
        inner: instance.clone(),
    }) as sb::WGPUInstance;
    // We hand AdapterData out only for parity, but `BackendContext` doesn't
    // include an adapter handle, so this lives only inside DeviceData as a
    // keep-alive.
    let device_handle = Resource::into_handle(DeviceData {
        inner: device.clone(),
        queue: queue.clone(),
        _adapter: adapter,
        dummy_vertex_buffer: std::sync::OnceLock::new(),
    }) as sb::WGPUDevice;
    let queue_handle = Resource::into_handle(QueueData {
        inner: queue,
        _device: device,
    }) as sb::WGPUQueue;

    crate::graphite::dawn::BackendContext {
        instance: instance_handle,
        device: device_handle,
        queue: queue_handle,
    }
}

/// Builds a [`DawnProcTable`] whose entries route to wgpu.
///
/// Every entry is populated: real thunks for what we've implemented, and
/// named abort-on-call stubs for everything else, so that the first missing
/// thunk surfaces a clean error message identifying which WGPU function
/// Skia needs next.
pub fn wgpu_proc_table() -> DawnProcTable {
    let mut table: DawnProcTable = unsafe { core::mem::zeroed() };

    install_all_abort_stubs(&mut table);

    // Real thunks — these overwrite the corresponding abort-stubs above.
    table.createInstance = Some(create_instance);
    table.instanceAddRef = Some(instance_add_ref);
    table.instanceRelease = Some(instance_release);
    table.instanceProcessEvents = Some(instance_process_events);
    table.instanceWaitAny = Some(instance_wait_any);
    table.instanceRequestAdapter = Some(instance_request_adapter);

    table.adapterAddRef = Some(adapter_add_ref);
    table.adapterRelease = Some(adapter_release);
    table.adapterRequestDevice = Some(adapter_request_device);

    table.deviceAddRef = Some(device_add_ref);
    table.deviceRelease = Some(device_release);
    table.deviceGetQueue = Some(device_get_queue);

    table.queueAddRef = Some(queue_add_ref);
    table.queueRelease = Some(queue_release);

    table.deviceCreateRenderPipeline = Some(device_create_render_pipeline);
    table.deviceCreateRenderPipelineAsync = Some(device_create_render_pipeline_async);
    table.renderPipelineAddRef = Some(render_pipeline_add_ref);
    table.renderPipelineRelease = Some(render_pipeline_release);
    table.renderPipelineSetLabel = Some(render_pipeline_set_label);
    table.renderPipelineGetBindGroupLayout = Some(render_pipeline_get_bind_group_layout);

    table.deviceCreateShaderModule = Some(device_create_shader_module);
    table.shaderModuleAddRef = Some(shader_module_add_ref);
    table.shaderModuleRelease = Some(shader_module_release);
    table.shaderModuleSetLabel = Some(shader_module_set_label);
    table.shaderModuleGetCompilationInfo = Some(shader_module_get_compilation_info);

    table.deviceGetAdapter = Some(device_get_adapter);
    table.adapterGetInstance = Some(adapter_get_instance);
    table.adapterGetInfo = Some(adapter_get_info);
    table.adapterInfoFreeMembers = Some(adapter_info_free_members);
    table.deviceGetLimits = Some(device_get_limits);
    table.adapterGetLimits = Some(adapter_get_limits);
    table.deviceHasFeature = Some(device_has_feature);
    table.adapterHasFeature = Some(adapter_has_feature);
    table.deviceGetFeatures = Some(device_get_features);
    table.adapterGetFeatures = Some(adapter_get_features);
    table.supportedFeaturesFreeMembers = Some(supported_features_free_members);
    table.deviceTick = Some(device_tick);
    table.deviceSetLoggingCallback = Some(device_set_logging_callback);
    table.deviceSetLabel = Some(device_set_label);
    table.adapterGetFormatCapabilities = Some(adapter_get_format_capabilities);

    table.deviceCreateBindGroupLayout = Some(device_create_bind_group_layout);
    table.bindGroupLayoutAddRef = Some(bind_group_layout_add_ref);
    table.bindGroupLayoutRelease = Some(bind_group_layout_release);
    table.bindGroupLayoutSetLabel = Some(bind_group_layout_set_label);

    table.deviceCreateSampler = Some(device_create_sampler);
    table.samplerAddRef = Some(sampler_add_ref);
    table.samplerRelease = Some(sampler_release);
    table.samplerSetLabel = Some(sampler_set_label);

    table.deviceCreateBuffer = Some(device_create_buffer);
    table.bufferAddRef = Some(buffer_add_ref);
    table.bufferRelease = Some(buffer_release);
    table.bufferDestroy = Some(buffer_destroy);
    table.bufferGetSize = Some(buffer_get_size);
    table.bufferGetUsage = Some(buffer_get_usage);
    table.bufferSetLabel = Some(buffer_set_label);
    table.bufferGetMappedRange = Some(buffer_get_mapped_range);
    table.bufferGetConstMappedRange = Some(buffer_get_const_mapped_range);
    table.bufferUnmap = Some(buffer_unmap);
    table.bufferMapAsync = Some(buffer_map_async);
    table.bufferGetMapState = Some(buffer_get_map_state);

    table.queueWriteBuffer = Some(queue_write_buffer);
    table.queueSubmit = Some(queue_submit);

    table.deviceCreateCommandEncoder = Some(device_create_command_encoder);
    table.commandEncoderAddRef = Some(command_encoder_add_ref);
    table.commandEncoderRelease = Some(command_encoder_release);
    table.commandEncoderSetLabel = Some(command_encoder_set_label);
    table.commandEncoderFinish = Some(command_encoder_finish);
    table.commandEncoderInsertDebugMarker = Some(command_encoder_insert_debug_marker);
    table.commandEncoderPushDebugGroup = Some(command_encoder_push_debug_group);
    table.commandEncoderPopDebugGroup = Some(command_encoder_pop_debug_group);
    table.commandEncoderCopyBufferToBuffer = Some(command_encoder_copy_buffer_to_buffer);
    table.commandEncoderClearBuffer = Some(command_encoder_clear_buffer);
    table.commandEncoderCopyTextureToBuffer = Some(command_encoder_copy_texture_to_buffer);
    table.commandEncoderCopyBufferToTexture = Some(command_encoder_copy_buffer_to_texture);
    table.commandEncoderCopyTextureToTexture = Some(command_encoder_copy_texture_to_texture);

    table.commandBufferAddRef = Some(command_buffer_add_ref);
    table.commandBufferRelease = Some(command_buffer_release);
    table.commandBufferSetLabel = Some(command_buffer_set_label);

    table.queueOnSubmittedWorkDone = Some(queue_on_submitted_work_done);

    table.deviceCreateTexture = Some(device_create_texture);
    table.textureAddRef = Some(texture_add_ref);
    table.textureRelease = Some(texture_release);
    table.textureDestroy = Some(texture_destroy);
    table.textureSetLabel = Some(texture_set_label);
    table.textureGetWidth = Some(texture_get_width);
    table.textureGetHeight = Some(texture_get_height);
    table.textureGetDepthOrArrayLayers = Some(texture_get_depth_or_array_layers);
    table.textureGetMipLevelCount = Some(texture_get_mip_level_count);
    table.textureGetSampleCount = Some(texture_get_sample_count);
    table.textureGetDimension = Some(texture_get_dimension);
    table.textureGetUsage = Some(texture_get_usage);
    table.textureGetFormat = Some(texture_get_format);
    table.textureCreateView = Some(texture_create_view);
    table.textureViewAddRef = Some(texture_view_add_ref);
    table.textureViewRelease = Some(texture_view_release);
    table.textureViewSetLabel = Some(texture_view_set_label);

    table.commandEncoderBeginRenderPass = Some(command_encoder_begin_render_pass);
    table.renderPassEncoderAddRef = Some(render_pass_encoder_add_ref);
    table.renderPassEncoderRelease = Some(render_pass_encoder_release);
    table.renderPassEncoderSetLabel = Some(render_pass_encoder_set_label);
    table.renderPassEncoderEnd = Some(render_pass_encoder_end);
    table.renderPassEncoderInsertDebugMarker = Some(render_pass_encoder_insert_debug_marker);
    table.renderPassEncoderPushDebugGroup = Some(render_pass_encoder_push_debug_group);
    table.renderPassEncoderPopDebugGroup = Some(render_pass_encoder_pop_debug_group);
    table.renderPassEncoderSetViewport = Some(render_pass_encoder_set_viewport);
    table.renderPassEncoderSetScissorRect = Some(render_pass_encoder_set_scissor_rect);
    table.renderPassEncoderSetBlendConstant = Some(render_pass_encoder_set_blend_constant);
    table.renderPassEncoderSetStencilReference = Some(render_pass_encoder_set_stencil_reference);
    table.renderPassEncoderDraw = Some(render_pass_encoder_draw);
    table.renderPassEncoderDrawIndexed = Some(render_pass_encoder_draw_indexed);
    table.renderPassEncoderSetPipeline = Some(render_pass_encoder_set_pipeline);
    table.renderPassEncoderSetBindGroup = Some(render_pass_encoder_set_bind_group);
    table.renderPassEncoderSetVertexBuffer = Some(render_pass_encoder_set_vertex_buffer);
    table.renderPassEncoderSetIndexBuffer = Some(render_pass_encoder_set_index_buffer);

    table.deviceCreateBindGroup = Some(device_create_bind_group);
    table.bindGroupAddRef = Some(bind_group_add_ref);
    table.bindGroupRelease = Some(bind_group_release);
    table.bindGroupSetLabel = Some(bind_group_set_label);

    table.deviceCreateComputePipeline = Some(device_create_compute_pipeline);
    table.deviceCreateComputePipelineAsync = Some(device_create_compute_pipeline_async);
    table.computePipelineAddRef = Some(compute_pipeline_add_ref);
    table.computePipelineRelease = Some(compute_pipeline_release);
    table.computePipelineSetLabel = Some(compute_pipeline_set_label);
    table.computePipelineGetBindGroupLayout = Some(compute_pipeline_get_bind_group_layout);

    table.commandEncoderBeginComputePass = Some(command_encoder_begin_compute_pass);
    table.computePassEncoderAddRef = Some(compute_pass_encoder_add_ref);
    table.computePassEncoderRelease = Some(compute_pass_encoder_release);
    table.computePassEncoderSetLabel = Some(compute_pass_encoder_set_label);
    table.computePassEncoderEnd = Some(compute_pass_encoder_end);
    table.computePassEncoderSetPipeline = Some(compute_pass_encoder_set_pipeline);
    table.computePassEncoderSetBindGroup = Some(compute_pass_encoder_set_bind_group);
    table.computePassEncoderDispatchWorkgroups = Some(compute_pass_encoder_dispatch_workgroups);
    table.computePassEncoderDispatchWorkgroupsIndirect =
        Some(compute_pass_encoder_dispatch_workgroups_indirect);
    table.computePassEncoderInsertDebugMarker = Some(compute_pass_encoder_insert_debug_marker);
    table.computePassEncoderPushDebugGroup = Some(compute_pass_encoder_push_debug_group);
    table.computePassEncoderPopDebugGroup = Some(compute_pass_encoder_pop_debug_group);

    table.queueWriteTexture = Some(queue_write_texture);

    table.deviceCreateRenderBundleEncoder = Some(device_create_render_bundle_encoder);
    table.renderBundleEncoderAddRef = Some(render_bundle_encoder_add_ref);
    table.renderBundleEncoderRelease = Some(render_bundle_encoder_release);
    table.renderBundleEncoderSetLabel = Some(render_bundle_encoder_set_label);
    table.renderBundleEncoderFinish = Some(render_bundle_encoder_finish);
    table.renderBundleAddRef = Some(render_bundle_add_ref);
    table.renderBundleRelease = Some(render_bundle_release);
    table.renderBundleSetLabel = Some(render_bundle_set_label);

    table.deviceCreateQuerySet = Some(device_create_query_set);
    table.querySetAddRef = Some(query_set_add_ref);
    table.querySetRelease = Some(query_set_release);
    table.querySetDestroy = Some(query_set_destroy);
    table.querySetGetCount = Some(query_set_get_count);
    table.querySetGetType = Some(query_set_get_type);
    table.querySetSetLabel = Some(query_set_set_label);

    table.deviceCreatePipelineLayout = Some(device_create_pipeline_layout);
    table.pipelineLayoutAddRef = Some(pipeline_layout_add_ref);
    table.pipelineLayoutRelease = Some(pipeline_layout_release);
    table.pipelineLayoutSetLabel = Some(pipeline_layout_set_label);

    table
}

/// Generated by enumerating every field of `DawnProcTable`. Keep this in
/// sync with the bindgen output if the WebGPU header revision changes
/// (bindings count: 276 fields). The list is mechanical; nothing here is
/// load-bearing for correctness beyond `stringify!($field)` matching the
/// dispatcher-side WGPU function name.
fn install_all_abort_stubs(table: &mut DawnProcTable) {
    install_abort_stubs!(
        table,
        createInstance,
        getInstanceFeatures,
        getInstanceLimits,
        hasInstanceFeature,
        getProcAddress,
        adapterCreateDevice,
        adapterGetFeatures,
        adapterGetFormatCapabilities,
        adapterGetInfo,
        adapterGetInstance,
        adapterGetLimits,
        adapterHasFeature,
        adapterRequestDevice,
        adapterAddRef,
        adapterRelease,
        adapterInfoFreeMembers,
        adapterPropertiesMemoryHeapsFreeMembers,
        adapterPropertiesSubgroupMatrixConfigsFreeMembers,
        bindGroupSetLabel,
        bindGroupAddRef,
        bindGroupRelease,
        bindGroupLayoutSetLabel,
        bindGroupLayoutAddRef,
        bindGroupLayoutRelease,
        bufferCreateTexelView,
        bufferDestroy,
        bufferGetConstMappedRange,
        bufferGetMappedRange,
        bufferGetMapState,
        bufferGetSize,
        bufferGetUsage,
        bufferMapAsync,
        bufferReadMappedRange,
        bufferSetLabel,
        bufferUnmap,
        bufferWriteMappedRange,
        bufferAddRef,
        bufferRelease,
        commandBufferSetLabel,
        commandBufferAddRef,
        commandBufferRelease,
        commandEncoderBeginComputePass,
        commandEncoderBeginRenderPass,
        commandEncoderClearBuffer,
        commandEncoderCopyBufferToBuffer,
        commandEncoderCopyBufferToTexture,
        commandEncoderCopyTextureToBuffer,
        commandEncoderCopyTextureToTexture,
        commandEncoderFinish,
        commandEncoderInjectValidationError,
        commandEncoderInsertDebugMarker,
        commandEncoderPopDebugGroup,
        commandEncoderPushDebugGroup,
        commandEncoderResolveQuerySet,
        commandEncoderSetLabel,
        commandEncoderWriteBuffer,
        commandEncoderWriteTimestamp,
        commandEncoderAddRef,
        commandEncoderRelease,
        computePassEncoderDispatchWorkgroups,
        computePassEncoderDispatchWorkgroupsIndirect,
        computePassEncoderEnd,
        computePassEncoderInsertDebugMarker,
        computePassEncoderPopDebugGroup,
        computePassEncoderPushDebugGroup,
        computePassEncoderSetBindGroup,
        computePassEncoderSetImmediates,
        computePassEncoderSetLabel,
        computePassEncoderSetPipeline,
        computePassEncoderSetResourceTable,
        computePassEncoderWriteTimestamp,
        computePassEncoderAddRef,
        computePassEncoderRelease,
        computePipelineGetBindGroupLayout,
        computePipelineSetLabel,
        computePipelineAddRef,
        computePipelineRelease,
        dawnDrmFormatCapabilitiesFreeMembers,
        deviceCreateBindGroup,
        deviceCreateBindGroupLayout,
        deviceCreateBuffer,
        deviceCreateCommandEncoder,
        deviceCreateComputePipeline,
        deviceCreateComputePipelineAsync,
        deviceCreateErrorBuffer,
        deviceCreateErrorExternalTexture,
        deviceCreateErrorShaderModule,
        deviceCreateErrorTexture,
        deviceCreateExternalTexture,
        deviceCreatePipelineLayout,
        deviceCreateQuerySet,
        deviceCreateRenderBundleEncoder,
        deviceCreateRenderPipeline,
        deviceCreateRenderPipelineAsync,
        deviceCreateResourceTable,
        deviceCreateSampler,
        deviceCreateShaderModule,
        deviceCreateTexture,
        deviceDestroy,
        deviceForceLoss,
        deviceGetAdapter,
        deviceGetAdapterInfo,
        deviceGetAHardwareBufferProperties,
        deviceGetFeatures,
        deviceGetLimits,
        deviceGetLostFuture,
        deviceGetQueue,
        deviceHasFeature,
        deviceImportSharedBufferMemory,
        deviceImportSharedFence,
        deviceImportSharedTextureMemory,
        deviceInjectError,
        devicePopErrorScope,
        devicePushErrorScope,
        deviceSetLabel,
        deviceSetLoggingCallback,
        deviceTick,
        deviceValidateTextureDescriptor,
        deviceAddRef,
        deviceRelease,
        externalTextureDestroy,
        externalTextureExpire,
        externalTextureRefresh,
        externalTextureSetLabel,
        externalTextureAddRef,
        externalTextureRelease,
        instanceCreateSurface,
        instanceGetWGSLLanguageFeatures,
        instanceHasWGSLLanguageFeature,
        instanceProcessEvents,
        instanceRequestAdapter,
        instanceWaitAny,
        instanceAddRef,
        instanceRelease,
        pipelineLayoutSetLabel,
        pipelineLayoutAddRef,
        pipelineLayoutRelease,
        querySetDestroy,
        querySetGetCount,
        querySetGetType,
        querySetSetLabel,
        querySetAddRef,
        querySetRelease,
        queueCopyExternalTextureForBrowser,
        queueCopyTextureForBrowser,
        queueOnSubmittedWorkDone,
        queueSetLabel,
        queueSubmit,
        queueWriteBuffer,
        queueWriteTexture,
        queueAddRef,
        queueRelease,
        renderBundleSetLabel,
        renderBundleAddRef,
        renderBundleRelease,
        renderBundleEncoderDraw,
        renderBundleEncoderDrawIndexed,
        renderBundleEncoderDrawIndexedIndirect,
        renderBundleEncoderDrawIndirect,
        renderBundleEncoderFinish,
        renderBundleEncoderInsertDebugMarker,
        renderBundleEncoderPopDebugGroup,
        renderBundleEncoderPushDebugGroup,
        renderBundleEncoderSetBindGroup,
        renderBundleEncoderSetImmediates,
        renderBundleEncoderSetIndexBuffer,
        renderBundleEncoderSetLabel,
        renderBundleEncoderSetPipeline,
        renderBundleEncoderSetResourceTable,
        renderBundleEncoderSetVertexBuffer,
        renderBundleEncoderAddRef,
        renderBundleEncoderRelease,
        renderPassEncoderBeginOcclusionQuery,
        renderPassEncoderDraw,
        renderPassEncoderDrawIndexed,
        renderPassEncoderDrawIndexedIndirect,
        renderPassEncoderDrawIndirect,
        renderPassEncoderEnd,
        renderPassEncoderEndOcclusionQuery,
        renderPassEncoderExecuteBundles,
        renderPassEncoderInsertDebugMarker,
        renderPassEncoderMultiDrawIndexedIndirect,
        renderPassEncoderMultiDrawIndirect,
        renderPassEncoderPixelLocalStorageBarrier,
        renderPassEncoderPopDebugGroup,
        renderPassEncoderPushDebugGroup,
        renderPassEncoderSetBindGroup,
        renderPassEncoderSetBlendConstant,
        renderPassEncoderSetImmediates,
        renderPassEncoderSetIndexBuffer,
        renderPassEncoderSetLabel,
        renderPassEncoderSetPipeline,
        renderPassEncoderSetResourceTable,
        renderPassEncoderSetScissorRect,
        renderPassEncoderSetStencilReference,
        renderPassEncoderSetVertexBuffer,
        renderPassEncoderSetViewport,
        renderPassEncoderWriteTimestamp,
        renderPassEncoderAddRef,
        renderPassEncoderRelease,
        renderPipelineGetBindGroupLayout,
        renderPipelineSetLabel,
        renderPipelineAddRef,
        renderPipelineRelease,
        resourceTableDestroy,
        resourceTableGetSize,
        resourceTableInsertBinding,
        resourceTableRemoveBinding,
        resourceTableUpdate,
        resourceTableAddRef,
        resourceTableRelease,
        samplerSetLabel,
        samplerAddRef,
        samplerRelease,
        shaderModuleGetCompilationInfo,
        shaderModuleSetLabel,
        shaderModuleAddRef,
        shaderModuleRelease,
        sharedBufferMemoryBeginAccess,
        sharedBufferMemoryCreateBuffer,
        sharedBufferMemoryEndAccess,
        sharedBufferMemoryGetProperties,
        sharedBufferMemoryIsDeviceLost,
        sharedBufferMemorySetLabel,
        sharedBufferMemoryAddRef,
        sharedBufferMemoryRelease,
        sharedBufferMemoryEndAccessStateFreeMembers,
        sharedFenceExportInfo,
        sharedFenceSetLabel,
        sharedFenceAddRef,
        sharedFenceRelease,
        sharedTextureMemoryBeginAccess,
        sharedTextureMemoryCreateTexture,
        sharedTextureMemoryEndAccess,
        sharedTextureMemoryGetProperties,
        sharedTextureMemoryIsDeviceLost,
        sharedTextureMemorySetLabel,
        sharedTextureMemoryAddRef,
        sharedTextureMemoryRelease,
        sharedTextureMemoryEndAccessStateFreeMembers,
        supportedFeaturesFreeMembers,
        supportedInstanceFeaturesFreeMembers,
        supportedWGSLLanguageFeaturesFreeMembers,
        surfaceConfigure,
        surfaceGetCapabilities,
        surfaceGetCurrentTexture,
        surfacePresent,
        surfaceSetLabel,
        surfaceUnconfigure,
        surfaceAddRef,
        surfaceRelease,
        surfaceCapabilitiesFreeMembers,
        texelBufferViewSetLabel,
        texelBufferViewAddRef,
        texelBufferViewRelease,
        textureCreateErrorView,
        textureCreateView,
        textureDestroy,
        textureGetDepthOrArrayLayers,
        textureGetDimension,
        textureGetFormat,
        textureGetHeight,
        textureGetMipLevelCount,
        textureGetSampleCount,
        textureGetTextureBindingViewDimension,
        textureGetUsage,
        textureGetWidth,
        texturePin,
        textureSetLabel,
        textureSetOwnershipForMemoryDump,
        textureUnpin,
        textureAddRef,
        textureRelease,
        textureViewSetLabel,
        textureViewAddRef,
        textureViewRelease,
    );
}

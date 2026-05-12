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
}

struct PipelineLayoutData {
    inner: wgpu::PipelineLayout,
    _device: wgpu::Device,
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
    let buffer = &Resource::<BufferData>::inner(handle as _).inner;
    let (start, end) = buffer_slice_range(buffer, offset, size);
    let mut view = buffer.slice(start..end).get_mapped_range_mut();
    let ptr = view.slice(..).as_raw_element_ptr().as_ptr() as *mut core::ffi::c_void;
    // We hand the raw pointer back to Skia; the BufferViewMut's drop would
    // unregister it from wgpu's bookkeeping prematurely, so we leak it and
    // rely on `bufferUnmap` to do the actual release.
    core::mem::forget(view);
    ptr
}

unsafe extern "C" fn buffer_get_const_mapped_range(
    handle: sb::WGPUBuffer,
    offset: usize,
    size: usize,
) -> *const core::ffi::c_void {
    let buffer = &Resource::<BufferData>::inner(handle as _).inner;
    let (start, end) = buffer_slice_range(buffer, offset, size);
    let view = buffer.slice(start..end).get_mapped_range();
    let ptr = (*view).as_ptr() as *const core::ffi::c_void;
    core::mem::forget(view);
    ptr
}

unsafe extern "C" fn buffer_unmap(handle: sb::WGPUBuffer) {
    if handle.is_null() {
        return;
    }
    Resource::<BufferData>::inner(handle as _).inner.unmap();
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
    _command_count: usize,
    _commands: *const sb::WGPUCommandBuffer,
) {
    let _ = Resource::<QueueData>::inner(queue as _);
    // TODO: collect command buffers and call queue.submit(...). Stubbed for
    // now so the rest of the path keeps progressing.
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

    table.deviceCreateShaderModule = Some(device_create_shader_module);
    table.shaderModuleAddRef = Some(shader_module_add_ref);
    table.shaderModuleRelease = Some(shader_module_release);
    table.shaderModuleSetLabel = Some(shader_module_set_label);

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

    table.queueWriteBuffer = Some(queue_write_buffer);
    table.queueSubmit = Some(queue_submit);

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

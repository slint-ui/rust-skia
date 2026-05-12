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
    _inner: wgpu::ShaderModule,
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
        _inner: module,
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

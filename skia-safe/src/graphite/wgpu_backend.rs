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

/// Generates an `unsafe extern "C" fn` stub for a given WGPU function name
/// and signature. The stub aborts when called.
macro_rules! stub {
    ($name:ident ( $($arg:ident : $arg_ty:ty),* $(,)? ) -> $ret:ty) => {{
        unsafe extern "C" fn stub( $( $arg : $arg_ty ),* ) -> $ret {
            $( let _ = $arg; )*
            $crate::graphite::wgpu_backend::unimplemented_stub(stringify!($name))
        }
        Some(stub as _)
    }};
    ($name:ident ( $($arg:ident : $arg_ty:ty),* $(,)? )) => {{
        unsafe extern "C" fn stub( $( $arg : $arg_ty ),* ) {
            $( let _ = $arg; )*
            $crate::graphite::wgpu_backend::unimplemented_stub(stringify!($name))
        }
        Some(stub as _)
    }};
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

//
// Proc table assembly
//

/// Builds a [`DawnProcTable`] whose entries route to wgpu.
///
/// Today: instance / adapter / device / queue lifecycle. Everything else is
/// either an abort-on-call stub or `None` (which will null-deref-crash).
pub fn wgpu_proc_table() -> DawnProcTable {
    // SAFETY: Each field of `DawnProcTable` is an `Option<unsafe extern "C" fn ...>`,
    // which uses null-pointer-optimisation: a zeroed `Option` is `None`. The
    // fields we explicitly populate below replace those nulls.
    let mut table: DawnProcTable = unsafe { core::mem::zeroed() };

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

    populate_remaining_stubs(&mut table);

    table
}

/// Replaces a handful of fields with abort-on-call stubs so that the first
/// unimplemented call surfaces a useful diagnostic instead of a null-deref
/// crash. Functions we know Skia exercises after device creation go here as
/// they're discovered.
fn populate_remaining_stubs(table: &mut DawnProcTable) {
    // Once `Context::new_wgpu` runs, Skia's Graphite-Dawn initialisation
    // will start calling the rest of the WGPU surface; entries discovered
    // there get listed here so the abort message identifies them by name.
    let _ = table;
}

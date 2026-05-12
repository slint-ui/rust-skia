//! Dawn backend factory for Graphite [`Context`].

use std::{fmt, mem::MaybeUninit, ptr};

use skia_bindings::{self as sb, WGPUDevice, WGPUInstance, WGPUQueue, WGPUTexture};

use super::{BackendTexture, Context, ContextOptions};
use crate::prelude::*;

/// Re-export of Dawn's `DawnProcTable` — the function-pointer table that the
/// `dawn_proc` dispatcher consults for every `wgpu*` call. Custom backends or
/// instrumentation layers can build their own table and install it via
/// [`install_proc_table`].
pub use sb::DawnProcTable;

/// Raw WebGPU handles that Graphite needs to construct a Dawn-backed
/// [`Context`].
///
/// The caller is responsible for creating valid `WGPUInstance`, `WGPUDevice`,
/// and `WGPUQueue` handles (e.g. via `wgpuCreateInstance` and the adapter
/// request flow from [`skia_safe::webgpu`](crate::webgpu)). When the Context
/// is constructed, rust-skia bumps the reference count on each handle, so the
/// caller retains ownership of its own references.
#[derive(Copy, Clone, Debug)]
pub struct BackendContext {
    pub instance: WGPUInstance,
    pub device: WGPUDevice,
    pub queue: WGPUQueue,
}

impl Context {
    /// Creates a Graphite Context backed by Dawn.
    ///
    /// # Safety
    ///
    /// `backend.instance`, `backend.device`, and `backend.queue` must be
    /// valid, currently-alive WebGPU handles. They must come from a Dawn
    /// build (typically the one bundled inside this crate's static Skia
    /// library) — passing handles from a different WebGPU implementation
    /// will not work.
    pub unsafe fn new_dawn(
        backend: &BackendContext,
        options: &ContextOptions,
    ) -> Option<Context> {
        Context::from_ptr(sb::C_SkgpuGraphite_ContextFactory_MakeDawn(
            backend.instance,
            backend.device,
            backend.queue,
            options.native(),
        ))
    }
}

/// Owns a Dawn `WGPUInstance`, `WGPUDevice`, and `WGPUQueue` created with
/// platform defaults. Designed for callers who just want "give me a working
/// Dawn setup" without managing the asynchronous WebGPU adapter/device
/// request flow themselves.
///
/// Dropping the device releases each handle. The Context returned by
/// [`Context::new_dawn`] takes its own references and remains valid after
/// this device is dropped (so the usual pattern is: create `DawnDevice`,
/// pass `.backend_context()` to `Context::new_dawn`, keep both alive for
/// the lifetime of the rendering session).
pub struct DawnDevice {
    instance: WGPUInstance,
    device: WGPUDevice,
    queue: WGPUQueue,
}

impl fmt::Debug for DawnDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DawnDevice")
            .field("instance", &self.instance)
            .field("device", &self.device)
            .field("queue", &self.queue)
            .finish()
    }
}

impl DawnDevice {
    /// Creates a default Dawn setup. Picks the platform's preferred GPU API
    /// (Vulkan on Linux/Android, Metal on macOS/iOS, D3D12 on Windows).
    /// Returns `None` if either the adapter or device request fails.
    pub fn new() -> Option<Self> {
        let mut instance: WGPUInstance = ptr::null_mut();
        let mut device: WGPUDevice = ptr::null_mut();
        let mut queue: WGPUQueue = ptr::null_mut();
        let ok = unsafe {
            sb::C_SkgpuGraphite_DawnDefaultSetup(&mut instance, &mut device, &mut queue)
        };
        if !ok {
            return None;
        }
        Some(Self {
            instance,
            device,
            queue,
        })
    }

    pub fn instance(&self) -> WGPUInstance {
        self.instance
    }

    pub fn device(&self) -> WGPUDevice {
        self.device
    }

    pub fn queue(&self) -> WGPUQueue {
        self.queue
    }

    /// Returns a [`BackendContext`] referencing this device's handles, ready
    /// to pass to [`Context::new_dawn`]. The handles remain owned by this
    /// `DawnDevice`; the returned struct is just a view.
    pub fn backend_context(&self) -> BackendContext {
        BackendContext {
            instance: self.instance,
            device: self.device,
            queue: self.queue,
        }
    }
}

impl Drop for DawnDevice {
    fn drop(&mut self) {
        unsafe {
            sb::wgpuQueueRelease(self.queue);
            sb::wgpuDeviceRelease(self.device);
            sb::wgpuInstanceRelease(self.instance);
        }
    }
}

/// Installs `procs` as the global `wgpu*` dispatcher table for the process.
///
/// Dawn's WebGPU C API is routed through a function-pointer table set by
/// `dawnProcSetProcs`. Calling this installs `procs` instead of the default
/// Dawn-native implementation. **First caller wins**: subsequent installs
/// (including the implicit install done by [`DawnDevice::new`]) are no-ops.
/// To take effect, this must be called before any `wgpu*` function is
/// invoked and before constructing a [`DawnDevice`] or [`Context`].
///
/// Typical use cases: routing calls to a different WebGPU implementation
/// (such as `wgpu-native`), instrumenting them, or capturing them for replay.
///
/// # Safety
///
/// `procs` must remain valid for the entire lifetime of the process — store
/// it in a `static` or leak it. Any field of `procs` that is `None` will
/// cause the corresponding `wgpu*` call to dereference null and crash, so
/// the table must be complete enough to cover everything Skia plus the rest
/// of your code uses. Concurrent calls are serialized internally; the first
/// one wins.
pub unsafe fn install_proc_table(procs: &'static DawnProcTable) {
    sb::C_SkgpuGraphite_SetProcTable(procs);
}

/// Returns the proc table that routes every call to Dawn's native
/// implementation. Useful as a starting point: clone, override individual
/// entries (for instrumentation or partial substitution), then install the
/// result via [`install_proc_table`].
///
/// The returned table is a value copy; the function pointers inside it
/// reference statically-linked Dawn symbols that are valid for the program's
/// lifetime.
pub fn default_dawn_proc_table() -> DawnProcTable {
    let mut table = MaybeUninit::<DawnProcTable>::uninit();
    unsafe {
        sb::C_SkgpuGraphite_GetDefaultDawnProcTable(table.as_mut_ptr());
        table.assume_init()
    }
}

/// Wraps an existing `WGPUTexture` (e.g. a swapchain image obtained from
/// `wgpuSurfaceGetCurrentTexture`) as a Graphite [`BackendTexture`]. The
/// resulting BackendTexture queries its metadata (size, format, ...) from the
/// underlying texture.
///
/// # Safety
///
/// `texture` must be a live `WGPUTexture` handle. The BackendTexture does
/// **not** add a reference to it; the caller is responsible for keeping the
/// texture alive for as long as the BackendTexture (and any wrapping Surface
/// or Image) is used.
pub unsafe fn backend_texture_from_wgpu_texture(texture: WGPUTexture) -> Option<BackendTexture> {
    BackendTexture::from_ptr(sb::C_SkgpuGraphite_BackendTextures_MakeDawn(texture))
}

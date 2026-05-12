//! Dawn backend factory for Graphite [`Context`].

use std::{fmt, ptr};

use skia_bindings::{self as sb, WGPUDevice, WGPUInstance, WGPUQueue};

use super::{Context, ContextOptions};
use crate::prelude::*;

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

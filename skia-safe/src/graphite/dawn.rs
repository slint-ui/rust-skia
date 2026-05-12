//! Dawn backend factory for Graphite [`Context`].

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

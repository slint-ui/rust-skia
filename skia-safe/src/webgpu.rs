//! Raw WebGPU C API bindings.
//!
//! When the `graphite` feature is enabled, rust-skia bundles Dawn (Google's
//! native WebGPU implementation) inside the Skia static library. This module
//! re-exports the common WebGPU handle types from `webgpu.h` so callers can
//! construct a `WGPUInstance`, request an adapter and device, and pass the
//! resulting handles into Graphite's context factories.
//!
//! These bindings are raw FFI. Only the most common handle types are re-exported
//! here for discoverability; the full WebGPU C API surface (functions like
//! `wgpuCreateInstance`, descriptor structs, enums, constants) lives in the
//! `skia_bindings` crate root and is accessed directly:
//!
//! ```ignore
//! use skia_safe::webgpu::WGPUInstance;
//! let instance: WGPUInstance =
//!     unsafe { skia_bindings::wgpuCreateInstance(core::ptr::null()) };
//! ```
//!
//! If you already have a `wgpu::Device` from the `wgpu` crate, you can extract
//! its raw `WGPUDevice` handle and pass it in directly without using this
//! module to construct anything.

pub use skia_bindings::{
    WGPUAdapter, WGPUBindGroup, WGPUBindGroupLayout, WGPUBuffer, WGPUCommandBuffer,
    WGPUCommandEncoder, WGPUComputePassEncoder, WGPUComputePipeline, WGPUDevice, WGPUInstance,
    WGPUPipelineLayout, WGPUQuerySet, WGPUQueue, WGPURenderBundle, WGPURenderBundleEncoder,
    WGPURenderPassEncoder, WGPURenderPipeline, WGPUSampler, WGPUShaderModule, WGPUSurface,
    WGPUTexture, WGPUTextureView,
};

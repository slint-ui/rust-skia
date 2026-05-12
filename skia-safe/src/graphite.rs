//! Skia's Graphite GPU backend.
//!
//! Graphite is Skia's newer GPU rendering backend. It records work into a
//! [`Recorder`], snaps it into a [`Recording`], and submits the recording to
//! the GPU via [`Context::submit`]. Under the hood Graphite uses Dawn (a
//! WebGPU implementation), which targets Metal on macOS, Vulkan on
//! Linux/Android, and D3D12 on Windows.
//!
//! This module currently exposes the backend-agnostic core. Backend factories
//! (Dawn, and platform-specific helpers) will be added in a follow-up.

mod context;
pub mod dawn;
mod options;
mod recorder;
mod recording;
pub mod surfaces;
mod types;

pub use context::Context;
pub use options::{ContextOptions, RecorderOptions};
pub use recorder::Recorder;
pub use recording::Recording;
pub use types::{
    BackendApi, CallbackResult, InsertStatus, MarkFrameBoundary, Mipmapped, SubmitInfo, SyncToCpu,
};

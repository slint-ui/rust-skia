use std::fmt;

use skia_bindings::{self as sb, skgpu_graphite_BackendTexture};

use super::BackendApi;
use crate::{prelude::*, ISize};

/// A handle to a GPU texture exposed by one of Graphite's backends. Wrap an
/// existing texture (e.g. a WebGPU swapchain image obtained from
/// `wgpuSurfaceGetCurrentTexture`) via the backend-specific constructor in
/// [`super::dawn`] and then call
/// [`super::surfaces::wrap_backend_texture`] to render onto it.
pub type BackendTexture = RefHandle<skgpu_graphite_BackendTexture>;
unsafe_send_sync!(BackendTexture);

impl NativeDrop for skgpu_graphite_BackendTexture {
    fn drop(&mut self) {
        unsafe { sb::C_SkgpuGraphite_BackendTexture_delete(self) }
    }
}

impl fmt::Debug for BackendTexture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BackendTexture")
            .field("is_valid", &self.is_valid())
            .field("backend", &self.backend())
            .field("dimensions", &self.dimensions())
            .finish()
    }
}

impl BackendTexture {
    pub fn is_valid(&self) -> bool {
        unsafe { sb::C_SkgpuGraphite_BackendTexture_isValid(self.native()) }
    }

    pub fn backend(&self) -> BackendApi {
        unsafe { sb::C_SkgpuGraphite_BackendTexture_backend(self.native()) }
    }

    pub fn width(&self) -> i32 {
        unsafe { sb::C_SkgpuGraphite_BackendTexture_width(self.native()) }
    }

    pub fn height(&self) -> i32 {
        unsafe { sb::C_SkgpuGraphite_BackendTexture_height(self.native()) }
    }

    pub fn dimensions(&self) -> ISize {
        ISize::new(self.width(), self.height())
    }
}

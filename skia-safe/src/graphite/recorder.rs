use std::fmt;

use skia_bindings::{self as sb, skgpu_graphite_Recorder};

use super::{BackendApi, Recording};
use crate::prelude::*;

pub type Recorder = RefHandle<skgpu_graphite_Recorder>;
unsafe_send_sync!(Recorder);

impl NativeDrop for skgpu_graphite_Recorder {
    fn drop(&mut self) {
        unsafe { sb::C_SkgpuGraphiteRecorder_delete(self) }
    }
}

impl fmt::Debug for Recorder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Recorder")
            .field("backend", &self.backend())
            .field("max_texture_size", &self.max_texture_size())
            .finish()
    }
}

impl Recorder {
    pub fn backend(&self) -> BackendApi {
        unsafe { sb::C_SkgpuGraphiteRecorder_backend(self.native()) }
    }

    /// Captures the work recorded so far into a [`Recording`] and resets the
    /// recorder. The recording can then be passed to
    /// [`Context::insert_recording`](super::Context) and submitted.
    pub fn snap(&mut self) -> Option<Recording> {
        Recording::from_ptr(unsafe { sb::C_SkgpuGraphiteRecorder_snap(self.native_mut()) })
    }

    pub fn max_texture_size(&self) -> i32 {
        unsafe { sb::C_SkgpuGraphiteRecorder_maxTextureSize(self.native()) }
    }

    pub fn free_gpu_resources(&mut self) {
        unsafe { sb::C_SkgpuGraphiteRecorder_freeGpuResources(self.native_mut()) }
    }

    pub fn current_budgeted_bytes(&self) -> usize {
        unsafe { sb::C_SkgpuGraphiteRecorder_currentBudgetedBytes(self.native()) }
    }

    pub fn current_purgeable_bytes(&self) -> usize {
        unsafe { sb::C_SkgpuGraphiteRecorder_currentPurgeableBytes(self.native()) }
    }

    pub fn max_budgeted_bytes(&self) -> usize {
        unsafe { sb::C_SkgpuGraphiteRecorder_maxBudgetedBytes(self.native()) }
    }

    pub fn set_max_budgeted_bytes(&mut self, bytes: usize) {
        unsafe { sb::C_SkgpuGraphiteRecorder_setMaxBudgetedBytes(self.native_mut(), bytes) }
    }
}

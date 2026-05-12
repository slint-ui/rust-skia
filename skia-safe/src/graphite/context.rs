use std::fmt;

use skia_bindings::{self as sb, skgpu_graphite_Context};

use super::{BackendApi, Recorder, RecorderOptions, SubmitInfo};
use crate::prelude::*;

pub type Context = RefHandle<skgpu_graphite_Context>;
unsafe_send_sync!(Context);

impl NativeDrop for skgpu_graphite_Context {
    fn drop(&mut self) {
        unsafe { sb::C_SkgpuGraphiteContext_delete(self) }
    }
}

impl fmt::Debug for Context {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Context")
            .field("backend", &self.backend())
            .field("max_texture_size", &self.max_texture_size())
            .field("is_device_lost", &self.is_device_lost())
            .finish()
    }
}

impl Context {
    pub fn backend(&self) -> BackendApi {
        unsafe { sb::C_SkgpuGraphiteContext_backend(self.native()) }
    }

    pub fn make_recorder(&mut self, options: &RecorderOptions) -> Option<Recorder> {
        Recorder::from_ptr(unsafe {
            sb::C_SkgpuGraphiteContext_makeRecorder(self.native_mut(), options.native())
        })
    }

    pub fn submit(&mut self, info: &SubmitInfo) -> bool {
        unsafe { sb::C_SkgpuGraphiteContext_submit(self.native_mut(), info) }
    }

    pub fn has_unfinished_gpu_work(&self) -> bool {
        unsafe { sb::C_SkgpuGraphiteContext_hasUnfinishedGpuWork(self.native()) }
    }

    pub fn has_pending_gpu_work(&self) -> bool {
        unsafe { sb::C_SkgpuGraphiteContext_hasPendingGPUWork(self.native()) }
    }

    pub fn check_async_work_completion(&mut self) {
        unsafe { sb::C_SkgpuGraphiteContext_checkAsyncWorkCompletion(self.native_mut()) }
    }

    pub fn is_device_lost(&self) -> bool {
        unsafe { sb::C_SkgpuGraphiteContext_isDeviceLost(self.native()) }
    }

    pub fn max_texture_size(&self) -> i32 {
        unsafe { sb::C_SkgpuGraphiteContext_maxTextureSize(self.native()) }
    }

    pub fn supports_protected_content(&self) -> bool {
        unsafe { sb::C_SkgpuGraphiteContext_supportsProtectedContent(self.native()) }
    }

    pub fn free_gpu_resources(&mut self) {
        unsafe { sb::C_SkgpuGraphiteContext_freeGpuResources(self.native_mut()) }
    }

    pub fn current_budgeted_bytes(&self) -> usize {
        unsafe { sb::C_SkgpuGraphiteContext_currentBudgetedBytes(self.native()) }
    }

    pub fn current_purgeable_bytes(&self) -> usize {
        unsafe { sb::C_SkgpuGraphiteContext_currentPurgeableBytes(self.native()) }
    }

    pub fn max_budgeted_bytes(&self) -> usize {
        unsafe { sb::C_SkgpuGraphiteContext_maxBudgetedBytes(self.native()) }
    }

    pub fn set_max_budgeted_bytes(&mut self, bytes: usize) {
        unsafe { sb::C_SkgpuGraphiteContext_setMaxBudgetedBytes(self.native_mut(), bytes) }
    }
}

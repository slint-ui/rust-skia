use std::fmt;

use skia_bindings::{self as sb, skgpu_graphite_Context};

use super::{BackendApi, InsertStatus, Recorder, RecorderOptions, Recording, SubmitInfo};
use crate::{prelude::*, IRect, ImageInfo, Surface};

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
        unsafe { sb::C_SkgpuGraphiteContext_submit(self.native_mut(), info.native()) }
    }

    /// Replays a Recording against the optional target Surface and queues
    /// resulting GPU work for the next [`Self::submit`]. The Recording is
    /// borrowed; the caller retains ownership.
    pub fn insert_recording(
        &mut self,
        recording: &mut Recording,
        target_surface: Option<&mut Surface>,
    ) -> InsertStatus {
        unsafe {
            sb::C_SkgpuGraphite_Context_insertRecording(
                self.native_mut(),
                recording.native_mut(),
                target_surface.map_or(core::ptr::null_mut(), |s| s.native_mut()),
            )
        }
    }

    /// Reads a rectangle of pixels from a Graphite-backed Surface into a
    /// caller-supplied buffer. Internally schedules an asynchronous read
    /// against `Context::asyncRescaleAndReadPixels` and then drives any
    /// pending GPU work to completion via `submit(SyncToCpu::Yes)`, returning
    /// once the callback has copied the data (or signalled failure).
    ///
    /// `dst_pixels` must be at least `dst_row_bytes * dst_info.height()` bytes
    /// long. Returns `false` if the buffer is too small, if the underlying
    /// read fails, or if the submit fails.
    pub fn read_pixels(
        &mut self,
        surface: &Surface,
        dst_info: &ImageInfo,
        dst_pixels: &mut [u8],
        dst_row_bytes: usize,
        src: impl Into<IRect>,
    ) -> bool {
        let height: usize = dst_info.height().max(0) as usize;
        if dst_row_bytes
            .checked_mul(height)
            .map_or(true, |needed| dst_pixels.len() < needed)
        {
            return false;
        }
        let src = src.into();
        unsafe {
            sb::C_SkgpuGraphite_Context_readPixelsBlocking(
                self.native_mut(),
                surface.native(),
                dst_info.native(),
                src.native(),
                dst_pixels.as_mut_ptr() as _,
                dst_row_bytes,
            )
        }
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

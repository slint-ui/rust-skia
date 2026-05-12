use core::ptr;

use skia_bindings as sb;

pub use sb::skgpu_BackendApi as BackendApi;
variant_name!(BackendApi::Dawn);
variant_name!(BackendApi::Metal);
variant_name!(BackendApi::Vulkan);
variant_name!(BackendApi::Unsupported);

pub use sb::skgpu_CallbackResult as CallbackResult;
variant_name!(CallbackResult::Success);
variant_name!(CallbackResult::Failed);

pub use sb::skgpu_graphite_SyncToCpu as SyncToCpu;
variant_name!(SyncToCpu::Yes);
variant_name!(SyncToCpu::No);

pub use sb::skgpu_graphite_MarkFrameBoundary as MarkFrameBoundary;
variant_name!(MarkFrameBoundary::Yes);
variant_name!(MarkFrameBoundary::No);

/// Parameters for [`Context::submit`](super::Context::submit).
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct SubmitInfo(sb::skgpu_graphite_SubmitInfo);

impl Default for SubmitInfo {
    fn default() -> Self {
        Self(sb::skgpu_graphite_SubmitInfo {
            fSync: SyncToCpu::No,
            fMarkBoundary: MarkFrameBoundary::No,
            fFrameID: 0,
            fFinishedProc: None,
            fFinishedContext: ptr::null_mut(),
        })
    }
}

impl SubmitInfo {
    pub fn sync_to_cpu() -> Self {
        Self::default().with_sync_to_cpu(SyncToCpu::Yes)
    }

    pub fn with_sync_to_cpu(mut self, sync: SyncToCpu) -> Self {
        self.0.fSync = sync;
        self
    }

    pub fn with_frame_boundary(mut self, mark: MarkFrameBoundary, frame_id: u64) -> Self {
        self.0.fMarkBoundary = mark;
        self.0.fFrameID = frame_id;
        self
    }

    pub(crate) fn native(&self) -> &sb::skgpu_graphite_SubmitInfo {
        &self.0
    }
}

pub use sb::skgpu_Mipmapped as Mipmapped;
variant_name!(Mipmapped::Yes);
variant_name!(Mipmapped::No);

pub use sb::skgpu_graphite_InsertStatus_V as InsertStatus;
variant_name!(InsertStatus::Success);
variant_name!(InsertStatus::InvalidRecording);
variant_name!(InsertStatus::PromiseImageInstantiationFailed);
variant_name!(InsertStatus::AddCommandsFailed);
variant_name!(InsertStatus::AsyncShaderCompilesFailed);
variant_name!(InsertStatus::OutOfOrderRecording);

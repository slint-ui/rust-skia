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

pub use sb::skgpu_graphite_SubmitInfo as SubmitInfo;

use skia_bindings::{
    self as sb, skgpu_graphite_ContextOptions, skgpu_graphite_RecorderOptions,
};

use crate::prelude::*;

pub type ContextOptions = Handle<skgpu_graphite_ContextOptions>;
unsafe_send_sync!(ContextOptions);

impl NativeDrop for skgpu_graphite_ContextOptions {
    fn drop(&mut self) {
        unsafe { sb::C_SkgpuGraphiteContextOptions_destruct(self) }
    }
}

impl Default for ContextOptions {
    fn default() -> Self {
        Self::construct(|opts| unsafe { sb::C_SkgpuGraphiteContextOptions_Construct(opts) })
    }
}

pub type RecorderOptions = Handle<skgpu_graphite_RecorderOptions>;
unsafe_send_sync!(RecorderOptions);

impl NativeDrop for skgpu_graphite_RecorderOptions {
    fn drop(&mut self) {
        unsafe { sb::C_SkgpuGraphiteRecorderOptions_destruct(self) }
    }
}

impl Default for RecorderOptions {
    fn default() -> Self {
        Self::construct(|opts| unsafe { sb::C_SkgpuGraphiteRecorderOptions_Construct(opts) })
    }
}

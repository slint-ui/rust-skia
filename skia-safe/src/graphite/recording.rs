use std::fmt;

use skia_bindings::{self as sb, skgpu_graphite_Recording};

use crate::prelude::*;

pub type Recording = RefHandle<skgpu_graphite_Recording>;
unsafe_send_sync!(Recording);

impl NativeDrop for skgpu_graphite_Recording {
    fn drop(&mut self) {
        unsafe { sb::C_SkgpuGraphiteRecording_delete(self) }
    }
}

impl fmt::Debug for Recording {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Recording").finish_non_exhaustive()
    }
}

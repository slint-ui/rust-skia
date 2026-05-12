//! Factory functions for Graphite-backed [`Surface`](crate::Surface) objects.

use skia_bindings as sb;

use super::{Mipmapped, Recorder};
use crate::{prelude::*, ImageInfo, Surface, SurfaceProps};

/// Allocates a renderable surface backed by a Graphite texture sized to
/// `image_info`. The surface is owned by the caller and must outlive any
/// Recording that targets it.
pub fn render_target(
    recorder: &mut Recorder,
    image_info: &ImageInfo,
    mipmapped: impl Into<Option<Mipmapped>>,
    surface_props: Option<&SurfaceProps>,
) -> Option<Surface> {
    Surface::from_ptr(unsafe {
        sb::C_SkgpuGraphite_Surfaces_RenderTarget(
            recorder.native_mut(),
            image_info.native(),
            mipmapped.into().unwrap_or(Mipmapped::No),
            surface_props.native_ptr_or_null(),
        )
    })
}

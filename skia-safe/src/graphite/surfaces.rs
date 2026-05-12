//! Factory functions for Graphite-backed [`Surface`](crate::Surface) objects.

use skia_bindings as sb;

use super::{BackendTexture, Mipmapped, Recorder};
use crate::{prelude::*, ColorSpace, ImageInfo, Surface, SurfaceProps};

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

/// Wraps an existing [`BackendTexture`] as a Surface. The caller is
/// responsible for keeping the underlying GPU texture alive for as long as the
/// Surface is in use. The Surface's color type is inferred from the texture's
/// format.
pub fn wrap_backend_texture(
    recorder: &mut Recorder,
    backend_texture: &BackendTexture,
    color_space: impl Into<Option<ColorSpace>>,
    surface_props: Option<&SurfaceProps>,
) -> Option<Surface> {
    Surface::from_ptr(unsafe {
        sb::C_SkgpuGraphite_Surfaces_WrapBackendTexture(
            recorder.native_mut(),
            backend_texture.native(),
            color_space.into().into_ptr_or_null(),
            surface_props.native_ptr_or_null(),
        )
    })
}

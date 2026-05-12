//! Windowed Graphite demo that shares a single `wgpu` device with Skia.
//!
//! Runs winit + wgpu to open a window and own the swapchain. Each frame, the
//! current swapchain texture is wrapped as a Skia `BackendTexture` via the
//! `wgpu` feature's proc-table dispatch, drawn into with the Skia canvas, and
//! then presented through wgpu. The point of this example is to show how a
//! host application that already drives its rendering through `wgpu` can pull
//! Skia/Graphite into the same device — no separate Dawn instance, no extra
//! GPU device.
//!
//! Run with:
//! `cargo run -p skia-safe --no-default-features --features graphite,wgpu --example graphite-wgpu-window`

#[cfg(not(all(feature = "graphite", feature = "wgpu")))]
fn main() {
    eprintln!(
        "This example requires the `graphite` and `wgpu` features. Run with:\n  \
         cargo run -p skia-safe --no-default-features --features graphite,wgpu \
         --example graphite-wgpu-window"
    );
}

#[cfg(all(feature = "graphite", feature = "wgpu"))]
use std::sync::Arc;

#[cfg(all(feature = "graphite", feature = "wgpu"))]
use skia_safe::{
    graphite::{
        self,
        dawn::{self, BackendContext},
        wgpu_backend,
    },
    Color, Color4f, Paint, PaintStyle, Point, Rect,
};
#[cfg(all(feature = "graphite", feature = "wgpu"))]
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowAttributes, WindowId},
};

#[cfg(all(feature = "graphite", feature = "wgpu"))]
fn main() {
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App::default();
    event_loop.run_app(&mut app).expect("event loop run");
}

#[cfg(all(feature = "graphite", feature = "wgpu"))]
#[derive(Default)]
struct App {
    state: Option<State>,
}

#[cfg(all(feature = "graphite", feature = "wgpu"))]
struct State {
    window: Arc<Window>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    skia_context: graphite::Context,
    recorder: graphite::Recorder,
}

#[cfg(all(feature = "graphite", feature = "wgpu"))]
impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let attrs = WindowAttributes::default()
            .with_title("Skia / Graphite / wgpu")
            .with_inner_size(winit::dpi::LogicalSize::new(800.0, 600.0));
        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("create_window"),
        );

        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let surface = instance
            .create_surface(window.clone())
            .expect("create_surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("request_adapter");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("request_device");

        // Pick a non-sRGB surface format. Skia maps Bgra8Unorm / Rgba8Unorm to
        // its own ColorType cleanly. sRGB formats also work but apply gamma
        // twice when combined with Skia's color management, so we steer clear.
        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps
            .formats
            .iter()
            .copied()
            .find(|f| {
                matches!(
                    f,
                    wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm
                )
            })
            .unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: caps.present_modes[0],
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        // Hand wgpu objects to Skia. install_and_wrap clones what it needs;
        // we keep our own device/queue/surface clones for swapchain work.
        let backend: BackendContext = wgpu_backend::install_and_wrap(
            instance,
            adapter,
            device.clone(),
            queue.clone(),
        );
        let mut skia_context = unsafe {
            graphite::Context::new_dawn(&backend, &graphite::ContextOptions::default())
        }
        .expect("graphite::Context::new_dawn");
        let recorder = skia_context
            .make_recorder(&graphite::RecorderOptions::default())
            .expect("make_recorder");

        self.state = Some(State {
            window,
            device,
            queue,
            surface,
            surface_config,
            skia_context,
            recorder,
        });
        self.state.as_ref().unwrap().window.request_redraw();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(new_size) => {
                state.surface_config.width = new_size.width.max(1);
                state.surface_config.height = new_size.height.max(1);
                state.surface.configure(&state.device, &state.surface_config);
                state.window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                state.render();
                state.window.request_redraw();
            }
            _ => {}
        }
    }
}

#[cfg(all(feature = "graphite", feature = "wgpu"))]
impl State {
    fn render(&mut self) {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f)
            | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.surface_config);
                return;
            }
            other => {
                eprintln!("get_current_texture: {other:?}");
                return;
            }
        };

        // The texture is reference-counted on the wgpu side, so cloning is
        // cheap and the swapchain image stays live until both the wgpu
        // SurfaceTexture and our Skia wrapper drop their refs.
        let texture_clone = frame.texture.clone();
        let wgpu_handle = wgpu_backend::wrap_wgpu_texture(texture_clone, self.device.clone());
        let Some(backend_tex) =
            (unsafe { dawn::backend_texture_from_wgpu_texture(wgpu_handle) })
        else {
            unsafe { wgpu_backend::release_wgpu_texture(wgpu_handle) };
            return;
        };

        let Some(mut skia_surface) = graphite::surfaces::wrap_backend_texture(
            &mut self.recorder,
            &backend_tex,
            None,
            None,
        ) else {
            drop(backend_tex);
            unsafe { wgpu_backend::release_wgpu_texture(wgpu_handle) };
            return;
        };

        draw_scene(skia_surface.canvas());

        if let Some(mut recording) = self.recorder.snap() {
            let status = self
                .skia_context
                .insert_recording(&mut recording, Some(&mut skia_surface));
            if status != graphite::InsertStatus::Success {
                eprintln!("insert_recording: {status:?}");
            }
        }

        drop(skia_surface);
        drop(backend_tex);
        unsafe { wgpu_backend::release_wgpu_texture(wgpu_handle) };

        // Submit any work the recording deferred and present.
        self.queue.submit(std::iter::empty::<wgpu::CommandBuffer>());
        frame.present();
    }
}

#[cfg(all(feature = "graphite", feature = "wgpu"))]
fn draw_scene(canvas: &skia_safe::Canvas) {
    let (w, h) = {
        let s = canvas.base_layer_size();
        (s.width as f32, s.height as f32)
    };
    canvas.clear(Color4f::new(0.95, 0.95, 0.97, 1.0));

    let mut fill = Paint::default();
    fill.set_anti_alias(true);
    fill.set_color(Color::from_argb(255, 0x33, 0x66, 0xCC));
    canvas.draw_rect(
        Rect::from_xywh(w * 0.1, h * 0.2, w * 0.35, h * 0.6),
        &fill,
    );

    fill.set_color(Color::from_argb(255, 0xCC, 0x33, 0x66));
    canvas.draw_circle(Point::new(w * 0.7, h * 0.5), w.min(h) * 0.2, &fill);

    let mut stroke = Paint::default();
    stroke.set_anti_alias(true);
    stroke.set_style(PaintStyle::Stroke);
    stroke.set_stroke_width(4.0);
    stroke.set_color(Color::from_argb(255, 0x22, 0x22, 0x22));
    canvas.draw_rect(
        Rect::from_xywh(w * 0.1, h * 0.2, w * 0.35, h * 0.6),
        &stroke,
    );
}

//! Headless Graphite demo: renders into a Dawn-backed Surface and writes a
//! PNG. Useful as a smoke test before integrating Graphite into a real app.
//!
//! Run with `cargo run -p skia-safe --features graphite --no-default-features
//! --example graphite-png`. The output is written to `graphite-output.png` in
//! the current directory.

use skia_safe::{
    graphite::{self, dawn::DawnDevice},
    images, AlphaType, Color, ColorType, Data, EncodedImageFormat, IRect, ImageInfo, Paint,
    PaintStyle, PathBuilder, Point, Rect,
};

const WIDTH: i32 = 512;
const HEIGHT: i32 = 512;

fn main() {
    let Some(dawn) = DawnDevice::new() else {
        eprintln!("No Dawn-compatible GPU available; aborting.");
        std::process::exit(1);
    };

    let backend = dawn.backend_context();
    let mut ctx = unsafe {
        graphite::Context::new_dawn(&backend, &graphite::ContextOptions::default())
    }
    .expect("Context::new_dawn");

    let mut recorder = ctx
        .make_recorder(&graphite::RecorderOptions::default())
        .expect("make_recorder");

    let info = ImageInfo::new(
        (WIDTH, HEIGHT),
        ColorType::RGBA8888,
        AlphaType::Premul,
        None,
    );
    let mut surface = graphite::surfaces::render_target(&mut recorder, &info, None, None)
        .expect("render_target");

    draw_scene(surface.canvas());

    let mut recording = recorder.snap().expect("snap");
    let status = ctx.insert_recording(&mut recording, Some(&mut surface));
    assert_eq!(status, graphite::InsertStatus::Success, "insert_recording");

    let mut pixels = vec![0u8; (WIDTH * HEIGHT * 4) as usize];
    let ok = ctx.read_pixels(
        &surface,
        &info,
        &mut pixels,
        (WIDTH * 4) as usize,
        IRect::new(0, 0, WIDTH, HEIGHT),
    );
    assert!(ok, "read_pixels");

    // Build a raster Image from the readback buffer and encode as PNG.
    let data = Data::new_copy(&pixels);
    let image =
        images::raster_from_data(&info, data, (WIDTH * 4) as usize).expect("raster image");
    let png = image
        .encode(None, EncodedImageFormat::PNG, None)
        .expect("encode PNG");
    let path = "graphite-output.png";
    std::fs::write(path, png.as_bytes()).expect("write PNG");
    println!("Wrote {path} ({WIDTH}x{HEIGHT})");
}

fn draw_scene(canvas: &skia_safe::Canvas) {
    // Dark slate background.
    canvas.clear(Color::from_argb(0xFF, 0x1E, 0x1E, 0x2E));

    // Four colored corner discs.
    let radius = 80.0;
    let inset = 20.0;
    let discs = [
        (Point::new(inset + radius, inset + radius), Color::RED),
        (
            Point::new(WIDTH as f32 - inset - radius, inset + radius),
            Color::GREEN,
        ),
        (
            Point::new(inset + radius, HEIGHT as f32 - inset - radius),
            Color::BLUE,
        ),
        (
            Point::new(
                WIDTH as f32 - inset - radius,
                HEIGHT as f32 - inset - radius,
            ),
            Color::YELLOW,
        ),
    ];
    for (center, color) in discs {
        let mut paint = Paint::default();
        paint.set_anti_alias(true);
        paint.set_color(color);
        canvas.draw_circle(center, radius, &paint);
    }

    // White rounded rectangle in the middle with a thicker outline.
    let rect_w = 260.0;
    let rect_h = 140.0;
    let rect = Rect::new(
        (WIDTH as f32 - rect_w) * 0.5,
        (HEIGHT as f32 - rect_h) * 0.5,
        (WIDTH as f32 + rect_w) * 0.5,
        (HEIGHT as f32 + rect_h) * 0.5,
    );
    let mut fill = Paint::default();
    fill.set_anti_alias(true);
    fill.set_color(Color::WHITE);
    canvas.draw_round_rect(rect, 16.0, 16.0, &fill);

    let mut stroke = Paint::default();
    stroke.set_anti_alias(true);
    stroke.set_color(Color::from_argb(0xFF, 0x33, 0x33, 0x55));
    stroke.set_style(PaintStyle::Stroke);
    stroke.set_stroke_width(6.0);
    canvas.draw_round_rect(rect, 16.0, 16.0, &stroke);

    // A simple geometric shape inside the rounded rect (a chevron path) to
    // exercise path rendering beyond circles and rects.
    let cx = WIDTH as f32 * 0.5;
    let cy = HEIGHT as f32 * 0.5;
    let mut path_builder = PathBuilder::new();
    path_builder.move_to((cx - 80.0, cy + 20.0));
    path_builder.line_to((cx, cy - 30.0));
    path_builder.line_to((cx + 80.0, cy + 20.0));
    let path = path_builder.detach();
    let mut chevron = Paint::default();
    chevron.set_anti_alias(true);
    chevron.set_color(Color::from_argb(0xFF, 0x66, 0x33, 0xCC));
    chevron.set_style(PaintStyle::Stroke);
    chevron.set_stroke_width(10.0);
    canvas.draw_path(&path, &chevron);
}

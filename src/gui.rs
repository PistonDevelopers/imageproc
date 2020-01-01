//! Displays an image in a window created by sdl2.

use image::{imageops::resize, RgbaImage};
use sdl2::{
    event::{Event, WindowEvent},
    keyboard::Keycode,
    pixels::{Color, PixelFormatEnum},
    rect::Rect,
    render::{Canvas, TextureCreator, WindowCanvas},
    surface::Surface,
    video::{Window, WindowContext},
};

/// Displays the provided RGBA image in a new window.
///
/// The minimum window width or height is 150 pixels - input values less than this
/// will be rounded up to the minimum.
pub fn display_image(title: &str, image: &RgbaImage, window_width: u32, window_height: u32) {
    display_multiple_images(title, &vec![image], window_width, window_height);
}

/// Displays the provided RGBA images in new windows.
///
/// The minimum window width or height is 150 pixels - input values less than this
/// will be rounded up to the minimum.
pub fn display_multiple_images(
    title: &str,
    images: &[&RgbaImage],
    window_width: u32,
    window_height: u32,
) {
    if images.len() == 0 {
        return;
    }

    // Enforce minimum window size
    const MIN_WINDOW_DIMENSION: u32 = 150;
    let window_width = window_width.max(MIN_WINDOW_DIMENSION);
    let window_height = window_height.max(MIN_WINDOW_DIMENSION);

    // Initialise sdl2 window
    let sdl = sdl2::init().expect("couldn't create sdl2 context");
    let video_subsystem = sdl.video().expect("couldn't create video subsystem");

    let mut windows: Vec<Window> = Vec::with_capacity(images.len());
    let mut window_visibility: Vec<bool> = Vec::with_capacity(images.len());
    for _ in 0..images.len() {
        let mut window = video_subsystem
            .window(title, window_width, window_height)
            .resizable()
            .allow_highdpi()
            .build()
            .expect("couldn't create window");

        window
            .set_minimum_size(MIN_WINDOW_DIMENSION, MIN_WINDOW_DIMENSION)
            .expect("invalid minimum size for window");

        windows.push(window);
        window_visibility.push(true);
    }

    {
        use sdl2::video::WindowPos::Positioned;

        let (base_position_x, base_position_y) = windows[0].position();
        for (i, window) in windows.iter_mut().enumerate() {
            let multiplier = 1.0 + i as f32 / 20.0;
            window.set_position(
                Positioned((base_position_x as f32 * multiplier) as i32),
                Positioned(base_position_y),
            );

            let (window_pos_x, _window_pos_y) = window.position();
            let display_bounds = video_subsystem
                .display_bounds(0)
                .expect("No bounds found for that display.");
            let screen_width = display_bounds.w;
            if window_pos_x + window_width as i32 > screen_width {
                window.set_position(
                    Positioned(screen_width - window_width as i32),
                    Positioned(base_position_y),
                );
            }
        }
    }

    let mut canvases: Vec<WindowCanvas> = Vec::with_capacity(images.len());
    for window in windows.into_iter() {
        let canvas = window
            .into_canvas()
            .software()
            .build()
            .expect("couldn't create canvas");
        canvases.push(canvas);
    }

    let mut texture_creators: Vec<TextureCreator<WindowContext>> = Vec::with_capacity(images.len());
    for canvas in canvases.iter() {
        let texture_creator = canvas.texture_creator();
        texture_creators.push(texture_creator);
    }

    // Shrinks input image to fit if required and renders to the sdl canvas
    let render_image_to_canvas =
        |image,
         window_width,
         window_height,
         canvas: &mut Canvas<Window>,
         texture_creator: &TextureCreator<WindowContext>| {
            let scaled_image = resize_to_fit(image, window_width, window_height);

            let (image_width, image_height) = scaled_image.dimensions();
            let mut buffer = scaled_image.into_raw();
            const CHANNEL_COUNT: u32 = 4;
            let surface = Surface::from_data(
                &mut buffer,
                image_width,
                image_height,
                image_width * CHANNEL_COUNT,
                PixelFormatEnum::ABGR8888, // sdl2 expects bits from highest to lowest
            )
            .expect("couldn't create surface");

            let texture = texture_creator
                .create_texture_from_surface(surface)
                .expect("couldn't create texture from surface");

            canvas.set_draw_color(Color::RGB(255, 255, 255));
            canvas.clear();

            let left = ((window_width - image_width) as f32 / 2f32) as i32;
            let top = ((window_height - image_height) as f32 / 2f32) as i32;
            canvas
                .copy(
                    &texture,
                    None,
                    Rect::new(left, top, image_width, image_height),
                )
                .unwrap();
            canvas.present();
        };

    for (i, (canvas, texture_creator)) in
        canvases.iter_mut().zip(texture_creators.iter()).enumerate()
    {
        render_image_to_canvas(
            images[i],
            window_width,
            window_height,
            canvas,
            texture_creator,
        );
    }

    let mut hidden_count = 0;

    // Create and start event loop to keep window open until Esc
    let mut event_pump = sdl.event_pump().unwrap();
    event_pump.enable_event(sdl2::event::EventType::Window);
    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'running,
                Event::KeyDown {
                    keycode: Some(Keycode::Q),
                    window_id,
                    ..
                } => {
                    for (i, canvas) in canvases.iter_mut().enumerate() {
                        if window_id == canvas.window().id() {
                            canvas.window_mut().hide();
                            window_visibility[i] = false;
                            hidden_count += 1;
                        }
                        if hidden_count == images.len() {
                            break 'running;
                        }
                    }
                }
                Event::Window {
                    win_event: WindowEvent::Close,
                    window_id,
                    ..
                } => {
                    for (i, canvas) in canvases.iter_mut().enumerate() {
                        if window_id == canvas.window().id() {
                            canvas.window_mut().hide();
                            window_visibility[i] = false;
                            hidden_count += 1;
                        }
                        if hidden_count == images.len() {
                            break 'running;
                        }
                    }
                }
                Event::Window {
                    win_event: WindowEvent::Resized(w, h),
                    window_id,
                    ..
                } => {
                    for (i, (canvas, texture_creator)) in
                        canvases.iter_mut().zip(texture_creators.iter()).enumerate()
                    {
                        if window_id == canvas.window().id() {
                            render_image_to_canvas(
                                images[i],
                                w as u32,
                                h as u32,
                                canvas,
                                texture_creator,
                            );
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

// Scale input image down if required so that it fits within a window of the given dimensions
fn resize_to_fit(image: &RgbaImage, window_width: u32, window_height: u32) -> RgbaImage {
    if image.height() < window_height && image.width() < window_width {
        return image.clone();
    }

    let scale = {
        let width_scale = window_width as f32 / image.width() as f32;
        let height_scale = window_height as f32 / image.height() as f32;
        width_scale.min(height_scale)
    };

    let height = (scale * image.height() as f32) as u32;
    let width = (scale * image.width() as f32) as u32;

    resize(image, width, height, image::FilterType::Triangle)
}

//! An example of template matching in a greyscale image.

use image::{open, GenericImage, GrayImage, Luma, Rgb, RgbImage};
use imageproc::definitions::Image;
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::map::map_colors;
use imageproc::rect::Rect;
use imageproc::template_matching::{match_template, MatchTemplateMethod};
use std::env;
use std::f32;
use std::fs;
use std::path::PathBuf;

struct TemplateMatchingArgs {
    input_path: PathBuf,
    output_dir: PathBuf,
    template_x: u32,
    template_y: u32,
    template_w: u32,
    template_h: u32,
}

impl TemplateMatchingArgs {
    fn parse(args: Vec<String>) -> TemplateMatchingArgs {
        if args.len() != 7 {
            panic!(
                r#"
Usage:

     cargo run --example template_matching input_path output_dir template_x template_y template_w template_h

Loads the image at input_path and extracts a region with the given location and size to use as the matching
template. Calls match_template on the input image and this template, and saves the results to output_dir.
"#
            );
        }

        let input_path = PathBuf::from(&args[1]);
        let output_dir = PathBuf::from(&args[2]);
        let template_x = args[3].parse().unwrap();
        let template_y = args[4].parse().unwrap();
        let template_w = args[5].parse().unwrap();
        let template_h = args[6].parse().unwrap();

        TemplateMatchingArgs {
            input_path,
            output_dir,
            template_x,
            template_y,
            template_w,
            template_h,
        }
    }
}

/// Convert an f32-valued image to a 8 bit depth, covering the whole
/// available intensity range.
fn convert_to_gray_image(image: &Image<Luma<f32>>) -> GrayImage {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;

    for p in image.iter() {
        lo = if *p < lo { *p } else { lo };
        hi = if *p > hi { *p } else { hi };
    }

    let range = hi - lo;
    let scale = |x| (255.0 * (x - lo) / range) as u8;
    map_colors(image, |p| Luma([scale(p[0])]))
}

fn copy_sub_image(image: &GrayImage, x: u32, y: u32, w: u32, h: u32) -> GrayImage {
    assert!(
        x + w < image.width() && y + h < image.height(),
        "invalid sub-image"
    );

    let mut result = GrayImage::new(w, h);
    for sy in 0..h {
        for sx in 0..w {
            result.put_pixel(sx, sy, *image.get_pixel(x + sx, y + sy));
        }
    }

    result
}

fn draw_green_rect(image: &GrayImage, rect: Rect) -> RgbImage {
    let mut color_image = map_colors(image, |p| Rgb([p[0], p[0], p[0]]));
    draw_hollow_rect_mut(&mut color_image, rect, Rgb([0, 255, 0]));
    color_image
}

fn run_match_template(
    args: &TemplateMatchingArgs,
    image: &GrayImage,
    template: &GrayImage,
    method: MatchTemplateMethod,
) -> RgbImage {
    // Match the template and convert to u8 depth to display
    let result = match_template(image, template, method);
    let result_scaled = convert_to_gray_image(&result);

    // Pad the result to the same size as the input image, to make them easier to compare
    let mut result_padded = GrayImage::new(image.width(), image.height());
    result_padded
        .copy_from(&result_scaled, args.template_w / 2, args.template_h / 2)
        .unwrap();

    // Show location the template was extracted from
    let roi = Rect::at(args.template_x as i32, args.template_y as i32)
        .of_size(args.template_w, args.template_h);

    draw_green_rect(&result_padded, roi)
}

fn main() {
    let args = TemplateMatchingArgs::parse(env::args().collect());

    let input_path = &args.input_path;
    let output_dir = &args.output_dir;

    if !output_dir.is_dir() {
        fs::create_dir(output_dir).expect("Failed to create output directory")
    }

    if !input_path.is_file() {
        panic!("Input file does not exist");
    }

    // Load image and convert to grayscale
    let image = open(input_path)
        .expect(&format!("Could not load image at {:?}", input_path))
        .to_luma();

    // Extract the requested image sub-region to use as the template
    let template = copy_sub_image(
        &image,
        args.template_x,
        args.template_y,
        args.template_w,
        args.template_h,
    );

    // Match using all available match methods
    let sse = run_match_template(
        &args,
        &image,
        &template,
        MatchTemplateMethod::SumOfSquaredErrors,
    );
    let sse_norm = run_match_template(
        &args,
        &image,
        &template,
        MatchTemplateMethod::SumOfSquaredErrorsNormalized,
    );

    // Show location the template was extracted from
    let roi = Rect::at(args.template_x as i32, args.template_y as i32)
        .of_size(args.template_w, args.template_h);

    let image_with_roi = draw_green_rect(&image, roi);

    // Save images to output_dir
    let template_path = output_dir.join("template.png");
    template.save(&template_path).unwrap();
    let source_path = output_dir.join("image.png");
    image_with_roi.save(&source_path).unwrap();
    let sse_path = output_dir.join("result_sse.png");
    sse.save(&sse_path).unwrap();
    let sse_path = output_dir.join("result_sse_norm.png");
    sse_norm.save(&sse_path).unwrap();
}

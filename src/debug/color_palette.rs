//! Color palette related visualization utilities.
mod colourado;

use std::collections::{HashMap};
use image::{ImageBuffer};
use image::{Luma, Rgb, Pixel};


#[doc(hidden)]
macro_rules! random_color_map {
    ($image:expr) => {{
        use colourado::{ColorPalette, PaletteType};
        let keys: Vec<_> = $image
            .pixels()
            .map(|p| p[0]).collect();
        let palette = ColorPalette::new(keys.len() as u32, PaletteType::Random, false);
        let mut output: HashMap<_, image::Rgb<u8>> = HashMap::new();
        for (key, ix) in keys.iter().zip(0 .. keys.len()) {
            let key = key.clone();
            if key == 0 {
                output.insert(key, image::Rgb([0, 0, 0]));
            } else {
                fn convert(x: f32) -> u8 {
                    (x * 255.0) as u8
                }
                let red = convert(palette.colors[ix].red);
                let green = convert(palette.colors[ix].green);
                let blue = convert(palette.colors[ix].blue);

                output.insert(key, image::Rgb([red, green, blue]));
            }
        }
        output
    }};
}

#[doc(hidden)]
macro_rules! to_pretty_rgb_palette {
    ($image:expr) => {{
        let colors = random_color_map!($image);
        let media = ImageBuffer::from_fn($image.width(), $image.height(), |x, y| {
            let px_key = $image.get_pixel(x, y).channels()[0];
            let color = colors.get(&px_key).expect("missing color entry");
            color.clone()
        });
        media
    }};
}


/// Remaps colors into some random RGB color palette, presumably is some
/// aesthetically pleasing arrangement. 
/// 
/// This is useful for visualizing the results of certain functions like,
/// e.g. `connected_components`. 
pub trait ToPrettyRgbPalette {
    /// Remap colors into some random RGB color palette.
    fn to_pretty_rgb_palette(&self) -> image::RgbImage;
}

impl ToPrettyRgbPalette for crate::definitions::Image<Luma<usize>> {
    fn to_pretty_rgb_palette(&self) -> image::RgbImage {
        to_pretty_rgb_palette!(self)
    }
}
impl ToPrettyRgbPalette for crate::definitions::Image<Luma<u8>> {
    fn to_pretty_rgb_palette(&self) -> image::RgbImage {
        to_pretty_rgb_palette!(self)
    }
}
impl ToPrettyRgbPalette for crate::definitions::Image<Luma<u16>> {
    fn to_pretty_rgb_palette(&self) -> image::RgbImage {
        to_pretty_rgb_palette!(self)
    }
}
impl ToPrettyRgbPalette for crate::definitions::Image<Luma<u32>> {
    fn to_pretty_rgb_palette(&self) -> image::RgbImage {
        to_pretty_rgb_palette!(self)
    }
}
impl ToPrettyRgbPalette for crate::definitions::Image<Luma<u64>> {
    fn to_pretty_rgb_palette(&self) -> image::RgbImage {
        to_pretty_rgb_palette!(self)
    }
}
impl ToPrettyRgbPalette for crate::definitions::Image<Luma<isize>> {
    fn to_pretty_rgb_palette(&self) -> image::RgbImage {
        to_pretty_rgb_palette!(self)
    }
}
impl ToPrettyRgbPalette for crate::definitions::Image<Luma<i8>> {
    fn to_pretty_rgb_palette(&self) -> image::RgbImage {
        to_pretty_rgb_palette!(self)
    }
}
impl ToPrettyRgbPalette for crate::definitions::Image<Luma<i16>> {
    fn to_pretty_rgb_palette(&self) -> image::RgbImage {
        to_pretty_rgb_palette!(self)
    }
}
impl ToPrettyRgbPalette for crate::definitions::Image<Luma<i32>> {
    fn to_pretty_rgb_palette(&self) -> image::RgbImage {
        to_pretty_rgb_palette!(self)
    }
}
impl ToPrettyRgbPalette for crate::definitions::Image<Luma<i64>> {
    fn to_pretty_rgb_palette(&self) -> image::RgbImage {
        to_pretty_rgb_palette!(self)
    }
}


///////////////////////////////////////////////////////////////////////////////
// MISC UTILS
///////////////////////////////////////////////////////////////////////////////

/// This function is useful for cleaning up very noisy image regions or data.
/// E.g. for better looking visualizations.
pub fn filter_rgb_regions(image: &::image::RgbImage, min_occurrence: usize) -> ::image::RgbImage {
    let mut image = image.clone();
    filter_rgb_regions_mut(&mut image, min_occurrence);
    image
}

/// This function is useful for cleaning up very noisy image regions or data.
/// E.g. for better looking visualizations.
pub fn filter_rgb_regions_mut(image: &mut ::image::RgbImage, min_occurrence: usize) {
    use crate::definitions::HasBlack;
    
    // INIT COUNTER
    let mut counter: HashMap<Rgb<u8>, usize> = HashMap::new();
    for px in image.pixels() {
        match counter.get_mut(&px) {
            Some(x) => {
                *x = *x + 1;
            }
            None => {
                counter.insert(px.clone(), 0);
            }
        }
    }

    // FILTER PIXELS
    for px in image.pixels_mut() {
        if let Some(count) = counter.get_mut(px) {
            if *count < min_occurrence {
                *px = Rgb::black();
            }
        }
    }
}

/// This function is useful for cleaning up very noisy image regions or data.
/// E.g. for better looking visualizations.
pub fn filter_luma_u32_regions(
    image: &crate::definitions::Image<Luma<u32>>,
    min_occurrence: usize,
) -> crate::definitions::Image<Luma<u32>> {
    let mut image = image.clone();
    filter_luma_u32_regions_mut(&mut image, min_occurrence);
    image
}

/// This function is useful for cleaning up very noisy image regions or data.
/// E.g. for better looking visualizations.
pub fn filter_luma_u32_regions_mut(
    image: &mut crate::definitions::Image<Luma<u32>>,
    min_occurrence: usize,
) { 
    // INIT COUNTER
    let mut counter: HashMap<Luma<u32>, usize> = HashMap::new();
    for px in image.pixels() {
        match counter.get_mut(&px) {
            Some(x) => {
                *x = *x + 1;
            }
            None => {
                counter.insert(px.clone(), 0);
            }
        }
    }

    // FILTER PIXELS
    for px in image.pixels_mut() {
        if let Some(count) = counter.get_mut(px) {
            if *count < min_occurrence {
                *px = Luma([0]);
            }
        }
    }
}

//! > original credits: the [colourado](https://github.com/BrandtM/colourado) crate, MIT.
//! 
//! TODO: Maybe use upstream crate? Both `imageproc` and `colourado` are using
//! the older `0.6` version of rand. The follow code should build just fine
//! for ‘0.7’.
//! 
//! The `colourado` API isn’t publicly exposed.
use rand::Rng;


///////////////////////////////////////////////////////////////////////////////
// BASIC DATA TYPES
///////////////////////////////////////////////////////////////////////////////

trait InRange {
    fn in_range(&self, begin: Self, end: Self) -> bool;
}

impl InRange for f32 {
    fn in_range(&self, begin: f32, end: f32) -> bool {
        *self >= begin && *self < end
    }
}

/// A simple struct containing the three main color components of RGB color space.
/// Colors are stored as f32 values ranging from 0.0 to 1.0 
#[derive(Copy, Clone)]
pub(crate) struct Color {
    pub red: f32,
    pub green: f32,
    pub blue: f32
}

impl Color {
    #![allow(dead_code)]
    fn to_array(&self) -> [f32; 3] {
        [self.red, self.green, self.blue]
    }

    /// Convert HSV to RGB. Plain and simple
    fn hsv_to_rgb(hue: f32, saturation: f32, value: f32) -> Color {
        let chroma = value * saturation;
        let hue2 = hue / 60.0;
        let tmp = chroma * (1.0 - ((hue2 % 2.0) - 1.0).abs());
        let color2: (f32, f32, f32);

        match hue2 {
            h if h.in_range(0.0, 1.0) => color2 = (chroma, tmp, 0.0),
            h if h.in_range(1.0, 2.0) => color2 = (tmp, chroma, 0.0),
            h if h.in_range(2.0, 3.0) => color2 = (0.0, chroma, tmp),
            h if h.in_range(3.0, 4.0) => color2 = (0.0, tmp, chroma),
            h if h.in_range(4.0, 5.0) => color2 = (tmp, 0.0, chroma),
            h if h.in_range(5.0, 6.0) => color2 = (chroma, 0.0, tmp),
            _ => color2 = (0.0, 0.0, 0.0)
        }

        let m = value - chroma;
        let red = color2.0 + m;
        let green = color2.1 + m;
        let blue = color2.2 + m;

        Color {
            red, 
            green, 
            blue
        }
    }
}


///////////////////////////////////////////////////////////////////////////////
// COLOR-PALETTE
///////////////////////////////////////////////////////////////////////////////


/// Container for a vector of colors.
/// You can also use it to store your own custom palette if you so desire. 
pub(crate) struct ColorPalette {
    pub colors: Vec<Color>
}

#[allow(dead_code)]
pub(crate) enum PaletteType {
    Random,
    Pastel,
    Dark,
}

impl ColorPalette {
    pub fn new(count: u32, palette_type: PaletteType, adjacent_colors: bool) -> ColorPalette {
        let mut rng = rand::thread_rng();

        // generate a random color but prevent it from being completely white or black
        let mut hue: f32;
        let mut saturation: f32;
        let mut value: f32;

        match palette_type {
            PaletteType::Random => {
                hue = rng.gen_range(0.0, 360.0);
                saturation = rng.gen_range(0.5, 1.0);
                value = rng.gen_range(0.3, 1.0);
            },
            PaletteType::Pastel => {
                hue = rng.gen_range(0.0, 360.0);
                saturation = rng.gen_range(0.1, 0.4);
                value = rng.gen_range(0.7, 1.0);
            },
            PaletteType::Dark => {
                hue = rng.gen_range(0.0, 360.0);
                saturation = rng.gen_range(0.5, 1.0);
                value = rng.gen_range(0.0, 0.4);
            }
        }

        let mut palette: Vec<Color> = vec![];
        let mut base_divergence = 80.0;

        if adjacent_colors == true {
            base_divergence = 25.0;
        }

        base_divergence -= (count as f32) / 2.6;

        for i in 0..count {
            let rgb = Color::hsv_to_rgb(hue, saturation, value);

            match palette_type {
                PaletteType::Random => {
                    ColorPalette::palette_random(&mut hue, &mut saturation, &mut value, i as f32, base_divergence);
                },
                PaletteType::Pastel => {
                    ColorPalette::palette_pastel(&mut hue, &mut saturation, &mut value, i as f32, base_divergence);
                },
                PaletteType::Dark => {
                    ColorPalette::palette_dark(&mut hue, &mut saturation, &mut value, i as f32, base_divergence);
                }
            }

            palette.push(Color {
                red: rgb.red,
                green: rgb.green,
                blue: rgb.blue
            });
        }

        ColorPalette {
            colors: palette
        }
    }

    fn palette_dark(hue: &mut f32, saturation: &mut f32, value: &mut f32, iteration: f32, divergence: f32) {
        let f = (iteration * 43.0).cos().abs();
        let mut div = divergence;

        if div < 15.0 {
            div = 15.0;
        }

        *hue = (*hue + div + f).abs() % 360.0;
        *saturation = 0.32 + ((iteration * 0.75).sin() / 2.0).abs();
        *value = 0.1 + (iteration.cos() / 6.0).abs();
    }

    fn palette_pastel(hue: &mut f32, saturation: &mut f32, value: &mut f32, iteration: f32, divergence: f32) {
        let f = (iteration * 25.0).cos().abs();
        let mut div = divergence;

        if div < 15.0 {
            div = 15.0;
        }

        *hue = (*hue + div + f).abs() % 360.0;
        *saturation = ((iteration * 0.35).cos() / 5.0).abs();
        *value = 0.5 + (iteration.cos() / 2.0).abs();
    }

    fn palette_random(hue: &mut f32, saturation: &mut f32, value: &mut f32, iteration: f32, divergence: f32) {
        let f = (iteration * 55.0).tan().abs();
        let mut div = divergence;

        if div < 15.0 {
            div = 15.0;
        }

        *hue = (*hue + div + f).abs() % 360.0;
        *saturation = (iteration * 0.35).sin().abs();
        *value = ((6.33 * iteration) * 0.5).cos().abs();

        if *saturation < 0.4 {
            *saturation = 0.4;
        }

        if *value < 0.2 {
            *value = 0.2;
        } else if *value > 0.85 {
            *value = 0.85;
        }        
    }
}



///////////////////////////////////////////////////////////////////////////////
// TESTS
///////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::ColorPalette;
    use super::PaletteType;

    #[test]
    fn generates_palette() {
        let palette = ColorPalette::new(7, PaletteType::Random, false);

        for color in palette.colors {
            assert!(color.red >= 0.0);
            assert!(color.red <= 1.0);

            assert!(color.green >= 0.0);
            assert!(color.green <= 1.0);

            assert!(color.blue >= 0.0);
            assert!(color.blue <= 1.0);
        }        
    }
}


//! Functions for manipulating the contrast of images.

use crate::definitions::{HasBlack, HasWhite};
use crate::integral_image::{integral_image, sum_image_pixels};
use crate::stats::{cumulative_histogram, histogram};
use image::{GrayImage, ImageBuffer, Luma};
use rayon::prelude::*;
use std::cmp::{max, min};

/// Applies an adaptive threshold to an image.
///
/// This algorithm compares each pixel's brightness with the average brightness of the pixels
/// in the (2 * `block_radius` + 1) square block centered on it. If the pixel if at least as bright
/// as the threshold then it will have a value of 255 in the output image, otherwise 0.
pub fn adaptive_threshold(image: &GrayImage, block_radius: u32) -> GrayImage {
    assert!(block_radius > 0);
    let integral = integral_image::<_, u32>(image);
    let mut out = ImageBuffer::from_pixel(image.width(), image.height(), Luma::black());

    for y in 0..image.height() {
        for x in 0..image.width() {
            let current_pixel = image.get_pixel(x, y);
            // Traverse all neighbors in (2 * block_radius + 1) x (2 * block_radius + 1)
            let (y_low, y_high) = (
                max(0, y as i32 - (block_radius as i32)) as u32,
                min(image.height() - 1, y + block_radius),
            );
            let (x_low, x_high) = (
                max(0, x as i32 - (block_radius as i32)) as u32,
                min(image.width() - 1, x + block_radius),
            );

            // Number of pixels in the block, adjusted for edge cases.
            let w = (y_high - y_low + 1) * (x_high - x_low + 1);
            let mean = sum_image_pixels(&integral, x_low, y_low, x_high, y_high)[0] / w;

            if current_pixel[0] as u32 >= mean as u32 {
                out.put_pixel(x, y, Luma::white());
            }
        }
    }
    out
}

/// Returns the [Otsu threshold level] of an 8bpp image.
///
/// [Otsu threshold level]: https://en.wikipedia.org/wiki/Otsu%27s_method
pub fn otsu_level(image: &GrayImage) -> u8 {
    let hist = histogram(image);
    let (width, height) = image.dimensions();
    let total_weight = width * height;

    // Sum of all pixel intensities, to use when calculating means.
    let total_pixel_sum = hist.channels[0]
        .iter()
        .enumerate()
        .fold(0f64, |sum, (t, h)| sum + (t as u32 * h) as f64);

    // Sum of all pixel intensities in the background class.
    let mut background_pixel_sum = 0f64;

    // The weight of a class (background or foreground) is
    // the number of pixels which belong to that class at
    // the current threshold.
    let mut background_weight = 0u32;
    let mut foreground_weight;

    let mut largest_variance = 0f64;
    let mut best_threshold = 0u8;

    for (threshold, hist_count) in hist.channels[0].iter().enumerate() {
        background_weight += hist_count;
        if background_weight == 0 {
            continue;
        };

        foreground_weight = total_weight - background_weight;
        if foreground_weight == 0 {
            break;
        };

        background_pixel_sum += (threshold as u32 * hist_count) as f64;
        let foreground_pixel_sum = total_pixel_sum - background_pixel_sum;

        let background_mean = background_pixel_sum / (background_weight as f64);
        let foreground_mean = foreground_pixel_sum / (foreground_weight as f64);

        let mean_diff_squared = (background_mean - foreground_mean).powi(2);
        let intra_class_variance =
            (background_weight as f64) * (foreground_weight as f64) * mean_diff_squared;

        if intra_class_variance > largest_variance {
            largest_variance = intra_class_variance;
            best_threshold = threshold as u8;
        }
    }

    best_threshold
}

/// Returns a binarized image from an input 8bpp grayscale image
/// obtained by applying the given threshold. Pixels with intensity
/// equal to the threshold are assigned to the background.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use imageproc::contrast::threshold;
///
/// let image = gray_image!(
///     10, 80, 20;
///     50, 90, 70);
///
/// let thresholded = gray_image!(
///     0, 255,   0;
///     0, 255, 255);
///
/// assert_pixels_eq!(threshold(&image, 50), thresholded);
/// # }
/// ```
pub fn threshold(image: &GrayImage, thresh: u8) -> GrayImage {
    let mut out = image.clone();
    threshold_mut(&mut out, thresh);
    out
}

/// Mutates given image to form a binarized version produced by applying
/// the given threshold. Pixels with intensity
/// equal to the threshold are assigned to the background.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use imageproc::contrast::threshold_mut;
///
/// let mut image = gray_image!(
///     10, 80, 20;
///     50, 90, 70);
///
/// let thresholded = gray_image!(
///     0, 255,   0;
///     0, 255, 255);
///
/// threshold_mut(&mut image, 50);
///
/// assert_pixels_eq!(image, thresholded);
/// # }
/// ```
pub fn threshold_mut(image: &mut GrayImage, thresh: u8) {
    for p in image.iter_mut() {
        *p = if *p <= thresh { 0 } else { 255 };
    }
}

/// Equalises the histogram of an 8bpp grayscale image in place. See also
/// [histogram equalization (wikipedia)](https://en.wikipedia.org/wiki/Histogram_equalization).
pub fn equalize_histogram_mut(image: &mut GrayImage) {
    let hist = cumulative_histogram(image).channels[0];
    let total = hist[255] as f32;

    image.par_iter_mut().for_each(|p| {
        // UNSAFE JUSTIFICATION:
        //  - bench_equalize_histogram_mut shows a ~10% regression from using checked indexing here.
        //  - correct because each channel of hist has length 256, and GrayImage has 8 bits per pixel.
        let fraction = unsafe { *hist.get_unchecked(*p as usize) as f32 / total };
        *p = (f32::min(255f32, 255f32 * fraction)) as u8;
    });
}

/// Equalises the histogram of an 8bpp grayscale image. See also
/// [histogram equalization (wikipedia)](https://en.wikipedia.org/wiki/Histogram_equalization).
pub fn equalize_histogram(image: &GrayImage) -> GrayImage {
    let mut out = image.clone();
    equalize_histogram_mut(&mut out);
    out
}

/// Adjusts contrast of an 8bpp grayscale image in place so that its
/// histogram is as close as possible to that of the target image.
pub fn match_histogram_mut(image: &mut GrayImage, target: &GrayImage) {
    let image_histc = cumulative_histogram(image).channels[0];
    let target_histc = cumulative_histogram(target).channels[0];
    let lut = histogram_lut(&image_histc, &target_histc);

    for p in image.iter_mut() {
        *p = lut[*p as usize] as u8;
    }
}

/// Adjusts contrast of an 8bpp grayscale image so that its
/// histogram is as close as possible to that of the target image.
pub fn match_histogram(image: &GrayImage, target: &GrayImage) -> GrayImage {
    let mut out = image.clone();
    match_histogram_mut(&mut out, target);
    out
}

/// `l = histogram_lut(s, t)` is chosen so that `target_histc[l[i]] / sum(target_histc)`
/// is as close as possible to `source_histc[i] / sum(source_histc)`.
fn histogram_lut(source_histc: &[u32; 256], target_histc: &[u32; 256]) -> [usize; 256] {
    let source_total = source_histc[255] as f32;
    let target_total = target_histc[255] as f32;

    let mut lut = [0usize; 256];
    let mut y = 0usize;
    let mut prev_target_fraction = 0f32;

    for s in 0..256 {
        let source_fraction = source_histc[s] as f32 / source_total;
        let mut target_fraction = target_histc[y] as f32 / target_total;

        while source_fraction > target_fraction && y < 255 {
            y += 1;
            prev_target_fraction = target_fraction;
            target_fraction = target_histc[y] as f32 / target_total;
        }

        if y == 0 {
            lut[s] = y;
        } else {
            let prev_dist = f32::abs(prev_target_fraction - source_fraction);
            let dist = f32::abs(target_fraction - source_fraction);
            if prev_dist < dist {
                lut[s] = y - 1;
            } else {
                lut[s] = y;
            }
        }
    }

    lut
}

/// Linearly stretches the contrast in an image, sending `lower` to `0u8` and `upper` to `2558u8`.
///
/// Is it common to choose `upper` and `lower` values using image percentiles - see [`percentile`](../stats/fn.percentile.html).
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use imageproc::contrast::stretch_contrast;
///
/// let image = gray_image!(
///      0,   20,  50;
///     80,  100, 255);
///
/// let lower = 20;
/// let upper = 100;
///
/// // Pixel intensities between 20 and 100 are linearly
/// // scaled so that 20 is mapped to 0 and 100 is mapped to 255.
/// // Pixel intensities less than 20 are sent to 0 and pixel
/// // intensities greater than 100 are sent to 255.
/// let stretched = stretch_contrast(&image, lower, upper);
///
/// let expected = gray_image!(
///       0,   0,  95;
///     191, 255, 255);
///
/// assert_pixels_eq!(stretched, expected);
/// # }
/// ```
pub fn stretch_contrast(image: &GrayImage, lower: u8, upper: u8) -> GrayImage {
    let mut out = image.clone();
    stretch_contrast_mut(&mut out, lower, upper);
    out
}

/// Linearly stretches the contrast in an image in place, sending `lower` to `0u8` and `upper` to `2558u8`.
///
/// See the [`stretch_contrast`](fn.stretch_contrast.html) documentation for more.
pub fn stretch_contrast_mut(image: &mut GrayImage, lower: u8, upper: u8) {
    assert!(upper > lower, "upper must be strictly greater than lower");
    let len = (upper - lower) as u16;
    for p in image.iter_mut() {
        if *p >= upper {
            *p = 255;
        } else if *p <= lower {
            *p = 0;
        } else {
            let scaled = (255 * (*p as u16 - lower as u16)) / len;
            *p = scaled as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::definitions::{HasBlack, HasWhite};
    use crate::utils::gray_bench_image;
    use image::{GrayImage, Luma};
    use test::{black_box, Bencher};

    #[test]
    fn adaptive_threshold_constant() {
        let image = GrayImage::from_pixel(3, 3, Luma([100u8]));
        let binary = adaptive_threshold(&image, 1);
        let expected = GrayImage::from_pixel(3, 3, Luma::white());
        assert_pixels_eq!(expected, binary);
    }

    #[test]
    fn adaptive_threshold_one_darker_pixel() {
        for y in 0..3 {
            for x in 0..3 {
                let mut image = GrayImage::from_pixel(3, 3, Luma([200u8]));
                image.put_pixel(x, y, Luma([100u8]));
                let binary = adaptive_threshold(&image, 1);
                // All except the dark pixel have brightness >= their local mean
                let mut expected = GrayImage::from_pixel(3, 3, Luma::white());
                expected.put_pixel(x, y, Luma::black());
                assert_pixels_eq!(binary, expected);
            }
        }
    }

    #[test]
    fn adaptive_threshold_one_lighter_pixel() {
        for y in 0..5 {
            for x in 0..5 {
                let mut image = GrayImage::from_pixel(5, 5, Luma([100u8]));
                image.put_pixel(x, y, Luma([200u8]));

                let binary = adaptive_threshold(&image, 1);

                for yb in 0..5 {
                    for xb in 0..5 {
                        let output_intensity = binary.get_pixel(xb, yb)[0];

                        let is_light_pixel = xb == x && yb == y;

                        let local_mean_includes_light_pixel =
                            (yb as i32 - y as i32).abs() <= 1 && (xb as i32 - x as i32).abs() <= 1;

                        if is_light_pixel {
                            assert_eq!(output_intensity, 255);
                        } else if local_mean_includes_light_pixel {
                            assert_eq!(output_intensity, 0);
                        } else {
                            assert_eq!(output_intensity, 255);
                        }
                    }
                }
            }
        }
    }

    #[bench]
    fn bench_adaptive_threshold(b: &mut Bencher) {
        let image = gray_bench_image(200, 200);
        let block_radius = 10;
        b.iter(|| {
            let thresholded = adaptive_threshold(&image, block_radius);
            black_box(thresholded);
        });
    }

    #[bench]
    fn bench_match_histogram(b: &mut Bencher) {
        let target = GrayImage::from_pixel(200, 200, Luma([150]));
        let image = gray_bench_image(200, 200);
        b.iter(|| {
            let matched = match_histogram(&image, &target);
            black_box(matched);
        });
    }

    #[bench]
    fn bench_match_histogram_mut(b: &mut Bencher) {
        let target = GrayImage::from_pixel(200, 200, Luma([150]));
        let mut image = gray_bench_image(200, 200);
        b.iter(|| {
            match_histogram_mut(&mut image, &target);
        });
    }

    #[test]
    fn test_histogram_lut_source_and_target_equal() {
        let mut histc = [0u32; 256];
        for i in 1..histc.len() {
            histc[i] = 2 * i as u32;
        }

        let lut = histogram_lut(&histc, &histc);
        let expected = (0..256).collect::<Vec<_>>();

        assert_eq!(&lut[0..256], &expected[0..256]);
    }

    #[test]
    fn test_histogram_lut_gradient_to_step_contrast() {
        let mut grad_histc = [0u32; 256];
        for i in 0..grad_histc.len() {
            grad_histc[i] = i as u32;
        }

        let mut step_histc = [0u32; 256];
        for i in 30..130 {
            step_histc[i] = 100;
        }
        for i in 130..256 {
            step_histc[i] = 200;
        }

        let lut = histogram_lut(&grad_histc, &step_histc);
        let mut expected = [0usize; 256];

        // No black pixels in either image
        expected[0] = 0;

        for i in 1..64 {
            expected[i] = 29;
        }
        for i in 64..128 {
            expected[i] = 30;
        }
        for i in 128..192 {
            expected[i] = 129;
        }
        for i in 192..256 {
            expected[i] = 130;
        }

        assert_eq!(&lut[0..256], &expected[0..256]);
    }

    fn constant_image(width: u32, height: u32, intensity: u8) -> GrayImage {
        GrayImage::from_pixel(width, height, Luma([intensity]))
    }

    #[test]
    fn test_otsu_constant() {
        // Variance is 0 at any threshold, and we
        // only increase the current threshold if we
        // see a strictly greater variance
        assert_eq!(otsu_level(&constant_image(10, 10, 0)), 0);
        assert_eq!(otsu_level(&constant_image(10, 10, 128)), 0);
        assert_eq!(otsu_level(&constant_image(10, 10, 255)), 0);
    }

    #[test]
    fn test_otsu_level_gradient() {
        let contents = (0u8..26u8).map(|x| x * 10u8).collect();
        let image = GrayImage::from_raw(26, 1, contents).unwrap();
        let level = otsu_level(&image);
        assert_eq!(level, 120);
    }

    #[bench]
    fn bench_otsu_level(b: &mut Bencher) {
        let image = gray_bench_image(200, 200);
        b.iter(|| {
            let level = otsu_level(&image);
            black_box(level);
        });
    }

    #[test]
    fn test_threshold_0_image_0() {
        let expected = 0u8;
        let actual = threshold(&constant_image(10, 10, 0), 0);
        assert_pixels_eq!(actual, constant_image(10, 10, expected));
    }

    #[test]
    fn test_threshold_0_image_1() {
        let expected = 255u8;
        let actual = threshold(&constant_image(10, 10, 1), 0);
        assert_pixels_eq!(actual, constant_image(10, 10, expected));
    }

    #[test]
    fn test_threshold_threshold_255_image_255() {
        let expected = 0u8;
        let actual = threshold(&constant_image(10, 10, 255), 255);
        assert_pixels_eq!(actual, constant_image(10, 10, expected));
    }

    #[test]
    fn test_threshold() {
        let original_contents = (0u8..26u8).map(|x| x * 10u8).collect();
        let original = GrayImage::from_raw(26, 1, original_contents).unwrap();

        let expected_contents = vec![0u8; 13].into_iter().chain(vec![255u8; 13]).collect();

        let expected = GrayImage::from_raw(26, 1, expected_contents).unwrap();

        let actual = threshold(&original, 125u8);
        assert_pixels_eq!(expected, actual);
    }

    #[bench]
    fn bench_equalize_histogram(b: &mut Bencher) {
        let image = gray_bench_image(500, 500);
        b.iter(|| {
            let equalized = equalize_histogram(&image);
            black_box(equalized);
        });
    }

    #[bench]
    fn bench_equalize_histogram_mut(b: &mut Bencher) {
        let mut image = gray_bench_image(500, 500);
        b.iter(|| {
            black_box(equalize_histogram_mut(&mut image));
        });
    }

    #[bench]
    fn bench_threshold(b: &mut Bencher) {
        let image = gray_bench_image(500, 500);
        b.iter(|| {
            let thresholded = threshold(&image, 125);
            black_box(thresholded);
        });
    }

    #[bench]
    fn bench_threshold_mut(b: &mut Bencher) {
        let mut image = gray_bench_image(500, 500);
        b.iter(|| {
            black_box(threshold_mut(&mut image, 125));
        });
    }

    #[bench]
    fn bench_stretch_contrast(b: &mut Bencher) {
        let image = gray_bench_image(500, 500);
        b.iter(|| {
            let stretched = stretch_contrast(&image, 20, 80);
            black_box(stretched);
        });
    }

    #[bench]
    fn bench_stretch_contrast_mut(b: &mut Bencher) {
        let mut image = gray_bench_image(500, 500);
        b.iter(|| {
            black_box(stretch_contrast_mut(&mut image, 20, 80));
        });
    }
}

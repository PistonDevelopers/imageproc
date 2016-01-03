//! Functions for detecting corners, also known as interest points.

use image::{
    GenericImage,
    Luma
};

use definitions::{
    Position,
    Score
};

/// A location and score for a detected corner.
/// The scores need not be comparable between different
/// corner detectors.
#[derive(Copy,Clone,Debug,PartialEq)]
pub struct Corner {
    pub x: u32,
    pub y: u32,
    pub score: f32
}

impl Corner {
    pub fn new(x: u32, y: u32, score: f32) -> Corner {
        Corner {x: x, y: y, score: score}
    }
}

impl Position for Corner {
    fn x(&self) -> u32 {
        self.x
    }

    fn y(&self) -> u32 {
        self.y
    }
}

impl Score for Corner {
    fn score(&self) -> f32 {
        self.score
    }
}

/// Variants of the [FAST](https://en.wikipedia.org/wiki/Features_from_accelerated_segment_test)
/// corner detector. These classify a point based on its intensity relative to the 16 pixels
/// in the Bresenham circle of radius 3 around it. A point P with intensity I is detected as a
/// corner if all pixels in a sufficiently long contiguous section of this circle either
/// all have intensity greater than I + t or all have intensity less than
/// I - t, for some user-provided threshold t. The score of a corner is
/// the greatest threshold for which the given pixel still qualifies as
/// a corner.
pub enum Fast {
    /// Corners require a section of length as least nine.
    Nine,
    /// Corners require a section of length as least twelve.
    Twelve
}

/// Finds corners using FAST-12 features. See comment on Fast enum.
pub fn corners_fast12<I>(image: &I, threshold: u8) -> Vec<Corner>
    where I: GenericImage<Pixel=Luma<u8>> {

    let (width, height) = image.dimensions();
    let mut corners = vec![];

    for y in 0..height {
        for x in 0..width {
            if is_corner_fast12(image, threshold, x, y) {
                let score = fast_corner_score(image, threshold, x, y, Fast::Twelve);
                corners.push(Corner::new(x,y, score as f32));
            }
        }
    }

    corners
}

/// Finds corners using FAST-9 features. See comment on Fast enum.
pub fn corners_fast9<I>(image: &I, threshold: u8) -> Vec<Corner>
    where I: GenericImage<Pixel=Luma<u8>> {

    let (width, height) = image.dimensions();
    let mut corners = vec![];

    for y in 0..height {
        for x in 0..width {
            if is_corner_fast9(image, threshold, x, y) {
                let score = fast_corner_score(image, threshold, x, y, Fast::Nine);
                corners.push(Corner::new(x,y, score as f32));
            }
        }
    }

    corners
}

/// The score of a corner detected using the FAST
/// detector is the largest threshold for which this
/// pixel is still a corner. We input the threshold at which
/// the corner was detected as a lower bound on the search.
/// Note that the corner check uses a strict inequality, so if
/// the smallest intensity difference between the center pixel
/// and a corner pixel is n then the corner will have a score of n - 1.
pub fn fast_corner_score<I>(image: &I, threshold: u8, x: u32, y: u32, variant: Fast) -> u8
    where I: GenericImage<Pixel=Luma<u8>> {

    let mut max = 255u8;
    let mut min = threshold;

    loop {
        if max == min {
            return max;
        }

        let mean = ((max as u16 + min as u16) / 2u16) as u8;
        let probe = if max == min + 1 { max } else { mean };

        let is_corner = match variant {
            Fast::Nine => is_corner_fast9(image, probe, x, y),
            Fast::Twelve => is_corner_fast12(image, probe, x, y),
        };

        if is_corner {
            min = probe;
        }
        else {
            max = probe - 1;
        }
    }
}

// Note [FAST circle labels]
//
//          15 00 01
//       14          02
//     13              03
//     12       p      04
//     11              05
//       10          06
//          09 08 07

/// Checks if the given pixel is a corner according to the FAST9 detector.
/// The current implementation is extremely inefficient.
// TODO: Make this much faster!
fn is_corner_fast9<I>(image: &I, threshold: u8, x: u32, y: u32) -> bool
    where I: GenericImage<Pixel=Luma<u8>> {

    let (width, height) = image.dimensions();
    if x < 3 || y < 3 || x >= width - 3 || y >= height - 3 {
        return false;
    }

    let c = image.get_pixel(x, y)[0];
    let low_thresh: i16  = c as i16 - threshold as i16;
    let high_thresh: i16 = c as i16 + threshold as i16;

    // See Note [FAST circle labels]
    let p0  = image.get_pixel(x, y - 3)[0] as i16;
    let p8  = image.get_pixel(x, y + 3)[0] as i16;
    let p4  = image.get_pixel(x + 3, y)[0] as i16;
    let p12 = image.get_pixel(x - 3, y)[0] as i16;

    let above = (p0  > high_thresh && p4  > high_thresh) ||
                (p4  > high_thresh && p8  > high_thresh) ||
                (p8  > high_thresh && p12 > high_thresh) ||
                (p12 > high_thresh && p0  > high_thresh);

    let below = (p0  < low_thresh && p4  < low_thresh) ||
                (p4  < low_thresh && p8  < low_thresh) ||
                (p8  < low_thresh && p12 < low_thresh) ||
                (p12 < low_thresh && p0  < low_thresh);

    if !above && !below {
        return false;
    }

    let mut pixels = [0i16; 16];

    pixels[0]  = p0;
    pixels[1]  = image.get_pixel(x + 1, y - 3)[0] as i16;
    pixels[2]  = image.get_pixel(x + 2, y - 2)[0] as i16;
    pixels[3]  = image.get_pixel(x + 3, y - 1)[0] as i16;
    pixels[4]  = p4;
    pixels[5]  = image.get_pixel(x + 3, y + 1)[0] as i16;
    pixels[6]  = image.get_pixel(x + 2, y + 2)[0] as i16;
    pixels[7]  = image.get_pixel(x + 1, y + 3)[0] as i16;
    pixels[8]  = p8;
    pixels[9]  = image.get_pixel(x - 1, y + 3)[0] as i16;
    pixels[10] = image.get_pixel(x - 2, y + 2)[0] as i16;
    pixels[11] = image.get_pixel(x - 3, y + 1)[0] as i16;
    pixels[12] = p12;
    pixels[13] = image.get_pixel(x - 3, y - 1)[0] as i16;
    pixels[14] = image.get_pixel(x - 2, y - 2)[0] as i16;
    pixels[15] = image.get_pixel(x - 1, y - 3)[0] as i16;

    // above and below could both be true
    (above && has_bright_span(&pixels, 9, high_thresh)) ||
    (below && has_dark_span(&pixels, 9, low_thresh))
}

/// Checks if the given pixel is a corner according to the FAST12 detector.
fn is_corner_fast12<I>(image: &I, threshold: u8, x: u32, y: u32) -> bool
    where I: GenericImage<Pixel=Luma<u8>> {

    let (width, height) = image.dimensions();
    if x < 3 || y < 3 || x >= width - 3 || y >= height - 3 {
        return false;
    }

    let c = image.get_pixel(x, y)[0];
    let low_thresh: i16  = c as i16 - threshold as i16;
    let high_thresh: i16 = c as i16 + threshold as i16;

    // See Note [FAST circle labels]
    let p0 = image.get_pixel(x, y - 3)[0] as i16;
    let p8 = image.get_pixel(x, y + 3)[0] as i16;

    let mut above = p0 > high_thresh && p8 > high_thresh;
    let mut below = p0 < low_thresh  && p8 < low_thresh;

    if !above && !below {
        return false;
    }

    let p4  = image.get_pixel(x + 3, y)[0] as i16;
    let p12 = image.get_pixel(x - 3, y)[0] as i16;

    above = above && ((p4 > high_thresh) || (p12 > high_thresh));
    below = below && ((p4 < low_thresh)  || (p12 < low_thresh));

    if !above && !below {
        return false;
    }

    // TODO: Generate a list of pixel offsets once per image,
    // TODO: and use those offsets directly when reading pixels.
    // TODO: This is a little tricky as we can't always do it - we'd
    // TODO: need to distinguish between GenericImages and ImageBuffers.
    // TODO: We can also reduce the number of checks we do below.

    let mut pixels = [0i16; 16];

    pixels[0]  = p0;
    pixels[1]  = image.get_pixel(x + 1, y - 3)[0] as i16;
    pixels[2]  = image.get_pixel(x + 2, y - 2)[0] as i16;
    pixels[3]  = image.get_pixel(x + 3, y - 1)[0] as i16;
    pixels[4]  = p4;
    pixels[5]  = image.get_pixel(x + 3, y + 1)[0] as i16;
    pixels[6]  = image.get_pixel(x + 2, y + 2)[0] as i16;
    pixels[7]  = image.get_pixel(x + 1, y + 3)[0] as i16;
    pixels[8]  = p8;
    pixels[9]  = image.get_pixel(x - 1, y + 3)[0] as i16;
    pixels[10] = image.get_pixel(x - 2, y + 2)[0] as i16;
    pixels[11] = image.get_pixel(x - 3, y + 1)[0] as i16;
    pixels[12] = p12;
    pixels[13] = image.get_pixel(x - 3, y - 1)[0] as i16;
    pixels[14] = image.get_pixel(x - 2, y - 2)[0] as i16;
    pixels[15] = image.get_pixel(x - 1, y - 3)[0] as i16;

    // Exactly one of above or below is true
    if above {
        has_bright_span(&pixels, 12, high_thresh)
    }
    else {
        has_dark_span(&pixels, 12, low_thresh)
    }
}

/// True if the circle has a contiguous section of at least the given length, all
/// of whose pixels have intensities strictly greater than the threshold.
fn has_bright_span(circle: &[i16; 16], length: u8, threshold: i16) -> bool {
    search_span(circle, length, |c| *c > threshold)
}

/// True if the circle has a contiguous section of at least the given length, all
/// of whose pixels have intensities strictly less than the threshold.
fn has_dark_span(circle: &[i16; 16], length: u8, threshold: i16) -> bool {
    search_span(circle, length, |c| *c < threshold)   
}

/// True if the circle has a contiguous section of at least the given length, all
/// of whose pixels match f condition.
fn search_span<F>(circle: &[i16; 16], length: u8, f: F) -> bool 
    where F: Fn(&i16) -> bool {
    
    if length > 16 { return false; }

    let mut nb_ok = 0u8;
    let mut nb_ok_start = None;
    let mut nb_ko = 16 - length;

    for c in circle.iter() {
        if f(c) {
            nb_ok += 1;
            if nb_ok == length { return true; }
        } else {
            if nb_ok_start.is_none() {
                nb_ok_start = Some(nb_ok);
            }
            nb_ok = 0;
	    nb_ko -= 1;
	    if nb_ko == 0 { return false; }
        }
    }

    nb_ok + nb_ok_start.unwrap() >= length
}

#[cfg(test)]
mod test {

    use super::{
        Corner,
        fast_corner_score,
        is_corner_fast9,
        is_corner_fast12,
        Fast
    };
    use image::{
        GrayImage,
        ImageBuffer
    };
    use test::Bencher;

    #[test]
    fn test_is_corner_fast12_12_contiguous_darker_pixels() {
        let image: GrayImage = ImageBuffer::from_raw(7, 7, vec![
            10, 10, 00, 00, 00, 10, 10,
            10, 00, 10, 10, 10, 00, 10,
            00, 10, 10, 10, 10, 10, 10,
            00, 10, 10, 10, 10, 10, 10,
            00, 10, 10, 10, 10, 10, 10,
            10, 00, 10, 10, 10, 10, 10,
            10, 10, 00, 00, 00, 10, 10]).unwrap();

        assert_eq!(is_corner_fast12(&image, 8, 3, 3), true);
    }

    #[test]
    fn test_is_corner_fast12_12_contiguous_darker_pixels_large_threshold() {
        let image: GrayImage = ImageBuffer::from_raw(7, 7, vec![
            10, 10, 00, 00, 00, 10, 10,
            10, 00, 10, 10, 10, 00, 10,
            00, 10, 10, 10, 10, 10, 10,
            00, 10, 10, 10, 10, 10, 10,
            00, 10, 10, 10, 10, 10, 10,
            10, 00, 10, 10, 10, 10, 10,
            10, 10, 00, 00, 00, 10, 10]).unwrap();

        assert_eq!(is_corner_fast12(&image, 15, 3, 3), false);
    }

    #[test]
    fn test_is_corner_fast12_12_contiguous_lighter_pixels() {
        let image: GrayImage = ImageBuffer::from_raw(7, 7, vec![
            00, 00, 10, 10, 10, 00, 00,
            00, 10, 00, 00, 00, 10, 00,
            10, 00, 00, 00, 00, 00, 00,
            10, 00, 00, 00, 00, 00, 00,
            10, 00, 00, 00, 00, 00, 00,
            00, 10, 00, 00, 00, 00, 00,
            00, 00, 10, 10, 10, 00, 00]).unwrap();

        assert_eq!(is_corner_fast12(&image, 8, 3, 3), true);
    }

    #[test]
    fn test_is_corner_fast12_12_noncontiguous() {
        let image: GrayImage = ImageBuffer::from_raw(7, 7, vec![
            10, 10, 00, 00, 00, 10, 10,
            10, 00, 10, 10, 10, 00, 10,
            00, 10, 10, 10, 10, 10, 10,
            00, 10, 10, 10, 10, 10, 10,
            10, 10, 10, 10, 10, 10, 00,
            10, 00, 10, 10, 10, 10, 10,
            10, 10, 00, 00, 00, 10, 10]).unwrap();

        assert_eq!(is_corner_fast12(&image, 8, 3, 3), false);
    }

    #[bench]
    fn bench_is_corner_fast12_12_noncontiguous(b: &mut Bencher) {
        let image: GrayImage = ImageBuffer::from_raw(7, 7, vec![
            10, 10, 00, 00, 00, 10, 10,
            10, 00, 10, 10, 10, 00, 10,
            00, 10, 10, 10, 10, 10, 10,
            00, 10, 10, 10, 10, 10, 10,
            10, 10, 10, 10, 10, 10, 00,
            10, 00, 10, 10, 10, 10, 10,
            10, 10, 00, 00, 00, 10, 10]).unwrap();

	b.iter(||is_corner_fast12(&image, 8, 3, 3));
    }

    #[test]
    fn test_is_corner_fast12_near_image_boundary() {
        let image: GrayImage = ImageBuffer::from_raw(7, 7, vec![
            10, 10, 00, 00, 00, 10, 10,
            10, 00, 10, 10, 10, 00, 10,
            00, 10, 10, 10, 10, 10, 10,
            00, 10, 10, 10, 10, 10, 10,
            00, 10, 10, 10, 10, 10, 10,
            10, 00, 10, 10, 10, 10, 10,
            10, 10, 00, 00, 00, 10, 10]).unwrap();

        assert_eq!(is_corner_fast12(&image, 8, 1, 1), false);
    }

    #[test]
    fn test_fast_corner_score_12() {
        let image: GrayImage = ImageBuffer::from_raw(7, 7, vec![
            10, 10, 00, 00, 00, 10, 10,
            10, 00, 10, 10, 10, 00, 10,
            00, 10, 10, 10, 10, 10, 10,
            00, 10, 10, 10, 10, 10, 10,
            00, 10, 10, 10, 10, 10, 10,
            10, 00, 10, 10, 10, 10, 10,
            10, 10, 00, 00, 00, 10, 10]).unwrap();

        let score = fast_corner_score(&image, 5, 3, 3, Fast::Twelve);
        assert_eq!(score, 9);

        let score = fast_corner_score(&image, 9, 3, 3, Fast::Twelve);
        assert_eq!(score, 9);
    }

    #[test]
    fn test_is_corner_fast9_9_contiguous_darker_pixels() {
        let image: GrayImage = ImageBuffer::from_raw(7, 7, vec![
            10, 10, 00, 00, 00, 10, 10,
            10, 00, 10, 10, 10, 00, 10,
            00, 10, 10, 10, 10, 10, 10,
            00, 10, 10, 10, 10, 10, 10,
            00, 10, 10, 10, 10, 10, 10,
            10, 00, 10, 10, 10, 10, 10,
            10, 10, 10, 10, 10, 10, 10]).unwrap();

        assert_eq!(is_corner_fast9(&image, 8, 3, 3), true);
    }

    #[test]
    fn test_is_corner_fast9_9_contiguous_lighter_pixels() {
        let image: GrayImage = ImageBuffer::from_raw(7, 7, vec![
            00, 00, 10, 10, 10, 00, 00,
            00, 10, 00, 00, 00, 10, 00,
            10, 00, 00, 00, 00, 00, 00,
            10, 00, 00, 00, 00, 00, 00,
            10, 00, 00, 00, 00, 00, 00,
            00, 10, 00, 00, 00, 00, 00,
            00, 00, 00, 00, 00, 00, 00]).unwrap();

        assert_eq!(is_corner_fast9(&image, 8, 3, 3), true);
    }

    #[bench]
    fn bench_is_corner_fast9_9_contiguous_lighter_pixels(b: &mut Bencher) {
        let image: GrayImage = ImageBuffer::from_raw(7, 7, vec![
            00, 00, 10, 10, 10, 00, 00,
            00, 10, 00, 00, 00, 10, 00,
            10, 00, 00, 00, 00, 00, 00,
            10, 00, 00, 00, 00, 00, 00,
            10, 00, 00, 00, 00, 00, 00,
            00, 10, 00, 00, 00, 00, 00,
            00, 00, 00, 00, 00, 00, 00]).unwrap();

        b.iter(|| is_corner_fast9(&image, 8, 3, 3));
    }

    #[test]
    fn test_is_corner_fast9_12_noncontiguous() {
        let image: GrayImage = ImageBuffer::from_raw(7, 7, vec![
            10, 10, 00, 00, 00, 10, 10,
            10, 00, 10, 10, 10, 00, 10,
            00, 10, 10, 10, 10, 10, 10,
            00, 10, 10, 10, 10, 10, 10,
            10, 10, 10, 10, 10, 10, 00,
            10, 00, 10, 10, 10, 10, 10,
            10, 10, 00, 00, 00, 10, 10]).unwrap();

        assert_eq!(is_corner_fast9(&image, 8, 3, 3), false);
    }

    #[test]
    fn test_corner_score_fast9() {
        // 8 pixels with an intensity diff of 20, then 1 with a diff of 10
        let image: GrayImage = ImageBuffer::from_raw(7, 7, vec![
            10, 10, 00, 00, 00, 10, 10,
            10, 00, 10, 10, 10, 00, 10,
            00, 10, 10, 10, 10, 10, 10,
            00, 10, 10, 20, 10, 10, 10,
            00, 10, 10, 10, 10, 10, 10,
            10, 10, 10, 10, 10, 10, 10,
            10, 10, 10, 10, 10, 10, 10]).unwrap();

        let score = fast_corner_score(&image, 5, 3, 3, Fast::Nine);
        assert_eq!(score, 9);

        let score = fast_corner_score(&image, 9, 3, 3, Fast::Nine);
        assert_eq!(score, 9);
    }
}

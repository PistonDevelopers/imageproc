//! Functions for mapping over pixels, colors or subpixels of images.

use image::{
    GenericImage,
    ImageBuffer,
    Luma,
    Pixel,
    Primitive,
    Rgb,
    Rgba
};

use definitions::{
    VecBuffer
};

use num::Zero;

/// The type obtained by replacing the channel type of a given Pixel type.
pub trait WithChannel<C: Primitive>: Pixel {
    type Pixel: Pixel<Subpixel=C> + 'static;
}

/// Alias to make uses of WithChannel less syntactically noisy.
pub type ChannelMap<Pix, Sub> = <Pix as WithChannel<Sub>>::Pixel;

impl<T, U> WithChannel<U> for Rgb<T>
    where T: Primitive + 'static,
          U: Primitive + 'static {
    type Pixel = Rgb<U>;
}

impl<T, U> WithChannel<U> for Rgba<T>
    where T: Primitive + 'static,
          U: Primitive + 'static {
    type Pixel = Rgba<U>;
}

impl<T, U> WithChannel<U> for Luma<T>
    where T: Primitive + 'static,
          U: Primitive + 'static {
    type Pixel = Luma<U>;
}

/// Applies f to each subpixel of the input image.
pub fn map_subpixels<I, P, F, S>(image: &I, f: F) -> VecBuffer<ChannelMap<P, S>>
    where I: GenericImage<Pixel=P>,
          P: WithChannel<S> + 'static,
          S: Primitive + 'static,
          F: Fn(P::Subpixel) -> S
{
    let (width, height) = image.dimensions();
    let mut out = ImageBuffer::<ChannelMap<P, S>, Vec<S>>::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let mut out_channels = out.get_pixel_mut(x, y).channels_mut();
            for c in 0..P::channel_count() {
                out_channels[c as usize]
                    = f(image.get_pixel(x, y).channels()[c as usize]);
            }
        }
    }

    out
}

/// Applies f to each subpixel of the input image.
pub fn map_subpixels_vec<P, F, S>(image: &VecBuffer<P>, f: F) -> VecBuffer<ChannelMap<P, S>>
    where P: WithChannel<S> + 'static,
          S: Primitive + 'static,
          F: Fn(P::Subpixel) -> S
{
    let mut out = Vec::with_capacity(image.len());
    out.extend(image.iter().map(|c| f(*c)));

    let (width, height) = image.dimensions();
    ImageBuffer::from_raw(width, height, out).unwrap()
}

/// Applies f to the color of each pixel in the input image.
pub fn map_colors<I, P, Q, F>(image: &I, f: F) -> VecBuffer<Q>
    where I: GenericImage<Pixel=P>,
          P: Pixel,
          Q: Pixel + 'static,
          F: Fn(P) -> Q
{
    map_pixels(image, |_, _, p| f(p))
}

/// Applies f to each pixel in the input image.
pub fn map_pixels<I, P, Q, F>(image: &I, f: F) -> VecBuffer<Q>
    where I: GenericImage<Pixel=P>,
          P: Pixel,
          Q: Pixel + 'static,
          F: Fn(u32, u32, P) -> Q
{
    let (width, height) = image.dimensions();
    let no_channel = Q::channel_count() as u32;
    let mut out = Vec::with_capacity((width * height * no_channel) as usize);

    for y in 0..height {
        for x in 0..width {
            let pix = f(x, y, image.get_pixel(x, y));
            for c in pix.channels().into_iter() {
                out.push(*c);
            }
        }
    }

    ImageBuffer::from_raw(width, height, out).unwrap()
}

macro_rules! implement_channel_extraction {
    ($extract_name: ident, $embed_name: ident, $idx: expr) => (
        /// Create a grayscale image by extracting a channel of an RGB image.
        pub fn $extract_name<I, C>(image: &I) -> VecBuffer<Luma<C>>
            where I: GenericImage<Pixel=Rgb<C>>,
                  C: Primitive + 'static
        {
            map_colors(image, |p| Luma([p[$idx]]))
        }

        /// Create an RGB image by embedding a grayscale image in a single channel.
        pub fn $embed_name<I, C>(image: &I) -> VecBuffer<Rgb<C>>
            where I: GenericImage<Pixel=Luma<C>>,
                  C: Primitive + 'static
        {
            map_colors(image, |p| {
                let mut cs = [C::zero(); 3];
                cs[$idx] = p[0];
                Rgb(cs)
            })
        }
    )
}

implement_channel_extraction!(red_channel, as_red_channel, 0);
implement_channel_extraction!(green_channel, as_green_channel, 1);
implement_channel_extraction!(blue_channel, as_blue_channel, 2);

#[cfg(test)]
mod test {
    use super::{
        map_colors,
        map_pixels,
        map_subpixels,
        map_subpixels_vec,
        red_channel,
        green_channel,
        blue_channel,
        as_red_channel,
        as_green_channel,
        as_blue_channel
    };
    use image::{
        GrayImage,
        ImageBuffer,
        Rgb,
        RgbImage
    };
    use test::Bencher;

    #[test]
    fn test_map_subpixels() {
        let image: GrayImage = ImageBuffer::from_raw(2, 2, vec![
            1, 2,
            3, 4]).unwrap();

        let expected = ImageBuffer::from_raw(2, 2, vec![
            -2i16, -4i16,
            -6i16, -8i16]).unwrap();

        let mapped = map_subpixels(&image, |x| -2 * (x as i16));
        assert_pixels_eq!(mapped, expected);
    }
    
    #[test]
    fn test_map_subpixels_vec() {
        let image: GrayImage = ImageBuffer::from_raw(2, 2, vec![
            1, 2,
            3, 4]).unwrap();

        let expected = ImageBuffer::from_raw(2, 2, vec![
            -2i16, -4i16,
            -6i16, -8i16]).unwrap();

        let mapped = map_subpixels_vec(&image, |x| -2 * (x as i16));
        assert_pixels_eq!(mapped, expected);
    }

    #[test]
    fn test_map_colors() {
        let image: GrayImage = ImageBuffer::from_raw(2, 2, vec![
            1, 2,
            3, 4]).unwrap();

        let expected: ImageBuffer<Rgb<i16>, Vec<i16>> = ImageBuffer::from_raw(2, 2, vec![
            1i16, 2i16, 3i16, 2i16, 4i16, 6i16,
            3i16, 6i16, 9i16, 4i16, 8i16, 12i16]).unwrap();

        let mapped = map_colors(&image, |p| {
            let intensity = p[0] as i16;
            Rgb([intensity, (2 * intensity), (3 * intensity)])
        });
        assert_pixels_eq!(mapped, expected);
    }

    #[test]
    fn test_map_pixels() {
        let image: GrayImage = ImageBuffer::from_raw(2, 2, vec![
            1, 2,
            3, 4]).unwrap();

        let expected: ImageBuffer<Rgb<i16>, Vec<i16>> = ImageBuffer::from_raw(2, 2, vec![
            1i16, 2i16, 3i16, 3i16, 5i16, 7i16,
            4i16, 7i16, 10i16, 6i16, 10i16, 14i16]).unwrap();

        let mapped = map_pixels(&image, |x, y, p| {
            let intensity = p[0] as i16;
            let offset = (x + y) as i16;
            Rgb([intensity + offset, 2 * intensity + offset, 3 * intensity + offset])
        });
        assert_pixels_eq!(mapped, expected);
    }

    #[test]
    fn test_red_channel() {
        let image: RgbImage = ImageBuffer::from_raw(2, 2, vec![
            1, 2, 3, 2, 4, 6,
            3, 6, 9, 4, 8, 12]).unwrap();

        let expected: GrayImage = ImageBuffer::from_raw(2, 2, vec![
            1, 2,
            3, 4]).unwrap();

        let actual = red_channel(&image);
        assert_pixels_eq!(actual, expected);
    }

    #[test]
    fn test_green_channel() {
        let image: RgbImage = ImageBuffer::from_raw(2, 2, vec![
            1, 2, 3, 2, 4, 6,
            3, 6, 9, 4, 8, 12]).unwrap();

        let expected: GrayImage = ImageBuffer::from_raw(2, 2, vec![
            2, 4,
            6, 8]).unwrap();

        let actual = green_channel(&image);
        assert_pixels_eq!(actual, expected);
    }

    #[test]
    fn test_blue_channel() {
        let image: RgbImage = ImageBuffer::from_raw(2, 2, vec![
            1, 2, 3, 2, 4, 6,
            3, 6, 9, 4, 8, 12]).unwrap();

        let expected: GrayImage = ImageBuffer::from_raw(2, 2, vec![
            3, 6,
            9, 12]).unwrap();

        let actual = blue_channel(&image);
        assert_pixels_eq!(actual, expected);
    }

    #[test]
    fn test_as_red_channel() {
        let image: GrayImage = ImageBuffer::from_raw(2, 2, vec![
            1, 2,
            3, 4]).unwrap();

        let expected: RgbImage = ImageBuffer::from_raw(2, 2, vec![
            1, 0, 0, 2, 0, 0,
            3, 0, 0, 4, 0, 0]).unwrap();

        let actual = as_red_channel(&image);
        assert_pixels_eq!(actual, expected);
    }

    #[test]
    fn test_as_green_channel() {
        let image: GrayImage = ImageBuffer::from_raw(2, 2, vec![
            1, 2,
            3, 4]).unwrap();

        let expected: RgbImage = ImageBuffer::from_raw(2, 2, vec![
            0, 1, 0, 0, 2, 0,
            0, 3, 0, 0, 4, 0]).unwrap();

        let actual = as_green_channel(&image);
        assert_pixels_eq!(actual, expected);
    }

    #[test]
    fn test_as_blue_channel() {
        let image: GrayImage = ImageBuffer::from_raw(2, 2, vec![
            1, 2,
            3, 4]).unwrap();

        let expected: RgbImage = ImageBuffer::from_raw(2, 2, vec![
            0, 0, 1, 0, 0, 2,
            0, 0, 3, 0, 0, 4]).unwrap();

        let actual = as_blue_channel(&image);
        assert_pixels_eq!(actual, expected);
    }
}

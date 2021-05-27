//! Functions for mapping over pixels, colors or subpixels of images.

use image::{Bgr, Bgra, GenericImage, ImageBuffer, Luma, LumaA, Pixel, Primitive, Rgb, Rgba};

use crate::definitions::Image;

/// The type obtained by replacing the channel type of a given `Pixel` type.
/// The output type must have the same name of channels as the input type, or
/// several algorithms will produce incorrect results or panic.
pub trait WithChannel<C: Primitive>: Pixel {
    /// The new pixel type.
    type Pixel: Pixel<Subpixel = C> + 'static;
}

/// Alias to make uses of `WithChannel` less syntactically noisy.
pub type ChannelMap<Pix, Sub> = <Pix as WithChannel<Sub>>::Pixel;

impl<T, U> WithChannel<U> for Rgb<T>
where
    T: Primitive + 'static,
    U: Primitive + 'static,
{
    type Pixel = Rgb<U>;
}

impl<T, U> WithChannel<U> for Rgba<T>
where
    T: Primitive + 'static,
    U: Primitive + 'static,
{
    type Pixel = Rgba<U>;
}

impl<T, U> WithChannel<U> for Bgr<T>
where
    T: Primitive + 'static,
    U: Primitive + 'static,
{
    type Pixel = Bgr<U>;
}

impl<T, U> WithChannel<U> for Bgra<T>
where
    T: Primitive + 'static,
    U: Primitive + 'static,
{
    type Pixel = Bgra<U>;
}

impl<T, U> WithChannel<U> for Luma<T>
where
    T: Primitive + 'static,
    U: Primitive + 'static,
{
    type Pixel = Luma<U>;
}

impl<T, U> WithChannel<U> for LumaA<T>
where
    T: Primitive + 'static,
    U: Primitive + 'static,
{
    type Pixel = LumaA<U>;
}

/// Applies `f` to each subpixel of the input image.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use imageproc::map::map_subpixels;
///
/// let image = gray_image!(
///     1, 2;
///     3, 4);
///
/// let scaled = gray_image!(type: i16,
///     -2, -4;
///     -6, -8);
///
/// assert_pixels_eq!(
///     map_subpixels(&image, |x| -2 * (x as i16)),
///     scaled);
/// # }
/// ```
pub fn map_subpixels<I, P, F, S>(image: &I, f: F) -> Image<ChannelMap<P, S>>
where
    I: GenericImage<Pixel = P>,
    P: WithChannel<S> + 'static,
    S: Primitive + 'static,
    F: Fn(P::Subpixel) -> S,
{
    let (width, height) = image.dimensions();
    let mut out: ImageBuffer<ChannelMap<P, S>, Vec<S>> = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let out_channels = out.get_pixel_mut(x, y).channels_mut();
            for c in 0..P::CHANNEL_COUNT {
                out_channels[c as usize] = f(unsafe {
                    *image
                        .unsafe_get_pixel(x, y)
                        .channels()
                        .get_unchecked(c as usize)
                });
            }
        }
    }

    out
}

/// Applies `f` to each subpixel of the input image in place.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use imageproc::map::map_subpixels_mut;
///
/// let mut image = gray_image!(
///     1, 2;
///     3, 4);
///
/// let want = gray_image!(
///     2, 4;
///     6, 8);
///
/// map_subpixels_mut(&mut image, |x| 2 * x);
///
/// assert_pixels_eq!(
///     image,
///     want);
/// # }
/// ```
pub fn map_subpixels_mut<I, P, F>(image: &mut I, f: F)
where
    I: GenericImage<Pixel = P>,
    P: Pixel,
    F: Fn(P::Subpixel) -> P::Subpixel,
{
    let (width, height) = image.dimensions();

    for y in 0..height {
        for x in 0..width {
            for c in 0..P::CHANNEL_COUNT {
                image.get_pixel_mut(x, y).channels_mut()[c as usize] = f(unsafe {
                    *image
                        .unsafe_get_pixel(x, y)
                        .channels()
                        .get_unchecked(c as usize)
                });
            }
        }
    }
}

/// Applies `f` to the color of each pixel in the input image.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use image::Rgb;
/// use imageproc::map::map_colors;
///
/// let image = gray_image!(
///     1, 2;
///     3, 4);
///
/// let rgb = rgb_image!(
///     [1, 2, 3], [2, 4, 6];
///     [3, 6, 9], [4, 8, 12]);
///
/// assert_pixels_eq!(
///     map_colors(&image, |p| { Rgb([p[0], (2 * p[0]), (3 * p[0])]) }),
///     rgb);
/// # }
/// ```
pub fn map_colors<I, P, Q, F>(image: &I, f: F) -> Image<Q>
where
    I: GenericImage<Pixel = P>,
    P: Pixel,
    Q: Pixel + 'static,
    F: Fn(P) -> Q,
{
    let (width, height) = image.dimensions();
    let mut out: ImageBuffer<Q, Vec<Q::Subpixel>> = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            unsafe {
                let pix = image.unsafe_get_pixel(x, y);
                out.unsafe_put_pixel(x, y, f(pix));
            }
        }
    }

    out
}

/// Applies `f` to the color of each pixel in the input image in place.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use image::Luma;
/// use imageproc::map::map_colors_mut;
///
/// let mut image = gray_image!(
///     1, 2;
///     3, 4);
///
/// let want = gray_image!(
///     2, 4;
///     6, 8);
///
/// map_colors_mut(&mut image, |p| Luma([2 * p[0]]));
///
/// assert_pixels_eq!(
///     image,
///     want);
/// # }
/// ```
pub fn map_colors_mut<I, P, F>(image: &mut I, f: F)
where
    I: GenericImage<Pixel = P>,
    P: Pixel,
    F: Fn(P) -> P,
{
    let (width, height) = image.dimensions();

    for y in 0..height {
        for x in 0..width {
            unsafe {
                let pix = image.unsafe_get_pixel(x, y);
                image.unsafe_put_pixel(x, y, f(pix));
            }
        }
    }
}

/// Applies `f` to the colors of the pixels in the input images.
///
/// Requires `image1` and `image2` to have the same dimensions.
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use image::Luma;
/// use imageproc::map::map_colors2;
///
/// let image1 = gray_image!(
///     1, 2,
///     3, 4
/// );
///
/// let image2 = gray_image!(
///     10, 20,
///     30, 40
/// );
///
/// let sum = gray_image!(
///     11, 22,
///     33, 44
/// );
///
/// assert_pixels_eq!(
///     map_colors2(&image1, &image2, |p, q| Luma([p[0] + q[0]])),
///     sum
/// );
/// # }
/// ```
pub fn map_colors2<I, J, P, Q, R, F>(image1: &I, image2: &J, f: F) -> Image<R>
where
    I: GenericImage<Pixel = P>,
    J: GenericImage<Pixel = Q>,
    P: Pixel,
    Q: Pixel,
    R: Pixel + 'static,
    F: Fn(P, Q) -> R,
{
    assert_eq!(image1.dimensions(), image2.dimensions());

    let (width, height) = image1.dimensions();
    let mut out: ImageBuffer<R, Vec<R::Subpixel>> = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            unsafe {
                let p = image1.unsafe_get_pixel(x, y);
                let q = image2.unsafe_get_pixel(x, y);
                out.unsafe_put_pixel(x, y, f(p, q));
            }
        }
    }

    out
}

/// Applies `f` to each pixel in the input image.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use image::Rgb;
/// use imageproc::map::map_pixels;
///
/// let image = gray_image!(
///     1, 2;
///     3, 4);
///
/// let rgb = rgb_image!(
///     [1, 0, 0], [2, 1, 0];
///     [3, 0, 1], [4, 1, 1]);
///
/// assert_pixels_eq!(
///     map_pixels(&image, |x, y, p| {
///         Rgb([p[0], x as u8, y as u8])
///     }),
///     rgb);
/// # }
/// ```
pub fn map_pixels<I, P, Q, F>(image: &I, f: F) -> Image<Q>
where
    I: GenericImage<Pixel = P>,
    P: Pixel,
    Q: Pixel + 'static,
    F: Fn(u32, u32, P) -> Q,
{
    let (width, height) = image.dimensions();
    let mut out: ImageBuffer<Q, Vec<Q::Subpixel>> = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            unsafe {
                let pix = image.unsafe_get_pixel(x, y);
                out.unsafe_put_pixel(x, y, f(x, y, pix));
            }
        }
    }

    out
}

/// Applies `f` to each pixel in the input image in place.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use image::Luma;
/// use imageproc::map::map_pixels_mut;
///
/// let mut image = gray_image!(
///     1, 2;
///     3, 4);
///
/// let want = gray_image!(
///     1, 3;
///     4, 6);
///
/// map_pixels_mut(&mut image, |x, y, p| {
///     Luma([p[0] + x as u8 + y as u8])
/// });
///
/// assert_pixels_eq!(
///     image,
///     want);
/// # }
/// ```
pub fn map_pixels_mut<I, P, F>(image: &mut I, f: F)
where
    I: GenericImage<Pixel = P>,
    P: Pixel,
    F: Fn(u32, u32, P) -> P,
{
    let (width, height) = image.dimensions();

    for y in 0..height {
        for x in 0..width {
            unsafe {
                let pix = image.unsafe_get_pixel(x, y);
                image.unsafe_put_pixel(x, y, f(x, y, pix));
            }
        }
    }
}

/// Creates a grayscale image by extracting the red channel of an RGB image.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use image::Luma;
/// use imageproc::map::red_channel;
///
/// let image = rgb_image!(
///     [1, 2, 3], [2, 4, 6];
///     [3, 6, 9], [4, 8, 12]);
///
/// let expected = gray_image!(
///     1, 2;
///     3, 4);
///
/// let actual = red_channel(&image);
/// assert_pixels_eq!(actual, expected);
/// # }
/// ```
pub fn red_channel<I, C>(image: &I) -> Image<Luma<C>>
where
    I: GenericImage<Pixel = Rgb<C>>,
    C: Primitive + 'static,
{
    map_colors(image, |p| Luma([p[0]]))
}

/// Creates an RGB image by embedding a grayscale image in its red channel.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use image::Luma;
/// use imageproc::map::as_red_channel;
///
/// let image = gray_image!(
///     1, 2;
///     3, 4);
///
/// let expected = rgb_image!(
///     [1, 0, 0], [2, 0, 0];
///     [3, 0, 0], [4, 0, 0]);
///
/// let actual = as_red_channel(&image);
/// assert_pixels_eq!(actual, expected);
/// # }
/// ```
pub fn as_red_channel<I, C>(image: &I) -> Image<Rgb<C>>
where
    I: GenericImage<Pixel = Luma<C>>,
    C: Primitive + 'static,
{
    map_colors(image, |p| {
        let mut cs = [C::zero(); 3];
        cs[0] = p[0];
        Rgb(cs)
    })
}

/// Creates a grayscale image by extracting the green channel of an RGB image.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use image::Luma;
/// use imageproc::map::green_channel;
///
/// let image = rgb_image!(
///     [1, 2, 3], [2, 4, 6];
///     [3, 6, 9], [4, 8, 12]);
///
/// let expected = gray_image!(
///     2, 4;
///     6, 8);
///
/// let actual = green_channel(&image);
/// assert_pixels_eq!(actual, expected);
/// # }
/// ```
pub fn green_channel<I, C>(image: &I) -> Image<Luma<C>>
where
    I: GenericImage<Pixel = Rgb<C>>,
    C: Primitive + 'static,
{
    map_colors(image, |p| Luma([p[1]]))
}

/// Creates an RGB image by embedding a grayscale image in its green channel.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use image::Luma;
/// use imageproc::map::as_green_channel;
///
/// let image = gray_image!(
///     1, 2;
///     3, 4);
///
/// let expected = rgb_image!(
///     [0, 1, 0], [0, 2, 0];
///     [0, 3, 0], [0, 4, 0]);
///
/// let actual = as_green_channel(&image);
/// assert_pixels_eq!(actual, expected);
/// # }
/// ```
pub fn as_green_channel<I, C>(image: &I) -> Image<Rgb<C>>
where
    I: GenericImage<Pixel = Luma<C>>,
    C: Primitive + 'static,
{
    map_colors(image, |p| {
        let mut cs = [C::zero(); 3];
        cs[1] = p[0];
        Rgb(cs)
    })
}

/// Creates a grayscale image by extracting the blue channel of an RGB image.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use image::Luma;
/// use imageproc::map::blue_channel;
///
/// let image = rgb_image!(
///     [1, 2, 3], [2, 4, 6];
///     [3, 6, 9], [4, 8, 12]);
///
/// let expected = gray_image!(
///     3, 6;
///     9, 12);
///
/// let actual = blue_channel(&image);
/// assert_pixels_eq!(actual, expected);
/// # }
/// ```
pub fn blue_channel<I, C>(image: &I) -> Image<Luma<C>>
where
    I: GenericImage<Pixel = Rgb<C>>,
    C: Primitive + 'static,
{
    map_colors(image, |p| Luma([p[2]]))
}

/// Creates an RGB image by embedding a grayscale image in its blue channel.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use image::Luma;
/// use imageproc::map::as_blue_channel;
///
/// let image = gray_image!(
///     1, 2;
///     3, 4);
///
/// let expected = rgb_image!(
///     [0, 0, 1], [0, 0, 2];
///     [0, 0, 3], [0, 0, 4]);
///
/// let actual = as_blue_channel(&image);
/// assert_pixels_eq!(actual, expected);
/// # }
/// ```
pub fn as_blue_channel<I, C>(image: &I) -> Image<Rgb<C>>
where
    I: GenericImage<Pixel = Luma<C>>,
    C: Primitive + 'static,
{
    map_colors(image, |p| {
        let mut cs = [C::zero(); 3];
        cs[2] = p[0];
        Rgb(cs)
    })
}

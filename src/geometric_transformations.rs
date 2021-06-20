//! Geometric transformations of images. This includes rotations, translation, and general
//! projective transformations.

use crate::definitions::{Clamp, Image};
use crate::math::cast;
use conv::ValueInto;
use image::{GenericImageView, ImageBuffer, Pixel};
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::{cmp, ops::Mul};

#[derive(Copy, Clone, Debug)]
enum TransformationClass {
    Translation,
    Affine,
    Projection,
}

/// A 2d projective transformation, stored as a row major 3x3 matrix.
///
/// Transformations combine by pre-multiplication, i.e. applying `P * Q` is equivalent to
/// applying `Q` and then applying `P`. For example, the following defines a rotation
/// about the point (320.0, 240.0).
///
/// ```
/// use imageproc::geometric_transformations::*;
/// use std::f32::consts::PI;
///
/// let (cx, cy) = (320.0, 240.0);
///
/// let c_rotation = Projection::translate(cx, cy)
///     * Projection::rotate(PI / 6.0)
///     * Projection::translate(-cx, -cy);
/// ```
///
/// See ./examples/projection.rs for more examples.
#[derive(Copy, Clone, Debug)]
pub struct Projection {
    transform: [f32; 9],
    inverse: [f32; 9],
    class: TransformationClass,
}

impl Projection {
    /// Creates a 2d projective transform from a row-major 3x3 matrix in homogeneous coordinates.
    ///
    /// Returns `None` if the matrix is not invertible.
    pub fn from_matrix(transform: [f32; 9]) -> Option<Projection> {
        let transform = normalize(transform);
        let class = class_from_matrix(transform);
        try_inverse(&transform).map(|inverse| Projection {
            transform,
            inverse,
            class,
        })
    }

    /// Combine the transformation with another one. The resulting transformation is equivalent to
    /// applying this transformation followed by the `other` transformation.
    pub fn and_then(self, other: Projection) -> Projection {
        other * self
    }

    /// A translation by (tx, ty).
    #[rustfmt::skip]
    pub fn translate(tx: f32, ty: f32) -> Projection {
        Projection {
            transform: [
                1.0, 0.0, tx,
                0.0, 1.0, ty,
                0.0, 0.0, 1.0
            ],
            inverse: [
                1.0, 0.0, -tx,
                0.0, 1.0, -ty,
                0.0, 0.0, 1.0
            ],
            class: TransformationClass::Translation,
        }
    }

    /// A clockwise rotation around the top-left corner of the image by theta radians.
    #[rustfmt::skip]
    pub fn rotate(theta: f32) -> Projection {
        let (s, c) = theta.sin_cos();
        Projection {
            transform: [
                  c,  -s, 0.0,
                  s,   c, 0.0,
                0.0, 0.0, 1.0
            ],
            inverse: [
                  c,   s, 0.0,
                 -s,   c, 0.0,
                0.0, 0.0, 1.0
            ],
            class: TransformationClass::Affine,
        }
    }

    /// An anisotropic scaling (sx, sy).
    ///
    /// Note that the `warp` function does not change the size of the input image.
    /// If you want to resize an image then use the `imageops` module in the `image` crate.
    #[rustfmt::skip]
    pub fn scale(sx: f32, sy: f32) -> Projection {
        Projection {
            transform: [
                 sx, 0.0, 0.0,
                0.0,  sy, 0.0,
                0.0, 0.0, 1.0
            ],
            inverse: [
                1.0 / sx, 0.0,      0.0,
                0.0,      1.0 / sy, 0.0,
                0.0,      0.0,      1.0
            ],
            class: TransformationClass::Affine,
        }
    }

    /// Inverts the transformation.
    pub fn invert(self) -> Projection {
        Projection {
            transform: self.inverse,
            inverse: self.transform,
            class: self.class,
        }
    }

    /// Calculates a projection from a set of four control point pairs.
    pub fn from_control_points(from: [(f32, f32); 4], to: [(f32, f32); 4]) -> Option<Projection> {
        use approx::AbsDiffEq;
        use nalgebra::{linalg::SVD, OMatrix, OVector, U8};

        let (xf1, yf1, xf2, yf2, xf3, yf3, xf4, yf4) = (
            from[0].0 as f64,
            from[0].1 as f64,
            from[1].0 as f64,
            from[1].1 as f64,
            from[2].0 as f64,
            from[2].1 as f64,
            from[3].0 as f64,
            from[3].1 as f64,
        );

        let (x1, y1, x2, y2, x3, y3, x4, y4) = (
            to[0].0 as f64,
            to[0].1 as f64,
            to[1].0 as f64,
            to[1].1 as f64,
            to[2].0 as f64,
            to[2].1 as f64,
            to[3].0 as f64,
            to[3].1 as f64,
        );

        #[rustfmt::skip]
        let a = OMatrix::<_, U8, U8>::from_row_slice(&[
            0.0, 0.0, 0.0, -xf1, -yf1, -1.0,  y1 * xf1,  y1 * yf1,
            xf1, yf1, 1.0,  0.0,  0.0,  0.0, -x1 * xf1, -x1 * yf1,
            0.0, 0.0, 0.0, -xf2, -yf2, -1.0,  y2 * xf2,  y2 * yf2,
            xf2, yf2, 1.0,  0.0,  0.0,  0.0, -x2 * xf2, -x2 * yf2,
            0.0, 0.0, 0.0, -xf3, -yf3, -1.0,  y3 * xf3,  y3 * yf3,
            xf3, yf3, 1.0,  0.0,  0.0,  0.0, -x3 * xf3, -x3 * yf3,
            0.0, 0.0, 0.0, -xf4, -yf4, -1.0,  y4 * xf4,  y4 * yf4,
            xf4, yf4, 1.0,  0.0,  0.0,  0.0, -x4 * xf4, -x4 * yf4,
        ]);

        let b = OVector::<_, U8>::from_row_slice(&[-y1, x1, -y2, x2, -y3, x3, -y4, x4]);

        SVD::try_new(a, true, true, f64::default_epsilon(), 0)
            .and_then(|svd| svd.solve(&b, f64::default_epsilon()).ok())
            .and_then(|h| {
                let mut transform = [
                    h[0] as f32,
                    h[1] as f32,
                    h[2] as f32,
                    h[3] as f32,
                    h[4] as f32,
                    h[5] as f32,
                    h[6] as f32,
                    h[7] as f32,
                    1.0,
                ];

                transform = normalize(transform);
                let class = class_from_matrix(transform);

                try_inverse(&transform).map(|inverse| Projection {
                    transform,
                    inverse,
                    class,
                })
            })
    }

    // Helper functions used as optimization in warp.
    #[inline(always)]
    fn map_projective(&self, x: f32, y: f32) -> (f32, f32) {
        let t = &self.transform;
        let d = t[6] * x + t[7] * y + t[8];
        (
            (t[0] * x + t[1] * y + t[2]) / d,
            (t[3] * x + t[4] * y + t[5]) / d,
        )
    }

    #[inline(always)]
    fn map_affine(&self, x: f32, y: f32) -> (f32, f32) {
        let t = &self.transform;
        ((t[0] * x + t[1] * y + t[2]), (t[3] * x + t[4] * y + t[5]))
    }

    #[inline(always)]
    fn map_translation(&self, x: f32, y: f32) -> (f32, f32) {
        let t = &self.transform;
        let tx = t[2];
        let ty = t[5];
        (x + tx, y + ty)
    }
}

impl Mul<Projection> for Projection {
    type Output = Projection;

    fn mul(self, rhs: Projection) -> Projection {
        use TransformationClass as TC;
        let t = mul3x3(self.transform, rhs.transform);
        let i = mul3x3(rhs.inverse, self.inverse);

        let class = match (self.class, rhs.class) {
            (TC::Translation, TC::Translation) => TC::Translation,
            (TC::Translation, TC::Affine) => TC::Affine,
            (TC::Affine, TC::Translation) => TC::Affine,
            (TC::Affine, TC::Affine) => TC::Affine,
            (_, _) => TC::Projection,
        };

        Projection {
            transform: t,
            inverse: i,
            class,
        }
    }
}

impl<'a, 'b> Mul<&'b Projection> for &'a Projection {
    type Output = Projection;

    fn mul(self, rhs: &Projection) -> Projection {
        *self * *rhs
    }
}

impl Mul<(f32, f32)> for Projection {
    type Output = (f32, f32);

    fn mul(self, rhs: (f32, f32)) -> (f32, f32) {
        let (x, y) = rhs;
        match self.class {
            TransformationClass::Translation => self.map_translation(x, y),
            TransformationClass::Affine => self.map_affine(x, y),
            TransformationClass::Projection => self.map_projective(x, y),
        }
    }
}

impl<'a, 'b> Mul<&'b (f32, f32)> for &'a Projection {
    type Output = (f32, f32);

    fn mul(self, rhs: &(f32, f32)) -> (f32, f32) {
        *self * *rhs
    }
}

/// Rotates an image clockwise about its center.
/// The output image has the same dimensions as the input. Output pixels
/// whose pre-image lies outside the input image are set to `default`.
pub fn rotate_about_center<P>(
    image: &Image<P>,
    theta: f32,
    interpolation: Interpolation,
    default: P,
) -> Image<P>
where
    P: Pixel + Send + Sync + 'static,
    <P as Pixel>::Subpixel: Send + Sync,
    <P as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>,
{
    let (w, h) = image.dimensions();
    rotate(
        image,
        (w as f32 / 2.0, h as f32 / 2.0),
        theta,
        interpolation,
        default,
    )
}

/// Rotates an image clockwise about the provided center by theta radians.
/// The output image has the same dimensions as the input. Output pixels
/// whose pre-image lies outside the input image are set to `default`.
pub fn rotate<P>(
    image: &Image<P>,
    center: (f32, f32),
    theta: f32,
    interpolation: Interpolation,
    default: P,
) -> Image<P>
where
    P: Pixel + Send + Sync + 'static,
    <P as Pixel>::Subpixel: Send + Sync,
    <P as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>,
{
    let (cx, cy) = center;
    let projection =
        Projection::translate(cx, cy) * Projection::rotate(theta) * Projection::translate(-cx, -cy);
    warp(image, &projection, interpolation, default)
}

/// Translates the input image by t. Note that image coordinates increase from
/// top left to bottom right. Output pixels whose pre-image are not in the input
/// image are set to the boundary pixel in the input image nearest to their pre-image.
// TODO: it's confusing that this has different behaviour to
// TODO: attempting the equivalent transformation using Projection.
pub fn translate<P>(image: &Image<P>, t: (i32, i32)) -> Image<P>
where
    P: Pixel + 'static,
{
    let (width, height) = image.dimensions();
    let (tx, ty) = t;
    let (w, h) = (width as i32, height as i32);
    let num_channels = P::CHANNEL_COUNT as usize;
    let mut out = ImageBuffer::new(width, height);

    for y in 0..height {
        let y_in = cmp::max(0, cmp::min(y as i32 - ty, h - 1));

        if tx > 0 {
            let p_min = *image.get_pixel(0, y_in as u32);
            for x in 0..tx.min(w) {
                out.put_pixel(x as u32, y, p_min);
            }

            if tx < w {
                let in_base = (y_in as usize * width as usize) * num_channels;
                let out_base = (y as usize * width as usize + (tx as usize)) * num_channels;
                let len = (w - tx) as usize * num_channels;
                (*out)[out_base..][..len].copy_from_slice(&(**image)[in_base..][..len]);
            }
        } else {
            let p_max = *image.get_pixel(width - 1, y_in as u32);
            for x in (w + tx).max(0)..w {
                out.put_pixel(x as u32, y, p_max);
            }

            if w + tx > 0 {
                let in_base = (y_in as usize * width as usize + (tx.abs() as usize)) * num_channels;
                let out_base = (y as usize * width as usize) * num_channels;
                let len = (w + tx) as usize * num_channels;
                (*out)[out_base..][..len].copy_from_slice(&(**image)[in_base..][..len]);
            }
        }
    }

    out
}

/// Applies a projective transformation to an image.
///
/// The returned image has the same dimensions as `image`. Output pixels
/// whose pre-image lies outside the input image are set to `default`.
///
/// The provided projection defines a mapping from locations in the input image to their
/// corresponding location in the output image.
pub fn warp<P>(
    image: &Image<P>,
    projection: &Projection,
    interpolation: Interpolation,
    default: P,
) -> Image<P>
where
    P: Pixel + Send + Sync + 'static,
    <P as Pixel>::Subpixel: Send + Sync,
    <P as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>,
{
    let (width, height) = image.dimensions();
    let mut out = ImageBuffer::new(width, height);
    warp_into(image, projection, interpolation, default, &mut out);
    out
}

/// Applies a projective transformation to an image, writing to a provided output.
///
/// See the [`warp`](fn.warp.html) documentation for more information.
pub fn warp_into<P>(
    image: &Image<P>,
    projection: &Projection,
    interpolation: Interpolation,
    default: P,
    out: &mut Image<P>,
) where
    P: Pixel + Send + Sync + 'static,
    <P as Pixel>::Subpixel: Send + Sync,
    <P as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32> + Sync,
{
    let projection = projection.invert();
    let nn = |x, y| interpolate_nearest(image, x, y, default);
    let bl = |x, y| interpolate_bilinear(image, x, y, default);
    let bc = |x, y| interpolate_bicubic(image, x, y, default);
    let wp = |x, y| projection.map_projective(x, y);
    let wa = |x, y| projection.map_affine(x, y);
    let wt = |x, y| projection.map_translation(x, y);
    use Interpolation as I;
    use TransformationClass as TC;

    match (interpolation, projection.class) {
        (I::Nearest, TC::Translation) => warp_inner(out, wt, nn),
        (I::Nearest, TC::Affine) => warp_inner(out, wa, nn),
        (I::Nearest, TC::Projection) => warp_inner(out, wp, nn),
        (I::Bilinear, TC::Translation) => warp_inner(out, wt, bl),
        (I::Bilinear, TC::Affine) => warp_inner(out, wa, bl),
        (I::Bilinear, TC::Projection) => warp_inner(out, wp, bl),
        (I::Bicubic, TC::Translation) => warp_inner(out, wt, bc),
        (I::Bicubic, TC::Affine) => warp_inner(out, wa, bc),
        (I::Bicubic, TC::Projection) => warp_inner(out, wp, bc),
    }
}

/// Warps an image using the provided function to define the pre-image of each output pixel.
///
/// # Examples
/// Applying a wave pattern.
/// ```
/// use image::{ImageBuffer, Luma};
/// use imageproc::utils::gray_bench_image;
/// use imageproc::geometric_transformations::*;
///
/// let image = gray_bench_image(300, 300);
/// let warped = warp_with(
///     &image,
///     |x, y| (x, y + (x / 30.0).sin()),
///     Interpolation::Nearest,
///     Luma([0u8])
/// );
/// ```
pub fn warp_with<P, F>(
    image: &Image<P>,
    mapping: F,
    interpolation: Interpolation,
    default: P,
) -> Image<P>
where
    F: Fn(f32, f32) -> (f32, f32) + Sync + Send,
    P: Pixel + Send + Sync + 'static,
    <P as Pixel>::Subpixel: Send + Sync,
    <P as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>,
{
    let (width, height) = image.dimensions();
    let mut out = ImageBuffer::new(width, height);
    warp_into_with(image, mapping, interpolation, default, &mut out);
    out
}

/// Warps an image using the provided function to define the pre-image of each output pixel,
/// writing into a preallocated output.
///
/// See the [`warp_with`](fn.warp_with.html) documentation for more information.
pub fn warp_into_with<P, F>(
    image: &Image<P>,
    mapping: F,
    interpolation: Interpolation,
    default: P,
    out: &mut Image<P>,
) where
    F: Fn(f32, f32) -> (f32, f32) + Send + Sync,
    P: Pixel + Send + Sync + 'static,
    <P as Pixel>::Subpixel: Send + Sync,
    <P as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>,
{
    let nn = |x, y| interpolate_nearest(image, x, y, default);
    let bl = |x, y| interpolate_bilinear(image, x, y, default);
    let bc = |x, y| interpolate_bicubic(image, x, y, default);
    use Interpolation as I;

    match interpolation {
        I::Nearest => warp_inner(out, mapping, nn),
        I::Bilinear => warp_inner(out, mapping, bl),
        I::Bicubic => warp_inner(out, mapping, bc),
    }
}

// Work horse of all warp functions
// TODO: make faster by avoiding boundary checks in inner section of src image
fn warp_inner<P, Fc, Fi>(out: &mut Image<P>, mapping: Fc, get_pixel: Fi)
where
    P: Pixel + 'static,
    <P as Pixel>::Subpixel: Send + Sync,
    <P as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>,
    Fc: Fn(f32, f32) -> (f32, f32) + Send + Sync,
    Fi: Fn(f32, f32) -> P + Send + Sync,
{
    let width = out.width();
    let raw_out = out.as_mut();
    let pitch = P::CHANNEL_COUNT as usize * width as usize;

    #[cfg(feature = "rayon")]
    let chunks = raw_out.par_chunks_mut(pitch);
    #[cfg(not(feature = "rayon"))]
    let chunks = raw_out.chunks_mut(pitch);

    chunks.enumerate().for_each(|(y, row)| {
        for (x, slice) in row.chunks_mut(P::CHANNEL_COUNT as usize).enumerate() {
            let (px, py) = mapping(x as f32, y as f32);
            *P::from_slice_mut(slice) = get_pixel(px, py);
        }
    });
}

// Classifies transformation by looking up transformation matrix coefficients
fn class_from_matrix(mx: [f32; 9]) -> TransformationClass {
    if (mx[6] - 0.0).abs() < 1e-10 && (mx[7] - 0.0).abs() < 1e-10 && (mx[8] - 1.0).abs() < 1e-10 {
        if (mx[0] - 1.0).abs() < 1e-10
            && (mx[1] - 0.0).abs() < 1e-10
            && (mx[3] - 0.0).abs() < 1e-10
            && (mx[4] - 1.0).abs() < 1e-10
        {
            TransformationClass::Translation
        } else {
            TransformationClass::Affine
        }
    } else {
        TransformationClass::Projection
    }
}

fn normalize(mx: [f32; 9]) -> [f32; 9] {
    [
        mx[0] / mx[8],
        mx[1] / mx[8],
        mx[2] / mx[8],
        mx[3] / mx[8],
        mx[4] / mx[8],
        mx[5] / mx[8],
        mx[6] / mx[8],
        mx[7] / mx[8],
        1.0,
    ]
}

// TODO: write me in f64
fn try_inverse(t: &[f32; 9]) -> Option<[f32; 9]> {
    let [t00, t01, t02, t10, t11, t12, t20, t21, t22] = t;

    let m00 = t11 * t22 - t12 * t21;
    let m01 = t10 * t22 - t12 * t20;
    let m02 = t10 * t21 - t11 * t20;

    let det = t00 * m00 - t01 * m01 + t02 * m02;

    if det.abs() < 1e-10 {
        return None;
    }

    let m10 = t01 * t22 - t02 * t21;
    let m11 = t00 * t22 - t02 * t20;
    let m12 = t00 * t21 - t01 * t20;
    let m20 = t01 * t12 - t02 * t11;
    let m21 = t00 * t12 - t02 * t10;
    let m22 = t00 * t11 - t01 * t10;

    #[rustfmt::skip]
    let inv = [
         m00 / det, -m10 / det,  m20 / det,
        -m01 / det,  m11 / det, -m21 / det,
         m02 / det, -m12 / det,  m22 / det,
    ];

    Some(normalize(inv))
}

fn mul3x3(a: [f32; 9], b: [f32; 9]) -> [f32; 9] {
    let [a00, a01, a02, a10, a11, a12, a20, a21, a22] = a;
    let [b00, b01, b02, b10, b11, b12, b20, b21, b22] = b;
    [
        a00 * b00 + a01 * b10 + a02 * b20,
        a00 * b01 + a01 * b11 + a02 * b21,
        a00 * b02 + a01 * b12 + a02 * b22,
        a10 * b00 + a11 * b10 + a12 * b20,
        a10 * b01 + a11 * b11 + a12 * b21,
        a10 * b02 + a11 * b12 + a12 * b22,
        a20 * b00 + a21 * b10 + a22 * b20,
        a20 * b01 + a21 * b11 + a22 * b21,
        a20 * b02 + a21 * b12 + a22 * b22,
    ]
}

fn blend_cubic<P>(px0: &P, px1: &P, px2: &P, px3: &P, x: f32) -> P
where
    P: Pixel,
    P::Subpixel: ValueInto<f32> + Clamp<f32>,
{
    let mut outp = *px0;

    for i in 0..(P::CHANNEL_COUNT as usize) {
        let p0 = cast(px0.channels()[i]);
        let p1 = cast(px1.channels()[i]);
        let p2 = cast(px2.channels()[i]);
        let p3 = cast(px3.channels()[i]);
        #[rustfmt::skip]
        let pval = p1 + 0.5 * x * (p2 - p0 + x * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + x * (3.0 * (p1 - p2) + p3 - p0)));
        outp.channels_mut()[i] = <P as Pixel>::Subpixel::clamp(pval);
    }

    outp
}

fn interpolate_bicubic<P>(image: &Image<P>, x: f32, y: f32, default: P) -> P
where
    P: Pixel + 'static,
    <P as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>,
{
    let left = x.floor() - 1f32;
    let right = left + 4f32;
    let top = y.floor() - 1f32;
    let bottom = top + 4f32;

    let x_weight = x - (left + 1f32);
    let y_weight = y - (top + 1f32);

    let mut col: [P; 4] = [default, default, default, default];

    let (width, height) = image.dimensions();
    if left < 0f32 || right >= width as f32 || top < 0f32 || bottom >= height as f32 {
        default
    } else {
        for row in top as u32..bottom as u32 {
            let (p0, p1, p2, p3): (P, P, P, P) = unsafe {
                (
                    image.unsafe_get_pixel(left as u32, row),
                    image.unsafe_get_pixel(left as u32 + 1, row),
                    image.unsafe_get_pixel(left as u32 + 2, row),
                    image.unsafe_get_pixel(left as u32 + 3, row),
                )
            };

            let c = blend_cubic(&p0, &p1, &p2, &p3, x_weight);
            col[row as usize - top as usize] = c;
        }

        blend_cubic(&col[0], &col[1], &col[2], &col[3], y_weight)
    }
}

fn blend_bilinear<P>(
    top_left: P,
    top_right: P,
    bottom_left: P,
    bottom_right: P,
    right_weight: f32,
    bottom_weight: f32,
) -> P
where
    P: Pixel,
    P::Subpixel: ValueInto<f32> + Clamp<f32>,
{
    let top = top_left.map2(&top_right, |u, v| {
        P::Subpixel::clamp((1f32 - right_weight) * cast(u) + right_weight * cast(v))
    });

    let bottom = bottom_left.map2(&bottom_right, |u, v| {
        P::Subpixel::clamp((1f32 - right_weight) * cast(u) + right_weight * cast(v))
    });

    top.map2(&bottom, |u, v| {
        P::Subpixel::clamp((1f32 - bottom_weight) * cast(u) + bottom_weight * cast(v))
    })
}

fn interpolate_bilinear<P>(image: &Image<P>, x: f32, y: f32, default: P) -> P
where
    P: Pixel + 'static,
    <P as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>,
{
    let left = x.floor();
    let right = left + 1f32;
    let top = y.floor();
    let bottom = top + 1f32;

    let right_weight = x - left;
    let bottom_weight = y - top;

    // default if out of bound
    let (width, height) = image.dimensions();
    if left < 0f32 || right >= width as f32 || top < 0f32 || bottom >= height as f32 {
        default
    } else {
        let (tl, tr, bl, br) = unsafe {
            (
                image.unsafe_get_pixel(left as u32, top as u32),
                image.unsafe_get_pixel(right as u32, top as u32),
                image.unsafe_get_pixel(left as u32, bottom as u32),
                image.unsafe_get_pixel(right as u32, bottom as u32),
            )
        };
        blend_bilinear(tl, tr, bl, br, right_weight, bottom_weight)
    }
}

#[inline(always)]
fn interpolate_nearest<P: Pixel + 'static>(image: &Image<P>, x: f32, y: f32, default: P) -> P {
    let rx = x.round();
    let ry = y.round();

    let (width, height) = image.dimensions();
    if rx < 0f32 || rx >= width as f32 || ry < 0f32 || ry >= height as f32 {
        default
    } else {
        unsafe { image.unsafe_get_pixel(rx as u32, ry as u32) }
    }
}

/// How to handle pixels whose pre-image lies between input pixels.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum Interpolation {
    /// Choose the nearest pixel to the pre-image of the
    /// output pixel.
    Nearest,
    /// Bilinearly interpolate between the four pixels
    /// closest to the pre-image of the output pixel.
    Bilinear,
    /// Bicubicly interpolate between the four pixels
    /// closest to the pre-image of the output pixel.
    Bicubic,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::gray_bench_image;
    use image::{GrayImage, Luma};
    use test::{black_box, Bencher};

    #[test]
    fn test_rotate_nearest_zero_radians() {
        let image = gray_image!(
            00, 01, 02;
            10, 11, 12);

        let rotated = rotate(
            &image,
            (0f32, 0f32),
            0f32,
            Interpolation::Nearest,
            Luma([99u8]),
        );
        assert_pixels_eq!(rotated, image);
    }

    #[test]
    fn text_rotate_nearest_quarter_turn_clockwise() {
        let image = gray_image!(
            00, 01, 02;
            10, 11, 12);

        let expected = gray_image!(
            11, 01, 99;
            12, 02, 99);
        let c = Projection::translate(1.0, 0.0);
        let rot = c * Projection::rotate(90f32.to_radians()) * c.invert();

        let rotated = warp(&image, &rot, Interpolation::Nearest, Luma([99u8]));
        assert_pixels_eq!(rotated, expected);
    }

    #[test]
    fn text_rotate_nearest_half_turn_anticlockwise() {
        let image = gray_image!(
            00, 01, 02;
            10, 11, 12);

        let expected = gray_image!(
            12, 11, 10;
            02, 01, 00);
        let c = Projection::translate(1.0, 0.5);

        let rot = c * Projection::rotate((-180f32).to_radians()) * c.invert();

        let rotated = warp(&image, &rot, Interpolation::Nearest, Luma([99u8]));
        assert_pixels_eq!(rotated, expected);
    }

    #[bench]
    fn bench_rotate_nearest(b: &mut Bencher) {
        let image = GrayImage::from_pixel(200, 200, Luma([15u8]));
        let c = Projection::translate(3.0, 3.0);
        let rot = c * Projection::rotate(1f32.to_degrees()) * c.invert();
        b.iter(|| {
            let rotated = warp(&image, &rot, Interpolation::Nearest, Luma([98u8]));
            black_box(rotated);
        });
    }

    #[bench]
    fn bench_rotate_bilinear(b: &mut Bencher) {
        let image = GrayImage::from_pixel(200, 200, Luma([15u8]));
        let c = Projection::translate(3.0, 3.0);
        let rot = c * Projection::rotate(1f32.to_degrees()) * c.invert();
        b.iter(|| {
            let rotated = warp(&image, &rot, Interpolation::Bilinear, Luma([98u8]));
            black_box(rotated);
        });
    }

    #[bench]
    fn bench_rotate_bicubic(b: &mut Bencher) {
        let image = GrayImage::from_pixel(200, 200, Luma([15u8]));
        let c = Projection::translate(3.0, 3.0);
        let rot = c * Projection::rotate(1f32.to_degrees()) * c.invert();
        b.iter(|| {
            let rotated = warp(&image, &rot, Interpolation::Bicubic, Luma([98u8]));
            black_box(rotated);
        });
    }

    #[test]
    fn test_translate_positive_x_positive_y() {
        let image = gray_image!(
            00, 01, 02;
            10, 11, 12;
            20, 21, 22);

        let expected = gray_image!(
            00, 00, 01;
            00, 00, 01;
            10, 10, 11);

        let translated = translate(&image, (1, 1));
        assert_pixels_eq!(translated, expected);
    }

    #[test]
    fn test_translate_positive_x_negative_y() {
        let image = gray_image!(
            00, 01, 02;
            10, 11, 12;
            20, 21, 22);

        let expected = gray_image!(
            10, 10, 11;
            20, 20, 21;
            20, 20, 21);

        let translated = translate(&image, (1, -1));
        assert_pixels_eq!(translated, expected);
    }

    #[test]
    fn test_translate_negative_x() {
        let image = gray_image!(
            00, 01, 02;
            10, 11, 12;
            20, 21, 22);

        let expected = gray_image!(
            01, 02, 02;
            11, 12, 12;
            21, 22, 22);

        let translated = translate(&image, (-1, 0));
        assert_pixels_eq!(translated, expected);
    }

    #[test]
    fn test_translate_large_x_large_y() {
        let image = gray_image!(
            00, 01, 02;
            10, 11, 12;
            20, 21, 22);

        let expected = gray_image!(
            00, 00, 00;
            00, 00, 00;
            00, 00, 00);

        // Translating by more than the image width and height
        let translated = translate(&image, (5, 5));
        assert_pixels_eq!(translated, expected);
    }

    #[bench]
    fn bench_translate(b: &mut Bencher) {
        let image = gray_bench_image(500, 500);
        b.iter(|| {
            let translated = translate(&image, (30, 30));
            black_box(translated);
        });
    }

    #[test]
    fn test_translate_positive_x_positive_y_projection() {
        let image = gray_image!(
            00, 01, 02;
            10, 11, 12;
            20, 21, 22);

        let expected = gray_image!(
            00, 00, 00;
            00, 00, 01;
            00, 10, 11);

        let translated = warp(
            &image,
            &Projection::translate(1.0, 1.0),
            Interpolation::Nearest,
            Luma([0u8]),
        );
        assert_pixels_eq!(translated, expected);
    }

    #[test]
    fn test_translate_positive_x_negative_y_projection() {
        let image = gray_image!(
            00, 01, 02;
            10, 11, 12;
            20, 21, 22);

        let expected = gray_image!(
            00, 10, 11;
            00, 20, 21;
            00, 00, 00);

        let translated = warp(
            &image,
            &Projection::translate(1.0, -1.0),
            Interpolation::Nearest,
            Luma([0u8]),
        );
        assert_pixels_eq!(translated, expected);
    }

    #[test]
    fn test_translate_large_x_large_y_projection() {
        let image = gray_image!(
            00, 01, 02;
            10, 11, 12;
            20, 21, 22);

        let expected = gray_image!(
            00, 00, 00;
            00, 00, 00;
            00, 00, 00);

        // Translating by more than the image width and height
        let translated = warp(
            &image,
            &Projection::translate(5.0, 5.0),
            Interpolation::Nearest,
            Luma([0u8]),
        );
        assert_pixels_eq!(translated, expected);
    }

    #[bench]
    fn bench_translate_projection(b: &mut Bencher) {
        let image = gray_bench_image(500, 500);
        let t = Projection::translate(-30.0, -30.0);

        b.iter(|| {
            let translated = warp(&image, &t, Interpolation::Nearest, Luma([0u8]));
            black_box(translated);
        });
    }

    #[bench]
    fn bench_translate_with(b: &mut Bencher) {
        let image = gray_bench_image(500, 500);

        b.iter(|| {
            let (width, height) = image.dimensions();
            let mut out = ImageBuffer::new(width, height);
            warp_into_with(
                &image,
                |x, y| (x - 30.0, y - 30.0),
                Interpolation::Nearest,
                Luma([0u8]),
                &mut out,
            );
            black_box(out);
        });
    }

    #[test]
    fn test_affine() {
        let image = gray_image!(
            00, 01, 02;
            10, 11, 12;
            20, 21, 22);

        let expected = gray_image!(
            00, 00, 00;
            00, 00, 01;
            00, 10, 11);

        #[rustfmt::skip]
        let aff = Projection::from_matrix([
            1.0, 0.0, 1.0,
            0.0, 1.0, 1.0,
            0.0, 0.0, 1.0
        ]).unwrap();

        let translated_nearest = warp(&image, &aff, Interpolation::Nearest, Luma([0u8]));
        assert_pixels_eq!(translated_nearest, expected);
        let translated_bilinear = warp(&image, &aff, Interpolation::Bilinear, Luma([0u8]));
        assert_pixels_eq!(translated_bilinear, expected);
    }

    #[test]
    fn test_affine_bicubic() {
        let image = gray_image!(
            99, 01, 02, 03, 04;
            10, 11, 12, 13, 14;
            20, 21, 22, 23, 24;
            30, 31, 32, 33, 34;
            40, 41, 42, 43, 44);

        // Expect 2 pixels each side lost due to kernel size
        let expected = gray_image!(
            00, 00, 00, 00, 00;
            00, 00, 00, 00, 00;
            00, 00, 11, 00, 00;
            00, 00, 00, 00, 00;
            00, 00, 00, 00, 00);

        #[rustfmt::skip]
        let aff = Projection::from_matrix([
            1.0, 0.0, 1.0,
            0.0, 1.0, 1.0,
            0.0, 0.0, 1.0
        ]).unwrap();

        let translated_bicubic = warp(&image, &aff, Interpolation::Bicubic, Luma([0u8]));
        assert_pixels_eq!(translated_bicubic, expected);
    }

    #[bench]
    fn bench_affine_nearest(b: &mut Bencher) {
        let image = GrayImage::from_pixel(200, 200, Luma([15u8]));

        #[rustfmt::skip]
        let aff = Projection::from_matrix([
            1.0, 0.0, -1.0,
            0.0, 1.0, -1.0,
            0.0, 0.0,  1.0
        ]).unwrap();

        b.iter(|| {
            let transformed = warp(&image, &aff, Interpolation::Nearest, Luma([0u8]));
            black_box(transformed);
        });
    }

    #[bench]
    fn bench_affine_bilinear(b: &mut Bencher) {
        let image = GrayImage::from_pixel(200, 200, Luma([15u8]));

        #[rustfmt::skip]
        let aff = Projection::from_matrix([
            1.8,      -0.2, 5.0,
            0.2,       1.9, 6.0,
            0.0002, 0.0003, 1.0
        ]).unwrap();

        b.iter(|| {
            let transformed = warp(&image, &aff, Interpolation::Bilinear, Luma([0u8]));
            black_box(transformed);
        });
    }

    #[bench]
    fn bench_affine_bicubic(b: &mut test::Bencher) {
        let image = GrayImage::from_pixel(200, 200, Luma([15u8]));

        #[rustfmt::skip]
        let aff = Projection::from_matrix([
            1.8,      -0.2, 5.0,
            0.2,       1.9, 6.0,
            0.0002, 0.0003, 1.0
        ]).unwrap();

        b.iter(|| {
            let transformed = warp(&image, &aff, Interpolation::Bicubic, Luma([0u8]));
            black_box(transformed);
        });
    }

    #[test]
    fn test_from_control_points_translate() {
        let from = [(0f32, 0.0), (50.0, 50.0), (50.0, 0.0), (0.0, 50.0)];
        let to = [(10f32, 5.0), (60.0, 55.0), (60.0, 5.0), (10.0, 55.0)];

        let p = Projection::from_control_points(from, to);
        assert!(p.is_some());

        let out = p.unwrap() * (0f32, 0f32);

        assert_approx_eq!(out.0, 10.0, 1e-10);
        assert_approx_eq!(out.1, 5.0, 1e-10);
    }

    #[test]
    fn test_from_control_points() {
        let from = [(0f32, 0.0), (50.0, 50.0), (50.0, 0.0), (0.0, 50.0)];
        let to = [(16f32, 20.0), (50.0, 50.0), (50.0, 0.0), (0.0, 50.0)];

        let p = Projection::from_control_points(from, to);
        assert!(p.is_some());

        let out = p.unwrap() * (0f32, 0f32);

        assert_approx_eq!(out.0, 16.0, 1e-10);
        assert_approx_eq!(out.1, 20.0, 1e-10);
    }

    #[test]
    fn test_from_control_points_2() {
        let from = [
            (67.24537, 427.96024),
            (65.51512, 67.96736),
            (569.6426, 62.33165),
            (584.4605, 425.33667),
        ];
        let to = [(0.0, 0.0), (640.0, 0.0), (640.0, 480.0), (0.0, 480.0)];

        let p = Projection::from_control_points(from, to);
        assert!(p.is_some());
    }

    #[test]
    /// Test case from https://github.com/image-rs/imageproc/issues/412
    fn test_from_control_points_nofreeze() {
        let from = [
            (0.0, 0.0),
            (250.0, 17.481735),
            (7.257017, 82.94814),
            (250.0, 104.18543),
        ];
        let to = [(0.0, 0.0), (249.0, 0.0), (0.0, 105.0), (249.0, 105.0)];

        Projection::from_control_points(from, to);
    }

    #[test]
    fn test_from_control_points_known_transform() {
        let t = Projection::translate(10f32, 10f32);
        let p = t * Projection::rotate(90f32.to_radians()) * t.invert();

        let from = [(0f32, 0.0), (50.0, 50.0), (50.0, 0.0), (0.0, 50.0)];
        let to = [p * from[0], p * from[1], p * from[2], p * from[3]];

        let p_est = Projection::from_control_points(from, to);
        assert!(p_est.is_some());
        let p_est = p_est.unwrap();

        for i in 0..50 {
            for j in 0..50 {
                let pt = (i as f32, j as f32);
                assert_approx_eq!((p * pt).0, (p_est * pt).0, 1e-3);
                assert_approx_eq!((p * pt).1, (p_est * pt).1, 1e-3);
            }
        }
    }

    #[test]
    fn test_from_control_points_colinear() {
        let from = [(0f32, 0.0), (50.0, 50.0), (50.0, 0.0), (0.0, 50.0)];
        let to = [(0f32, 5.0), (0.0, 55.0), (0.0, 5.0), (10.0, 55.0)];

        let p = Projection::from_control_points(from, to);
        // Should fail if 3 points are colinear
        assert!(p.is_none());
    }

    #[test]
    fn test_from_control_points_translation() {
        let p = Projection::translate(10f32, 15f32);

        let from = [(0f32, 0.0), (50.0, 50.0), (50.0, 0.0), (0.0, 50.0)];
        let to = [(10f32, 15.0), (60.0, 65.0), (60.0, 15.0), (10.0, 65.0)];

        let p_est = Projection::from_control_points(from, to).unwrap();

        for i in 0..50 {
            for j in 0..50 {
                let pt = (i as f32, j as f32);
                assert_approx_eq!((p * pt).0, (p_est * pt).0, 1e-3);
                assert_approx_eq!((p * pt).1, (p_est * pt).1, 1e-3);
            }
        }
    }

    #[test]
    fn test_from_control_points_underdetermined() {
        let from = [
            (307.12073f32, 3.2),
            (330.89783, 3.2),
            (21.333334, 248.17337),
            (21.333334, 230.34056),
        ];
        let to = [(0.0f32, 0.0), (3.0, 0.0), (3.0, 3.0), (0.0, 3.0)];

        let p = Projection::from_control_points(from, to);
        p.unwrap();
    }

    #[bench]
    fn bench_from_control_points(b: &mut Bencher) {
        let from = [(0f32, 0.0), (50.0, 50.0), (50.0, 0.0), (0.0, 50.0)];
        let to = [(10f32, 5.0), (60.0, 55.0), (60.0, 5.0), (10.0, 55.0)];

        b.iter(|| {
            let proj = Projection::from_control_points(from, to);
            black_box(proj);
        });
    }
}

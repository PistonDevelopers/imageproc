//! TODO: Doc
#![allow(missing_docs)]
use image::GenericImageView;
use image::{ImageBuffer, Pixel};
use itertools::{Itertools, Product};
use std::ops::{Deref, Range};

pub struct Window<I, P> {
    pub image: I,
    pub xoffset: i32,
    pub yoffset: i32,
    pub xstride: u32,
    pub ystride: u32,
    pub pad_option: PadOption<P>,
}

type DerefPixel<I> = <<I as Deref>::Target as GenericImageView>::Pixel;

impl<I> Window<I, DerefPixel<I>>
where
    I: Deref,
    I::Target: GenericImageView + Sized,
{
    pub fn dimensions(&self) -> (u32, u32) {
        (self.xstride, self.ystride)
    }

    pub fn get_pixel(&self, x: u32, y: u32) -> DerefPixel<I> {
        use PadOption::*;
        let (width, height) = self.image.dimensions();
        let (width, height) = (width as i32, height as i32);
        let (x, y) = (x as i32 + self.xoffset, y as i32 + self.yoffset);

        // TODO: check transformed `x`, `y` in bound
        match self.pad_option {
            Symmetric => {
                let x = clamp(x, -x, width * 2 - 2 - x) as u32;
                let y = clamp(y, -y, height * 2 - 2 - y) as u32;
                self.image.get_pixel(x, y)
            }
            Replicate => {
                let x = clamp(x, 0, width - 1) as u32;
                let y = clamp(y, 0, height - 1) as u32;
                self.image.get_pixel(x, y)
            }
            Circular => {
                let x = ((x + width) % width) as u32;
                let y = ((y + height) % height) as u32;
                self.image.get_pixel(x, y)
            }
            Constant(p) => {
                if x < 0 || x >= width || y < 0 || y >= height {
                    p
                } else {
                    self.image.get_pixel(x as u32, y as u32)
                }
            }
        }
    }
}

fn clamp<T: std::cmp::Ord>(v: T, lo: T, hi: T) -> T {
    if v < lo {
        lo
    } else if hi < v {
        hi
    } else {
        v
    }
}

#[derive(Debug, Copy, Clone)]
pub enum PadOption<P> {
    Symmetric,
    Replicate,
    Circular,
    Constant(P),
}

#[derive(Debug, Copy, Clone)]
pub enum SizeOption {
    Same,
    Full,
    Valid,
}

pub struct WindowIter<I, P> {
    image: I,
    width: u32,
    height: u32,
    pad_option: PadOption<P>,
    offset_iter: Product<Range<i32>, Range<i32>>,
}

impl<I: Copy, P: Pixel> Iterator for WindowIter<I, P> {
    type Item = Window<I, P>;
    fn next(&mut self) -> Option<Window<I, P>> {
        if let Some((xoffset, yoffset)) = self.offset_iter.next() {
            Some(Window {
                image: self.image,
                xoffset,
                yoffset,
                xstride: self.width,
                ystride: self.height,
                pad_option: self.pad_option,
            })
        } else {
            None
        }
    }
}

pub trait ImageWindow<P: Pixel> {
    fn windows(
        &self,
        width: u32,
        height: u32,
        size_option: SizeOption,
        pad_option: PadOption<P>,
    ) -> WindowIter<&Self, P>;
}

impl<P, C> ImageWindow<P> for ImageBuffer<P, C>
where
    P: Pixel + 'static,
    C: Deref<Target = [P::Subpixel]> + Deref,
    P::Subpixel: 'static,
{
    fn windows(
        &self,
        width: u32,
        height: u32,
        size_option: SizeOption,
        pad_option: PadOption<P>,
    ) -> WindowIter<&Self, P> {
        use SizeOption::*;
        let (image_width, image_height) = self.dimensions();
        let (image_width, image_height) = (image_width as i32, image_height as i32);
        let (width, height) = (width as i32, height as i32);
        let offset_iter = match size_option {
            Valid => {
                let xrange = 0..(image_width - width + 1);
                let yrange = 0..(image_height - height + 1);
                Itertools::cartesian_product(yrange, xrange)
            }
            Full => {
                let xrange = (-width + 1)..image_width;
                let yrange = (-height + 1)..image_height;
                Itertools::cartesian_product(yrange, xrange)
            }
            Same => {
                let xrange = (-(width - 1) / 2)..(image_width - (width - 1) / 2);
                let yrange = (-(height - 1) / 2)..(image_height - (height - 1) / 2);
                Itertools::cartesian_product(yrange, xrange)
            }
        };
        WindowIter {
            image: self,
            width: width as u32,
            height: height as u32,
            pad_option,
            offset_iter,
        }
    }
}

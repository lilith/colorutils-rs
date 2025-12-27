/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::utils::fmla;
use crate::{EuclideanDistance, Rgb, TaxicabDistance, TransferFunction};
use num_traits::Pow;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// XYB is a color space that was designed for use with the JPEG XL Image Coding System.
///
/// It is an LMS-based color model inspired by the human visual system, facilitating perceptually uniform quantization.
/// It uses a gamma of 3 for computationally efficient decoding.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialOrd, PartialEq)]
pub struct Xyb {
    pub x: f32,
    pub y: f32,
    pub b: f32,
}

impl Xyb {
    #[inline]
    pub fn new(x: f32, y: f32, b: f32) -> Xyb {
        Xyb { x, y, b }
    }

    #[inline]
    /// Converts [Rgb] to [Xyb] using provided [TransferFunction]
    pub fn from_rgb(rgb: Rgb<u8>, transfer_function: TransferFunction) -> Xyb {
        let linear_rgb = rgb.to_linear(transfer_function);
        Self::from_linear_rgb(linear_rgb)
    }

    #[inline]
    /// Converts linear [Rgb] to [Xyb]
    pub fn from_linear_rgb(rgb: Rgb<f32>) -> Xyb {
        const BIAS_CBRT: f32 = 0.155954200549248620f32;
        const BIAS: f32 = 0.00379307325527544933;
        let lgamma = fmla(
            0.3f32,
            rgb.r,
            fmla(0.622f32, rgb.g, fmla(0.078f32, rgb.b, BIAS)),
        )
        .cbrt()
            - BIAS_CBRT;
        let mgamma = fmla(
            0.23f32,
            rgb.r,
            fmla(0.692f32, rgb.g, fmla(0.078f32, rgb.b, BIAS)),
        )
        .cbrt()
            - BIAS_CBRT;
        let sgamma = fmla(
            0.24342268924547819f32,
            rgb.r,
            fmla(
                0.20476744424496821f32,
                rgb.g,
                fmla(0.55180986650955360f32, rgb.b, BIAS),
            ),
        )
        .cbrt()
            - BIAS_CBRT;
        let x = (lgamma - mgamma) * 0.5f32;
        let y = (lgamma + mgamma) * 0.5f32;
        let b = sgamma - mgamma;
        Xyb::new(x, y, b)
    }

    #[inline]
    /// Converts [Xyb] to linear [Rgb]
    pub fn to_linear_rgb(&self) -> Rgb<f32> {
        const BIAS_CBRT: f32 = 0.155954200549248620f32;
        const BIAS: f32 = 0.00379307325527544933;
        let x_lms = (self.x + self.y) + BIAS_CBRT;
        let y_lms = (-self.x + self.y) + BIAS_CBRT;
        let b_lms = (-self.x + self.y + self.b) + BIAS_CBRT;
        let x_c_lms = (x_lms * x_lms * x_lms) - BIAS;
        let y_c_lms = (y_lms * y_lms * y_lms) - BIAS;
        let b_c_lms = (b_lms * b_lms * b_lms) - BIAS;
        let r = fmla(
            11.031566901960783,
            x_c_lms,
            fmla(-9.866943921568629, y_c_lms, -0.16462299647058826 * b_c_lms),
        );
        let g = fmla(
            -3.254147380392157,
            x_c_lms,
            fmla(4.418770392156863, y_c_lms, -0.16462299647058826 * b_c_lms),
        );
        let b = fmla(
            -3.6588512862745097,
            x_c_lms,
            fmla(2.7129230470588235, y_c_lms, 1.9459282392156863 * b_c_lms),
        );
        Rgb::new(r, g, b)
    }

    #[inline]
    /// Converts [Xyb] to [Rgb] using provided [TransferFunction]
    pub fn to_rgb(&self, transfer_function: TransferFunction) -> Rgb<u8> {
        let linear_rgb = self.to_linear_rgb();
        linear_rgb.gamma(transfer_function).to_u8()
    }
}

impl Add<f32> for Xyb {
    type Output = Xyb;

    #[inline]
    fn add(self, rhs: f32) -> Self::Output {
        Xyb::new(self.x + rhs, self.y + rhs, self.b + rhs)
    }
}

impl Add<Xyb> for Xyb {
    type Output = Xyb;

    #[inline]
    fn add(self, rhs: Xyb) -> Self::Output {
        Xyb::new(self.x + rhs.x, self.y + rhs.y, self.b + rhs.b)
    }
}

impl Sub<f32> for Xyb {
    type Output = Xyb;

    #[inline]
    fn sub(self, rhs: f32) -> Self::Output {
        Xyb::new(self.x - rhs, self.y - rhs, self.b - rhs)
    }
}

impl Sub<Xyb> for Xyb {
    type Output = Xyb;

    #[inline]
    fn sub(self, rhs: Xyb) -> Self::Output {
        Xyb::new(self.x - rhs.x, self.y - rhs.y, self.b - rhs.b)
    }
}

impl Mul<f32> for Xyb {
    type Output = Xyb;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Xyb::new(self.x * rhs, self.y * rhs, self.b * rhs)
    }
}

impl Mul<Xyb> for Xyb {
    type Output = Xyb;

    #[inline]
    fn mul(self, rhs: Xyb) -> Self::Output {
        Xyb::new(self.x * rhs.x, self.y * rhs.y, self.b * rhs.b)
    }
}

impl Div<f32> for Xyb {
    type Output = Xyb;

    #[inline]
    fn div(self, rhs: f32) -> Self::Output {
        Xyb::new(self.x / rhs, self.y / rhs, self.b / rhs)
    }
}

impl Div<Xyb> for Xyb {
    type Output = Xyb;

    #[inline]
    fn div(self, rhs: Xyb) -> Self::Output {
        Xyb::new(self.x / rhs.x, self.y / rhs.y, self.b / rhs.b)
    }
}

impl Neg for Xyb {
    type Output = Xyb;

    #[inline]
    fn neg(self) -> Self::Output {
        Xyb::new(-self.x, -self.y, -self.b)
    }
}

impl Pow<f32> for Xyb {
    type Output = Xyb;

    #[inline]
    fn pow(self, rhs: f32) -> Self::Output {
        Xyb::new(self.x.powf(rhs), self.y.powf(rhs), self.b.powf(rhs))
    }
}

impl Pow<Xyb> for Xyb {
    type Output = Xyb;

    #[inline]
    fn pow(self, rhs: Xyb) -> Self::Output {
        Xyb::new(self.x.powf(rhs.x), self.y.powf(rhs.y), self.b.powf(rhs.b))
    }
}

impl MulAssign<f32> for Xyb {
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        self.x *= rhs;
        self.y *= rhs;
        self.b *= rhs;
    }
}

impl MulAssign<Xyb> for Xyb {
    #[inline]
    fn mul_assign(&mut self, rhs: Xyb) {
        self.x *= rhs.x;
        self.y *= rhs.y;
        self.b *= rhs.b;
    }
}

impl AddAssign<f32> for Xyb {
    #[inline]
    fn add_assign(&mut self, rhs: f32) {
        self.x += rhs;
        self.y += rhs;
        self.b += rhs;
    }
}

impl AddAssign<Xyb> for Xyb {
    #[inline]
    fn add_assign(&mut self, rhs: Xyb) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.b += rhs.b;
    }
}

impl SubAssign<f32> for Xyb {
    #[inline]
    fn sub_assign(&mut self, rhs: f32) {
        self.x -= rhs;
        self.y -= rhs;
        self.b -= rhs;
    }
}

impl SubAssign<Xyb> for Xyb {
    #[inline]
    fn sub_assign(&mut self, rhs: Xyb) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.b -= rhs.b;
    }
}

impl DivAssign<f32> for Xyb {
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        self.x /= rhs;
        self.y /= rhs;
        self.b /= rhs;
    }
}

impl DivAssign<Xyb> for Xyb {
    #[inline]
    fn div_assign(&mut self, rhs: Xyb) {
        self.x /= rhs.x;
        self.y /= rhs.y;
        self.b /= rhs.b;
    }
}

impl Xyb {
    #[inline]
    pub fn sqrt(&self) -> Xyb {
        Xyb::new(
            if self.x < 0. { 0. } else { self.x.sqrt() },
            if self.y < 0. { 0. } else { self.y.sqrt() },
            if self.b < 0. { 0. } else { self.b.sqrt() },
        )
    }

    #[inline]
    pub fn cbrt(&self) -> Xyb {
        Xyb::new(self.x.cbrt(), self.y.cbrt(), self.b.cbrt())
    }
}

impl EuclideanDistance for Xyb {
    fn euclidean_distance(&self, other: Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let db = self.b - other.b;
        (dx * dx + dy * dy + db * db).sqrt()
    }
}

impl TaxicabDistance for Xyb {
    fn taxicab_distance(&self, other: Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let db = self.b - other.b;
        dx.abs() + dy.abs() + db.abs()
    }
}

#[cfg(test)]
mod tests {
    use crate::{Rgb, TransferFunction, Xyb};

    #[test]
    fn test_xyb_black_should_be_zero() {
        // Black (0,0,0) should produce XYB values of (0,0,0)
        let black = Xyb::from_rgb(Rgb::<u8>::new(0, 0, 0), TransferFunction::Srgb);

        assert!(
            black.x.abs() < 0.001,
            "Black X should be ~0, got {}",
            black.x
        );
        assert!(
            black.y.abs() < 0.001,
            "Black Y should be ~0, got {}",
            black.y
        );
        assert!(
            black.b.abs() < 0.001,
            "Black B should be ~0, got {}",
            black.b
        );
    }

    #[test]
    fn test_xyb_colors_should_be_distinct() {
        // Different colors should produce different XYB values
        let black = Xyb::from_rgb(Rgb::<u8>::new(0, 0, 0), TransferFunction::Srgb);
        let green = Xyb::from_rgb(Rgb::<u8>::new(0, 255, 0), TransferFunction::Srgb);
        let blue = Xyb::from_rgb(Rgb::<u8>::new(0, 0, 255), TransferFunction::Srgb);

        // These should all be different!
        assert!(
            (black.y - green.y).abs() > 0.1,
            "Black and Green Y should differ significantly. Black Y={}, Green Y={}",
            black.y,
            green.y
        );
        assert!(
            (black.y - blue.y).abs() > 0.1,
            "Black and Blue Y should differ significantly. Black Y={}, Blue Y={}",
            black.y,
            blue.y
        );
        assert!(
            (green.b - blue.b).abs() > 0.1,
            "Green and Blue B channel should differ. Green B={}, Blue B={}",
            green.b,
            blue.b
        );
    }

    #[test]
    fn test_xyb_round_trip_primaries() {
        // Round-trip through XYB should preserve colors (within rounding error)
        let test_colors = [
            (255u8, 0u8, 0u8, "Red"),
            (0, 255, 0, "Green"),
            (0, 0, 255, "Blue"),
            (255, 255, 255, "White"),
            (0, 0, 0, "Black"),
            (128, 128, 128, "Gray"),
        ];

        for (r, g, b, name) in test_colors {
            let rgb = Rgb::<u8>::new(r, g, b);
            let xyb = Xyb::from_rgb(rgb, TransferFunction::Srgb);
            let rgb2 = xyb.to_rgb(TransferFunction::Srgb);

            let max_error = (r as i32 - rgb2.r as i32)
                .abs()
                .max((g as i32 - rgb2.g as i32).abs())
                .max((b as i32 - rgb2.b as i32).abs());

            assert!(
                max_error <= 1,
                "{} round-trip failed: [{},{},{}] -> XYB({:.4},{:.4},{:.4}) -> [{},{},{}], error={}",
                name,
                r,
                g,
                b,
                xyb.x,
                xyb.y,
                xyb.b,
                rgb2.r,
                rgb2.g,
                rgb2.b,
                max_error
            );
        }
    }

    #[test]
    fn test_xyb_neutral_colors_have_zero_opponents() {
        // Neutral colors (grays) should have X ≈ 0 (no red-green) and B ≈ 0 (no blue)
        for v in [0u8, 64, 128, 192, 255] {
            let gray = Xyb::from_rgb(Rgb::<u8>::new(v, v, v), TransferFunction::Srgb);
            assert!(
                gray.x.abs() < 0.001,
                "Gray {} X should be ~0, got {}",
                v,
                gray.x
            );
            assert!(
                gray.b.abs() < 0.001,
                "Gray {} B should be ~0, got {}",
                v,
                gray.b
            );
        }
    }

    #[test]
    fn test_xyb_red_green_opponent() {
        // Red should have positive X (L > M), Green should have negative X (M > L)
        let red = Xyb::from_rgb(Rgb::<u8>::new(255, 0, 0), TransferFunction::Srgb);
        let green = Xyb::from_rgb(Rgb::<u8>::new(0, 255, 0), TransferFunction::Srgb);

        assert!(red.x > 0.0, "Red X should be positive, got {}", red.x);
        assert!(green.x < 0.0, "Green X should be negative, got {}", green.x);
    }

    #[test]
    fn test_xyb_blue_channel() {
        // Pure blue should have high positive B channel (S - M)
        let blue = Xyb::from_rgb(Rgb::<u8>::new(0, 0, 255), TransferFunction::Srgb);

        assert!(blue.b > 0.3, "Blue B should be > 0.3, got {}", blue.b);
    }

    #[test]
    fn test_xyb_white_known_values() {
        // White should produce known XYB values
        // Expected: X ≈ 0, Y ≈ 0.845, B ≈ 0
        let white = Xyb::from_rgb(Rgb::<u8>::new(255, 255, 255), TransferFunction::Srgb);

        assert!(
            white.x.abs() < 0.001,
            "White X should be ~0, got {}",
            white.x
        );
        assert!(
            (white.y - 0.845).abs() < 0.01,
            "White Y should be ~0.845, got {}",
            white.y
        );
        assert!(
            white.b.abs() < 0.001,
            "White B should be ~0, got {}",
            white.b
        );
    }
}

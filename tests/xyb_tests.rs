//! XYB color space tests
//!
//! These tests verify the XYB implementation matches the JPEG XL specification.

use colorutils_rs::{Rgb, TransferFunction, Xyb};

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

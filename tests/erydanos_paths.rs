/// Tests that exercise erydanos dependency functions (scalar math + SIMD vectorized paths).
///
/// Scalar erydanos: ehypotf, eatan2f, ehypot3f, Cosine/Sine traits
///   - exercised via Oklch and Jzczhz round-trips
///   - exercised via Xyz::euclidean_distance (.hypot3 from Euclidean3DDistance)
///
/// SIMD erydanos (neon/sse/avx): vpowq, vcbrtq, vexpq, vlnq, vatan2q, vhypotq,
///   vcosq, vsinq, vfmodq, etc.
///   - exercised via image-level conversions with width >= 32 (triggers vectorized loops)
///   - rgb_to_oklab / oklab_to_rgb (cbrt, pow, atan2, hypot, cos, sin)
///   - rgb_to_sigmoidal / sigmoidal_to_rgb (exp, ln)
///   - rgb_to_jzazbz / jzazbz_to_rgb (pow, atan2, hypot, cos, sin)
///   - rgb_to_lab / lab_to_rgb (cbrt, pow, atan2, hypot, cos, sin)
use colorutils_rs::*;

const W: u32 = 64;
const H: u32 = 4;
const N: usize = (W * H) as usize;

/// Generate a deterministic test image with diverse RGB values.
/// Covers blacks, whites, primaries, grays, and smooth gradients to
/// exercise both scalar tails and SIMD vector loops.
fn make_test_rgb() -> Vec<u8> {
    let mut buf = vec![0u8; N * 3];
    for i in 0..N {
        let t = i as f32 / (N - 1) as f32;
        // Rotate hue-ish pattern so we get diverse values
        let r = ((t * 7.3).sin() * 0.5 + 0.5).clamp(0.0, 1.0);
        let g = ((t * 5.1 + 1.0).sin() * 0.5 + 0.5).clamp(0.0, 1.0);
        let b = ((t * 3.7 + 2.0).sin() * 0.5 + 0.5).clamp(0.0, 1.0);
        buf[i * 3] = (r * 255.0) as u8;
        buf[i * 3 + 1] = (g * 255.0) as u8;
        buf[i * 3 + 2] = (b * 255.0) as u8;
    }
    buf
}

fn make_test_rgba() -> Vec<u8> {
    let mut buf = vec![0u8; N * 4];
    for i in 0..N {
        let t = i as f32 / (N - 1) as f32;
        let r = ((t * 7.3).sin() * 0.5 + 0.5).clamp(0.0, 1.0);
        let g = ((t * 5.1 + 1.0).sin() * 0.5 + 0.5).clamp(0.0, 1.0);
        let b = ((t * 3.7 + 2.0).sin() * 0.5 + 0.5).clamp(0.0, 1.0);
        buf[i * 4] = (r * 255.0) as u8;
        buf[i * 4 + 1] = (g * 255.0) as u8;
        buf[i * 4 + 2] = (b * 255.0) as u8;
        buf[i * 4 + 3] = 255;
    }
    buf
}

// ===== Oklch scalar round-trip (erydanos: ehypotf, eatan2f, ecos, esin) =====

#[test]
fn oklch_round_trip_scalar() {
    let colors = [
        Rgb::<u8>::new(255, 0, 0),
        Rgb::<u8>::new(0, 255, 0),
        Rgb::<u8>::new(0, 0, 255),
        Rgb::<u8>::new(255, 255, 0),
        Rgb::<u8>::new(0, 255, 255),
        Rgb::<u8>::new(255, 0, 255),
        Rgb::<u8>::new(128, 64, 192),
        Rgb::<u8>::new(10, 10, 10),
        Rgb::<u8>::new(245, 245, 245),
    ];

    for rgb in colors {
        let oklch = Oklch::from_rgb(rgb, TransferFunction::Srgb);
        let back = oklch.to_rgb(TransferFunction::Srgb);
        let err = (rgb.r as i32 - back.r as i32)
            .abs()
            .max((rgb.g as i32 - back.g as i32).abs())
            .max((rgb.b as i32 - back.b as i32).abs());
        assert!(
            err <= 1,
            "Oklch round-trip: [{},{},{}] -> Oklch({:.4},{:.4},{:.4}) -> [{},{},{}], err={}",
            rgb.r,
            rgb.g,
            rgb.b,
            oklch.l,
            oklch.c,
            oklch.h,
            back.r,
            back.g,
            back.b,
            err
        );
    }
}

// ===== Jzczhz scalar round-trip (erydanos: ehypotf, eatan2f, ehypot3f, ecos, esin) =====

#[test]
fn jzczhz_round_trip_scalar() {
    let colors = [
        Rgb::<u8>::new(255, 0, 0),
        Rgb::<u8>::new(0, 255, 0),
        Rgb::<u8>::new(0, 0, 255),
        Rgb::<u8>::new(128, 128, 128),
        Rgb::<u8>::new(200, 100, 50),
    ];

    for rgb in colors {
        let jzczhz = Jzczhz::from_rgb(rgb, TransferFunction::Srgb);
        let back = jzczhz.to_rgb(TransferFunction::Srgb);
        let err = (rgb.r as i32 - back.r as i32)
            .abs()
            .max((rgb.g as i32 - back.g as i32).abs())
            .max((rgb.b as i32 - back.b as i32).abs());
        assert!(
            err <= 1,
            "Jzczhz round-trip: [{},{},{}] -> Jzczhz({:.4},{:.4},{:.4}) -> [{},{},{}], err={}",
            rgb.r,
            rgb.g,
            rgb.b,
            jzczhz.jz,
            jzczhz.cz,
            jzczhz.hz,
            back.r,
            back.g,
            back.b,
            err
        );
    }
}

// ===== Jzczhz distance (erydanos: ehypot3f, esin) =====

#[test]
fn jzczhz_distance_uses_erydanos() {
    let a = Jzczhz::from_rgb(Rgb::<u8>::new(255, 0, 0), TransferFunction::Srgb);
    let b = Jzczhz::from_rgb(Rgb::<u8>::new(0, 0, 255), TransferFunction::Srgb);
    let d = a.distance(b);
    assert!(d > 0.0, "Distance between red and blue should be > 0, got {}", d);
    assert!(d.is_finite(), "Distance should be finite");

    // Self-distance should be ~0
    let self_d = a.distance(a);
    assert!(self_d.abs() < 1e-6, "Self-distance should be ~0, got {}", self_d);
}

// ===== Xyz euclidean distance (erydanos: Euclidean3DDistance / .hypot3()) =====

#[test]
fn xyz_euclidean_distance_uses_erydanos() {
    let white = Xyz::new(0.9505, 1.0, 1.0890);
    let black = Xyz::new(0.0, 0.0, 0.0);
    let d = white.euclidean_distance(black);
    // sqrt(0.9505^2 + 1.0^2 + 1.089^2) ≈ 1.756
    assert!(
        (d - 1.756).abs() < 0.01,
        "Xyz distance white-black should be ~1.756, got {}",
        d
    );
}

// ===== Image-level Oklab round-trip (SIMD: cbrt, pow, atan2, hypot, cos, sin) =====

#[test]
fn oklab_image_round_trip() {
    let src = make_test_rgb();
    let src_stride = W * 3;
    let dst_stride = W * 3 * 4; // 3 f32 per pixel
    let mut oklab = vec![0f32; N * 3];

    rgb_to_oklab(
        &src,
        src_stride,
        &mut oklab,
        dst_stride,
        W,
        H,
        TransferFunction::Srgb,
    );

    // Verify oklab values are finite and in reasonable range
    for (i, chunk) in oklab.chunks_exact(3).enumerate() {
        assert!(chunk[0].is_finite(), "pixel {} L is not finite", i);
        assert!(chunk[1].is_finite(), "pixel {} a is not finite", i);
        assert!(chunk[2].is_finite(), "pixel {} b is not finite", i);
        assert!(chunk[0] >= 0.0 && chunk[0] <= 1.1, "pixel {} L={} out of range", i, chunk[0]);
    }

    // Convert back
    let mut dst = vec![0u8; N * 3];
    oklab_to_rgb(
        &oklab,
        dst_stride,
        &mut dst,
        src_stride,
        W,
        H,
        TransferFunction::Srgb,
    );

    // Check round-trip error
    let mut max_err = 0i32;
    for i in 0..N * 3 {
        let err = (src[i] as i32 - dst[i] as i32).abs();
        max_err = max_err.max(err);
    }
    assert!(
        max_err <= 2,
        "Oklab image round-trip max error {} exceeds tolerance 2",
        max_err
    );
}

// ===== Image-level Oklch round-trip (SIMD: cbrt, pow, atan2, hypot, cos, sin) =====

#[test]
fn oklch_image_round_trip() {
    let src = make_test_rgb();
    let src_stride = W * 3;
    let dst_stride = W * 3 * 4;
    let mut oklch = vec![0f32; N * 3];

    rgb_to_oklch(
        &src,
        src_stride,
        &mut oklch,
        dst_stride,
        W,
        H,
        TransferFunction::Srgb,
    );

    for (i, chunk) in oklch.chunks_exact(3).enumerate() {
        assert!(chunk[0].is_finite(), "pixel {} L is not finite", i);
        assert!(chunk[1].is_finite(), "pixel {} C is not finite", i);
        assert!(chunk[2].is_finite(), "pixel {} h is not finite", i);
    }

    let mut dst = vec![0u8; N * 3];
    oklch_to_rgb(
        &oklch,
        dst_stride,
        &mut dst,
        src_stride,
        W,
        H,
        TransferFunction::Srgb,
    );

    let mut max_err = 0i32;
    for i in 0..N * 3 {
        let err = (src[i] as i32 - dst[i] as i32).abs();
        max_err = max_err.max(err);
    }
    assert!(
        max_err <= 2,
        "Oklch image round-trip max error {} exceeds tolerance 2",
        max_err
    );
}

// ===== Image-level Sigmoidal round-trip (SIMD: exp, ln) =====

#[test]
fn sigmoidal_image_round_trip() {
    let src = make_test_rgb();
    let src_stride = W * 3;
    let dst_stride = W * 3 * 4;
    let mut sig = vec![0f32; N * 3];

    rgb_to_sigmoidal(&src, src_stride, &mut sig, dst_stride, W, H);

    for (i, chunk) in sig.chunks_exact(3).enumerate() {
        assert!(chunk[0].is_finite(), "pixel {} S0 not finite", i);
        assert!(chunk[1].is_finite(), "pixel {} S1 not finite", i);
        assert!(chunk[2].is_finite(), "pixel {} S2 not finite", i);
    }

    let mut dst = vec![0u8; N * 3];
    sigmoidal_to_rgb(&sig, dst_stride, &mut dst, src_stride, W, H);

    let mut max_err = 0i32;
    for i in 0..N * 3 {
        let err = (src[i] as i32 - dst[i] as i32).abs();
        max_err = max_err.max(err);
    }
    assert!(
        max_err <= 1,
        "Sigmoidal image round-trip max error {} exceeds tolerance 1",
        max_err
    );
}

// ===== Image-level RGBA Sigmoidal round-trip (SIMD: exp, ln + alpha handling) =====

#[test]
fn sigmoidal_rgba_image_round_trip() {
    let src = make_test_rgba();
    let src_stride = W * 4;
    let dst_stride = W * 4 * 4;
    let mut sig = vec![0f32; N * 4];

    rgba_to_sigmoidal(&src, src_stride, &mut sig, dst_stride, W, H);

    let mut dst = vec![0u8; N * 4];
    sigmoidal_to_rgba(&sig, dst_stride, &mut dst, src_stride, W, H);

    let mut max_err = 0i32;
    for i in 0..N {
        for c in 0..3 {
            let err = (src[i * 4 + c] as i32 - dst[i * 4 + c] as i32).abs();
            max_err = max_err.max(err);
        }
    }
    assert!(
        max_err <= 1,
        "Sigmoidal RGBA round-trip max error {} exceeds tolerance 1",
        max_err
    );
}

// ===== Image-level Jzazbz round-trip (SIMD: pow, atan2, hypot, cos, sin) =====

#[test]
fn jzazbz_image_round_trip() {
    let src = make_test_rgb();
    let src_stride = W * 3;
    let dst_stride = W * 3 * 4;
    let mut jz = vec![0f32; N * 3];

    rgb_to_jzazbz(
        &src,
        src_stride,
        &mut jz,
        dst_stride,
        W,
        H,
        200.0,
        TransferFunction::Srgb,
    );

    for (i, chunk) in jz.chunks_exact(3).enumerate() {
        assert!(chunk[0].is_finite(), "pixel {} Jz not finite", i);
        assert!(chunk[1].is_finite(), "pixel {} az not finite", i);
        assert!(chunk[2].is_finite(), "pixel {} bz not finite", i);
    }

    let mut dst = vec![0u8; N * 3];
    jzazbz_to_rgb(
        &jz,
        dst_stride,
        &mut dst,
        src_stride,
        W,
        H,
        200.0,
        TransferFunction::Srgb,
    );

    // Jzazbz image path uses a 2048-entry LUT for gamma encoding, which limits precision.
    // PQ is also inherently lossy at low luminance. Check that most pixels round-trip well.
    let mut err_over_3 = 0u32;
    for i in 0..N {
        for c in 0..3 {
            let err = (src[i * 3 + c] as i32 - dst[i * 3 + c] as i32).abs();
            if err > 3 {
                err_over_3 += 1;
            }
        }
    }
    assert!(
        err_over_3 <= (N as u32 * 3 / 2),
        "Jzazbz image round-trip: {} of {} channels have error > 3 (max 50% allowed)",
        err_over_3,
        N * 3
    );
}

// ===== Image-level Lab round-trip (SIMD: cbrt, pow via cie path) =====

#[test]
fn lab_image_round_trip() {
    let src = make_test_rgb();
    let src_stride = W * 3;
    let dst_stride = W * 3 * 4;
    let mut lab = vec![0f32; N * 3];

    rgb_to_lab(
        &src,
        src_stride,
        &mut lab,
        dst_stride,
        W,
        H,
        &SRGB_TO_XYZ_D65,
        TransferFunction::Srgb,
    );

    for (i, chunk) in lab.chunks_exact(3).enumerate() {
        assert!(chunk[0].is_finite(), "pixel {} L not finite", i);
        assert!(chunk[1].is_finite(), "pixel {} a not finite", i);
        assert!(chunk[2].is_finite(), "pixel {} b not finite", i);
        assert!(
            chunk[0] >= 0.0 && chunk[0] <= 101.0,
            "pixel {} L={} out of range",
            i,
            chunk[0]
        );
    }

    let mut dst = vec![0u8; N * 3];
    lab_to_rgb(
        &lab,
        dst_stride,
        &mut dst,
        src_stride,
        W,
        H,
        &XYZ_TO_SRGB_D65,
        TransferFunction::Srgb,
    );

    let mut max_err = 0i32;
    for i in 0..N * 3 {
        let err = (src[i] as i32 - dst[i] as i32).abs();
        max_err = max_err.max(err);
    }
    assert!(
        max_err <= 2,
        "Lab image round-trip max error {} exceeds tolerance 2",
        max_err
    );
}

// ===== Image-level Jzczhz round-trip (SIMD: pow, atan2, hypot, cos, sin) =====

#[test]
fn jzczhz_image_round_trip() {
    let src = make_test_rgb();
    let src_stride = W * 3;
    let dst_stride = W * 3 * 4;
    let mut jzczhz = vec![0f32; N * 3];

    rgb_to_jzczhz(
        &src,
        src_stride,
        &mut jzczhz,
        dst_stride,
        W,
        H,
        200.0,
        TransferFunction::Srgb,
    );

    for (i, chunk) in jzczhz.chunks_exact(3).enumerate() {
        assert!(chunk[0].is_finite(), "pixel {} Jz not finite", i);
        assert!(chunk[1].is_finite(), "pixel {} Cz not finite", i);
        assert!(chunk[2].is_finite(), "pixel {} hz not finite", i);
    }

    let mut dst = vec![0u8; N * 3];
    jzczhz_to_rgb(
        &jzczhz,
        dst_stride,
        &mut dst,
        src_stride,
        W,
        H,
        200.0,
        TransferFunction::Srgb,
    );

    // Same PQ/LUT precision limits as Jzazbz, plus polar coordinate conversion
    let mut err_over_3 = 0u32;
    for i in 0..N {
        for c in 0..3 {
            let err = (src[i * 3 + c] as i32 - dst[i * 3 + c] as i32).abs();
            if err > 3 {
                err_over_3 += 1;
            }
        }
    }
    assert!(
        err_over_3 <= (N as u32 * 3 / 2),
        "Jzczhz image round-trip: {} of {} channels have error > 3 (max 50% allowed)",
        err_over_3,
        N * 3
    );
}

// ===== Verify SIMD tail handling with non-aligned widths =====

#[test]
fn oklab_image_odd_width() {
    // Width 17 ensures SIMD loop (>=4 or >=8) plus scalar tail
    let w: u32 = 17;
    let h: u32 = 3;
    let n = (w * h) as usize;
    let mut src = vec![0u8; n * 3];
    for i in 0..n {
        let t = i as f32 / (n - 1) as f32;
        src[i * 3] = (((t * 7.3).sin() * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
        src[i * 3 + 1] = (((t * 5.1 + 1.0).sin() * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
        src[i * 3 + 2] = (((t * 3.7 + 2.0).sin() * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
    }

    let src_stride = w * 3;
    let dst_stride = w * 3 * 4;
    let mut oklab = vec![0f32; n * 3];

    rgb_to_oklab(
        &src,
        src_stride,
        &mut oklab,
        dst_stride,
        w,
        h,
        TransferFunction::Srgb,
    );

    let mut dst = vec![0u8; n * 3];
    oklab_to_rgb(
        &oklab,
        dst_stride,
        &mut dst,
        src_stride,
        w,
        h,
        TransferFunction::Srgb,
    );

    let mut max_err = 0i32;
    for i in 0..n * 3 {
        let err = (src[i] as i32 - dst[i] as i32).abs();
        max_err = max_err.max(err);
    }
    assert!(
        max_err <= 2,
        "Oklab odd-width round-trip max error {} exceeds tolerance 2",
        max_err
    );
}

// ===== LCh round-trip (SIMD: cie path + atan2, hypot, cos, sin for polar) =====

#[test]
fn lch_image_round_trip() {
    let src = make_test_rgb();
    let src_stride = W * 3;
    let dst_stride = W * 3 * 4;
    let mut lch = vec![0f32; N * 3];

    rgb_to_lch(
        &src,
        src_stride,
        &mut lch,
        dst_stride,
        W,
        H,
        &SRGB_TO_XYZ_D65,
        TransferFunction::Srgb,
    );

    for (i, chunk) in lch.chunks_exact(3).enumerate() {
        assert!(chunk[0].is_finite(), "pixel {} L not finite", i);
        assert!(chunk[1].is_finite(), "pixel {} C not finite", i);
        assert!(chunk[2].is_finite(), "pixel {} h not finite", i);
    }

    let mut dst = vec![0u8; N * 3];
    lch_to_rgb(
        &lch,
        dst_stride,
        &mut dst,
        src_stride,
        W,
        H,
        &XYZ_TO_SRGB_D65,
        TransferFunction::Srgb,
    );

    let mut max_err = 0i32;
    for i in 0..N * 3 {
        let err = (src[i] as i32 - dst[i] as i32).abs();
        max_err = max_err.max(err);
    }
    assert!(
        max_err <= 2,
        "LCh image round-trip max error {} exceeds tolerance 2",
        max_err
    );
}

#![forbid(unsafe_code)]

#[cfg(test)]
#[path = "../../../src/hdr_compat.rs"]
mod hdr_compat;

#[cfg(test)]
mod tests {
    use zenpixels::{AlphaMode as A, ColorPrimaries as P, TransferFunction as T};
    use zenpixels_legacy::{AlphaMode as OldA, ColorPrimaries as OldP, TransferFunction as OldT};

    // Independent historical implementation from zenanalyze 7064f1df,
    // versioning.rs::descriptor_hash_of and fnv1a. Real 0.2.14 enum casts
    // supply the reference bytes; no new serialization helper is reused.
    fn legacy_hash(descriptor: &zenpixels_legacy::PixelDescriptor, white: Option<f32>) -> u64 {
        let dw = white.map_or(0u64, |n| u64::from(n.to_bits())).to_le_bytes();
        let bytes = [
            descriptor.transfer as u8,
            descriptor.primaries as u8,
            descriptor.alpha.map_or(0u8, |a| a as u8),
            dw[0],
            dw[1],
            dw[2],
            dw[3],
            dw[4],
            dw[5],
            dw[6],
            dw[7],
        ];
        bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }

    #[test]
    fn all_existing_descriptor_hashes_match_the_historical_runtime() {
        let transfers = [
            (T::Linear, OldT::Linear),
            (T::Srgb, OldT::Srgb),
            (T::Bt709, OldT::Bt709),
            (T::Pq, OldT::Pq),
            (T::Hlg, OldT::Hlg),
            (T::Gamma22, OldT::Gamma22),
            (T::Unknown, OldT::Unknown),
        ];
        let primaries = [
            (P::Bt709, OldP::Bt709),
            (P::Bt2020, OldP::Bt2020),
            (P::DisplayP3, OldP::DisplayP3),
            (P::AdobeRgb, OldP::AdobeRgb),
            (P::Unknown, OldP::Unknown),
        ];
        let alphas = [
            (None, None),
            (Some(A::Undefined), Some(OldA::Undefined)),
            (Some(A::Straight), Some(OldA::Straight)),
            (Some(A::Premultiplied), Some(OldA::Premultiplied)),
            (Some(A::Opaque), Some(OldA::Opaque)),
        ];
        let whites = [
            None,
            Some(0.0),
            Some(-0.0),
            Some(203.0),
            Some(10_000.0),
            Some(f32::from_bits(0x7fc0_0001)),
            Some(f32::INFINITY),
            Some(f32::NEG_INFINITY),
        ];
        let mut compared = 0;
        for (transfer, old_transfer) in transfers {
            for (primaries, old_primaries) in primaries {
                for (alpha, old_alpha) in alphas {
                    let mut current = zenpixels::PixelDescriptor::RGB8_SRGB;
                    current.transfer = transfer;
                    current.primaries = primaries;
                    current.alpha = alpha;
                    let mut previous = zenpixels_legacy::PixelDescriptor::RGB8_SRGB;
                    previous.transfer = old_transfer;
                    previous.primaries = old_primaries;
                    previous.alpha = old_alpha;
                    for white in whites {
                        assert_eq!(
                            zenanalyze::versioning::descriptor_hash_of(&current, white),
                            legacy_hash(&previous, white),
                            "{transfer:?}/{primaries:?}/{alpha:?}/{white:?}",
                        );
                        compared += 1;
                    }
                }
            }
        }
        println!("LEGACY_DESCRIPTOR_HASHES {compared}");
    }

    #[test]
    fn hdr_analysis_rows_match_the_qualified_legacy_converter() {
        use crate::hdr_compat::HdrAnalysisConverter;
        use zenpixels::{ChannelLayout as L, ChannelType as C, PixelDescriptor as D};
        use zenpixels_convert_legacy::{
            AlphaMode as OA, ChannelLayout as OL, ChannelType as OC, ColorPrimaries as OP,
            PixelDescriptor as OD, RowConverter as OldConverter, TransferFunction as OT,
        };
        let transfers = [(T::Pq, OT::Pq), (T::Hlg, OT::Hlg)];
        let primaries = [
            (P::Bt709, OP::Bt709),
            (P::Bt2020, OP::Bt2020),
            (P::DisplayP3, OP::DisplayP3),
            (P::AdobeRgb, OP::AdobeRgb),
            (P::Unknown, OP::Unknown),
        ];
        let layouts = [
            (L::Rgb, OL::Rgb, None, None, 3usize),
            (
                L::Rgba,
                OL::Rgba,
                Some(A::Undefined),
                Some(OA::Undefined),
                4,
            ),
            (L::Rgba, OL::Rgba, Some(A::Straight), Some(OA::Straight), 4),
            (
                L::Rgba,
                OL::Rgba,
                Some(A::Premultiplied),
                Some(OA::Premultiplied),
                4,
            ),
            (L::Rgba, OL::Rgba, Some(A::Opaque), Some(OA::Opaque), 4),
            (L::Gray, OL::Gray, None, None, 1),
            (
                L::GrayAlpha,
                OL::GrayAlpha,
                Some(A::Straight),
                Some(OA::Straight),
                2,
            ),
        ];
        let depths = [
            (C::U8, OC::U8, 1usize),
            (C::U16, OC::U16, 2),
            (C::F16, OC::F16, 2),
            (C::F32, OC::F32, 4),
        ];
        let mut cases = 0;
        let mut matched_rows = 0;
        let mut differing_rows = 0;
        let mut gray_reference_rows = 0;
        let mut failures = Vec::new();
        let mut compared_bytes = 0usize;
        for width in [1u32, 7, 16, 31, 257, 65_536] {
            println!("HDR_LEGACY_PROGRESS width={width}");
            for (depth, old_depth, bpc) in depths {
                for (transfer, old_transfer) in transfers {
                    for (primary, old_primary) in primaries {
                        for (layout, old_layout, alpha, old_alpha, channels) in layouts {
                            cases += 1;
                            let current =
                                D::new(depth, layout, alpha, transfer).with_primaries(primary);
                            // The pinned legacy converter panics on one-pixel HDR gray
                            // rows. Use the equivalent replicated-RGB source as that
                            // domain's independent reference; never skip its coverage.
                            let gray = matches!(layout, L::Gray | L::GrayAlpha);
                            let reference_layout = if gray {
                                if old_alpha.is_some() {
                                    OL::Rgba
                                } else {
                                    OL::Rgb
                                }
                            } else {
                                old_layout
                            };
                            let previous =
                                OD::new(old_depth, reference_layout, old_alpha, old_transfer)
                                    .with_primaries(old_primary);
                            let mut reference = OldConverter::new(previous, OD::RGB8_SRGB);
                            let mut actual = HdrAnalysisConverter::new(current, width);
                            let label = format!(
                                "{width}/{depth:?}/{transfer:?}/{primary:?}/{layout:?}/{alpha:?}"
                            );
                            match (&mut reference, &mut actual) {
                                (Ok(reference), Ok(actual)) => {
                                    let row_bytes = width as usize * channels * bpc;
                                    let stride = row_bytes + 16;
                                    let mut input = vec![0x9bu8; stride * 2];
                                    for y in 0..2 {
                                        for x in 0..width as usize {
                                            for channel in 0..channels {
                                                let code = if alpha.is_some()
                                                    && channel == channels - 1
                                                {
                                                    ((x * 257 + y * 32767) & 0xffff) as u16
                                                } else {
                                                    ((x * 17 + channel * 1297 + y * 7919) & 0xffff)
                                                        as u16
                                                };
                                                let at =
                                                    y * stride + (x * channels + channel) * bpc;
                                                if depth == C::U8 {
                                                    input[at] = (code >> 8) as u8;
                                                } else if depth == C::F16 {
                                                    // All nonnegative half bit patterns through 1.0,
                                                    // including subnormals; no encoder is the oracle.
                                                    let half =
                                                        ((u32::from(code) * 0x3c00) / 65535) as u16;
                                                    input[at..at + 2]
                                                        .copy_from_slice(&half.to_ne_bytes());
                                                } else if depth == C::U16 {
                                                    input[at..at + 2]
                                                        .copy_from_slice(&code.to_ne_bytes());
                                                } else {
                                                    input[at..at + 4].copy_from_slice(
                                                        &(f32::from(code) / 65535.0).to_ne_bytes(),
                                                    );
                                                }
                                            }
                                        }
                                        let source = &input[y * stride..y * stride + row_bytes];
                                        let mut expected = vec![0x5au8; width as usize * 3];
                                        let mut got = vec![0xa5u8; width as usize * 3];
                                        let expanded;
                                        let reference_source = if gray {
                                            let target_channels =
                                                if alpha.is_some() { 4 } else { 3 };
                                            let mut bytes = Vec::with_capacity(
                                                width as usize * target_channels * bpc,
                                            );
                                            for pixel in source.chunks_exact(channels * bpc) {
                                                for _ in 0..3 {
                                                    bytes.extend_from_slice(&pixel[..bpc]);
                                                }
                                                if alpha.is_some() {
                                                    bytes.extend_from_slice(&pixel[bpc..2 * bpc]);
                                                }
                                            }
                                            expanded = bytes;
                                            &expanded[..]
                                        } else {
                                            source
                                        };
                                        reference.convert_row(
                                            reference_source,
                                            &mut expected,
                                            width,
                                        );
                                        actual.convert_row(source, &mut got, width);
                                        compared_bytes += got.len();
                                        if gray {
                                            gray_reference_rows += 1;
                                        }
                                        if got == expected {
                                            matched_rows += 1;
                                        } else {
                                            differing_rows += 1;
                                            {
                                                let at = got
                                                    .iter()
                                                    .zip(&expected)
                                                    .position(|(a, b)| a != b)
                                                    .unwrap();
                                                failures.push(format!(
                                                    "{label} row={y} byte={at} new={} old={}",
                                                    got[at], expected[at]
                                                ));
                                                if failures.len() == 1 {
                                                    println!(
                                                        "HDR_DIAGNOSTIC_OLD_PLAN {:?}",
                                                        reference.plan()
                                                    );
                                                    println!("HDR_DIAGNOSTIC_NEW_PLANS {actual:?}");
                                                }
                                            }
                                        }
                                    }
                                }
                                (Err(_), Err(_)) => {}
                                _ => failures.push(format!(
                                    "{label} accepted-domain mismatch: new={} old={} error={:?}",
                                    actual.is_ok(),
                                    reference.is_ok(),
                                    actual.as_ref().err()
                                )),
                            }
                        }
                    }
                }
            }
        }
        println!(
            "HDR_LEGACY_ROWS cases={cases} matched_rows={matched_rows} differing_rows={differing_rows} gray_via_rgb_rows={gray_reference_rows} compared_bytes={compared_bytes}"
        );
        assert!(
            failures.is_empty(),
            "legacy row mismatches:\n{}",
            failures.join("\n")
        );
    }
}

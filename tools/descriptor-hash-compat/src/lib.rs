#![forbid(unsafe_code)]

#[cfg(test)]
mod tests {
    use zenpixels::{AlphaMode as A, ColorPrimaries as P, TransferFunction as T};
    use zenpixels_legacy::{AlphaMode as OldA, ColorPrimaries as OldP, TransferFunction as OldT};

    // Independent historical implementation from zenanalyze 7064f1df,
    // versioning.rs::descriptor_hash_of and fnv1a. Real 0.2.16 enum casts
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
}

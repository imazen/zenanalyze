//! Historical relative-light representation for analyzer model features.
//!
//! Retains zenpixels-convert 0.2.14's EOTF, matrix operation order and
//! quantizers. Rendering HDR policy remains with the caller; this operation
//! introduces no scene peak, exposure adjustment or tone curve.

use zenpixels::{ChannelLayout, ChannelType, ColorPrimaries, PixelDescriptor, TransferFunction};
use zenpixels_convert::{ConvertPlan, RowConverter, gamut};

pub(crate) struct HdrAnalysisConverter {
    first: RowConverter,
    colors: Option<StoredColors>,
    second: Option<RowConverter>,
    matrix: Option<gamut::GamutMatrix>,
    scratch: Vec<u8>,
}

impl HdrAnalysisConverter {
    pub(crate) fn new(from: PixelDescriptor, width: u32) -> Result<Self, String> {
        let linear = PixelDescriptor::RGBF32_LINEAR;
        let matrix = gamut::conversion_matrix(from.primaries, ColorPrimaries::Bt709);
        // Decode in the source primaries. Applying the matrix separately below
        // preserves the historical non-FMA sum order, still available through
        // gamut::apply_matrix_f32. New rendering plans use SIMD FMA here.
        let source_linear = if matrix.is_some() {
            linear.with_primaries(from.primaries)
        } else {
            linear
        };
        // Historical content features measure the stored color channels,
        // independently of alpha (the alpha tier reads the original source).
        // Extract them explicitly: modern rendering conversion unassociates
        // premultiplied colors, which would change the trained feature contract.
        let colors = StoredColors::new(from, width)?;
        let rgb = PixelDescriptor::new(
            from.channel_type(),
            ChannelLayout::Rgb,
            None,
            from.transfer(),
        )
        .with_primaries(from.primaries)
        .with_signal_range(from.signal_range);
        // 0.2.14's U8 wide-gamut plan inserts its matrix immediately after
        // widening, before the EOTF. Preserve that historical feature definition
        // explicitly; this encoded-domain operation is not a rendering transform.
        let encoded_matrix = from.channel_type() == ChannelType::U8 && matrix.is_some();
        let intermediate = if encoded_matrix {
            source_linear.with_transfer(from.transfer())
        } else {
            source_linear
        };
        let first = RowConverter::from_plan(
            ConvertPlan::new(rgb, intermediate)
                .map_err(|e| format!("HDR analysis linear stage failed: {e:?}"))?,
        );
        let second = if from.channel_type() == ChannelType::U16 {
            // The old U16 path uses the fused linear→sRGB8 quantizer.
            RowConverter::new(linear, PixelDescriptor::RGB8_SRGB)
                .map_err(|e| format!("HDR analysis U16 quantizer failed: {e:?}"))?
        } else {
            // Float/F16 paths keep a materialized F32 OETF then U8 quantization.
            let encoded = linear.with_transfer(TransferFunction::Srgb);
            let gamma = RowConverter::new(linear, encoded)
                .map_err(|e| format!("HDR analysis encoding stage failed: {e:?}"))?;
            let quantize = RowConverter::new(encoded, PixelDescriptor::RGB8_SRGB)
                .map_err(|e| format!("HDR analysis float quantizer failed: {e:?}"))?;
            gamma
                .compose_preserving(&quantize)
                .ok_or_else(|| "HDR analysis quantizer stages are incompatible".to_string())?
        };
        let second = if encoded_matrix {
            let decode = RowConverter::from_plan(
                ConvertPlan::new(linear.with_transfer(from.transfer()), linear)
                    .map_err(|e| format!("HDR analysis legacy U8 decode failed: {e:?}"))?,
            );
            decode
                .compose_preserving(&second)
                .ok_or_else(|| "HDR analysis legacy U8 stages are incompatible".to_string())?
        } else {
            second
        };
        if matrix.is_none()
            && let Some(composed) = first.compose_preserving(&second)
        {
            return Ok(Self {
                first: composed,
                colors,
                second: None,
                matrix: None,
                scratch: Vec::new(),
            });
        }
        let bytes = (width as usize)
            .checked_mul(12)
            .ok_or_else(|| "HDR analysis row size overflow".to_string())?;
        let mut scratch = Vec::new();
        scratch
            .try_reserve_exact(bytes)
            .map_err(|e| format!("HDR analysis row allocation failed: {e}"))?;
        scratch.resize(bytes, 0);
        Ok(Self {
            first,
            colors,
            second: Some(second),
            matrix,
            scratch,
        })
    }

    pub(crate) fn convert_row(&mut self, from: &[u8], to: &mut [u8], width: u32) {
        let from = if let Some(colors) = &mut self.colors {
            colors.extract(from)
        } else {
            from
        };
        if let Some(second) = &mut self.second {
            self.first.convert_row(from, &mut self.scratch, width);
            if let Some(matrix) = &self.matrix {
                for bytes in self.scratch.as_chunks_mut::<12>().0 {
                    let mut rgb = [
                        f32::from_ne_bytes(bytes[0..4].try_into().unwrap()),
                        f32::from_ne_bytes(bytes[4..8].try_into().unwrap()),
                        f32::from_ne_bytes(bytes[8..12].try_into().unwrap()),
                    ];
                    gamut::apply_matrix_f32(&mut rgb, matrix);
                    for (channel, value) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(rgb) {
                        channel.copy_from_slice(&value.to_ne_bytes());
                    }
                }
            }
            second.convert_row(&self.scratch, to, width);
        } else {
            self.first.convert_row(from, to, width);
        }
    }
}

impl core::fmt::Debug for HdrAnalysisConverter {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HdrAnalysisConverter")
            .field("first", self.first.plan())
            .field("matrix", &self.matrix)
            .field("second", &self.second.as_ref().map(RowConverter::plan))
            .finish()
    }
}

/// Extract encoded color samples without alpha arithmetic or transfer changes.
/// This is an analysis representation, not an alpha conversion for rendering.
struct StoredColors {
    bytes: Vec<u8>,
    source_pixel_bytes: usize,
    channel_bytes: usize,
    indices: [usize; 3],
}

impl StoredColors {
    fn new(from: PixelDescriptor, width: u32) -> Result<Option<Self>, String> {
        let (channels, indices) = match from.layout() {
            ChannelLayout::Rgb => return Ok(None),
            ChannelLayout::Rgba => (4, [0, 1, 2]),
            ChannelLayout::Bgra => (4, [2, 1, 0]),
            ChannelLayout::Gray => (1, [0, 0, 0]),
            ChannelLayout::GrayAlpha => (2, [0, 0, 0]),
            other => return Err(format!("Unsupported HDR analysis layout: {other:?}")),
        };
        let channel_bytes = match from.channel_type() {
            ChannelType::U8 => 1,
            ChannelType::U16 | ChannelType::F16 => 2,
            ChannelType::F32 => 4,
            other => return Err(format!("Unsupported HDR analysis depth: {other:?}")),
        };
        let len = (width as usize)
            .checked_mul(3 * channel_bytes)
            .ok_or_else(|| "HDR stored-color row size overflow".to_string())?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(len)
            .map_err(|e| format!("HDR stored-color row allocation failed: {e}"))?;
        bytes.resize(len, 0);
        Ok(Some(Self {
            bytes,
            source_pixel_bytes: channels * channel_bytes,
            channel_bytes,
            indices,
        }))
    }

    fn extract(&mut self, source: &[u8]) -> &[u8] {
        let required = self.bytes.len() / (3 * self.channel_bytes) * self.source_pixel_bytes;
        assert!(
            source.len() >= required,
            "HDR analysis source row is too short"
        );
        for (pixel, rgb) in source[..required]
            .chunks_exact(self.source_pixel_bytes)
            .zip(self.bytes.chunks_exact_mut(3 * self.channel_bytes))
        {
            for (to, index) in rgb.chunks_exact_mut(self.channel_bytes).zip(self.indices) {
                let at = index * self.channel_bytes;
                to.copy_from_slice(&pixel[at..at + self.channel_bytes]);
            }
        }
        &self.bytes
    }
}

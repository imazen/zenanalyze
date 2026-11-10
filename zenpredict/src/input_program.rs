//! Validated, ordered input composition before feature transforms and scaling.
//!
//! Node references address earlier nodes only. This small straight-line program
//! expresses shared products, monotone damage shaping, and reference gates
//! without embedding metric-specific feature IDs in the inference engine.
use crate::{PredictError, limits::MAX_DIM};
use alloc::vec::Vec;

/// Required metadata entry for an input-program bake (bytes, never UTF-8).
pub const METADATA_KEY: &str = "zenpredict.input_program";
/// Same physical header as v3; required input composition makes old readers reject it.
pub const FORMAT_VERSION: u16 = 4;
fn invalid(reason: &'static str) -> PredictError {
    PredictError::InvalidInputProgram { reason }
}

/// Zero-anchored, nondecreasing, three-segment damage shaper.
/// Negative inputs clamp to zero; NaN propagates. Exact identity slopes preserve -0.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DamageShape([f32; 5]);
impl DamageShape {
    /// Knots must be finite with `0 < k0 < k1`; slopes finite and nonnegative.
    pub fn new(knots: [f32; 2], slopes: [f32; 3]) -> Result<Self, PredictError> {
        let p = [knots[0], knots[1], slopes[0], slopes[1], slopes[2]];
        if p.iter().any(|x| !x.is_finite() || *x < 0.) || p[0] <= 0. || p[1] <= p[0] {
            return Err(invalid("invalid damage knots or slopes"));
        }
        Ok(Self(p))
    }
    /// `[k0, k1, s0, s1, s2]` in raw, pre-standardization coordinates.
    #[cfg(feature = "baking")]
    pub fn parameters(self) -> [f32; 5] {
        self.0
    }
    /// The research wire arithmetic, including its exact identity branch.
    pub fn apply(self, x: f32) -> f32 {
        let [a, b, s0, s1, s2] = self.0;
        if x.is_nan() {
            return x;
        }
        let x = if x < 0. { 0. } else { x };
        if s0 == 1. && s1 == 1. && s2 == 1. {
            return x;
        }
        let first = x.min(a);
        let second = (x - a).max(0.).min(b - a);
        let third = (x - b).max(0.);
        (first * s0 + second * s1) + third * s2
    }
}

/// Nonnegative reference gate, linearly interpolated at 0, 0.5 and 1.
/// Endpoints clamp outside [0,1]. Nonnegativity does not imply monotonicity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReferenceGate([f32; 3]);
impl ReferenceGate {
    /// All three values must be finite and nonnegative.
    pub fn new(values: [f32; 3]) -> Result<Self, PredictError> {
        if values.iter().any(|x| !x.is_finite() || *x < 0.) {
            return Err(invalid("invalid reference gate"));
        }
        Ok(Self(values))
    }
    /// Values at 0, 0.5 and 1.
    #[cfg(feature = "baking")]
    pub fn parameters(self) -> [f32; 3] {
        self.0
    }
    /// Preserve the research interpolation order; NaN propagates.
    pub fn apply(self, x: f32) -> f32 {
        let [a, b, c] = self.0;
        if x <= 0. {
            return a;
        }
        if x >= 1. {
            return c;
        }
        let (lo, hi, t) = if x < 0.5 {
            (a, b, 2. * x)
        } else {
            (b, c, 2. * x - 1.)
        };
        (1. - t) * lo + t * hi
    }
}

/// A node in an ordered input program. All references except Source's index
/// address earlier nodes, not source columns. Unused source columns are allowed.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum InputOp {
    /// Read a source column.
    Source { index: u32 },
    /// f32 product, without reassociation with adjacent shaping operations.
    Product { left: u32, right: u32 },
    /// Monotone damage shaping of an earlier node.
    Shape { input: u32, shape: DamageShape },
    /// Reference-only modulation of an earlier node.
    Gate { input: u32, gate: ReferenceGate },
}

/// Validated input program. Immutable after construction; scratch belongs to the caller.
#[derive(Clone, Debug, PartialEq)]
pub struct InputProgram {
    source_count: u32,
    nodes: Vec<InputOp>,
    outputs: Vec<u32>,
}
impl InputProgram {
    /// Validate dimensions, source indices, backward-only node references and outputs.
    /// Counts are bounded by the model parser's MAX_DIM before scratch allocation.
    pub fn new(
        source_count: u32,
        nodes: Vec<InputOp>,
        outputs: Vec<u32>,
    ) -> Result<Self, PredictError> {
        if source_count == 0
            || source_count as usize > MAX_DIM
            || nodes.is_empty()
            || nodes.len() > MAX_DIM
            || outputs.is_empty()
            || outputs.len() > MAX_DIM
        {
            return Err(invalid("invalid program dimensions"));
        }
        for (i, op) in nodes.iter().enumerate() {
            let valid = match *op {
                InputOp::Source { index } => index < source_count,
                InputOp::Product { left, right } => (left as usize) < i && (right as usize) < i,
                InputOp::Shape { input, .. } | InputOp::Gate { input, .. } => (input as usize) < i,
            };
            if !valid {
                return Err(invalid("invalid source or forward node reference"));
            }
        }
        if outputs.iter().any(|&x| x as usize >= nodes.len()) {
            return Err(invalid("invalid output reference"));
        }
        Ok(Self {
            source_count,
            nodes,
            outputs,
        })
    }
    /// Number of caller-supplied source columns.
    pub fn source_count(&self) -> usize {
        self.source_count as usize
    }
    /// Ordered, validated operations.
    #[cfg(feature = "baking")]
    pub fn nodes(&self) -> &[InputOp] {
        &self.nodes
    }
    /// Node indices in compact model-input order, before feature transforms.
    pub fn outputs(&self) -> &[u32] {
        &self.outputs
    }
    /// Required number of f32 scratch elements.
    pub fn scratch_len(&self) -> usize {
        self.nodes.len()
    }
    /// Evaluate without allocation. Lengths are checked before any writes.
    /// IEEE NaN/infinity semantics are retained; this is not a finite-score guarantee.
    pub fn apply(
        &self,
        source: &[f32],
        scratch: &mut [f32],
        output: &mut [f32],
    ) -> Result<(), PredictError> {
        for (expected, got) in [
            (self.source_count(), source.len()),
            (self.outputs.len(), output.len()),
        ] {
            if expected != got {
                return Err(PredictError::FeatureLenMismatch { expected, got });
            }
        }
        if scratch.len() < self.scratch_len() {
            return Err(invalid("program scratch too short"));
        }
        for (i, op) in self.nodes.iter().enumerate() {
            scratch[i] = match *op {
                InputOp::Source { index } => source[index as usize],
                InputOp::Product { left, right } => {
                    scratch[left as usize] * scratch[right as usize]
                }
                InputOp::Shape { input, shape } => shape.apply(scratch[input as usize]),
                InputOp::Gate { input, gate } => gate.apply(scratch[input as usize]),
            };
        }
        for (dst, &i) in output.iter_mut().zip(&self.outputs) {
            *dst = scratch[i as usize];
        }
        Ok(())
    }
    /// Canonical little-endian bytes, suitable for METADATA_KEY.
    #[cfg(any(feature = "baking", test))]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"ZINP\x01\x00\x00\x00");
        for n in [
            self.source_count,
            self.nodes.len() as u32,
            self.outputs.len() as u32,
        ] {
            b.extend_from_slice(&n.to_le_bytes());
        }
        for op in &self.nodes {
            let (code, refs, params): (u8, &[u32], &[f32]) = match op {
                InputOp::Source { index } => (0, core::slice::from_ref(index), &[]),
                InputOp::Product { left, right } => {
                    b.extend_from_slice(&[1, 0, 0, 0]);
                    b.extend_from_slice(&left.to_le_bytes());
                    b.extend_from_slice(&right.to_le_bytes());
                    continue;
                }
                InputOp::Shape { input, shape } => (2, core::slice::from_ref(input), &shape.0),
                InputOp::Gate { input, gate } => (3, core::slice::from_ref(input), &gate.0),
            };
            b.extend_from_slice(&[code, 0, 0, 0]);
            for n in refs {
                b.extend_from_slice(&n.to_le_bytes());
            }
            for x in params {
                b.extend_from_slice(&x.to_bits().to_le_bytes());
            }
        }
        for x in &self.outputs {
            b.extend_from_slice(&x.to_le_bytes());
        }
        b
    }
    /// Strict decoder: unknown codes, reserved bits, trailing bytes, nonfinite
    /// parameters, excessive counts and invalid references are errors.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, PredictError> {
        let mut c = Cursor(bytes);
        if c.take(8)? != b"ZINP\x01\x00\x00\x00" {
            return Err(invalid("unknown program schema"));
        }
        let source_count = c.u32()?;
        let n = c.u32()? as usize;
        let out = c.u32()? as usize;
        if source_count == 0
            || source_count as usize > MAX_DIM
            || n > MAX_DIM
            || out > MAX_DIM
            || n.checked_mul(8)
                .and_then(|x| out.checked_mul(4).and_then(|y| x.checked_add(y)))
                .is_none_or(|min| min > c.0.len())
        {
            return Err(invalid("program counts exceed input or limits"));
        }
        let mut nodes = Vec::new();
        let mut outputs = Vec::new();
        nodes
            .try_reserve_exact(n)
            .map_err(|_| invalid("program allocation failed"))?;
        outputs
            .try_reserve_exact(out)
            .map_err(|_| invalid("program allocation failed"))?;
        for _ in 0..n {
            let code = c.take(4)?;
            if code[1..] != [0, 0, 0] {
                return Err(invalid("reserved node flags"));
            }
            let input = c.u32()?;
            nodes.push(match code[0] {
                0 => InputOp::Source { index: input },
                1 => InputOp::Product {
                    left: input,
                    right: c.u32()?,
                },
                2 => InputOp::Shape {
                    input,
                    shape: DamageShape::new([c.f32()?, c.f32()?], [c.f32()?, c.f32()?, c.f32()?])?,
                },
                3 => InputOp::Gate {
                    input,
                    gate: ReferenceGate::new([c.f32()?, c.f32()?, c.f32()?])?,
                },
                _ => return Err(invalid("unknown input opcode")),
            });
        }
        for _ in 0..out {
            outputs.push(c.u32()?);
        }
        if !c.0.is_empty() {
            return Err(invalid("trailing program bytes"));
        }
        Self::new(source_count, nodes, outputs)
    }
}
struct Cursor<'a>(&'a [u8]);
impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], PredictError> {
        if n > self.0.len() {
            return Err(invalid("truncated input program"));
        }
        let (a, b) = self.0.split_at(n);
        self.0 = b;
        Ok(a)
    }
    fn u32(&mut self) -> Result<u32, PredictError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, PredictError> {
        Ok(f32::from_bits(self.u32()?))
    }
}

#[cfg(test)]
#[path = "input_program_tests.rs"]
mod tests;

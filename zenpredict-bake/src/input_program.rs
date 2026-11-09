//! Typed JSON and Rust composition helpers for required input programs.
use crate::{BakeError, BakeMetadataEntry, BakeRequest};
use alloc::vec::Vec;
use serde::Deserialize;
use zenpredict::input_program::{DamageShape, InputOp, InputProgram, METADATA_KEY, ReferenceGate};

/// JSON input program. Unknown fields are errors, including misspelled parameters.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct InputProgramJson {
    pub source_count: u32,
    pub nodes: Vec<InputOpJson>,
    pub outputs: Vec<u32>,
}
/// Tagged operations; indices have the same meaning as runtime InputOp.
#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
#[non_exhaustive]
pub enum InputOpJson {
    Source {
        index: u32,
    },
    Product {
        left: u32,
        right: u32,
    },
    Shape {
        input: u32,
        knots: [f32; 2],
        slopes: [f32; 3],
    },
    Gate {
        input: u32,
        values: [f32; 3],
    },
}
impl InputProgramJson {
    /// Validate using the runtime's constructors; no duplicate wire/math parser.
    pub fn build(&self) -> Result<InputProgram, zenpredict::PredictError> {
        let nodes = self
            .nodes
            .iter()
            .map(|op| {
                Ok(match *op {
                    InputOpJson::Source { index } => InputOp::Source { index },
                    InputOpJson::Product { left, right } => InputOp::Product { left, right },
                    InputOpJson::Shape {
                        input,
                        knots,
                        slopes,
                    } => InputOp::Shape {
                        input,
                        shape: DamageShape::new(knots, slopes)?,
                    },
                    InputOpJson::Gate { input, values } => InputOp::Gate {
                        input,
                        gate: ReferenceGate::new(values)?,
                    },
                })
            })
            .collect::<Result<Vec<_>, zenpredict::PredictError>>()?;
        InputProgram::new(self.source_count, nodes, self.outputs.clone())
    }
}

/// Bake a request with required composition. Existing caller metadata is retained.
/// A duplicate program key is rejected instead of silently choosing precedence.
/// The resulting v4 bake is rejected by v3-only readers before inference.
pub fn bake_with_input_program(
    req: &BakeRequest<'_>,
    program: &InputProgram,
) -> Result<Vec<u8>, BakeError> {
    let bytes = program.to_bytes();
    let mut metadata: Vec<_> = req
        .metadata
        .iter()
        .map(|e| BakeMetadataEntry {
            key: e.key,
            kind: e.kind,
            value: e.value,
        })
        .collect();
    metadata.push(BakeMetadataEntry {
        key: METADATA_KEY,
        kind: zenpredict::MetadataType::Bytes,
        value: &bytes,
    });
    crate::bake(&BakeRequest {
        metadata: &metadata,
        schema_hash: req.schema_hash,
        flags: req.flags,
        scaler_mean: req.scaler_mean,
        scaler_scale: req.scaler_scale,
        layers: req.layers,
        feature_bounds: req.feature_bounds,
        output_specs: req.output_specs,
        discrete_sets: req.discrete_sets,
        sparse_overrides: req.sparse_overrides,
        feature_order: req.feature_order,
        output_order: req.output_order,
        compressed: req.compressed,
        hu_permutations: req.hu_permutations,
    })
}

/// Attach or replace composition on an existing bake without reordering hidden
/// units or requantizing any weights. The compact network/scaler/transforms are
/// preserved byte for byte. The caller supplies source ordering and provenance.
pub fn attach_input_program(
    bytes: &[u8],
    program: &InputProgram,
) -> Result<Vec<u8>, crate::AppendError> {
    crate::append::attach_program(bytes, &program.to_bytes())
}

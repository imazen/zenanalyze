//! Opt-in research rewrite of a certified tied ReLU pair to a direct maximum.
//! Unmarked models keep their original dense evaluation and bits.
use crate::{Activation, Model, PredictError, WeightStorage};
use alloc::vec::Vec;

const KEY: &str = "zenpredict.research_max_pairs";
fn invalid(reason: &'static str) -> PredictError {
    PredictError::InvalidResearchMaxPair { reason }
}

pub(crate) fn parse(model: &Model) -> Result<Vec<[usize; 2]>, PredictError> {
    let metadata = model.metadata();
    if metadata.get(KEY).is_none() {
        if model
            .layers()
            .any(|l| l.activation == Activation::ResearchReluMaxPairs)
        {
            return Err(invalid("research activation requires pair metadata"));
        }
        return Ok(Vec::new());
    }
    let text = metadata.get_utf8(KEY)?;
    let mut lines = text.lines();
    if lines.next() != Some("relu-max-pairs-v1") {
        return Err(invalid("unknown schema"));
    }
    if model.n_layers() < 2 {
        return Err(invalid("requires two adjacent dense layers"));
    }
    let first = model.layer(0);
    let next = model.layer(1);
    if first.activation != Activation::ResearchReluMaxPairs
        || model
            .layers()
            .skip(1)
            .any(|l| l.activation == Activation::ResearchReluMaxPairs)
    {
        return Err(invalid(
            "research activation must occur only at the first layer",
        ));
    }
    let (WeightStorage::F32(w), WeightStorage::F32(mid)) = (&first.weights, &next.weights) else {
        return Err(invalid("requires f32 weights"));
    };
    if model.scaler_mean().iter().any(|&v| v != 0.)
        || model
            .scaler_scale()
            .iter()
            .any(|&v| !v.is_finite() || v <= 0.)
    {
        return Err(invalid("requires positive scale-only inputs"));
    }
    let mut pairs = Vec::new();
    let mut used = alloc::vec![false;first.out_dim];
    for line in lines {
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.len() != 2 {
            return Err(invalid("pair requires two indices"));
        }
        let u = tokens[0]
            .parse::<usize>()
            .map_err(|_| invalid("invalid index"))?;
        let v = tokens[1]
            .parse::<usize>()
            .map_err(|_| invalid("invalid index"))?;
        if u >= first.out_dim || v >= first.out_dim || u == v || used[u] || used[v] {
            return Err(invalid("overlapping or out-of-range pair"));
        }
        if first.biases[u] != 0. || !first.biases[v].is_finite() || first.biases[v] > 0. {
            return Err(invalid("pair biases must be zero and nonpositive"));
        }
        for i in 0..first.in_dim {
            let a = w[i * first.out_dim + u];
            let correction = w[i * first.out_dim + v];
            if !a.is_finite()
                || !correction.is_finite()
                || a < 0.
                || f64::from(a) + f64::from(correction) < 0.
            {
                return Err(invalid("pair detector coefficients must be nonnegative"));
            }
        }
        for j in 0..next.out_dim {
            let a = mid[u * next.out_dim + j];
            let b = mid[v * next.out_dim + j];
            if !a.is_finite() || a.to_bits() != b.to_bits() {
                return Err(invalid("following weights must be exactly tied"));
            }
        }
        used[u] = true;
        used[v] = true;
        pairs.push([u, v]);
    }
    if pairs.is_empty() {
        return Err(invalid("empty declaration"));
    }
    Ok(pairs)
}

pub(crate) fn apply(model: &Model, src: &[f32], dst: &mut [f32]) -> Result<(), PredictError> {
    if model.research_max_pairs.is_empty() {
        return Ok(());
    }
    if src.iter().any(|&x| !x.is_finite() || x < 0.) {
        return Err(invalid("inputs must be finite and nonnegative"));
    }
    let first = model.layer(0);
    let WeightStorage::F32(w) = first.weights else {
        return Err(invalid("requires f32 weights"));
    };
    for &[u, v] in &model.research_max_pairs {
        let mut a = 0f64;
        let mut b = f64::from(first.biases[v]);
        for (i, &x) in src.iter().enumerate() {
            let aw = f64::from(w[i * first.out_dim + u]);
            let bw = aw + f64::from(w[i * first.out_dim + v]);
            // Nonnegative detector coefficients preserve order under each rounded
            // operation. Reconstruct in f64; don't round a+c back to f32 first.
            a += f64::from(x) * aw;
            b += f64::from(x) * bw;
        }
        dst[u] = a.max(b) as f32;
        dst[v] = 0.;
    }
    Ok(())
}

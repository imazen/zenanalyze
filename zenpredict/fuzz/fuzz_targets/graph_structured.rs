//! Fuzz target: structured ZNPR v4 graphs, differential against the
//! per-node reference evaluator in `graph_core.rs`. `arbitrary` picks the node
//! list (op, activation, dtype, width, input indices, payload bytes) and
//! the builder lays out a well-formed file around it — sections in range,
//! sizes either exact for the declared op or deliberately off — then
//! applies a few raw byte flips. Reaches executor paths (slot planning,
//! every op, every activation) that random bytes almost never validate
//! into.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

include!("graph_core.rs");

#[derive(Arbitrary, Debug)]
struct FuzzNode {
    op: u8,
    act: u8,
    dtype: u8,
    out_dim: u8,
    inputs: Vec<u8>,
    exact_sizes: bool,
    bias: bool,
    data: Vec<u8>,
}

#[derive(Arbitrary, Debug)]
struct FuzzGraph {
    n_inputs: u8,
    nodes: Vec<FuzzNode>,
    n_outputs_override: Option<u8>,
    flips: Vec<(u16, u8)>,
}

fn sec(buf: &mut [u8], at: usize, off: usize, len: usize) {
    buf[at..at + 4].copy_from_slice(&(off as u32).to_le_bytes());
    buf[at + 4..at + 8].copy_from_slice(&(len as u32).to_le_bytes());
}

fn pad4(buf: &mut Vec<u8>) {
    while buf.len() % 4 != 0 {
        buf.push(0);
    }
}

/// Append `len` bytes cycled from `src` (zeros if empty), 4-aligned.
fn blob(buf: &mut Vec<u8>, src: &[u8], len: usize) -> usize {
    pad4(buf);
    let off = buf.len();
    for k in 0..len {
        buf.push(if src.is_empty() {
            0
        } else {
            src[k % src.len()]
        });
    }
    off
}

fn build(g: &FuzzGraph) -> Vec<u8> {
    use core::mem::offset_of;
    use zenpredict::NodeEntry;
    use zenpredict::wire::*;
    let n_in = g.n_inputs as usize % 16 + 1;
    let nodes = &g.nodes[..g.nodes.len().min(40)];
    let n = nodes.len().max(1);
    let mut widths = vec![n_in; n];
    let mut buf = vec![0u8; HEADER_SIZE + n * NODE_ENTRY_SIZE];
    buf[0..4].copy_from_slice(b"ZNPR");
    buf[4..6].copy_from_slice(&4u16.to_le_bytes());
    buf[8..12].copy_from_slice(&(n_in as u32).to_le_bytes());
    buf[16..20].copy_from_slice(&(n as u32).to_le_bytes());
    sec(
        &mut buf,
        SECTION_OFF_LAYER_TABLE,
        HEADER_SIZE,
        n * NODE_ENTRY_SIZE,
    );
    // Scaler: mean 0, scale 1.
    pad4(&mut buf);
    let m = buf.len();
    buf.extend(std::iter::repeat_n(0u8, n_in * 4));
    sec(&mut buf, SECTION_OFF_SCALER_MEAN, m, n_in * 4);
    let s = buf.len();
    for _ in 0..n_in {
        buf.extend_from_slice(&1.0f32.to_le_bytes());
    }
    sec(&mut buf, SECTION_OFF_SCALER_SCALE, s, n_in * 4);

    for (i, node) in nodes.iter().enumerate() {
        // Node 0 is Input except for a rare deliberately-wrong op; later
        // nodes cover every op plus one unknown byte (7).
        let op = if i == 0 {
            if node.op == 255 { 2 } else { 0 }
        } else {
            node.op % 8
        };
        let prev = widths.clone();
        let width_of = |j: u32| prev.get(j as usize).copied().unwrap_or(1);
        let ins: Vec<u32> = node
            .inputs
            .iter()
            .take(6)
            .map(|&b| b as u32 % (i as u32 + 1))
            .collect();
        let mut out = node.out_dim as usize % 24 + 1;
        // With exact sizes, make widths agree with the op so validation
        // passes and the executor runs.
        if node.exact_sizes {
            match op {
                0 => out = n_in,
                2 => out = ins.first().map_or(out, |&j| width_of(j)),
                4 | 5 => {
                    out = ins.first().map_or(out, |&j| width_of(j));
                }
                6 => out = ins.iter().map(|&j| width_of(j)).sum::<usize>().max(1),
                _ => {}
            }
        }
        if i < widths.len() {
            widths[i] = out;
        }
        let in_list: Vec<u32> = if node.exact_sizes {
            match op {
                0 => vec![],
                1..=3 => ins.iter().take(1).copied().collect(),
                4 | 5 => {
                    let a = ins.first().copied().unwrap_or(0);
                    let b = ins
                        .get(1)
                        .copied()
                        .filter(|&b| width_of(b) == width_of(a))
                        .unwrap_or(a);
                    vec![a, b]
                }
                _ => ins.clone(),
            }
        } else {
            ins.clone()
        };
        pad4(&mut buf);
        let in_off = buf.len();
        for &j in &in_list {
            buf.extend_from_slice(&j.to_le_bytes());
        }
        let entry = HEADER_SIZE + i * NODE_ENTRY_SIZE;
        let act = node.act % 6;
        let dtype = node.dtype % 4;
        let (mut d0, mut d1, mut d2) = ((0, 0), (0, 0), (0, 0));
        match op {
            1 => {
                let in_dim = in_list.first().map_or(1, |&j| width_of(j));
                let elem = match dtype {
                    1 => 2,
                    2 => 1,
                    _ => 4,
                };
                let wlen = if node.exact_sizes {
                    in_dim * out * elem
                } else {
                    node.data.len()
                };
                d0 = (blob(&mut buf, &node.data, wlen), wlen);
                if dtype == 2 {
                    let slen = if node.exact_sizes {
                        out * 4
                    } else {
                        node.data.len() % 64
                    };
                    d1 = (blob(&mut buf, &[0, 0, 128, 63], slen), slen);
                }
                if node.bias {
                    let blen = if node.exact_sizes {
                        out * 4
                    } else {
                        node.data.len() % 32
                    };
                    d2 = (blob(&mut buf, &node.data, blen), blen);
                }
            }
            3 => {
                let src_w = in_list.first().map_or(1, |&j| width_of(j)) as u32;
                pad4(&mut buf);
                let off = buf.len();
                let count = if node.exact_sizes {
                    out
                } else {
                    node.data.len() % 32
                };
                for k in 0..count {
                    let b = node.data.get(k).copied().unwrap_or(k as u8) as u32;
                    let idx = if node.exact_sizes { b % src_w } else { b };
                    buf.extend_from_slice(&idx.to_le_bytes());
                }
                d0 = (off, count * 4);
            }
            _ => {}
        }
        let e = &mut buf[entry..entry + NODE_ENTRY_SIZE];
        e[0] = op;
        e[1] = if node.exact_sizes && !matches!(op, 1 | 2) {
            0
        } else {
            act
        };
        e[2] = if node.exact_sizes && op != 1 {
            0
        } else {
            dtype
        };
        e[4..8].copy_from_slice(&(out as u32).to_le_bytes());
        sec(e, offset_of!(NodeEntry, inputs), in_off, in_list.len() * 4);
        sec(e, offset_of!(NodeEntry, data0), d0.0, d0.1);
        sec(e, offset_of!(NodeEntry, data1), d1.0, d1.1);
        sec(e, offset_of!(NodeEntry, data2), d2.0, d2.1);
    }
    let n_out = match g.n_outputs_override {
        Some(o) => o as usize % 32 + 1,
        None => widths.last().copied().unwrap_or(1),
    };
    buf[12..16].copy_from_slice(&(n_out as u32).to_le_bytes());
    let len = buf.len();
    for &(at, val) in g.flips.iter().take(4) {
        buf[at as usize % len] ^= val;
    }
    buf
}

// Differential: every graph that loads must give the same outputs as the
// per-node reference evaluator (bits, or NaN for NaN).
fuzz_target!(|g: FuzzGraph| {
    exercise_model_bytes(&build(&g), true);
});

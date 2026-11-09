use super::*;
use alloc::{vec, vec::Vec};
fn program() -> InputProgram {
    InputProgram::new(
        2,
        vec![
            InputOp::Source { index: 0 },
            InputOp::Source { index: 1 },
            InputOp::Product { left: 0, right: 1 },
            InputOp::Shape {
                input: 2,
                shape: DamageShape::new([0.25, 1.], [2., 0.5, 3.]).unwrap(),
            },
        ],
        vec![3],
    )
    .unwrap()
}
#[test]
fn explicit_shaper_values_identity_boundaries_and_order() {
    let shape = DamageShape::new([0.25, 1.], [2., 0.5, 3.]).unwrap();
    for (x, y) in [
        (-1., 0.),
        (0., 0.),
        (0.125, 0.25),
        (0.25, 0.5),
        (0.5, 0.625),
        (1., 0.875),
        (2., 3.875),
    ] {
        assert_eq!(shape.apply(x), y);
    }
    let identity = DamageShape::new([0.3, 1.7], [1.; 3]).unwrap();
    for x in [-0.0, 0., f32::MIN_POSITIVE, 0.3, 1.7, 1e10, f32::MAX] {
        assert_eq!(identity.apply(x).to_bits(), x.to_bits());
    }
    assert!(shape.apply(f32::NAN).is_nan());
    let mut last = 0.;
    for i in 0..10000 {
        let y = shape.apply(i as f32 / 1000.);
        assert!(y >= last);
        last = y;
    }
    let gate = ReferenceGate::new([1., 0.25, 2.]).unwrap();
    for (x, y) in [
        (-1., 1.),
        (0., 1.),
        (0.25, 0.625),
        (0.5, 0.25),
        (0.75, 1.125),
        (1., 2.),
        (2., 2.),
    ] {
        assert_eq!(gate.apply(x), y);
    }
    assert!(gate.apply(f32::NAN).is_nan());
    for knots in [[0., 1.], [1., 1.], [2., 1.], [1., f32::INFINITY]] {
        assert!(DamageShape::new(knots, [1.; 3]).is_err());
    }
    assert!(DamageShape::new([1., 2.], [1., -1., 1.]).is_err());
    assert!(ReferenceGate::new([1., f32::NAN, 1.]).is_err());
}
#[test]
fn program_rejects_every_truncation_reserved_bits_bad_counts_and_references() {
    let p = program();
    let bytes = p.to_bytes();
    assert_eq!(InputProgram::from_bytes(&bytes).unwrap(), p);
    for n in 0..bytes.len() {
        assert!(InputProgram::from_bytes(&bytes[..n]).is_err(), "prefix {n}");
    }
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(InputProgram::from_bytes(&trailing).is_err());
    for (offset, value) in [(4, 2), (6, 1), (20, 255), (21, 1)] {
        let mut b = bytes.clone();
        b[offset] = value;
        assert!(InputProgram::from_bytes(&b).is_err());
    }
    for offset in [8, 12, 16, 24, 32, 40, 44] {
        let mut b = bytes.clone();
        b[offset..offset + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(InputProgram::from_bytes(&b).is_err(), "offset {offset}");
    }
    for offset in [52, 56, 60, 64, 68] {
        let mut b = bytes.clone();
        b[offset..offset + 4].copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
        assert!(InputProgram::from_bytes(&b).is_err());
    }
    assert!(InputProgram::new(1, vec![InputOp::Product { left: 0, right: 0 }], vec![0]).is_err());
    assert!(InputProgram::new(1, vec![InputOp::Source { index: 1 }], vec![0]).is_err());
    assert!(InputProgram::new(1, Vec::new(), vec![0]).is_err());
}
#[test]
fn evaluation_is_validated_before_writes_and_reuses_all_scratch() {
    let p = program();
    let mut scratch = vec![99.; p.scratch_len()];
    let mut out = [99.];
    assert!(p.apply(&[1.], &mut scratch, &mut out).is_err());
    assert_eq!(out, [99.]);
    assert!(scratch.iter().all(|&x| x == 99.));
    assert!(p.apply(&[1., 2.], &mut [], &mut out).is_err());
    assert_eq!(out, [99.]);
    p.apply(&[2., 0.25], &mut scratch, &mut out).unwrap();
    assert_eq!(out, [0.625]);
    p.apply(&[0., 0.25], &mut scratch, &mut out).unwrap();
    assert_eq!(out, [0.]);
}

//! Print output-transform semantics for comparison across Cargo feature sets.
//! This is an investigation tool; it does not bless incorrect values as expectations.

use zenpredict::OutputTransform;

fn main() {
    println!("std={}", cfg!(feature = "std"));
    for transform in [OutputTransform::Exp, OutputTransform::Sigmoid] {
        println!("{transform:?}(1)={}", transform.apply(1.0, [0.0, 0.0]));
    }
    for value in [16_777_215.0f32, f32::INFINITY] {
        println!(
            "Round({value})={}, reference={}",
            OutputTransform::Round.apply(value, [0.0, 0.0]),
            value.round()
        );
    }
}

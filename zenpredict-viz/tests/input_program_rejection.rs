use zenpredict::input_program::{InputOp, InputProgram};
use zenpredict::{Activation, WeightDtype};
use zenpredict_bake::input_program::bake_with_input_program;
use zenpredict_bake::{BakeLayer, BakeRequest, bake};
#[test]
fn unsupported_operators_are_errors_instead_of_identity_traces() {
    for activation in [Activation::Softplus, Activation::Silu] {
        let layers = [BakeLayer {
            in_dim: 1,
            out_dim: 1,
            activation,
            dtype: WeightDtype::F32,
            weights: &[1.],
            biases: &[0.],
        }];
        let bytes = bake(&BakeRequest::new(0, 0, &[0.], &[1.], &layers)).unwrap();
        let e = zenpredict_viz::forward_with_taps_native(&bytes, &[2.])
            .err()
            .expect("unsupported activation must be explicit");
        assert!(e.contains("activations"));
    }
    let layers = [BakeLayer {
        in_dim: 1,
        out_dim: 1,
        activation: Activation::Identity,
        dtype: WeightDtype::F32,
        weights: &[1.],
        biases: &[0.],
    }];
    let program = InputProgram::new(1, vec![InputOp::Source { index: 0 }], vec![0]).unwrap();
    let bytes =
        bake_with_input_program(&BakeRequest::new(0, 0, &[0.], &[1.], &layers), &program).unwrap();
    let e = zenpredict_viz::forward_with_taps_native(&bytes, &[2.])
        .err()
        .expect("required program must not be omitted");
    assert!(e.contains("input-program"));
}

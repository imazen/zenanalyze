//! PR144 canonical Full/HIGHP versus libm on feature-transform power inputs.
use magetypes::powf::{PowfOptions, try_powf_array_with, try_powf_with};
use zenbench::black_box;
use zenpredict::FeatureTransform;
const OPTIONS: u32 = PowfOptions::PORTABLE_HIGHP.bits();
zenbench::main!(|suite| {
    let inputs: [f32; 64] = core::array::from_fn(|i| libm::exp2f((i as f32 - 32.) * 0.45));
    for exponent in [-1.5f32, 0.37, 2.4] {
        suite.group(format!("full_power_{exponent}"), |g| {
            g.throughput(zenbench::Throughput::Elements(inputs.len() as u64));
            g.bench("libm_powf", move |b| {
                b.iter(move || {
                    let mut sum = 0.;
                    for x in black_box(inputs) {
                        sum += libm::powf(x, black_box(exponent));
                    }
                    black_box(sum)
                })
            });
            g.bench("scalar_highp_auto", move |b| {
                b.iter(move || {
                    let mut sum = 0.;
                    for x in black_box(inputs) {
                        sum += try_powf_with::<OPTIONS>(x, black_box(exponent)).unwrap();
                    }
                    black_box(sum)
                })
            });
            if exponent > 0. {
                g.bench("feature_signed_powf_highp", move |b| {
                    b.iter(move || {
                        let mut sum = 0.;
                        for x in black_box(inputs) {
                            sum += FeatureTransform::SignedPow
                                .apply_with_params(x, &[black_box(exponent)]);
                        }
                        black_box(sum)
                    })
                });
            }
            g.bench("array8_highp", move |b| {
                b.iter(move || {
                    let mut sum = 0.;
                    for &x in black_box(inputs).as_chunks::<8>().0 {
                        for y in try_powf_array_with::<OPTIONS, 8>(x, black_box(exponent)).unwrap()
                        {
                            sum += y;
                        }
                    }
                    black_box(sum)
                })
            });
        });
    }
});

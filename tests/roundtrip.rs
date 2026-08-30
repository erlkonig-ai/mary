//! shelley format round-trip: store a tensor + a Linear (weight+bias), reload by
//! id, assert bit-identical — proving the self-describing leaf/module format.

use burn::prelude::*;
use burn_ndarray::NdArray;
use mary::{load_linear, load_tensor, put_linear, put_tensor};
use triblespace::prelude::*;

type B = NdArray;

#[test]
fn tensor_and_linear_roundtrip() {
    let device = Default::default();
    let mut blobs = MemoryBlobStore::new();

    // a tensor leaf
    let t = Tensor::<B, 2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
    let tf = put_tensor(&mut blobs, &t).unwrap();
    let t_id = tf.root().unwrap();

    // a Linear module (weight [out,in] + bias [out])
    let w = Tensor::<B, 2>::from_floats([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], &device);
    let bias = Tensor::<B, 1>::from_floats([7.0, 8.0], &device);
    let lf = put_linear(&mut blobs, &w, Some(&bias)).unwrap();
    let l_id = lf.root().unwrap();

    let mut tribles = TribleSet::new();
    tribles += tf;
    tribles += lf;

    let reader = SnapshotSource::snapshot(&mut blobs).unwrap();

    let t2 = load_tensor::<B, 2>(&tribles, &reader, t_id, &device);
    assert_eq!(t2.dims(), [2, 3]);
    assert_eq!(
        t2.to_data().to_vec::<f32>().unwrap(),
        t.to_data().to_vec::<f32>().unwrap()
    );

    let (w2, b2) = load_linear::<B>(&tribles, &reader, l_id, &device);
    assert_eq!(
        w2.to_data().to_vec::<f32>().unwrap(),
        w.to_data().to_vec::<f32>().unwrap()
    );
    let b2 = b2.expect("bias present");
    assert_eq!(
        b2.to_data().to_vec::<f32>().unwrap(),
        bias.to_data().to_vec::<f32>().unwrap()
    );
}

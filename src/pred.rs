use crate::model::Model;
use candle::{DType, Device, Result, Tensor, D};
use candle_nn::{VarBuilder, VarMap};

pub fn predict<P: AsRef<std::path::Path>>(path: P, images: &Tensor) -> Result<Tensor> {
    let device = Device::new_cuda(0)?;
    let images = images.to_device(&device)?;

    let mut varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = Model::new(vs.clone())?;
    varmap.load(path)?;

    let logits = model.forward(&images, false)?;
    let preds = logits.argmax(D::Minus1)?;
    println!("Predictions: {:?}", preds.to_vec1::<u32>()?);

    Ok(preds)
}

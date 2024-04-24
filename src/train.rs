use crate::model::Model;
use candle::{DType, Device, Result, D};
use candle_datasets::vision::Dataset;
use candle_nn::{loss, ops, AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use rand::prelude::*;

pub struct TrainingArgs {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub save: Option<String>,
}

pub fn train_loop(m: Dataset, args: TrainingArgs) -> Result<()> {
    let device = Device::new_cuda(0)?;

    let train_labels = m.train_labels.to_dtype(DType::U32)?.to_device(&device)?;
    let train_images = m.train_images.to_device(&device)?;
    let test_images = m.test_images.to_device(&device)?;
    let test_labels = m.test_labels.to_dtype(DType::U32)?.to_device(&device)?;

    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = Model::new(vs.clone())?;

    let adamw_params = ParamsAdamW {
        lr: args.learning_rate,
        ..Default::default()
    };
    let mut opt = AdamW::new(varmap.all_vars(), adamw_params)?;

    let n_batches = train_labels.dim(0)? / args.batch_size;
    let mut batch_idxs = (0..n_batches).collect::<Vec<_>>();

    for epoch in 1..=args.epochs {
        let mut sum_loss = 0.0;
        batch_idxs.shuffle(&mut thread_rng());
        for batch_idx in batch_idxs.iter() {
            let start = batch_idx * args.batch_size;
            let batch_images = train_images.narrow(0, start, args.batch_size)?;
            let batch_labels = train_labels.narrow(0, start, args.batch_size)?;
            let logits = model.forward(&batch_images, true)?;
            let log_sm = ops::log_softmax(&logits, D::Minus1)?;
            let loss = loss::nll(&log_sm, &batch_labels)?;
            opt.backward_step(&loss)?;
            sum_loss += loss.to_vec0::<f32>()?;
        }

        let avg_loss = sum_loss / n_batches as f32;
        let test_logits = model.forward(&test_images, false)?;
        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(&test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let accuracy = sum_ok / test_labels.dims1()? as f32;
        println!(
            "Epoch: {}, Loss: {}, Accuracy: {}",
            epoch, avg_loss, accuracy
        );
    }
    if let Some(save) = &args.save {
        varmap.save(&save)?;
    }
    Ok(())
}

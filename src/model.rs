use candle::{Result, Tensor};
use candle_nn::{Conv2d, Dropout, Linear, ModuleT, VarBuilder};

const LABELS: usize = 10;

pub struct Model {
    conv1: Conv2d,
    conv2: Conv2d,
    fc1: Linear,
    fc2: Linear,
    dropout: candle_nn::Dropout,
}

impl Model {
    pub fn new(vs: VarBuilder) -> Result<Self> {
        let conv1 = candle_nn::conv2d(1, 32, 5, Default::default(), vs.pp("c1"))?;
        let conv2 = candle_nn::conv2d(32, 64, 5, Default::default(), vs.pp("c2"))?;
        let fc1 = candle_nn::linear(1024, 1024, vs.pp("fc1"))?;
        let fc2 = candle_nn::linear(1024, LABELS, vs.pp("fc2"))?;
        let dropout = Dropout::new(0.5);
        Ok(Self {
            conv1,
            conv2,
            fc1,
            fc2,
            dropout,
        })
    }

    pub fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let (b_size, _img_dim) = xs.dims2()?;
        let xs = xs
            .reshape((b_size, 1, 28, 28))?
            .apply(&self.conv1)?
            .max_pool2d(2)?
            .apply(&self.conv2)?
            .max_pool2d(2)?
            .flatten_from(1)?
            .apply(&self.fc1)?
            .relu()?;
        self.dropout.forward_t(&xs, train)?.apply(&self.fc2)
    }
}

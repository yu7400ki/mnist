use candle::Result;
use candle_datasets::vision::mnist;
use clap::{Args, Parser, Subcommand};
use train::{train_loop, TrainingArgs};

mod model;
mod pred;
mod train;

#[derive(Parser)]
struct Cli {
    #[clap(subcommand)]
    subcmd: SubCommand,
}

#[derive(Subcommand)]
enum SubCommand {
    Train(TrainArgs),
    Pred(PredArgs),
}

#[derive(Args)]
struct TrainArgs {
    #[clap(short, long)]
    epochs: usize,
    #[clap(short, long)]
    save: Option<String>,
}

#[derive(Args)]
struct PredArgs {
    #[clap(short, long)]
    model: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.subcmd {
        SubCommand::Train(args) => {
            let m = mnist::load()?;
            println!("train-images: {:?}", m.train_images.shape());
            println!("train-labels: {:?}", m.train_labels.shape());
            println!("test-images: {:?}", m.test_images.shape());
            println!("test-labels: {:?}", m.test_labels.shape());

            let training_args = TrainingArgs {
                epochs: args.epochs,
                batch_size: 64,
                learning_rate: 0.001,
                save: args.save,
            };
            train_loop(m, training_args)?;
        }
        SubCommand::Pred(args) => {
            let m = mnist::load()?;
            let test_iamges = m.test_images.narrow(0, 0, 10)?;
            let test_labels = m.test_labels.narrow(0, 0, 10)?;
            pred::predict(args.model, &test_iamges)?;
            println!("Labels: {:?}", test_labels.to_vec1::<u8>()?);
        }
    }

    Ok(())
}

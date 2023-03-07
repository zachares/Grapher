""" Script for training and testing text to graph generation models """

import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning import loggers as pl_loggers

from data.data_loader import init_dataloader
from data.graph_tokenizer import TokenizerFactory, GraphTokenizer
from data.preprocess_data import prepareWebNLG
from model.model_loader import init_and_load_model


def main(args):
    args.output_path = os.path.join(args.data_path, 'processed')
    tokenizer = GraphTokenizer(
        tokenizer=TokenizerFactory[args.language_model_type].value.from_pretrained(
            pretrained_model_name_or_path=args.language_model_name,
            cache_dir=args.model_dir
        )
    )
    prepareWebNLG(
        tokenizer=tokenizer,
        source_path=args.data_path,
        target_path=args.output_path,
        split_names=['train', 'dev', 'test']
    )
    model = init_and_load_model(
        tokenizer=tokenizer,
        language_model_type=args.language_model_type,
        language_model_name=args.language_model_name,
        model_dir=args.model_dir,
        learning_rate=args.learning_rate,
        load_model=args.load_model
    )
    logger = pl_loggers.TensorBoardLogger(
        save_dir=args.logging_dir,
        name='',
        version=f"{args.dataset}_version_{args.version}",
        default_hp_metric=False
    )
    if args.run == 'train':
        train_dataloader = init_dataloader(
            tokenizer=tokenizer,
            split_name='train',
            shuffle_data=True,
            data_path=args.output_path,
            batch_size=args.batch_size,
            num_data_workers=args.num_data_workers
        )
        val_dataloader = init_dataloader(
            tokenizer=tokenizer,
            split_name='dev',
            shuffle_data=False,
            data_path=args.output_path,
            batch_size=args.batch_size,
            num_data_workers=args.num_data_workers
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.model_dir,
            filename=args.model_name,
            save_last=True,
            save_top_k=-1,
            save_weights_only=True,
            every_n_epochs=1,
        )
        trainer = pl.Trainer.from_argparse_args(
            args,
            logger=logger,
            callbacks=[checkpoint_callback, RichProgressBar(10)],
            val_check_interval=1.0,
            check_val_every_n_epoch=1,
            devices=args.gpu_devices,
            accelerator="gpu"
        )
        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=None
        )

    elif args.run == 'test':
        test_dataloader = init_dataloader(
            tokenizer=tokenizer,
            split_name='dev',
            shuffle_data=True,
            data_path=args.output_path,
            batch_size=args.batch_size,
            num_data_workers=args.num_data_workers
        )
        trainer = pl.Trainer.from_argparse_args(
            args,
            logger=logger,
            devices=args.gpu_devices,
            accelerator="gpu"
        )
        trainer.test(model=model, dataloaders=test_dataloader)


if __name__ == "__main__":

    # Parsing arguments
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("--logging-dir", type=str, required=True)
    parser.add_argument("--eval_dump_only", type=int, default=0)

    parser.add_argument("--language-model-type", type=str, default='t5')
    parser.add_argument('--language-model-name', type=str, default='t5-large')
    parser.add_argument('--model-dir', type=str, required=True)
    parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument("--load-model", type=bool, default=True)

    parser.add_argument("--dataset", type=str, default='webnlg')
    parser.add_argument('--version', type=str, default='0')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--num_data_workers', type=int, default=3)

    parser.add_argument("--run", type=str, default='train')
    parser.add_argument("--gpu-devices", nargs="*", type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=55)
    parser.add_argument('--learning-rate', default=1e-5, type=float)

    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    main(args)

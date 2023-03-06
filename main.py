import argparse
import enum

from data.data_loader import GraphDataModule
from pytorch_lightning import loggers as pl_loggers

from model.litgrapher import LitGrapher
import pytorch_lightning as pl
from transformers import T5Tokenizer
import os
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from misc.utils import decode_graph

from data.data_loader import init_dataloader
from data.preprocess_data import prepareWebNLG
from data.graph_tokenizer import TokenizerFactory, GraphTokenizer
from model.model_loader import init_model, load_model


def main(args):
    args.output_path = os.path.join(args.data_path, 'processed')
    tokenizer = GraphTokenizer(
        tokenizer=TokenizerFactory[args.language_model_type].value.from_pretrained(
            pretrained_model_name_or_path=args.language_model_name,
            cache_dir=args.cache_dir
        )
    )
    prepareWebNLG(
        tokenizer=tokenizer,
        source_path=args.data_path,
        target_path=args.output_path,
        split_names=['train', 'dev', 'test']
    )
    model = load_model(args.model_dir) if args.load_model else init_model(
        tokenizer=GraphTokenizer,
        language_model_type=args.language_model_type,
        language_model_name=args.language_model_name,
        model_dir=args.model_dir,
        learning_rate=args.learning_rate
    )
    logger = pl_loggers.TensorBoardLogger(
        save_dir=args.default_root_dir,
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
            shuffle_data=True,
            data_path=args.output_path,
            batch_size=args.batch_size,
            num_data_workers=args.num_data_workers
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename='model-{step}',
            save_last=True,
            save_top_k=-1,
            every_n_train_steps=args.checkpoint_step_frequency,
        )
        trainer = pl.Trainer.from_argparse_args(
            args,
            logger=logger,
            callbacks=[checkpoint_callback, RichProgressBar(10)],
            val_check_interval=1.0,
            check_val_every_n_epoch=1,
            devices=[0,1,2,3],
            accelerator="gpu"
        )
        trainer.fit(model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader)

    elif args.run == 'test':
        model = load_model(args.model_dir)
        test_dataloader = init_dataloader(
            tokenizer=tokenizer,
            split_name='dev',
            shuffle_data=True,
            data_path=args.output_path,
            batch_size=args.batch_size,
            num_data_workers=args.num_data_workers
        )

        dm = GraphDataModule(tokenizer_class=T5Tokenizer,
                             tokenizer_name=grapher.transformer_name,
                             cache_dir=grapher.cache_dir,
                             data_path=args.data_path,
                             dataset=args.dataset,
                             batch_size=args.batch_size,
                             num_data_workers=args.num_data_workers,
                             max_nodes=grapher.max_nodes,
                             max_edges=grapher.max_edges,
                             edges_as_classes=grapher.edges_as_classes)

        dm.setup(stage='test')

        trainer = pl.Trainer.from_argparse_args(args, logger=TB, devices=[3], accelerator="gpu")

        trainer.test(grapher, datamodule=dm)

    else: # single inference

        assert os.path.exists(checkpoint_model_path), 'Provided checkpoint does not exists, cannot do inference'

        grapher = LitGrapher.load_from_checkpoint(checkpoint_path=checkpoint_model_path)

        tokenizer = T5Tokenizer.from_pretrained(grapher.transformer_name, cache_dir=grapher.cache_dir)
        tokenizer.add_tokens('__no_node__')
        tokenizer.add_tokens('__no_edge__')
        tokenizer.add_tokens('__node_sep__')

        text_tok = tokenizer([args.inference_input_text],
                             add_special_tokens=True,
                             padding=True,
                             return_tensors='pt')

        text_input_ids, mask = text_tok['input_ids'], text_tok['attention_mask']

        _, seq_nodes, _, seq_edges = grapher.model.sample(text_input_ids, mask)

        dec_graph = decode_graph(tokenizer, grapher.edge_classes, seq_nodes, seq_edges, grapher.edges_as_classes,
                                grapher.node_sep_id, grapher.max_nodes, grapher.noedge_cl, grapher.noedge_id,
                                grapher.bos_token_id, grapher.eos_token_id)

        graph_str = ['-->'.join(tri) for tri in dec_graph[0]]

        print(f'Generated Graph: {graph_str}')


if __name__ == "__main__":

    # Parsing arguments
    parser = argparse.ArgumentParser(description='Arguments')

    parser.add_argument("--dataset", type=str, default='webnlg')
    parser.add_argument("--run", type=str, default='train')
    parser.add_argument('--pretrained_model', type=str, default='t5-large')
    parser.add_argument('--version', type=str, default='0')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--num_data_workers', type=int, default=3)
    parser.add_argument('--checkpoint_step_frequency', type=int, default=-1)
    parser.add_argument('--checkpoint_model_id', type=int, default=-1)
    parser.add_argument('--max_nodes', type=int, default=8)
    parser.add_argument('--max_edges', type=int, default=7)
    parser.add_argument('--default_seq_len_node', type=int, default=20)
    parser.add_argument('--default_seq_len_edge', type=int, default=20)
    parser.add_argument('--edges_as_classes', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument("--focal_loss_gamma", type=float, default=0.0)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--eval_dump_only", type=int, default=0)
    parser.add_argument("--inference_input_text", type=str,
                        default='Danielle Harris had a main role in Super Capers, a 98 minute long movie.')
    parser.add_argument("--language-model", type=str, default='t5')

    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    main(args)

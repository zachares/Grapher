from data.dataset import GraphDataModule
from pytorch_lightning import loggers as pl_loggers
import argparse
from model.litgrapher import LitGrapher
import pytorch_lightning as pl
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from misc.utils import decode_graph

def main(args):

    args.eval_dir = os.path.join(args.default_root_dir, args.dataset + '_version_' + args.version)
    args.checkpoint_dir = os.path.join(args.eval_dir, 'checkpoints')

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(args.eval_dir, 'valid'), exist_ok=True)
    os.makedirs(os.path.join(args.eval_dir, 'test'), exist_ok=True)

    TB = pl_loggers.TensorBoardLogger(save_dir=args.default_root_dir, name='', version=args.dataset + '_version_' + args.version, default_hp_metric=False)

    if args.checkpoint_model_id < 0:
        checkpoint_model_path = os.path.join(args.checkpoint_dir, 'last.ckpt')
    else:
        checkpoint_model_path = os.path.join(args.checkpoint_dir, f"model-step={args.checkpoint_model_id}.ckpt")

    if args.run == 'train':
        dm = GraphDataModule(tokenizer_class=T5Tokenizer,
                             tokenizer_name=args.pretrained_model,
                             cache_dir=args.cache_dir,
                             data_path=args.data_path,
                             dataset=args.dataset,
                             batch_size=args.batch_size,
                             num_data_workers=args.num_data_workers,
                             max_nodes=args.max_nodes,
                             max_edges=args.max_edges,
                             edges_as_classes=args.edges_as_classes)

        dm.prepare_data()
        dm.setup(stage='fit')
        dm.setup(stage='validate')

        checkpoint_callback = ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename='model-{step}',
            save_last=True,
            save_top_k=-1,
            every_n_train_steps=args.checkpoint_step_frequency,
        )

        grapher = LitGrapher(
            transformer_class=T5ForConditionalGeneration,
            transformer_name=args.pretrained_model,
            tokenizer=dm.tokenizer,
            cache_dir=args.cache_dir,
            max_nodes=args.max_nodes,
            max_edges=args.max_edges,
            edges_as_classes=args.edges_as_classes,
            default_seq_len_edge=args.default_seq_len_edge,
            num_classes=len(dm.dataset_train.edge_classes),
            dropout_rate=args.dropout_rate,
            num_layers=args.num_layers,
            vocab_size=len(dm.tokenizer.get_vocab()),
            bos_token_id=dm.tokenizer.pad_token_id,
            eos_token_id=dm.tokenizer.eos_token_id,
            nonode_id=dm.tokenizer.convert_tokens_to_ids('__no_node__'),
            noedge_id=dm.tokenizer.convert_tokens_to_ids('__no_edge__'),
            node_sep_id=dm.tokenizer.convert_tokens_to_ids('__node_sep__'),
            noedge_cl=len(dm.dataset_train.edge_classes) - 1,
            edge_classes=dm.dataset_train.edge_classes,
            focal_loss_gamma=args.focal_loss_gamma,
            eval_dir=args.eval_dir,
            lr=args.lr,
            node_token=dm.node_token,
            edge_token=dm.edge_token,
            no_edge_token=dm.no_edge_token
        )

        if not os.path.exists(checkpoint_model_path):
            checkpoint_model_path = None

        trainer = pl.Trainer.from_argparse_args(
            args,
            logger=TB,
            callbacks=[checkpoint_callback, RichProgressBar(10)],
            val_check_interval=100,
            check_val_every_n_epoch=1,
            devices=[0,1,2,3],
            accelerator="gpu"
        )

        dm.setup(stage='validate')

        trainer.fit(model=grapher, datamodule=dm, ckpt_path=checkpoint_model_path)

    elif args.run == 'test':

        assert os.path.exists(checkpoint_model_path), 'Provided checkpoint does not exists, cannot run the test'

        grapher = LitGrapher.load_from_checkpoint(checkpoint_path=checkpoint_model_path)

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

    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    main(args)

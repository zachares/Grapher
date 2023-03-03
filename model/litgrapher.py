import torch
import pytorch_lightning as pl
from misc.utils import compute_loss, decode_text, decode_graph, compute_scores
import torch.nn.functional as F
import torch.nn as nn
import os
from model.grapher import Grapher


def sorted_ls(path):
    mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
    return list(sorted(os.listdir(path), key=mtime))


class FocalLoss(nn.modules.loss._WeightedLoss):

    def __init__(self, gamma, weight=None):
        super(FocalLoss, self).__init__(weight)
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):

        # input: batch_size x vocab_size x d1 x ... x dn x seq_len
        # target: batch_size x d1 x ... x dn x seq_len
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction='none')
        p = torch.exp(-ce_loss)
        focal_loss = ((1 - p) ** self.gamma * ce_loss)

        return focal_loss

class LitGrapher(pl.LightningModule):
    def __init__(self,
                 transformer_class,
                 transformer_name,
                 tokenizer,
                 cache_dir,
                 max_nodes,
                 max_edges,
                 edges_as_classes,
                 default_seq_len_edge,
                 num_classes,
                 dropout_rate,
                 num_layers,
                 vocab_size,
                 bos_token_id,
                 eos_token_id,
                 nonode_id,
                 noedge_id,
                 node_sep_id,
                 noedge_cl,
                 edge_classes,
                 focal_loss_gamma,
                 eval_dir,
                 lr,
                 node_token,
                 edge_token,
                 no_edge_token
                 ):
        super().__init__()
        self.save_hyperparameters()

        model = Grapher(
            transformer_class=transformer_class,
            transformer_name=transformer_name,
            cache_dir=cache_dir
        )

        self.model = model
        self.criterion = {'ce': nn.CrossEntropyLoss(reduction='none'), 'focal': FocalLoss(focal_loss_gamma)}
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.transformer_name = transformer_name
        self.edges_as_classes = edges_as_classes
        self.edge_classes = edge_classes
        self.focal_loss_gamma = focal_loss_gamma
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.node_sep_id = node_sep_id
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.noedge_cl = noedge_cl
        self.nonode_id = nonode_id
        self.noedge_id = noedge_id
        self.eval_dir=eval_dir
        self.lr = lr
        self.node_token = node_token
        self.edge_token = edge_token
        self.no_edge_token = no_edge_token

    def training_step(self, batch, batch_idx):

        # target_nodes: batch_size X seq_len_node
        # target_edges: num_nodes X num_nodes X batch_size X seq_len_edge [FULL]
        # target_edges: batch_size X num_nodes X num_nodes [CLASSES]
        text_input_ids, text_input_attn_mask, target_edges, target_edges_mask = batch

        # logits_nodes: batch_size X seq_len_node X vocab_size
        # logits_edges: num_nodes X num_nodes X batch_size X seq_len_edge X vocab_size [FULL]
        # logits_edges: num_nodes X num_nodes X batch_size X num_classes [CLASSES]
        logits_edges= self.model(
            text_input_ids,
            text_input_attn_mask,
            target_edges,
            target_edges_mask
        )
        loss = compute_loss(self.criterion, logits_edges, target_edges, target_edges_mask)

        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True, batch_size=text_input_ids.size(0))

        return loss

    def eval_step(self, batch, batch_idx, split):

        iteration = self.global_step
        rank = self.global_rank

        text_input_ids, text_input_attn_mask, target_edge_ids, _ = batch

        logits_edges, generated_edge_ids = self.model.sample(text_input_ids, text_input_attn_mask)

        target_text_dec = decode_text(
            self.tokenizer,
            target_edge_ids,
            self.bos_token_id,
            self.eos_token_id
        )
        pred_text_dec = decode_text(
            self.tokenizer,
            generated_edge_ids,
            self.bos_token_id,
            self.eos_token_id
        )

        dec_graph_target = decode_graph(
            target_text_dec,
            self.node_token,
            self.edge_token,
            self.no_edge_token
        )

        dec_graph_pred = decode_graph(
            pred_text_dec,
            self.node_token,
            self.edge_token,
            self.no_edge_token
        )
        decodes = {
            'text_dec': pred_text_dec,
            'dec_target': dec_graph_target,
            'dec_pred': dec_graph_pred
        }

        return decodes

    def eval_epoch_end(self, outputs, split):

        iteration = self.global_step
        rank = self.global_rank

        dec_target_all = []
        dec_pred_all = []

        for out in outputs:
            dec_target_all += out['dec_target']
            dec_pred_all += out['dec_pred']

        # make sure number of paths is smaller than 10
        dec_pred_all = [tr[:10] for tr in dec_pred_all]
        dec_target_all = [tr[:10] for tr in dec_target_all]

        # hack to avoid crashing the program if evaluation fails
        try:
            scores = compute_scores(dec_pred_all, dec_target_all, iteration, self.eval_dir, split, rank)
        except Exception:
            return

        for k, v in scores.items():
            self.logger.experiment.add_scalar(f'{split}_score/{k}', v, global_step=iteration)

        self.log_dict(scores)

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'valid')

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'test')

    def validation_epoch_end(self, outputs):
        self.eval_epoch_end(outputs, 'valid')

    def test_epoch_end(self, outputs):
        self.eval_epoch_end(outputs, 'test')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), self.lr)
        return optimizer

import torch
import torch.nn.functional as F
from torch import nn
from transformers import T5Tokenizer, T5ForConditionalGeneration


class Grapher(nn.Module):
    def __init__(
        self,
        transformer_class: T5ForConditionalGeneration,
        transformer_name: str,
        cache_dir: str
    ):
        super().__init__()

        self.transformer = transformer_class.from_pretrained(transformer_name, cache_dir=cache_dir)

    # def split_nodes(self, output_ids, features):

    #     # features: batch_size x seq_len x hidden_dim
    #     # output_ids: batch_size x seq_len

    #     batch_size, _ = output_ids.size()
    #     split_features = torch.zeros((self.max_nodes, batch_size, self.hidden_dim), device=features.device, dtype=features.dtype)  # num_nodes X batch_size X hidden_dim

    #     for n in range(self.max_nodes):
    #         mask_node_n = ((torch.cumsum((output_ids == self.node_sep_id), 1) == n) & (output_ids != self.node_sep_id)).unsqueeze(2)
    #         features_node_n = features*mask_node_n
    #         sum_features_node_n = torch.cumsum(features_node_n, 1)[:, -1]
    #         num_tokens_node_n = torch.sum(mask_node_n, 1)
    #         num_tokens_node_n[num_tokens_node_n == 0] = 1
    #         ave_features_node_n = sum_features_node_n / num_tokens_node_n
    #         split_features[n] = ave_features_node_n

    #     return split_features

    def forward(
        self,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        target_edges: torch.Tensor,
        target_edges_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.transformer(
            input_ids=text,
            attention_mask=text_mask,
            decoder_input_ids=target_edges,
            decoder_attention_mask=target_edges_mask
        ).logits

    def sample(self, text, text_mask):

        output = self.transformer.generate(
            input_ids=text,
            max_length=150,
            attention_mask=text_mask,
            output_hidden_states=True,
            output_scores=True,
            return_dict_in_generate=True
        )

        seq_nodes = output.sequences[:, 1:]

        logits_nodes = output.scores  # list of seq_len of batch_size x vocab_size
        logits_nodes = torch.cat([l.unsqueeze(0) for l in logits_nodes], 0).permute(1, 0, 2)
        return logits_nodes, seq_nodes


# class EdgesGen(nn.Module):
#     def __init__(self, hidden_dim, vocab_size, bos_token):
#         super(EdgesGen, self).__init__()

#         self.vocab_size = vocab_size
#         self.bos_token = bos_token
#         self.edgeDecoder = GRUDecoder(hidden_dim, vocab_size)

#     def forward(self, features, seq_len):

#         # features: num_nodes X batch_size X hidden_dim

#         device = features.device

#         num_nodes = features.size(0)
#         batch_size = features.size(1)
#         hidden_dim = features.size(2)

#         all_logits = torch.zeros(seq_len, num_nodes * num_nodes * batch_size, self.vocab_size, device=device)

#         input = torch.ones(num_nodes * num_nodes * batch_size, dtype=torch.long, device=device) * self.bos_token

#         # num_nodes X num_nodes X batch_size X hidden_dim
#         feats = features.unsqueeze(0).expand(num_nodes, -1, -1, -1)

#         # num_nodes*num_nodes*batch_size X hidden_dim
#         hidden = (feats.permute(1, 0, 2, 3) - feats).reshape(-1, hidden_dim).contiguous()

#         # set first token in output
#         all_logits[0, :, input] = 1.0

#         for t in range(1, seq_len):
#             output, hidden = self.edgeDecoder(input, hidden)
#             all_logits[t] = output
#             input = output.max(1)[1]

#         # num_nodes X num_nodes X batch_size X seq_len X vocab_size
#         all_logits = all_logits.reshape(seq_len, num_nodes, num_nodes, batch_size, -1).permute(1, 2, 3, 0, 4)

#         return all_logits


# class GRUDecoder(nn.Module):

#     def __init__(self, hidden_size, vocab_size):
#         super(GRUDecoder, self).__init__()

#         self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, num_layers=1)
#         self.out = nn.Linear(hidden_size, vocab_size)

#         self.embedding = nn.Embedding(vocab_size, hidden_size)

#     def forward(self, x, hidden):

#         # x: bsize
#         # hidden: bsize X  hidden_dim or 1 X bsize X hidden_dim
#         if len(hidden.size()) == 2:
#             hidden = hidden.unsqueeze(0)  # to imitate num_layers=1

#         emb_input = self.embedding(x)

#         if len(x.size()) == 1:
#             emb_input = emb_input.unsqueeze(1)  # bsize X 1 X emb_dim
#         # else bsize X sent_len X emb_dim

#         output = F.relu(emb_input)
#         output, hidden = self.gru(output, hidden)
#         output = self.out(output.squeeze())  # bsize X vocab_size OR bsize X sent_len X vocab_size

#         return output, hidden


# class EdgesClass(nn.Module):
#     def __init__(self, hidden_dim, num_classes, dropout_rate=0.5, num_layers=0):
#         super(EdgesClass, self).__init__()

#         self.hidden_dim = hidden_dim
#         self.num_classes = num_classes
#         self.layers = nn.Sequential()

#         dim = num_classes
#         self.layers.add_module('first', nn.Linear(hidden_dim, dim))
#         self.layers.add_module('firstrelu', nn.ReLU())
#         self.layers.add_module('firstdropout', nn.Dropout(dropout_rate))
#         for l in range(num_layers):
#             self.layers.add_module(f'lin{l}', nn.Linear(dim, dim))
#             self.layers.add_module(f'relu{l}', nn.ReLU())
#             self.layers.add_module(f'dropout{l}', nn.Dropout(dropout_rate))
#         self.layers.add_module('last', nn.Linear(dim, num_classes))

#     def forward(self, features):

#         # features: num_nodes X batch_size X hidden_dim

#         num_nodes = features.size(0)
#         batch_size = features.size(1)

#         # num_nodes_valid X num_nodes_valid X batch_size X hidden_dim
#         feats = features.unsqueeze(0).expand(num_nodes, -1, -1, -1)

#         # [featurs[i] - features[j]]: num_nodes_valid*num_nodes_valid*batch_size X hidden_dim
#         hidden = (feats.permute(1, 0, 2, 3) - feats).reshape(-1, self.hidden_dim)

#         # logits: num_nodes_valid*num_nodes_valid*batch_size X num_classes
#         logits = self.layers(hidden)

#         # num_nodes X num_nodes X batch_size X num_classes
#         all_logits = logits.reshape(num_nodes, num_nodes, batch_size, -1)

#         return all_logits

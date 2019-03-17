import torch
import torch.nn as nn

from onmt.encoders.encoder import EncoderBase
from onmt.modules import MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward


class MultiTransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, heads, d_ff, dropout,
                 max_relative_positions=0):
        super(MultiTransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout,
            max_relative_positions=max_relative_positions
        )
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask, type="self")
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class MultiTransformerEncoder(EncoderBase):
    def __init__(self, n_langs, n_words, num_layers, share_enc, d_model, heads,
                 d_ff, dropout, embeddings, max_relative_positions):
        super(MultiTransformerEncoder, self).__init__()

        self.embeddings = embeddings
        self.transformer = nn.ModuleList()
        for k in range(num_layers):
            layer_is_shared = (k >= num_layers - share_enc)

            self.transformer.append(nn.ModuleList([
                MultiTransformerEncoderLayer(d_model, heads, d_ff, dropout,
                                             max_relative_positions=max_relative_positions)
            ]))

            for i in range(1, self.n_langs):
                if layer_is_shared:
                    self.transformer[k].append(self.transformer[k][0])
                else:
                    self.transformer[k].append(MultiTransformerEncoderLayer(d_model, heads, d_ff, dropout,
                                                                            max_relative_positions=max_relative_positions))
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    @classmethod
    def from_opt(cls, opt, embeddings):
        return cls(
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.dropout,
            embeddings,
            opt.max_relative_positions
        )

    def forward(self, src, src_langid, lengths=None):
        self._check_args(src, lengths)

        emb = self.embeddings[src_langid](src)
        
        out = emb.transpose(0, 1).contiguous()
        words = src[:, :, 0].transpose(0, 1)
        w_batch, w_len = words.size()
        padding_idx = self.embeddings[src_langid].word_padding_idx
        mask = words.data.eq(padding_idx).unsqueeze(1)  # [B, 1, T]
        for layer in self.transformer:
            out = layer[src_langid](out, mask)
        out = self.layer_norm(out)

        return emb, out.transpose(0, 1).contiguous(), lengths
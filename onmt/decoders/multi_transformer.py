import torch
import torch.nn as nn

from onmt.decoders.decoder import DecoderBase
from onmt.modules import MultiHeadedAttention, AverageAttention
from onmt.modules.position_ffn import PositionwiseFeedForward


class MultiTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout,
                 self_attn_type="scaled-dot", max_relative_positions=0):
        super(MultiTransformerDecoderLayer, self).__init__()

        if self_attn_type == "scaled-dot":
            self.self_attn = MultiHeadedAttention(
                heads, d_model, dropout=dropout,
                max_relative_positions=max_relative_positions
            )
        elif self_attn_type == "average":
            self.self_attn = AverageAttention(d_model, dropout)

        self.context_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout
        )
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask,
                layer_cache=None, step=None):
        dec_mask = None
        if step is None:
            tgt_len = tgt_pad_mask.size(-1)
            future_mask = torch.ones(
                [tgt_len, tgt_len],
                device=tgt_pad_mask.device,
                dtype=torch.uint8
            )
            future_mask = future_mask.triu_(1).view(1, tgt_len, tgt_len)
            dec_mask = torch.gt(tgt_pad_mask + future_mask, 0)

        input_norm = self.layer_norm_1(inputs)

        if isinstance(self.self_attn, MultiHeadedAttention):
            query, attn = self.self_attn(input_norm, input_norm, input_norm,
                                         mask=dec_mask,
                                         layer_cache=layer_cache,
                                         type="self")
        elif isinstance(self.self_attn, AverageAttention):
            query, attn = self.self_attn(input_norm, mask=dec_mask,
                                         layer_cache=layer_cache, step=step)

        query = self.drop(query) + inputs

        query_norm = self.layer_norm_2(query)
        mid, attn = self.context_attn(memory_bank, memory_bank, query_norm,
                                      mask=src_pad_mask,
                                      layer_cache=layer_cache,
                                      type="context")
        output = self.feed_forward(self.drop(mid) + query)

        return output, attn


class MultiTransformerDecoder(DecoderBase):

    def __init__(self, n_langs, n_words, num_layers, share_dec, d_model, heads,
                 d_ff, copy_attn, self_attn_type, dropout, embeddings,
                 max_relative_positions):
        super(MultiTransformerDecoder, self).__init__()

        self.n_langs = n_langs
        self.n_words = n_words
        self.embeddings = embeddings

        self.transformer_layers = nn.ModuleList()
        for k in range(num_layers):
            layer_is_shared = (k < share_dec)

            self.transformer.append(nn.ModuleList([
                MultiTransformerDecoderLayer(d_model, heads, d_ff, dropout,
                                             max_relative_positions=max_relative_positions)
            ]))

            for i in range(1, self.n_langs):
                if layer_is_shared:
                    self.transformer_layers[k].append(self.transformer_layers[k][0])
                else:
                    self.transformer_layers[k].append(
                        MultiTransformerDecoderLayer(d_model, heads, d_ff, dropout,
                                                     max_relative_positions=max_relative_positions))

        self._copy = copy_attn
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    @classmethod
    def from_opt(cls, opt, embeddings):
        return cls(
            opt.dec_layers,
            opt.dec_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.copy_attn,
            opt.self_attn_type,
            opt.dropout,
            embeddings,
            opt.max_relative_positions
        )

    def init_state(self, src, memory_bank, enc_hidden):
        self.state["src"] = src
        self.state["cache"] = None

    def map_state(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.item():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        self.state["src"] = fn(self.state["src"], 1)
        if self.state["cache"] is not None:
            _recursive_map(self.state["cache"])

    def detach_state(self):
        self.state["src"] = self.state["src"].detach()

    def forward(self, tgt, tgt_langid, memory_bank, step=None, **kwargs):

        if step == 0:
            self._init_cache(memory_bank)

        src = self.state["src"]
        src_words = src[:, :, 0].transpose(0, 1)
        tgt_words = tgt[:, :, 0].transpose(0, 1)
        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()

        emb = self.embeddings[tgt_langid](tgt, step=step)
        assert emb.dim() == 3

        output = emb.transpose(0, 1).contiguous()
        src_memory_bank = memory_bank.transpose(0, 1).contiguous()

        pad_idx = self.embeddings[tgt_langid].word_padding_idx
        src_pad_mask = src_words.data.eq(pad_idx).unsqueeze(1)
        tgt_pad_mask = tgt_words.data.eq(pad_idx).unsqueeze(1)

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = self.state["cache"]["layer_{}".format(i)] \
                            if step is not None else None
            output, attn = layer[tgt_langid](
                output,
                src_memory_bank,
                src_pad_mask,
                tgt_pad_mask,
                layer_cache=layer_cache,
                step=step
            )

        output = self.layer_norm(output)
        dec_outs = output.transpose(0, 1).contiguous()
        attn = attn.transpose(0, 1).contiguous()

        attns = {"std": attn}
        if self._copy:
            attns["copy"] = attn

        return dec_outs, attns

    def _init_cache(self, memory_bank):
        self.state["cache"] = {}
        batch_size = memory_bank.size(1)
        depth = memory_bank.size(-1)

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = {"memory_keys": None, "memory_values": None}
            if isinstance(layer.self_attn, AverageAttention):
                layer_cache["prev_g"] = torch.zeros((batch_size, 1, depth))
            else:
                layer_cache["self_keys"] = None
                layer_cache["self_values"] = None
            self.state["cache"]["layer_{}".format(i)] = layer_cache

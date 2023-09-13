import torch.nn
import torch.nn as nn

from module import encoder, common, initializer
from util import consts
from . import config
from ..model_meta import MetaType, model

import logging
logger = logging.getLogger(__name__)


@model("meta_sasrec", MetaType.ModelBuilder)
class SasRec(nn.Module):
    def __init__(self, model_conf):
        super(SasRec, self).__init__()

        assert isinstance(model_conf, config.ModelConfig)

        self._position_embedding = encoder.IDEncoder(
            model_conf.id_vocab,
            model_conf.id_dimension
        )

        self._id_encoder = encoder.IDEncoder(
            model_conf.id_vocab,
            model_conf.id_dimension
        )

        self._target_trans = common.StackedDense(
            model_conf.id_dimension, [model_conf.id_dimension] * 2, [torch.nn.Tanh, None]
        )
        self._seq_trans = common.StackedDense(
            model_conf.id_dimension, [model_conf.id_dimension], [torch.nn.Tanh]
        )

        self._transformer = nn.TransformerEncoderLayer(
            d_model=model_conf.id_dimension,
            nhead=model_conf.nhead,
            dim_feedforward=4*model_conf.id_dimension,
            dropout=0
        )

        initializer.default_weight_init(self._transformer.self_attn.in_proj_weight)
        initializer.default_weight_init(self._transformer.self_attn.out_proj.weight)
        initializer.default_bias_init(self._transformer.self_attn.in_proj_bias)
        initializer.default_bias_init(self._transformer.self_attn.out_proj.bias)

        initializer.default_weight_init(self._transformer.linear1.weight)
        initializer.default_bias_init(self._transformer.linear1.bias)
        initializer.default_weight_init(self._transformer.linear2.weight)
        initializer.default_bias_init(self._transformer.linear2.bias)

        self._meta_classifier_param_list = common.HyperNetwork_FC(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None],
            batch=True,
            model_conf=model_conf
        )

    def forward(self, features, fig1=False):
        # Encode target item
        trigger_embed = self._id_encoder(features[consts.FIELD_TRIGGER_SEQUENCE])
        trigger_embed = self._seq_trans(trigger_embed)

        # B * D
        target_embed = self._id_encoder(features[consts.FIELD_TARGET_ID])
        target_embed = self._target_trans(target_embed)

        # Encode user historical behaviors
        with torch.no_grad():
            click_seq = features[consts.FIELD_CLK_SEQUENCE]
            batch_size = int(click_seq.shape[0])
            # B * L
            positions = torch.arange(0, int(click_seq.shape[1]), dtype=torch.int32).to(click_seq.device)
            positions = torch.tile(positions.unsqueeze(0), [batch_size, 1])
            mask = torch.not_equal(click_seq, 0)
            # B
            seq_length = torch.maximum(torch.sum(mask.to(torch.int32), dim=1) - 1, torch.Tensor([0]).to(device=mask.device))
            seq_length = seq_length.to(torch.long)

            trigger_mask = torch.not_equal(features[consts.FIELD_TRIGGER_SEQUENCE], 0).to(dtype=torch.int32)
            # B
            trigger_seq_length = torch.maximum(torch.sum(trigger_mask, dim=1) - 1,
                                               torch.Tensor([0]).to(device=trigger_mask.device))
            trigger_seq_length = trigger_seq_length.to(torch.long)

        # B * L * D
        hist_embed = self._id_encoder(click_seq)
        hist_pos_embed = self._position_embedding(positions)
        hist_embed = self._seq_trans(hist_embed + hist_pos_embed)

        atten_embed = self._transformer(
            torch.swapaxes(hist_embed, 0, 1)
        )
        user_state = torch.swapaxes(atten_embed, 0, 1)[range(batch_size), seq_length, :]

        user_embedding = self._meta_classifier_param_list(user_state, hist_embed,
                                         user_state.size()[0],
                                         seq_length)
        if fig1:
            user_embedding1 = self._meta_classifier_param_list(user_state, trigger_embed,
                                                              user_state.size()[0],
                                                              trigger_seq_length)
            output1 = torch.sum(user_embedding1 * target_embed, dim=1, keepdim=True)
            return output, mis_rec_pred, request_num, total_num, output1
        return torch.sum(user_embedding * target_embed, dim=1, keepdim=True)

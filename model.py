import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from allennlp.nn.util import batched_index_select
from transformers import BertPreTrainedModel, BertModel, BertConfig
from transformers.modeling_outputs import (
    MaskedLMOutput,
    SequenceClassifierOutput,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

from typing import Optional, Tuple, Union, List


def multilabel_onehot(multiple_labels, bs, num_labels, device=None):
    ones = torch.ones(bs, num_labels).to(multiple_labels).float()
    onehot = torch.zeros(bs, num_labels).to(ones).float()
    # i.e. onehot[b][multiple_labels[b, i]] = ones[b, i] = 1
    onehot.scatter_(dim=1, index=multiple_labels, src=ones)
    return onehot.float()


class BertTokenFusionPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear_a = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
        )
        self.gated_attention_w = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Sigmoid(),
        )
        self.transform = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
        )
        self.decoder = nn.Linear(config.hidden_size, config.num_idioms)

    def forward(self, masked_token_feat: torch.Tensor) -> torch.Tensor:
        gate_w = self.gated_attention_w(masked_token_feat)
        masked_token_feat = (
            gate_w * self.linear_a(masked_token_feat)
        ) + masked_token_feat  # [B, 4, F]
        fused_token_feat = self.transform(masked_token_feat.mean(dim=-2))  # [B, F]
        fused_prediction = self.decoder(fused_token_feat)  # [B, N_idiom]
        return fused_prediction

    def load_decoder_embeddings(self, embeddings: torch.Tensor):
        self.decoder.weight.data = embeddings
        self.decoder.bias.data.zero_()


# BertModel with additional prefix input
class BertPrefixModel(BertModel):
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        prefix_embeds: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)), device=device
            )

        if prefix_embeds is not None:
            attention_mask = torch.cat(
                [torch.ones((batch_size, prefix_embeds.size(0)), device=device), attention_mask], dim=1
            ).long()

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    batch_size, seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=device
                )

        # if prefix_embeds is not None:
        #     token_type_ids = torch.cat(
        #         [torch.zeros((batch_size, prefix_embeds.size(0)), device=device), token_type_ids], dim=1
        #     ).long()

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        # --- concat prefixLM embeddings
        if prefix_embeds is not None:
            if prefix_embeds.dim() == 2:
                prefix_embeds = prefix_embeds.unsqueeze(0).expand(
                    embedding_output.size(0), -1, -1
                )
            embedding_output = torch.cat(
                [prefix_embeds, embedding_output], dim=-2
            )  # [B, seq_len + prefix_len, F]

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        # if sequence_output
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class BertForChID(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertPrefixModel(config, add_pooling_layer=False)
        # self.bert = BertModel(config, add_pooling_layer=False)

        if self.config.use_prefixlm:
            self.prefix = nn.Parameter(
                torch.randn([self.config.prefix_len, self.config.hidden_size]), requires_grad=True
            )
        else:
            self.prefix = None

        self.cls = BertOnlyMLMHead(config)
        if self.config.use_fused_cls:
            self.fused_cls = BertTokenFusionPredictionHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     processor_class=_TOKENIZER_FOR_DOC,
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=MaskedLMOutput,
    #     config_class=_CONFIG_FOR_DOC,
    #     expected_output="'paris'",
    #     expected_loss=0.88,
    # )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        candidates: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        # idiom_labels: Optional[bool] = None,
        idiom_candidates: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels: torch.LongTensor of shape `(batch_size, )`
        candidates: torch.LongTensor of shape `(batch_size, num_choices, 4)`
        candidate_mask: torch.BooleanTensor of shape `(batch_size, seq_len)`
        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            prefix_embeds=self.prefix,
        )

        sequence_output = outputs[0]  # (Batch_size, Seq_len, F)

        if self.config.use_prefixlm:
            candidate_mask = torch.cat(
                [torch.zeros([candidate_mask.size(0), self.config.prefix_len], device=candidate_mask.device), candidate_mask], dim=1
            ).bool()

        if self.config.use_mlm:
            prediction_scores = self.cls(sequence_output)
            # (Batch_size, Seq_len, Vocab_size)
        elif self.config.use_fused_cls:
            masked_token_feat = torch.masked_select(
                sequence_output, candidate_mask.unsqueeze(-1)
            ).reshape(-1, 4, sequence_output.shape[-1])
            fused_scores = self.fused_cls(masked_token_feat)  # [B, N_idioms]
            prediction_scores = fused_scores
            idiom_candidate_mask = multilabel_onehot(
                idiom_candidates,
                bs=fused_scores.shape[0],
                num_labels=self.config.num_idioms,
            )  # [B, N_idioms]
            fused_scores += (idiom_candidate_mask - 1) * 1e6

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            if self.config.use_mlm:
                # --- MLM Loss
                
                candidate_prediction_scores = torch.masked_select(
                    prediction_scores, candidate_mask.unsqueeze(-1)
                ).reshape(
                    -1, prediction_scores.shape[-1], 1
                )  # (Batch_size x 4, Vocab_size, 1)
                candidate_indices = candidates.transpose(-1, -2).reshape(
                    -1, candidates.shape[1]
                )  # (Batch_size x 4, num_choices)
                candidate_logits = (
                    batched_index_select(candidate_prediction_scores, candidate_indices)
                    .squeeze(-1)
                    .reshape(prediction_scores.shape[0], 4, -1)
                    .transpose(-1, -2)
                )  # (Batch_size, num_choices, 4)

                candidate_labels = labels.reshape(labels.shape[0], 1).repeat(
                    1, 4
                )  # (Batch_size, 4)
                candidate_final_scores = torch.sum(
                    F.log_softmax(candidate_logits, dim=-2), dim=-1
                )  # (Batch_size, num_choices)

                loss = loss_fct(candidate_logits, candidate_labels)
            elif self.config.use_fused_cls:
                # --- Classification Loss
                loss = loss_fct(fused_scores, labels)
            else:
                raise NotImplementedError()

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        if self.config.use_mlm:
            return MaskedLMOutput(
                loss=loss,
                logits=candidate_final_scores,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        elif self.config.use_fused_cls:
            return SequenceClassifierOutput(
                loss=loss,
                logits=fused_scores,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            raise NotImplementedError()

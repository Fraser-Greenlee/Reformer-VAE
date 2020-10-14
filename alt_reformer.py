from transformers.modeling_reformer import (
    _get_least_common_mult_chunk_len,
    _get_min_chunk_len
)
from transformers import (
    ReformerModel
)


class AltReformerModel(ReformerModel):
    '''
        Putting initial start method in this class so it doesn't fill up the main one with copy & paste code.
    '''
    def start_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        num_hashes=None,
        past_buckets_states=None,
        use_cache=None,
        output_hidden_states=None,
        output_attentions=None,
        return_dict=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()  # noqa: F841
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]  # noqa: F841
            device = inputs_embeds.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        assert (
            len(input_shape) == 2
        ), "`input_ids` have be of shape `[batch_size, sequence_length]`, but got shape: {}".format(input_shape)

        if past_buckets_states is not None:
            assert not self.training, "`past_buckets_states` can only be used for inference, not for training`."

        # prepare head mask
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers, is_attention_chunked=True)

        # original sequence length for padding
        orig_sequence_length = input_shape[-1]

        # if needs padding
        least_common_mult_chunk_length = _get_least_common_mult_chunk_len(self.config)
        min_chunk_length = _get_min_chunk_len(self.config)

        must_pad_to_match_chunk_length = (
            input_shape[-1] % least_common_mult_chunk_length != 0
            and input_shape[-1] > min_chunk_length
            and past_buckets_states is None
        )

        if must_pad_to_match_chunk_length:
            padding_length = least_common_mult_chunk_length - input_shape[-1] % least_common_mult_chunk_length

            if self.training is True:
                raise ValueError(
                    "If training, sequence Length {} has to be a multiple of least common multiple chunk_length {}. Please consider padding the input to a length of {}.".format(
                        input_shape[-1], least_common_mult_chunk_length, input_shape[-1] + padding_length
                    )
                )

            # pad input
            input_ids, inputs_embeds, attention_mask, position_ids, input_shape = self._pad_to_mult_of_chunk_length(
                input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                input_shape=input_shape,
                padding_length=padding_length,
                padded_seq_length=least_common_mult_chunk_length,
                device=device,
            )

        # start index for postion encoding depends on incremental decoding
        if past_buckets_states is not None:
            start_idx_pos_encodings = past_buckets_states[0][1].shape[1]
        else:
            start_idx_pos_encodings = 0

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            start_idx_pos_encodings=start_idx_pos_encodings,
        )

        return embedding_output, orig_sequence_length, input_ids, inputs_embeds, attention_mask, must_pad_to_match_chunk_length

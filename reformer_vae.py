from dataclasses import dataclass
import logging
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (
    ReformerModelWithLMHead,
    ReformerEncoder,
    ReformerConfig,
    ReformerModelWithLMHeadOutput,
    ReformerModelOutput,
)
from alt_reformer import AltReformerModel


logger = logging.getLogger(__name__)


class LatentEncoderLargeTanh_1kLatent(nn.Module):
    def __init__(self, dim_m, set_input_size, latent_size):
        super().__init__()
        assert dim_m > 100
        self.shrink_tokens = nn.Linear(dim_m, 100)
        self.shrink_sequence = nn.Linear(100 * set_input_size, latent_size)
        self.tanh = nn.Tanh()

    def forward(self, encoding) -> torch.Tensor:
        batch_size = encoding.size(0)
        # shrink each tokens encoding
        encoding = self.shrink_tokens(encoding)
        encoding = self.shrink_sequence(encoding.view(batch_size, -1))
        return self.tanh(encoding)


class LatentDecoderLargeT5NormFF(nn.Module):
    def __init__(self, dim_m, set_input_size, latent_size):
        super().__init__()
        self.decode_latent = nn.Linear(latent_size, 10 * set_input_size)
        self.grow_sequence = nn.Linear(10 * set_input_size, 100 * set_input_size)
        self.grow_tokens = nn.Linear(100, dim_m)

    def forward(self, latent) -> torch.Tensor:
        batch_size = latent.size(0)
        # grow each tokens encoding
        latent = self.decode_latent(latent)
        latent = self.grow_sequence(latent)
        return self.grow_tokens(latent.view(batch_size, -1, 100))


class MMD_VAE(nn.Module):
    '''
        Runs an MMD_VAE on any given input.
        Pass a latent_predictor to have the model return an additional loss.
    '''
    def __init__(self, dim_model, seq_size, latent_size, latent_predictor=None):
        super().__init__()
        self.encoder = LatentEncoderLargeTanh_1kLatent(dim_model, seq_size, latent_size)
        self.decoder = LatentDecoderLargeT5NormFF(dim_model, seq_size, latent_size)
        self.latent_predictor = latent_predictor

    def forward(self, hidden, latent_pred_labels) -> torch.Tensor:
        latent = self.encoder(hidden)
        hidden = self.decoder(hidden)
        if self.latent_predictor:
            return hidden, self.latent_predictor(latent, latent_pred_labels)
        hidden, torch.tensor(0)


class ReformerVAE_Encoder(nn.Module):
    '''
        Combines 2 reformer encoders with an MMD-VAE inbetween.
    '''
    def __init__(self, config):
        self.encoder_1 = ReformerEncoder(config)
        self.vae = MMD_VAE(config.dim_model, config.seq_size, config.latent_size)
        self.encoder_2 = ReformerEncoder(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        num_hashes=None,
        past_buckets_states=None,
        use_cache=False,
        orig_sequence_length=None,
        output_hidden_states=False,
        output_attentions=False,
        latent_pred_labels=None,
    ):
        reformerEncoderOutput = self.encoder_1(
            hidden_states,
            attention_mask,
            head_mask,
            num_hashes,
            past_buckets_states,
            use_cache,
            orig_sequence_length,
            output_hidden_states,
            output_attentions,
        )
        hidden_state, loss = self.vae(reformerEncoderOutput.hidden_states[-1], latent_pred_labels)
        reformerEncoderOutput.hidden_states.append(hidden_state)
        return self.encoder_2(
            reformerEncoderOutput.hidden_states,
            attention_mask,
            head_mask,
            num_hashes,
            past_buckets_states,
            use_cache,
            orig_sequence_length,
            output_hidden_states,
            output_attentions,
        ), loss


@dataclass
class ReformerVAE_ModelOutput(ReformerModelOutput):
    loss: torch.FloatTensor


class ReformerVAE_Model(AltReformerModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = ReformerVAE_Encoder(config)
        self.init_weights()

    def forward(
        self,
        latent_pred_labels=None,
        # OLD ARGS
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
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        embedding_output, head_mask, attention_mask, orig_sequence_length, input_ids, inputs_embeds, attention_mask, must_pad_to_match_chunk_length = self.start_forward(
            input_ids,
            attention_mask,
            position_ids,
            head_mask,
            inputs_embeds,
            num_hashes,
            past_buckets_states,
            use_cache,
            output_hidden_states,
            output_attentions,
            return_dict,
        )

        encoder_outputs, loss = self.encoder(
            hidden_states=embedding_output,
            head_mask=head_mask,
            attention_mask=attention_mask,
            num_hashes=num_hashes,
            past_buckets_states=past_buckets_states,
            use_cache=use_cache,
            orig_sequence_length=orig_sequence_length,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            latent_pred_labels=latent_pred_labels
        )
        sequence_output = encoder_outputs.hidden_states

        # if padding was applied
        if must_pad_to_match_chunk_length:
            sequence_output = sequence_output[:, :orig_sequence_length]

        past_buckets_states = encoder_outputs.past_buckets_states if use_cache else None
        hidden_states = encoder_outputs.all_hidden_states if output_hidden_states else None
        attentions = encoder_outputs.all_attentions if output_attentions else None

        if not return_dict:
            return tuple(v for v in [sequence_output, past_buckets_states, hidden_states, attentions] if v is not None)
        return ReformerVAE_ModelOutput(
            last_hidden_state=sequence_output,
            past_buckets_states=past_buckets_states,
            hidden_states=hidden_states,
            attentions=attentions,
            loss=loss
        )


class ReformerVAE_ModelWithLMHead(ReformerModelWithLMHead):
    def __init__(self, config):
        super().__init__(config)
        self.reformer = ReformerVAE_Model(config)

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        num_hashes=None,
        past_buckets_states=None,
        use_cache=None,
        output_hidden_states=None,
        output_attentions=None,
        return_dict=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Labels for computing the sequence classification/regression loss.
                Indices should be in :obj:`[-100, 0, ..., config.vocab_size - 1]`.
                All labels set to ``-100`` are ignored (masked), the loss is only
                computed for labels in ``[0, ..., config.vocab_size]``
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        reformer_outputs = self.reformer(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            num_hashes=num_hashes,
            past_buckets_states=past_buckets_states,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        sequence_output = reformer_outputs[0]
        logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + reformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ReformerModelWithLMHeadOutput(
            loss=loss + reformer_outputs.loss,
            logits=logits,
            past_buckets_states=reformer_outputs.past_buckets_states,
            hidden_states=reformer_outputs.hidden_states,
            attentions=reformer_outputs.attentions,
        )


class ReformerVAEConfig(ReformerConfig):
    def __init__(self, set_seq_size, **kwargs):
        super().__init__(**kwargs)
        self.set_seq_size = set_seq_size


class NesTokenizer():
    def __init__(self, vocab_file):
        with open(vocab_file, "r") as f:
            tokens = f.read().split("\n")
        tokens = list(filter(None, tokens))
        tokens = [txt.strip() for txt in tokens]
        tokens = list(set(tokens))
        assert len(tokens) > 10
        logger.info(f"Using vocab size {len(tokens)}")
        self.index2vocab = ["<pad>", "</s>"] + tokens
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.vocab2index = {word: i for i, word in enumerate(self.index2vocab)}
        self.wait_amts = set()
        for word in self.index2vocab:
            if word[:2] == "WT":
                self.wait_amts.add(int(word.split("_")[1]))

    def get_index(self, word):
        if word in self.vocab2index:
            return self.vocab2index[word]
        else:
            assert word[:2] == "WT"
            wait_amt = int(word.split("_")[1])
            closest = min(self.wait_amts, key=lambda x: abs(x - wait_amt))
            return self.vocab2index["WT_{}".format(closest)]

    def tokenize(self, lines, max_length=None):
        samples = []
        for line in lines:
            line = line.strip()
            tokens = [self.get_index(word) for word in line.split(" ")]
            segments = []
            if max_length:
                for i in range(1 + (len(tokens) // max_length)):
                    segments.append(tokens[i*max_length:(i+1)*max_length])
            else:
                segments = [tokens]
            samples += segments
        return {
            'input_ids': samples
        }

    def decode(self, token_ids):
        assert(len(token_ids.size()) == 1)
        return ' '.join([self.index2vocab[index] for index in token_ids])

    def __len__(self):
        return len(self.index2vocab)

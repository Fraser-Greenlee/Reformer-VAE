import torch
from torch import nn
from transformers import (
    ReformerEncoder,
    ReformerModel,
    ReformerConfig,
    DataCollatorForLanguageModeling
)


class LatentEncoderLargeTanh_1kLatent(nn.Module):
    def __init__(self, dim_m, set_input_size, latent_size=1000):
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
    def __init__(self, dim_m, set_input_size, latent_size=1000):
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
    '''
    def __init__(self, dim_model, seq_size):
        super().__init__()
        self.encoder = LatentEncoderLargeTanh_1kLatent(dim_model, seq_size)
        self.decoder = LatentDecoderLargeT5NormFF(dim_model, seq_size, )


class ReformerVAE_Encoder(ReformerEncoder):
    def __init__(self, config):
        super().__init__()
        mid_point = len(self.layers) // 2
        self.layers = self.layers[mid_point:] + [MMD_VAE(config.dim_model, config.seq_size)] + self.layers[:mid_point]


class ReformerVAEModel(ReformerModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = ReformerVAE_Encoder(config)


class ReformerVAEConfig(ReformerConfig):
    def __init__(self, set_seq_size, **kwargs):
        super().__init__(**kwargs)
        self.set_seq_size = set_seq_size


# TODO: use DataCollatorForLanguageModeling to collate data

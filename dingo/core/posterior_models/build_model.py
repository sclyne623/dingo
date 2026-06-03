import torch

from dingo.core.posterior_models.base_model import BasePosteriorModel
from dingo.core.posterior_models.flow_matching import FlowMatchingPosteriorModel
from dingo.core.posterior_models.normalizing_flow import NormalizingFlowPosteriorModel
from dingo.core.posterior_models.score_matching import ScoreDiffusionPosteriorModel
from dingo.core.utils.backward_compatibility import (
    torch_load_with_fallback,
    update_model_config,
)


def build_model_from_kwargs(
    filename: str = None, settings: dict = None, **kwargs
) -> BasePosteriorModel:
    """
    Returns a PosteriorModel based on a saved network or settings dict.

    The function is careful to choose the appropriate PosteriorModel class (e.g.,
    for a normalizing flow, flow matching, or score matching).

    Parameters
    ----------
    filename: str
        Path to a saved network (.pt).
    settings: dict
        Settings dictionary.
    kwargs
        Arguments forwarded to the model constructor.

    Returns
    -------
    PosteriorModel
    """
    if (filename is None) == (settings is None):
        raise ValueError(
            "Either a filename or a settings dict must be provided, but not both."
        )

    models_dict = {
        "normalizing_flow": NormalizingFlowPosteriorModel,
        "flow_matching": FlowMatchingPosteriorModel,
        "score_matching": ScoreDiffusionPosteriorModel,
    }

    if filename is not None:
        d, _ = torch_load_with_fallback(filename, preferred_map_location="meta")
        update_model_config(d["metadata"]["train_settings"]["model"])  # Backward compat
        posterior_model_type = d["metadata"]["train_settings"]["model"][
            "posterior_model_type"
        ]
    else:
        update_model_config(settings["train_settings"]["model"])  # Backward compat
        posterior_model_type = settings["train_settings"]["model"][
            "posterior_model_type"
        ]

    if not posterior_model_type.lower() in models_dict:
        raise ValueError("No valid posterior model type specified.")

    model = models_dict[posterior_model_type.lower()]

    return model(model_filename=filename, metadata=settings, **kwargs)


def autocomplete_model_kwargs(model_kwargs: dict, data_sample: list):
    """
    Autocomplete the model kwargs from train_settings and data_sample from the dataloader:

    * set input dimension of embedding net to shape of data_sample[1]
    * set dimension of parameter space to len(data_sample[0])
    * set added_context flag of embedding net if required for gnpe proxies
    * set context dim of posterior model to output dim of embedding net + gnpe proxy dim

    Parameters
    ----------
    model_kwargs: dict
        Model settings, which are modified in-place.
    data_sample: list
        Sample from dataloader (e.g., wfd[0]) used for autocomplection.
        Should be of format [parameters, GW data, gnpe_proxies], where the
        last element is only there is GNPE proxies are required.
    """

    emb = model_kwargs["embedding_kwargs"]

    if "transformer_kwargs" in emb:
        # Transformer embedding path (lifted from dingo-t1). Note: GNPE is not
        # supported with the transformer (data_sample[2] is the `position`
        # tensor here, not a GNPE proxy), so we do not use the try/except below.
        d_model = emb["transformer_kwargs"]["d_model"]
        if "tokenizer_kwargs" in emb:
            # data_sample[1] is the tokenized waveform: (num_tokens, num_features)
            emb["tokenizer_kwargs"]["input_dims"] = list(data_sample[1].shape)
            emb["tokenizer_kwargs"]["output_dim"] = d_model
        if "final_net_kwargs" in emb:
            emb["final_net_kwargs"]["input_dim"] = d_model
            context_dim = emb["final_net_kwargs"]["output_dim"]
        else:
            context_dim = d_model
        # TODO(LISA): num_blocks (number of TDI channels, = 2 for A/E) is NOT
        # auto-derived here. Set it explicitly in the YAML, e.g.
        # block_encoder_kwargs.num_blocks: 2  (and tokenizer num_blocks if used).
        model_kwargs["posterior_kwargs"]["input_dim"] = len(data_sample[0])
        model_kwargs["posterior_kwargs"]["context_dim"] = context_dim
    else:
        # Original reduced-basis (SVD) embedding path.
        # set input dims from ifo_list and domain information
        emb["input_dims"] = list(data_sample[1].shape)
        # set dimension of parameter space of posterior model
        model_kwargs["posterior_kwargs"]["input_dim"] = len(data_sample[0])
        # set added_context flag of embedding net if GNPE proxies are required
        # set context dim of nsf to output dim of embedding net + GNPE proxy dim
        try:
            gnpe_proxy_dim = len(data_sample[2])
            emb["added_context"] = True
            model_kwargs["posterior_kwargs"]["context_dim"] = (
                emb["output_dim"] + gnpe_proxy_dim
            )
        except IndexError:
            emb["added_context"] = False
            model_kwargs["posterior_kwargs"]["context_dim"] = emb["output_dim"]

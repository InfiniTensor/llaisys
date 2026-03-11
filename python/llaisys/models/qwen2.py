from .decoder_only import DecoderOnlyModel
from ..libllaisys.qwen2 import LlaisysQwen2Meta


class Qwen2(DecoderOnlyModel):
    create_api = "llaisysQwen2ModelCreate"
    destroy_api = "llaisysQwen2ModelDestroy"
    weights_api = "llaisysQwen2ModelWeights"
    reset_api = "llaisysQwen2ModelReset"
    infer_api = "llaisysQwen2ModelInfer"
    infer_sample_api = "llaisysQwen2ModelInferSample"
    meta_cls = LlaisysQwen2Meta
    default_rope_theta = 1000000.0

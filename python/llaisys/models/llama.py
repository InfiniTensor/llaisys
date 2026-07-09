from .decoder_only import DecoderOnlyModel
from ..libllaisys.llama import LlaisysLlamaMeta


class Llama(DecoderOnlyModel):
    create_api = "llaisysLlamaModelCreate"
    destroy_api = "llaisysLlamaModelDestroy"
    weights_api = "llaisysLlamaModelWeights"
    reset_api = "llaisysLlamaModelReset"
    infer_api = "llaisysLlamaModelInfer"
    infer_sample_api = "llaisysLlamaModelInferSample"
    meta_cls = LlaisysLlamaMeta
    default_rope_theta = 10000.0

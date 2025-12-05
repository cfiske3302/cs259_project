import argparse
import os

import torch
from transformers import AutoTokenizer, AutoModelForVision2Seq
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextAttention,
    eager_attention_forward,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

# We will use the AWQ search implementation from llm-awq, but we need to
# make it understand Qwen3-VL's text backbone (Qwen3VLTextModel).
# To do that without modifying the library on disk, we monkey‑patch
# `get_blocks` and `move_embed` inside `awq.quantize.pre_quant` at runtime.

import awq.quantize.pre_quant as pre_quant


# Keep references to the original helpers so we can fall back to them
_orig_get_blocks = pre_quant.get_blocks
_orig_move_embed = pre_quant.move_embed


def _get_blocks_patched(model):
    """Extend AWQ's get_blocks to handle Qwen3VLTextModel.

    For Qwen3-VL, the language backbone is `Qwen3VLTextModel`, which owns
    `layers` (a ModuleList of decoder layers). We quantize ONLY this text
    model, not the vision encoder.
    """
    name = model.__class__.__name__

    # Text backbone of Qwen3-VL
    if name == "Qwen3VLTextModel":
        # This mirrors what AWQ does for LLaMA/Qwen2, etc., where it
        # returns the list of transformer layers.
        return model.layers

    # Fallback to the original implementation for all other models
    return _orig_get_blocks(model)


def _move_embed_patched(model, device: str):
    """Extend AWQ's move_embed to handle Qwen3VLTextModel.

    We follow the pattern used for LLaMA/Qwen2 in AWQ: move the token
    embeddings and rotary embedding module to the target device so that
    the calibration forward passes run correctly on GPU.
    """
    name = model.__class__.__name__

    if name == "Qwen3VLTextModel":
        # Qwen3VLTextModel defines `embed_tokens` and `rotary_emb` at the
        # top level of the module (not under `model.*`).
        model.embed_tokens = model.embed_tokens.to(device)
        model.rotary_emb = model.rotary_emb.to(device)
        return

    # Fallback for all other model types
    return _orig_move_embed(model, device)


# Monkey‑patch the helpers in the awq.quantize.pre_quant module
pre_quant.get_blocks = _get_blocks_patched
pre_quant.move_embed = _move_embed_patched


# Monkey‑patch Qwen3VLTextAttention.forward for AWQ so that it ignores
# rotary position embeddings entirely. For AWQ calibration we only need
# the attention block to behave consistently between the FP and
# quantized passes; exact RoPE behavior is not critical.

def _qwen3vl_attn_forward_patched(
    self,
    hidden_states,
    position_embeddings,
    attention_mask,
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    # This is adapted from Qwen3VLTextAttention.forward in
    # transformers' modeling_qwen3_vl.py, but we skip
    # apply_rotary_pos_emb and ignore past_key_values/cache.
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # Force use of the simple eager attention implementation during AWQ
    # calibration to avoid shape/kv-grouping assumptions in SDPA/Flash
    # attention wrappers.
    attn_output, attn_weights = eager_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling=self.scaling,
        dropout=0.0 if not self.training else self.attention_dropout,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


Qwen3VLTextAttention.forward = _qwen3vl_attn_forward_patched

# Now we can safely import run_awq, which will use the patched helpers
from awq.quantize.pre_quant import run_awq


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run AWQ search on the Qwen3-VL-2B-Instruct text backbone and save awq_results."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Hugging Face repo ID or local path for Qwen3-VL-2B-Instruct.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="awq_cache/qwen3-vl-2b-instruct-w4-g128.pt",
        help="Where to save the AWQ search results (awq_results .pt file).",
    )
    parser.add_argument(
        "--w_bit",
        type=int,
        default=4,
        help="Weight bit-width for AWQ (e.g., 4 for INT4).",
    )
    parser.add_argument(
        "--q_group_size",
        type=int,
        default=128,
        help="AWQ group size (typically 128). Use -1 for per-channel.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=128,
        help="Number of calibration samples (from the text calib dataset).",
    )
    parser.add_argument(
        "--seqlen",
        type=int,
        default=512,
        help="Sequence length for calibration samples.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16"],
        help="Base dtype to load the model with for AWQ search.",
    )
    parser.add_argument(
        "--calib_data",
        type=str,
        default="pileval",
        help=(
            "Name of calibration dataset to use (passed through AWQ's "
            "get_calib_dataset; default is 'pileval')."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run AWQ search; no GPU detected.")

    torch_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    print(f"Loading tokenizer from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=False,
        trust_remote_code=True,
    )

    print(f"Loading Qwen3-VL model from {args.model_path} (dtype={torch_dtype}) ...")
    # We load the full VLM, but we will only pass the text backbone to AWQ.
    full_model = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map=None,  # keep on CPU; AWQ will move parts to CUDA as needed
    )

    # Extract the language model part: Qwen3VLTextModel
    text_model = full_model.model.language_model
    print("Extracted text backbone:", text_model.__class__.__name__)

    # AWQ quantization config
    q_config = {
        "zero_point": True,
        "q_group_size": args.q_group_size,
    }
    print("AWQ quantization config:", q_config)

    print(
        f"Running AWQ search on Qwen3-VL text model with "
        f"w_bit={args.w_bit}, n_samples={args.n_samples}, seqlen={args.seqlen}, "
        f"calib_data='{args.calib_data}' ..."
    )

    # Run AWQ *only* on the text model; no images/videos are involved in search.
    awq_results = run_awq(
        text_model,
        tokenizer,
        w_bit=args.w_bit,
        q_config=q_config,
        n_samples=args.n_samples,
        seqlen=args.seqlen,
        calib_data=args.calib_data,
    )

    # Save awq_results for later use (apply_awq + real_quantize_model_weight)
    dirpath = os.path.dirname(args.output_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    torch.save(awq_results, args.output_path)
    print(f"AWQ results saved to {args.output_path}")


if __name__ == "__main__":
    main()

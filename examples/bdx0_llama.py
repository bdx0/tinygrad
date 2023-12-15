#!/usr/bin/env python3
import os

from examples.llama import Transformer, convert_from_huggingface
from tinygrad import Tensor, nn
from tinygrad.helpers import Timing

"""
tok_embeddings.weight (32000, 4096)
norm.weight (4096,)
output.weight (32000, 4096)
layers.0.attention.wq.weight (4096, 4096)
layers.0.attention.wk.weight (4096, 4096)
layers.0.attention.wv.weight (4096, 4096)
layers.0.attention.wo.weight (4096, 4096)
layers.0.feed_forward.w1.weight (11008, 4096)
layers.0.feed_forward.w2.weight (4096, 11008)
layers.0.feed_forward.w3.weight (11008, 4096)
layers.0.attention_norm.weight (4096,)
layers.0.ffn_norm.weight (4096,)
"""


# openhermes-2.5-mistral-7b
# llama2-7b
def main():
    Tensor.no_grad = True
    with Timing("load weight:"):
        # part = nn.state.torch_load("weights/LLaMA/7B/consolidated.00.pth")
        # for k, v in part.items():
        #     print(k)
        count = 0
        part = nn.state.torch_load(
            "weights/PhoGPT-Instruction/7.5B/pytorch_model-00001-of-00002.bin"
        )
        for k, v in part.items():
            print(count, k)
            count += 1

        part = nn.state.torch_load(
            "weights/PhoGPT-Instruction/7.5B/pytorch_model-00002-of-00002.bin"
        )
        for k, v in part.items():
            print(count, k)
            count += 1

    # with Timing("create model:"):
    #     model = Transformer(4096, 256, 32, 32, 1e-05, 32000)

    # with Timing("weight -> model: "):
    #     weights = convert_from_huggingface(part, model, 32, 8)
    #     nn.state.load_state_dict(model, weights, strict=False)


if __name__ == "__main__":
    main()

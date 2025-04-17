import argparse
import torch
import numpy as np
import zstandard as zstd
import model_sm as M
from typing import List, Union
import torch.nn as nn


def ascii_hist(name: str, x: Union[List[float], np.ndarray], bins: int = 10) -> None:
    if len(x) == 0:
        print(f"{name}\nNo data provided.")
        return

    N, edges = np.histogram(x, bins=bins)
    width = 50
    nmax = N.max() if N.max() != 0 else 1

    print(name)
    for i in range(len(N)):
        bin_range = f"[{edges[i]:.4g}, {edges[i + 1]:.4g})".ljust(20)
        bar = "#" * int(N[i] * width / nmax)
        count = f"({N[i]:d})".rjust(6)
        print(f"{bin_range}| {bar} {count}")


class NNWriter:
    def __init__(self, model: M.ReversiSmallModel, show_hist: bool = True) -> None:
        self.buf = bytearray()
        self.show_hist = show_hist

        self.write_input_layer(model)
        for (
            l1_ps,
            l2,
            output,
        ) in model.layer_stacks.get_coalesced_layer_stacks():
            self.write_fc_layer(model, l1_ps)
            self.write_fc_layer(model, l2)
            self.write_fc_layer(model, output, is_output=True)

    def write_input_layer(self, model: M.ReversiSmallModel) -> None:
        for p in model.ps_input.get_layers():
            bias = (
                p.bias.detach().cpu().mul(model.quantized_one).round().to(torch.int16)
            )
            weight = (
                p.weight.detach().cpu().mul(model.quantized_one).round().to(torch.int16)
            )

            self.buf.extend(bias.flatten().numpy().tobytes())
            self.buf.extend(weight.flatten().numpy().tobytes())

    def write_fc_layer(
        self, model: M.ReversiSmallModel, layer: nn.Module, is_output: bool = False
    ) -> None:
        kWeightScaleHidden = model.weight_scale_hidden
        kWeightScaleOut = (
            model.score_scale * model.weight_scale_out / model.quantized_one
        )
        kWeightScale = kWeightScaleOut if is_output else kWeightScaleHidden
        kBiasScaleOut = model.weight_scale_out * model.score_scale
        kBiasScaleHidden = model.weight_scale_hidden * model.quantized_one
        kBiasScale = kBiasScaleOut if is_output else kBiasScaleHidden
        kMaxWeight = model.quantized_one / kWeightScale

        bias = layer.bias.detach().cpu()
        bias = bias.mul(kBiasScale).round().to(torch.int32)

        weight = layer.weight.detach().cpu()
        clipped_diff = weight.clamp(-kMaxWeight, kMaxWeight) - weight
        clipped = torch.count_nonzero(clipped_diff)
        total_elements = torch.numel(weight)
        clipped_max = torch.max(torch.abs(clipped_diff)).item()

        weight = (
            weight.clamp(-kMaxWeight, kMaxWeight)
            .mul(kWeightScale)
            .round()
            .to(torch.int8)
        )

        print(
            f"Layer has {clipped}/{total_elements} clipped weights. "
            f"Maximum excess: {clipped_max} (limit: {kMaxWeight})."
        )

        num_input = weight.shape[1]
        if num_input % 32 != 0:
            padded_num = num_input + (32 - num_input % 32)
            print(f"Padding input from {num_input} to {padded_num} elements.")
            new_w = torch.zeros(weight.shape[0], padded_num, dtype=torch.int8)
            new_w[:, :num_input] = weight
            weight = new_w

        self.buf.extend(bias.flatten().numpy().tobytes())
        self.buf.extend(weight.flatten().numpy().tobytes())

        if is_output:
            print("Output layer parameters:")
            print(f"Weight: {weight.flatten()}")
            print(f"Bias: {bias.flatten()}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument("--cl", type=int, default=7)
    parser.add_argument(
        "--output",
        type=str,
        default="eval_sm.zst",
    )
    parser.add_argument("--no-hist", action="store_true")
    args = parser.parse_args()

    model = M.ReversiSmallModel.load_from_checkpoint(args.checkpoint)
    model.eval()

    writer = NNWriter(model, show_hist=not args.no_hist)
    cctx = zstd.ZstdCompressor(level=args.cl)
    compressed_data = cctx.compress(writer.buf)

    with open(args.output, "wb") as f:
        f.write(compressed_data)


if __name__ == "__main__":
    main()

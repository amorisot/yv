import torch
import torch.nn as nn

def main():
    transformer_model = nn.Transformer(
        d_model=32,
        num_encoder_layers=0,
        num_decoder_layers=2,
        )
    print(f"Number of parameters: {sum(p.numel() for p in transformer_model.parameters()):,}")

    src = torch.rand((10, 32, 32))
    tgt = torch.rand((10, 32, 32))
    out = transformer_model(src, tgt)
    breakpoint()

if __name__ == "__main__":
    main()
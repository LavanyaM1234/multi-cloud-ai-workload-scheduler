import torch
state = torch.load("lstm_extractor.pt", map_location="cpu", weights_only=True)
print({k: v.shape for k, v in state.items()})
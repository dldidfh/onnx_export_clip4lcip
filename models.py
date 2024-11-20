from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
)
import torch 

class VisualModel(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.transform = Compose(
            [
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).expand(1,3,224,224).to("cuda")
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).expand(1,3,224,224).to("cuda")
        self.visual = model.model.clip.visual

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = self.visual(x)
        x = self.visual.ln_post(x) @ self.visual.proj
        x = x[:, 0, :]
        # x = x / x.norm(dim=1, keepdim=True)
        return x
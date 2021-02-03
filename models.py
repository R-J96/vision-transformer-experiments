from timm.models.registry import register_model
from vit_pytorch.efficient import ViT
from performer_pytorch import Performer

__all__ = [
    "performer_tiny_patch25_500"
]


@register_model
def performer_tiny_patch25_500(pretrained=False, **kwargs):
    efficient_transformer = Performer(
        dim=512,
        depth=1,
        heads=8,
        causal=True
    )

    model = ViT(
        image_size=500,
        patch_size=25,
        num_classes=2,
        dim=512,
        transformer=efficient_transformer
    )

    # if pretrained:
    #     checkpoint = torch.load_state_dict(
    #         torch.load(PATH)
    #     )
    #     model.load_state_dict(checkpoint["model"])
    return model

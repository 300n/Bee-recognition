from .custom_cnn import CustomCNN
from .backbones import BackbonePoseModel


def build_model(name: str, num_classes: int = 2, num_keypoints: int = 3):
    """
    Factory that returns an initialised model.

    name: "custom_cnn" | "resnet18" | "mobilenet_v3" | "efficientnet_b0"
    """
    if name == "custom_cnn":
        return CustomCNN(num_classes=num_classes, num_keypoints=num_keypoints)
    elif name in ("resnet18", "mobilenet_v3", "efficientnet_b0"):
        pretrained = True
        return BackbonePoseModel(
            backbone_name=name,
            pretrained=pretrained,
            num_classes=num_classes,
            num_keypoints=num_keypoints,
        )
    else:
        raise ValueError(f"Unknown model name: {name}")

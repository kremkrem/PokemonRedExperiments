from lime import lime_image
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from numpy.typing import NDArray
from stable_baselines3.common.policies import ActorCriticPolicy
import matplotlib.pyplot as plt

_CLASS_CNT = 8

class ModelVisualization:
    IDX_TO_CLASS: list[str] = [
        "DOWN", "LEFT", "RIGHT", "UP", "A", "B", "START", "PASS"
    ]

    def __init__(self, policy: ActorCriticPolicy):
        self.policy = policy
        self.explainer = lime_image.LimeImageExplainer()

    def batch_predict(self, images: NDArray) -> NDArray:
        if len(images.shape) == 3:
            np.expand_dims(images, 0)
        logits = self.policy.action_net(
            self.policy.mlp_extractor(
                self.policy.extract_features(self.policy.obs_to_tensor(images)[0])
            )[0]
        )
        probs = F.softmax(F.normalize(logits), dim=1)
        return probs.detach().cpu().numpy()

    def explain(self, image: NDArray) -> lime_image.ImageExplanation:
        return self.explainer.explain_instance(image, self.batch_predict, top_labels=None, labels=range(_CLASS_CNT), batch_size=100, num_samples=1000)

    def explain_and_save_pic(self, image: NDArray, path: Path) -> None:
        explanation = self.explain(image)
        plt.imsave(path, np.concatenate(
            [explanation.get_image_and_mask(i, positive_only=False, num_features=5, hide_rest=False)[0] for i in range(_CLASS_CNT)],
            axis=1))

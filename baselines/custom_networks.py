from typing import Callable

import gymnasium as gym
import torch as th
from gymnasium import spaces
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class WorseNatureCNN(BaseFeaturesExtractor):

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256,
        normalized_image: bool = False,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "NatureCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space,
                              check_channels=False,
                              normalized_image=normalized_image), (
                                  "WorseNatureCNN isonly for images, "
                                  f"not for {observation_space}\n")
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels,
                      32,
                      kernel_size=6,
                      stride=3,
                      padding=1,
                      padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(
                    observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim),
                                    nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


def linear_schedule(init_value: float,
                    fin_value: float) -> Callable[[float], float]:

    def func(progress_remaining: float) -> float:
        return fin_value + (init_value - fin_value) * progress_remaining

    return func

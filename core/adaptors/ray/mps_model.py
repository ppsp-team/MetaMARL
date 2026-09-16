"""MPS-accelerated model wrapper for RLlib."""

import torch
from gymnasium.spaces import Space
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType

MPS_DEVICE = torch.device("mps") if torch.backends.mps.is_available() else None


class MPSFullyConnectedNetwork(TorchModelV2, torch.nn.Module):
    """FCNet wrapper that runs forward pass on MPS."""

    def __init__(
        self,
        obs_space: Space,
        action_space: Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        torch.nn.Module.__init__(self)

        self.device = MPS_DEVICE or torch.device("cpu")
        self._base_model = FullyConnectedNetwork(
            obs_space, action_space, num_outputs, model_config, name + "_base"
        )

        self._base_model.to(self.device)

        self._last_value = None

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: dict[str, TensorType],
        state: list[TensorType],
        seq_lens: TensorType,
    ) -> tuple[TensorType, list[TensorType]]:
        """Run the wrapped fully connected network on the MPS device.

        Parameters
        ----------
        input_dict : dict[str, TensorType]
            RLlib input dict; ``"obs"`` is converted to a tensor if needed and
            copied to ``self.device``, and ``"obs_flat"`` is set to the same
            tensor because ``FullyConnectedNetwork`` reads that key.
        state : list[TensorType]
            RNN state (unused by the FC net, passed through).
        seq_lens : TensorType
            Sequence lengths (passed through).

        Returns
        -------
        tuple[TensorType, list[TensorType]]
            ``(logits, new_state)`` with logits moved back to CPU, since the
            rest of the RLlib pipeline runs there. The value branch output is
            cached for ``value_function``.
        """

        obs = input_dict["obs"]

        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs)

        obs_mps = obs.to(self.device)
        input_dict_mps = {**input_dict, "obs": obs_mps, "obs_flat": obs_mps}
        output, new_state = self._base_model(input_dict_mps, state, seq_lens)

        # Cache value for value_function() call
        self._last_value = self._base_model.value_function()

        # Return to CPU for RLlib
        return output.cpu(), new_state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        """Return the value estimate cached by the last ``forward`` call.

        Returns
        -------
        TensorType
            Value tensor of shape ``[B]`` moved to CPU.

        Raises
        ------
        ValueError
            If ``forward`` has not been called yet.
        """

        if self._last_value is None:
            raise ValueError("forward() must be called before value_function()")

        return self._last_value.cpu()

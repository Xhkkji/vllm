# SPDX-License-Identifier: Apache-2.0
"""
KVConnectorBase Class for Distributed KV Cache & Hidden State communication

The class provides two primary abstract methods:
1. send_kv_caches_and_hidden_states(): Send KV caches and hidden states
2. recv_kv_caches_and_hidden_states(): Recv KV caches and hidden states
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, List, Tuple, Union

import torch

from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata


class KVConnectorBase(ABC):
    """
    Abstract base class for a KV connector.

    The class provides two primary abstract methods:
    1. send_kv_caches_and_hidden_states(): Send KV caches and hidden states
    2. recv_kv_caches_and_hidden_states(): Recv KV caches and hidden states
    """

    @abstractmethod
    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: "VllmConfig",
    ):
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Close the buffer and release resources.

        This method is responsible for cleaning up resources related to the 
        connector when it is no longer needed.

        Raises:
            NotImplementedError: This method must be implemented in subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
    ) -> None:
        """
        Send KV caches and hidden states to the connector.

        This method processes the input tokens, KV caches, and 
        hidden/intermediate states for a given model and sends the data to the 
        decode instance.

        Args:
            model_executable (torch.nn.Module): The model executable containing 
                start and end layer information.
            model_input (ModelInputForGPUWithSamplingMetadata): The input
                metadata from vLLM.
            kv_caches (List[torch.Tensor]): List of KV caches (keys and values) 
                for each layer.
            hidden_or_intermediate_states (Union[torch.Tensor, 
            IntermediateTensors]): 
                The hidden or intermediate states associated with the tokens.

        Returns:
            None

        """

        raise NotImplementedError

    @abstractmethod
    def recv_kv_caches_and_hidden_states(
        self, model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor]
    ) -> "KVReceiveResult":
        """
        Receive KV caches and hidden states from the connector.

        This method attempts to retrieve KV caches and hidden states for input
        tokens. If all required KV caches and hidden states are received, it
        will bypass model input, else it will fall back to normal vLLM model 
        forwarding.

        Args:
            model_executable (torch.nn.Module): 
                The model executable from vLLM modelrunner.
            model_input (ModelInputForGPUWithSamplingMetadata): 
                The model input from vLLM modelrunner.
            kv_caches (List[torch.Tensor]): 
                List of KV caches for each layer.

        Returns:
            `KVReceiveResult`，用于显式表达三种运行时语义：

            - `READY_BYPASS`
              当前 batch 的 KV/hidden state 已完整就绪，可以直接跳过
              当前前向。
            - `READY_FORWARD`
              当前 batch 需要继续走本轮正常前向，必要时会携带 rebuilt 后
              的 `model_input`。
            - `DEFERRED`
              当前 batch 的 retrieve 已经启动，但还不能安全继续前向；
              上层 runtime 应保持这批 request 原样挂起，并在下一轮继续 poll。

        """

        raise NotImplementedError


class KVReceiveStatus(Enum):
    """描述一次 KV receive 在当前调度轮次结束时的稳定运行时语义。

    这里的状态机是“request/batch 级”的，而不是 page 级的：

    - page/chunk 的 read_ready / cache_ready / consumable 仍然由底层
      BaM store 跟踪。
    - connector/runtime 这一层只关心：
      1. 这轮能不能直接 bypass
      2. 这轮要不要继续 forward
      3. 还是必须 defer 到下一轮
    """

    READY_BYPASS = auto()
    READY_FORWARD = auto()
    DEFERRED = auto()


@dataclass(frozen=True)
class KVReceiveResult:
    """统一承载 v0 connector `recv` 的返回结果。

    相比旧三元组，这里显式补上 `status`，避免把“当前不能 forward、但也
    不能直接 bypass”的 `DEFERRED` 语义硬塞进布尔值里。
    """

    status: KVReceiveStatus
    hidden_or_intermediate_states: Union[torch.Tensor, IntermediateTensors,
                                         None]
    bypass_model_exec: bool
    model_input: "ModelInputForGPUWithSamplingMetadata"

    @classmethod
    def ready_forward(
        cls,
        *,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors,
                                             None] = None,
    ) -> "KVReceiveResult":
        """当前 batch 需要继续执行本轮正常前向。"""
        return cls(
            status=KVReceiveStatus.READY_FORWARD,
            hidden_or_intermediate_states=hidden_or_intermediate_states,
            bypass_model_exec=False,
            model_input=model_input,
        )

    @classmethod
    def ready_bypass(
        cls,
        *,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors,
                                             None],
    ) -> "KVReceiveResult":
        """当前 batch 已完整命中，可以跳过这轮 model forward。"""
        return cls(
            status=KVReceiveStatus.READY_BYPASS,
            hidden_or_intermediate_states=hidden_or_intermediate_states,
            bypass_model_exec=True,
            model_input=model_input,
        )

    @classmethod
    def deferred(
        cls,
        *,
        model_input: "ModelInputForGPUWithSamplingMetadata",
    ) -> "KVReceiveResult":
        """当前 batch 已进入 in-flight retrieve，但本轮还不能安全继续前向。"""
        return cls(
            status=KVReceiveStatus.DEFERRED,
            hidden_or_intermediate_states=None,
            bypass_model_exec=False,
            model_input=model_input,
        )

    @classmethod
    def from_legacy_tuple(
        cls,
        legacy_result: Tuple[Union[torch.Tensor, IntermediateTensors, None],
                             bool, "ModelInputForGPUWithSamplingMetadata"],
    ) -> "KVReceiveResult":
        """兼容旧版 connector 的 `(hidden, bypass, model_input)` 三元组。"""
        hidden_or_intermediate_states, bypass_model_exec, model_input = (
            legacy_result)
        if bypass_model_exec:
            return cls.ready_bypass(
                model_input=model_input,
                hidden_or_intermediate_states=hidden_or_intermediate_states,
            )
        return cls.ready_forward(
            model_input=model_input,
            hidden_or_intermediate_states=hidden_or_intermediate_states,
        )


KVConnectorBaseType = Union[KVConnectorBase, KVConnectorBase_V1]

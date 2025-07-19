from typing import List, Tuple

try:
    import torch
except ImportError:
    raise ImportError(
        "`tensor_transport` requires PyTorch. "
        "Please install torch with 'pip install torch' to use this feature."
    )

import ray.util.collective as collective
from ray._private.custom_types import TensorTransportEnum
from ray._private.worker import global_worker
from ray.util.collective.types import Backend

TENSOR_TRANSPORT_TO_COLLECTIVE_BACKEND = {
    TensorTransportEnum.NCCL: Backend.NCCL,
    TensorTransportEnum.GLOO: Backend.TORCH_GLOO,
}

COLLECTIVE_BACKEND_TO_TORCH_DEVICE = {
    Backend.NCCL: torch.device("cuda"),
    Backend.TORCH_GLOO: torch.device("cpu"),
}


def tensor_transport_to_collective_backend(
    tensor_transport: TensorTransportEnum,
) -> Backend:
    try:
        return TENSOR_TRANSPORT_TO_COLLECTIVE_BACKEND[tensor_transport]
    except KeyError:
        raise ValueError(
            f"Invalid tensor transport {tensor_transport.name}, must be one of {list(TENSOR_TRANSPORT_TO_COLLECTIVE_BACKEND.keys())}."
        )


def __ray_send__(self, communicator_name: str, obj_id: str, dst_rank: int):
    """Helper function that runs on the src actor to send tensors to the dst actor."""
    gpu_object_manager = global_worker.gpu_object_manager
    assert gpu_object_manager.has_gpu_object(
        obj_id
    ), f"obj_id={obj_id} not found in GPU object store"
    tensors = gpu_object_manager.get_gpu_object(obj_id)

    backend = collective.get_group_handle(communicator_name).backend()
    device = COLLECTIVE_BACKEND_TO_TORCH_DEVICE[backend]

    for tensor in tensors:
        if tensor.device.type != device.type:
            # TODO(swang): Right now there is no way to catch this error
            # and the receiving Ray task will hang.
            raise ValueError(
                f"tensor device {tensor.device} does not match device {device}"
            )
        collective.send(tensor, dst_rank, group_name=communicator_name)
    # TODO(kevin85421): The current garbage collection implementation for the
    # in-actor object store is naive. We garbage collect each object after it
    # is consumed once.
    gpu_object_manager.remove_gpu_object(obj_id)

import socket
import struct

def parse_metadata(metadata: bytes):
    if len(metadata) == 10:
        # IPv4: 4 bytes IP, 2 bytes port, 4 bytes GPU idx
        ip_bytes = metadata[:4]
        port_bytes = metadata[4:6]
        gpu_idx_bytes = metadata[6:10]
        ip = socket.inet_ntop(socket.AF_INET, ip_bytes)
    elif len(metadata) == 22:
        # IPv6: 16 bytes IP, 2 bytes port, 4 bytes GPU idx
        ip_bytes = metadata[:16]
        port_bytes = metadata[16:18]
        gpu_idx_bytes = metadata[18:22]
        ip = socket.inet_ntop(socket.AF_INET6, ip_bytes)
    else:
        raise ValueError(f"Unexpected metadata length: {len(metadata)}")
    
    port = struct.unpack('!H', port_bytes)[0]
    remote_gpu_idx = struct.unpack('i', gpu_idx_bytes)[0]  # host byte order
    return ip, port, remote_gpu_idx

def __ray_recv__(
    self,
    communicator_name: str,
    obj_id: str,
    src_rank: int,
    tensor_meta: List[Tuple["torch.Size", "torch.dtype"]],
):
    """Helper function that runs on the dst actor to receive tensors from the src actor."""
    from ray._private.worker import global_worker

    backend = collective.get_group_handle(communicator_name).backend()
    device = COLLECTIVE_BACKEND_TO_TORCH_DEVICE[backend]

    gpu_object_manager = global_worker.gpu_object_manager
    tensors = []
    tensor_meta, uccl_endpoint_meta = tensor_meta
    for meta in tensor_meta:
        shape, dtype = meta
        tensor = torch.zeros(shape, dtype=dtype, device=device)
        # collective.recv(tensor, src_rank, group_name=communicator_name)
        tensors.append(tensor)
        
    uccl_endpoint = gpu_object_manager.init_uccl_endpoint()
    ip, port, r_gpu = parse_metadata(uccl_endpoint_meta)
    ok, conn_id = uccl_endpoint.connect(ip, r_gpu, remote_port=port)
    
    # TODO(MaoZiming): where to accept. 
    total_size = sum(t.numel() * t.element_size() for t in tensors)
    ok, mr_id = uccl_endpoint.reg(tensors, total_size)
    assert ok, "[Client] register failed"
    ok = uccl_endpoint.send(conn_id, mr_id, tensors, total_size)
    assert ok, "[Client] send error"
    
    gpu_object_manager.add_gpu_object(obj_id, tensors)


def __ray_fetch_gpu_object__(self, obj_id: str):
    """Helper function that runs on the src actor to fetch tensors from the GPU object store via the object store."""
    gpu_object_manager = global_worker.gpu_object_manager
    assert gpu_object_manager.has_gpu_object(
        obj_id
    ), f"obj_id={obj_id} not found in GPU object store"
    tensors = gpu_object_manager.get_gpu_object(obj_id)
    # TODO(kevin85421): The current garbage collection implementation for the
    # in-actor object store is naive. We garbage collect each object after it
    # is consumed once.
    gpu_object_manager.remove_gpu_object(obj_id)
    return tensors

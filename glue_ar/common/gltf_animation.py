from gltflib import AccessorType, ComponentType
from struct import calcsize

from ..gltf_utils import add_points_to_bytearray, add_values_to_bytearray
from .gltf_builder import GLTFBuilder


def one_hot_scales(length: int) -> tuple[tuple[int]]:
    return tuple((1, 1, 1) if idx == (length-1) else (0, 0, 0) for idx in range(2*length-1))


def set_up_flipbook_animation(
    builder: GLTFBuilder,
    n_snapshots: int,
    time_delta: float = 0.05,
    uri: str = "animation.bin",
    animation_name: str = "Flipbook",
    with_scales: bool = True,
) -> dict:

    timestamps = tuple(i * time_delta for i in range(1, n_snapshots))
    buffer = bytearray()
    add_values_to_bytearray(buffer, timestamps)
    time_bytelen = len(buffer)
    buffer_index = builder.buffer_count

    if with_scales:
        scale_accessor_indices = []
        scales = one_hot_scales(n_snapshots)
        add_points_to_bytearray(buffer, scales)
        scale_mins = (0, 0, 0)
        scale_maxes = (1, 1, 1)
        scales_len = len(buffer) - time_bytelen
        buffer_view_length = scales_len * n_snapshots // (2 * n_snapshots - 1)

        float_size = calcsize("f")
        for index in range(n_snapshots):
        
            offset = 3 * float_size * index + time_bytelen
            builder.add_buffer_view(
                buffer=buffer_index,
                byte_length=buffer_view_length,
                byte_offset=offset,
            )
            buffer_view = builder.buffer_view_count - 1
            builder.add_accessor(
                buffer_view=buffer_view,
                component_type=ComponentType.FLOAT,
                count=len(timestamps),
                type=AccessorType.VEC3,
                mins=scale_mins,
                maxes=scale_maxes,
            )
            scale_accessor_indices.append(builder.accessor_count - 1)

    builder.add_buffer(
        byte_length=len(buffer),
        uri=uri,
    )
    builder.add_file_resource(
        filename=uri,
        data=buffer,
    )
    builder.add_buffer_view(
        buffer=buffer_index,
        byte_length=time_bytelen,
        byte_offset=0,
    )

    time_buffer_view_index = builder.buffer_view_count - 1
    builder.add_accessor(
        buffer_view=time_buffer_view_index,
        component_type=ComponentType.FLOAT,
        count=len(timestamps),
        type=AccessorType.SCALAR,
        mins=[min(timestamps)],
        maxes=[max(timestamps)],
    )
    time_accessor_index = builder.accessor_count - 1

    builder.add_animation(name=animation_name)
    animation_index = builder.animation_count - 1

    data = dict(
        animation_index=animation_index,
        time_accessor_index=time_accessor_index,
    )
    if with_scales:
        data["scale_accessor_indices"] = scale_accessor_indices

    return data

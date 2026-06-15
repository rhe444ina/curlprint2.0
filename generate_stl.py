import numpy as np
import trimesh

def create_comb_stl(
    tooth_spacing,
    tooth_length,
    handle_angle,
    tip_roundness,
    filename="custom_comb.stl"
):

    parts = []

    # Handle
    handle = trimesh.creation.box(
        extents=[80, 15, 5]
    )

    handle.apply_translation([40, 0, 0])
    parts.append(handle)

    # Teeth
    for i in range(8):

        x = 10 + i * tooth_spacing

        tooth = trimesh.creation.cylinder(
            radius=1.5,
            height=tooth_length,
            sections=24
        )

        tooth.apply_transform(
            trimesh.transformations.rotation_matrix(
                np.radians(90),
                [1,0,0]
            )
        )

        tooth.apply_translation([x,-tooth_length/2,0])

        parts.append(tooth)

        # Rounded tip
        tip = trimesh.creation.icosphere(
            radius=1.5 * tip_roundness,
            subdivisions=2
        )

        tip.apply_translation([x,-tooth_length,0])

        parts.append(tip)

    comb = trimesh.util.concatenate(parts)

    comb.export(filename)

    print("Saved:", filename)

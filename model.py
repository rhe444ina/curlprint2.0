# CurlPrint prototype
# Mental map: 
Hair profile -> encoder model -> tool measurements -> simple 3D-printable STL

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import trimesh


# -------------------------------------
# 1. Dataset, collected from WOC @ Tech
# -------------------------------------

X = torch.tensor([
    [3, 2, 2, 0, 1, 1],
    [2, 2, 2, 1, 2, 1],
    [1, 1, 1, 1, 1, 2],
    [0, 0, 0, 2, 0, 0],
    [3, 1, 2, 0, 2, 1],
    [2, 1, 1, 1, 1, 0],
    [1, 2, 1, 2, 1, 2],
    [3, 2, 1, 1, 2, 1],
], dtype=torch.float32)

y = torch.tensor([
    [8.0, 45.0, 30.0, 0.90],
    [7.5, 42.0, 28.0, 0.95],
    [5.0, 30.0, 20.0, 0.70],
    [3.0, 20.0, 10.0, 0.50],
    [8.5, 43.0, 32.0, 0.98],
    [6.0, 35.0, 22.0, 0.75],
    [5.5, 32.0, 18.0, 0.80],
    [8.0, 40.0, 30.0, 0.95],
], dtype=torch.float32)


# -----------------------------
# 2. Model
# -----------------------------

class HairEncoder(nn.Module):
    def __init__(self, input_dim=6, embedding_dim=16):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)


class HairToolModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = HairEncoder()

        self.predictor = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        embedding = self.encoder(x)
        return self.predictor(embedding)


# -----------------------------
# 3. Training + prediction
# -----------------------------

def train_model(epochs=1000, learning_rate=0.01):
    model = HairToolModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        optimizer.zero_grad()

        predictions = model(X)
        loss = criterion(predictions, y)

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return model


def predict_tool_params(model, user_profile):
    model.eval()

    with torch.no_grad():
        tool_params = model(user_profile).numpy()[0]

    return tool_params


# -----------------------------
# 4. Generate basic 3D comb STL
# -----------------------------

def create_comb_stl(
    tooth_spacing=8,
    tooth_length=45,
    handle_length=80,
    handle_width=15,
    handle_thickness=5,
    num_teeth=8,
    filename="curlprint_comb.stl"
):
    parts = []

    handle = trimesh.creation.box(
        extents=[handle_length, handle_width, handle_thickness]
    )
    handle.apply_translation([handle_length / 2, 0, 0])
    parts.append(handle)

    start_x = 10

    for i in range(num_teeth):
        x = start_x + i * tooth_spacing

        tooth = trimesh.creation.cylinder(
            radius=1.5,
            height=tooth_length,
            sections=24
        )

        tooth.apply_transform(
            trimesh.transformations.rotation_matrix(
                np.radians(90),
                [1, 0, 0]
            )
        )

        tooth.apply_translation([x, -tooth_length / 2, 0])
        parts.append(tooth)

        tip = trimesh.creation.icosphere(
            radius=1.7,
            subdivisions=2
        )
        tip.apply_translation([x, -tooth_length, 0])
        parts.append(tip)

    comb = trimesh.util.concatenate(parts)
    comb.export(filename)

    print(f"\nSTL saved as {filename}")


# -----------------------------
# 5. Running the prototype!!
# -----------------------------

if __name__ == "__main__":
    model = train_model()

    new_user = torch.tensor([[3, 2, 2, 0, 1, 1]], dtype=torch.float32)

    tooth_spacing, tooth_length, handle_angle, tip_roundness = predict_tool_params(
        model,
        new_user
    )

    print("\nPredicted tool design:")
    print(f"Tooth spacing: {tooth_spacing:.2f} mm")
    print(f"Tooth length: {tooth_length:.2f} mm")
    print(f"Handle angle: {handle_angle:.2f} degrees")
    print(f"Tip roundness: {tip_roundness:.2f}")

    create_comb_stl(
        tooth_spacing=float(tooth_spacing),
        tooth_length=float(tooth_length),
        filename="curlprint_comb.stl"
    )

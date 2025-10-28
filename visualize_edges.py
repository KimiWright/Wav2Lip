import vvad_st_gcn_model_functions as st
import st_gcn_vvad as vvad
import PR_curve_mediapipe_vvad as m_vvad

from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION 
from torch.utils import data as data_utils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np

def draw_landmarks_with_edges(landmarks, edges, save_path=None, show=True):
    """
    Draw landmarks connected by edges.

    Args:
        landmarks (np.ndarray): shape [2, N], where landmarks[0] = x, landmarks[1] = y
        edges (list of tuple): each tuple (i, j) connects landmark i to landmark j
        save_path (str, optional): path to save the generated image (e.g., 'output.png')
        show (bool): whether to display the plot immediately

    Returns:
        np.ndarray: the image array (RGB)
    """
    landmarks = np.asarray(landmarks)
    if landmarks.shape[0] != 2:
        raise ValueError(f"Expected landmarks shape [2, N], got {landmarks.shape}")

    fig, ax = plt.subplots(figsize=(4, 4))
    canvas = FigureCanvas(fig)  # attach a proper Agg canvas
    ax.scatter(landmarks[0], landmarks[1], c='red', s=20)

    for i, j in edges:
        ax.plot(
            [landmarks[0, i], landmarks[0, j]],
            [landmarks[1, i], landmarks[1, j]],
            'b-', linewidth=1
        )

    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')

    # Draw and extract the image buffer
    canvas.draw()
    image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # RGBA

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)

    # Convert RGBA → RGB
    image = image[..., :3]
    return image



if __name__ == "__main__":

    data_limit = 5
    syncnet_T = 5
    test_dataset = vvad.Dataset_Frames("test", frames=syncnet_T, data_point_limit=data_limit)
    
    facial_edges = st.facial_edges()

    first_point = test_dataset[0]
    (x, x_rot, y) = first_point
    first_lmks = x[0].T
    knn_edges = st.knn_edges(first_lmks)

    landmarks = x[0]
    print(landmarks.shape)

    # save_path = "Figures_edges/Facial_Edges.png"
    # draw_landmarks_with_edges(landmarks, facial_edges, save_path)

    # save_path = "Figures_edges/Knn_Edges.png"
    # draw_landmarks_with_edges(landmarks, knn_edges, save_path)

    mediapipe_edges = list(FACEMESH_TESSELATION)

    save_path = "Figures_edges/Mediapipe_Edges.png"

    m_test_dataset = m_vvad.Dataset_Frames("test", frames=syncnet_T, data_point_limit=data_limit)
    (x, y) = m_test_dataset[0]

    landmarks = x[0].T
    landmarks = landmarks[:2]
    print(landmarks.shape)

    draw_landmarks_with_edges(landmarks, mediapipe_edges, save_path)
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
from sklearn.metrics import precision_recall_curve, auc

# List all your CSV file paths here
csv_files_cosine_reg = [
    "VVAD_PR_Curves\color\color_babble_noise_cosine_PR_data.csv",
    "VVAD_PR_Curves\color\color_silence_cosine_PR_data.csv",
    "VVAD_PR_Curves\color\color_still_face_cosine_PR_data.csv",
    "VVAD_PR_Curves\color\color_white_noise_cosine_PR_data.csv"
]

csv_files_fconfm_reg = [
    "VVAD_PR_Curves\color\color_babble_noise_fconfm_PR_data.csv",
    "VVAD_PR_Curves\color\color_silence_fconfm_PR_data.csv",
    "VVAD_PR_Curves\color\color_still_face_fconfm_PR_data.csv",
    "VVAD_PR_Curves\color\color_white_noise_fconfm_PR_data.csv"
]

csv_files_cosine_neg = [
    "VVAD_PR_Curves\color\color_babble_noise_neg_cosine_PR_data.csv",
    "VVAD_PR_Curves\color\color_silence_neg_cosine_PR_data.csv",
    "VVAD_PR_Curves\color\color_still_face_neg_cosine_PR_data.csv",
    "VVAD_PR_Curves\color\color_white_noise_neg_cosine_PR_data.csv"
]

csv_files_fconfm_neg = [
    "VVAD_PR_Curves\color\color_babble_noise_neg_fconfm_PR_data.csv",
    "VVAD_PR_Curves\color\color_silence_neg_fconfm_PR_data.csv",
    "VVAD_PR_Curves\color\color_still_face_neg_fconfm_PR_data.csv",
    "VVAD_PR_Curves\color\color_white_noise_neg_fconfm_PR_data.csv"
]

syncnet_csv_files = [
    "PR_curve_color_syncnet_personally_trained_PR_data.csv",
    "PR_curve_lmks_attn_PR_data.csv",
    "PR_curve_lmks_gru2_PR_data.csv",
    "PR_curve_lmks_ST_GCN with Rotation_2_PR_data.csv"
]

gru_syncnet_csv_files = [
    "PR_curve_lmks_attn_PR_data.csv",
    "PR_curve_lmks_gru2_PR_data.csv",
    "PR_curve_lmks_triplets_PR_data.csv"
]

my_models_syncnet_csv_files = [
    "PR_curve_lmks_ST_GCN with Rotation_2_PR_data.csv",
    "PR_curve_lmks_ST_GCN with Rotation using Knn_2_PR_data.csv",
    "PR_curve_lmks_ST_GCN without Rotation_2_PR_data.csv",
    "PR_curve_lmks_ST_GCN without Rotation using Knn_2_PR_data.csv",
    "Syncnet_task_PR_Curves\PR_curve_lmks_mediapipe_PR_data.csv"
]

rot_knn_reg_csv_files = [
    r"VVAD_PR_Curves\rot_knn_babble_noise_PR_data.csv",
    r"VVAD_PR_Curves\rot_knn_silence_PR_data.csv",
    r"VVAD_PR_Curves\rot_knn_still_face_PR_data.csv",
    r"VVAD_PR_Curves\rot_knn_white_noise_PR_data.csv"
]

rot_knn_neg_csv_files = [
    r"VVAD_PR_Curves\rot_knn_babble_noise_neg_PR_data.csv",
    r"VVAD_PR_Curves\rot_knn_silence_neg_PR_data.csv",
    r"VVAD_PR_Curves\rot_knn_still_face_neg_PR_data.csv",
    r"VVAD_PR_Curves\rot_knn_white_noise_neg_PR_data.csv"
]

norot_knn_reg_csv_files = [
    r"VVAD_PR_Curves\norot_knn_babble_noise_PR_data.csv",
    r"VVAD_PR_Curves\norot_knn_silence_PR_data.csv",
    r"VVAD_PR_Curves\norot_knn_still_face_PR_data.csv",
    r"VVAD_PR_Curves\norot_knn_white_noise_PR_data.csv"
]

norot_knn_neg_csv_files = [
    r"VVAD_PR_Curves\norot_knn_babble_noise_neg_PR_data.csv",
    r"VVAD_PR_Curves\norot_knn_silence_neg_PR_data.csv",
    r"VVAD_PR_Curves\norot_knn_still_face_neg_PR_data.csv",
    r"VVAD_PR_Curves\norot_knn_white_noise_neg_PR_data.csv"
]

rot_facial_reg_csv_files = [
    r"VVAD_PR_Curves\rot_facial_babble_noise_PR_data.csv",
    r"VVAD_PR_Curves\rot_facial_silence_PR_data.csv",
    r"VVAD_PR_Curves\rot_facial_still_face_PR_data.csv",
    r"VVAD_PR_Curves\rot_facial_white_noise_PR_data.csv"
]

rot_facial_neg_csv_files = [
    r"VVAD_PR_Curves\rot_facial_babble_noise_neg_PR_data.csv",
    r"VVAD_PR_Curves\rot_facial_silence_neg_PR_data.csv",
    r"VVAD_PR_Curves\rot_facial_still_face_neg_PR_data.csv",
    r"VVAD_PR_Curves\rot_facial_white_noise_neg_PR_data.csv"
]

norot_facial_reg_csv_files = [
    r"VVAD_PR_Curves\norot_facial_babble_noise_PR_data.csv",
    r"VVAD_PR_Curves\norot_facial_silence_PR_data.csv",
    r"VVAD_PR_Curves\norot_facial_still_face_PR_data.csv",
    r"VVAD_PR_Curves\norot_facial_white_noise_PR_data.csv"
]

norot_facial_neg_csv_files = [
    r"VVAD_PR_Curves\norot_facial_babble_noise_neg_PR_data.csv",
    r"VVAD_PR_Curves\norot_facial_silence_neg_PR_data.csv",
    r"VVAD_PR_Curves\norot_facial_still_face_neg_PR_data.csv",
    r"VVAD_PR_Curves\norot_facial_white_noise_neg_PR_data.csv"
]

mediapipe_reg_csv_files = [
    r"VVAD_PR_Curves\mediapipe_babble_noise_PR_data.csv",
    r"VVAD_PR_Curves\mediapipe_silence_PR_data.csv",
    r"VVAD_PR_Curves\mediapipe_still_face_PR_data.csv",
    r"VVAD_PR_Curves\mediapipe_white_noise_PR_data.csv"
]

mediapipe_neg_csv_files = [
    r"VVAD_PR_Curves\mediapipe_babble_noise_neg_PR_data.csv",
    r"VVAD_PR_Curves\mediapipe_silence_neg_PR_data.csv",
    r"VVAD_PR_Curves\mediapipe_still_face_neg_PR_data.csv",
    r"VVAD_PR_Curves\mediapipe_white_noise_neg_PR_data.csv"
]

vvad_trained = [
    r"VVAD_PR_Curves\trained\Facial Landmarks with Orientation_PR_data.csv",
    r"VVAD_PR_Curves\trained\Facial Landmarks without Orientation_PR_data.csv",
    r"VVAD_PR_Curves\trained\Knn Landmarks with Orientation_PR_data.csv",
    r"VVAD_PR_Curves\trained\Knn Landmarks without Orientation_PR_data.csv",
    r"VVAD_PR_Curves\trained\Mediapipe_PR_data.csv"
]

def plot_multiple_PR_curves(csv_files, title="Precision–Recall Curves", names=None, save_path=None):
    plt.figure(figsize=(8, 6))

    for i, csv_path in enumerate(csv_files):
        df = pd.read_csv(csv_path)
        if names is None:
            label = Path(csv_path).stem  # filename without extension
        else:
            label = names[i]
        # plt.plot(df["recall"], df["precision"], label=label, drawstyle="steps-post")
        plt.plot(df["recall"], df["precision"], label=label)
        auc_val = auc(df["recall"], df["precision"])
        print(f"{auc_val} {label}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    # plt.ylim(.5, 1)  # <-- set y-axis to 0–1
    # plt.xlim(0, 1)  # optional, keeps x-axis consistent too
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

def check_paths(csv_files):
    missing_file = False
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"{csv_file} does not exist\n")
            missing_file = True
    if not missing_file:
        print("All files present")

def get_auc(csv_files, names=None):
    for i, csv_path in enumerate(csv_files):
        df = pd.read_csv(csv_path)
        if names is None:
            label = Path(csv_path).stem  # filename without extension
        else:
            label = names[i]
        auc_val = auc(df["recall"], df["precision"])
        print(f"{auc_val} {label}")

if __name__ == "__main__":
    not_talking = ["Babble Noise", "Silence", "Still Face", "White Noise"]
    vvad_trained_models = ["Facial-OA", "Facial-Base", "KNN-OA", "KNN-Base", "MediaPipe"]
    syncnet_names = ["images", "GRU-Attention", "GRU", "OA-Facial"]

    plot_multiple_PR_curves(my_models_syncnet_csv_files, title=None, names=vvad_trained_models, save_path=r"Multi_PR_Curves\my_models_syncnet.png")
    plot_multiple_PR_curves(syncnet_csv_files, title=None, names=syncnet_names, save_path=r"Multi_PR_Curves\all_models_syncnet.png")

    # plot_multiple_PR_curves(mediapipe_reg_csv_files, title=None, names=not_talking, save_path=r"Multi_PR_Curves\mediapipe.png")
    # plot_multiple_PR_curves(vvad_trained, title=None, names=vvad_trained_models, save_path=r"Multi_PR_Curves\vvad_trained.png")
    # plot_multiple_PR_curves(norot_facial_neg_csv_files, title=None, names=not_talking, save_path=r"Multi_PR_Curves\norot_facial.png")
    # plot_multiple_PR_curves(rot_facial_neg_csv_files, title=None, names=not_talking, save_path=r"Multi_PR_Curves\rot_facial.png")
    # plot_multiple_PR_curves(norot_knn_neg_csv_files, title=None, names=not_talking, save_path=r"Multi_PR_Curves\norot_knn.png")
    # plot_multiple_PR_curves(rot_knn_neg_csv_files, title=None, names=not_talking, save_path=r"Multi_PR_Curves\rot_knn.png")
    # get_auc(csv_files_fconfm_neg)
    # plot_multiple_PR_curves(csv_files_fconfm_neg, title="PR Curve for Wav2Lip VVAD", names=not_talking, save_path="color_vvad.png")



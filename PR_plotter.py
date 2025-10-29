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

def plot_multiple_PR_curves(csv_files, title="Precision–Recall Curves", names=None, save_path=None):
    plt.figure(figsize=(8, 6))

    for i, csv_path in enumerate(csv_files):
        df = pd.read_csv(csv_path)
        print(len(df))
        if names is None:
            label = Path(csv_path).stem  # filename without extension
        else:
            label = names[i]
        # plt.plot(df["recall"], df["precision"], label=label, drawstyle="steps-post")
        plt.plot(df["recall"], df["precision"], label=label)

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
            print(f"{csv_file} does not exist")
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
    # get_auc(csv_files_fconfm_reg)
    get_auc(csv_files_fconfm_neg)
    # get_auc(csv_files_cosine_neg)
    plot_multiple_PR_curves(csv_files_fconfm_reg)
    plot_multiple_PR_curves(csv_files_cosine_neg)
    plot_multiple_PR_curves(csv_files_fconfm_neg, title="PR Curve for Wav2Lip VVAD", names=not_talking, save_path="color_vvad.png")

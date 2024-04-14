import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
from torchvision.transforms import Compose, Lambda, Resize, CenterCrop, Normalize, ToTensor

class YourVideoDataset(Dataset):
    def __init__(self, csv_file='../data/labels_filtered.csv', video_dir='../data/videos_filtered', transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with video paths and labels.
            video_dir (string): Directory with all the videos.
            transform (callable, optional): Optional transform to be applied on a video.
        """
        self.video_labels = pd.read_csv(csv_file)
        self.video_dir = video_dir
        self.transform = transform
        self.label_to_index = self._build_label_index(self.video_labels['label'])

    def _build_label_index(self, labels):
        """Build a dictionary mapping each unique label to a unique integer."""
        unique_labels = sorted(labels.unique())
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        return label_to_index
    
    def __len__(self):
        return len(self.video_labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        video_path = f"{self.video_dir}/{self.video_labels.iloc[idx, 1]}.mp4"
        label = self.video_labels.iloc[idx, 0]
        label_int = self.label_to_index[label]

        # video, _, _ = read_video(video_path, pts_unit='sec')  # Load video

        # if self.transform:
        #     video = self.transform(video)

        return video_path, label_int

# Define the transformation pipeline for your videos here
video_transforms = Compose([
    Lambda(lambda x: x / 255.),  # Normalize pixel values to [0, 1]
    Resize((128, 171)),  # Resize video frames
    CenterCrop(128),  # Crop video frames
    Lambda(lambda x: x.permute(3, 0, 1, 2)),  # Change shape from (T, H, W, C) to (C, T, H, W)
    # Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),  # Normalize
])



# Example usage
if __name__ == "__main__":
    dataset = YourVideoDataset(csv_file='../data/labels_filtered.csv',
                               video_dir='../data/videos_filtered',
                               transform=False)
    print(f"Dataset size: {len(dataset)}")
    # print(dataset[0])
    # Load and print details of the first video and its label
    # first_video, first_label = dataset[0]
    # print(f"First video: {first_video}, Label: {first_label}")

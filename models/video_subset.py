import os
import pandas as pd

# Path to the folder containing videos
folder_path = '../data/videos_subset_40'

# Path to your CSV file
csv_file_path = '../data/labels_filtered.csv'

# Get a list of all video names in the folder
video_files = {os.path.splitext(file)[0] for file in os.listdir(folder_path) if file.endswith(('.mp4'))}

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Filter the DataFrame to only include video names that exist in the folder
filtered_df = df[df['youtube_id'].isin(video_files)]

# Save the filtered DataFrame to a new CSV file (optional)
filtered_df.to_csv('subset_video_list.csv', index=False)

import pandas as pd
from timesformer_quantized import TimesformerQuantized
from timesformer_v2 import Timesformer
from resnet3d import predict
# Load models
timesformer_model = Timesformer()
timesformer_quantized = TimesformerQuantized()

# Load labels
labels_df = pd.read_csv('../data/labels_filtered.csv')
subset_labels_df = labels_df.head(100)
# Path to your video folder
video_folder_path = '../data/videos_filtered/'

def predict_and_evaluate(model, video_folder_path, labels_df):
    predictions = []
    for _, row in labels_df.iterrows():
        video_path = video_folder_path + row['youtube_id'] + '.mp4'
        if model:
            predicted_label = model.predict_label(video_path)
        else:
            predicted_label = predict(video_path)
        print(predicted_label)
        predictions.append(predicted_label)
    
    # Compare predictions with ground truth
    labels_df['predicted_label'] = predictions
    correct_predictions = (labels_df['label'] == labels_df['predicted_label']).sum()
    accuracy = correct_predictions / len(labels_df)
    
    return accuracy

# Calculate accuracies
timesformer_accuracy = predict_and_evaluate(timesformer_model, video_folder_path, subset_labels_df.copy())
resnet_3d_accuracy = predict_and_evaluate(timesformer_quantized, video_folder_path, subset_labels_df.copy())

print(f"Timesformer Accuracy: {timesformer_accuracy * 100} %")
print(f"ResNet 3D Accuracy: {resnet_3d_accuracy * 100} %")

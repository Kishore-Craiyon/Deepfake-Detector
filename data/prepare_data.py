import argparse
import json
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import wget
import zipfile

def download_datasets():
    """Download common deepfake datasets"""
    print("Note: You'll need to manually download datasets like:")
    print("1. FaceForensics++ (FF++): https://github.com/ondyari/FaceForensics")
    print("2. DFDC: https://ai.facebook.com/datasets/dfdc/")
    print("3. Celeb-DF: http://www.cs.albany.edu/~lsw/celeb-deepfakeforensics.html")
    print("\nPlace videos in: data/raw/real/ and data/raw/fake/")

def create_train_val_split(data_path, val_ratio=0.2, test_ratio=0.1):
    """Create train/validation/test splits"""
    data_path = Path(data_path)
    
    # Collect all video files
    real_videos = list((data_path / 'real').glob('*.mp4'))
    fake_videos = list((data_path / 'fake').glob('*.mp4'))
    
    # Create annotations
    annotations = []
    
    # Process real videos
    for video in real_videos:
        annotations.append({
            'video_path': str(video),
            'label': 0,  # Real
            'original_name': video.name
        })
    
    # Process fake videos
    for video in fake_videos:
        annotations.append({
            'video_path': str(video),
            'label': 1,  # Fake
            'original_name': video.name
        })
    
    # Split data
    train_data, temp_data = train_test_split(
        annotations, test_size=(val_ratio + test_ratio), 
        stratify=[a['label'] for a in annotations], random_state=42
    )
    
    val_data, test_data = train_test_split(
        temp_data, test_size=(test_ratio / (val_ratio + test_ratio)),
        stratify=[a['label'] for a in temp_data], random_state=42
    )
    
    # Add split information
    for item in train_data:
        item['split'] = 'train'
    for item in val_data:
        item['split'] = 'val'
    for item in test_data:
        item['split'] = 'test'
    
    all_annotations = train_data + val_data + test_data
    
    # Save annotations
    with open(data_path / 'annotations.json', 'w') as f:
        json.dump(all_annotations, f, indent=2)
    
    print(f"Created splits:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Validation: {len(val_data)} samples") 
    print(f"  Test: {len(test_data)} samples")
    
    return all_annotations

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help='Path to data directory')
    parser.add_argument('--download', action='store_true', help='Show download instructions')
    
    args = parser.parse_args()
    
    if args.download:
        download_datasets()
    else:
        create_train_val_split(args.data_path)

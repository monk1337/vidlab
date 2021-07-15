from tensorflow_docs.vis import embed
from tensorflow.keras import layers
from tensorflow import keras

import matplotlib.pyplot as plt
import shutil

import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import numpy as np
import imageio
import cv2
import os


MAX_SEQ_LENGTH = 20
NUM_FEATURES = 1024
IMG_SIZE = 128
EPOCHS = 5


def load_data(data_path, test_size = 0.3):
    
    train_data      = os.listdir(data_path)
    labels          = ['no_fight' if 'nofi' in tag else 'fight' for tag in train_data]
    

    # make dataframe
    train_df        = pd.DataFrame({'video_name': train_data, 'tag': labels})
    train_df        = train_df.sample(frac=1).reset_index(drop=True)
    print(f"Total videos for training: {len(train_df)}")
    
    # split the dataset into train and test
    train_size      = float(1.0 - test_size)
    msk             = np.random.rand(len(train_df)) < train_size
    train           = train_df[msk]
    test            = train_df[~msk]

    try:
      shutil.rmtree('train_data')
      shutil.rmtree('test_data')
    except Exception as e:
      pass

    os.makedirs('train_data')
    os.makedirs('test_data')

    real_path         = data_path
    new_train_path    = 'train_data'
    new_test_path     = 'test_data'

    for file in train['video_name']:
        shutil.copyfile(f"{real_path}/{file}", f"{new_train_path}/{file}")
        
    for file in test['video_name']:
        shutil.copyfile(f"{real_path}/{file}", f"{new_test_path}/{file}")

    return train, test


def get_features(data_path, test_size = 0.3):
    
    train_df, test_df = load_data(data_path, test_size)

    print(len(train_df), len(test_df))
    center_crop_layer = layers.experimental.preprocessing.CenterCrop(IMG_SIZE, IMG_SIZE)

    def crop_center(frame):
        cropped = center_crop_layer(frame[None, ...])
        cropped = cropped.numpy().squeeze()
        return cropped

    # Following method is modified from this tutorial:
    # https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub
    def load_video(path, max_frames=0):
        cap = cv2.VideoCapture(path)
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = crop_center(frame)
                frame = frame[:, :, [2, 1, 0]]
                frames.append(frame)

                if len(frames) == max_frames:
                    break
        finally:
            cap.release()
        return np.array(frames)


    def build_feature_extractor():
        feature_extractor = keras.applications.DenseNet121(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
        )
        preprocess_input = keras.applications.densenet.preprocess_input

        inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
        preprocessed = preprocess_input(inputs)

        outputs = feature_extractor(preprocessed)
        return keras.Model(inputs, outputs, name="feature_extractor")


    feature_extractor = build_feature_extractor()


    # Label preprocessing with StringLookup.
    label_processor = keras.layers.experimental.preprocessing.StringLookup(
        num_oov_indices=0, vocabulary=np.unique(train_df["tag"]), mask_token=None
    )
    print(label_processor.get_vocabulary())


    def prepare_all_videos(df, root_dir):
        num_samples = len(df)
        video_paths = df["video_name"].values.tolist()
        labels = df["tag"].values
        labels = label_processor(labels[..., None]).numpy()

        # `frame_features` are what we will feed to our sequence model.
        frame_features = np.zeros(
            shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )

        # For each video.
        for idx, path in tqdm(enumerate(video_paths)):
            # Gather all its frames and add a batch dimension.
            frames = load_video(os.path.join(root_dir, path))

            # Pad shorter videos.
            if len(frames) < MAX_SEQ_LENGTH:
                diff = MAX_SEQ_LENGTH - len(frames)
                padding = np.zeros((diff, IMG_SIZE, IMG_SIZE, 3))
                frames = np.concatenate(frames, padding)

            frames = frames[None, ...]

            # Initialize placeholder to store the features of the current video.
            temp_frame_featutes = np.zeros(
                shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
            )

            # Extract features from the frames of the current video.
            for i, batch in enumerate(frames):
                video_length = batch.shape[1]
                length = min(MAX_SEQ_LENGTH, video_length)
                for j in range(length):
                    if np.mean(batch[j, :]) > 0.0:
                        temp_frame_featutes[i, j, :] = feature_extractor.predict(
                            batch[None, j, :]
                        )

                    else:
                        temp_frame_featutes[i, j, :] = 0.0

            frame_features[idx,] = temp_frame_featutes.squeeze()

        return frame_features, labels

    train_data, train_labels = prepare_all_videos(train_df,   'train_data')
    test_data, test_labels   = prepare_all_videos(test_df,    'test_data')

    print(f"Frame features in train set: {train_data.shape}")
    print(f"Frame masks in train set: {test_data.shape}")

    np.savez('final_features.npz', train_data= train_data, train_labels= train_labels, 
    test_data = test_data, test_labels = test_labels)

    return train_data, train_labels, test_data, test_labels, label_processor

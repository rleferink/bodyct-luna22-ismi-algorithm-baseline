from typing import Dict

import SimpleITK
import numpy as np
from pathlib import Path
import json

import tensorflow.keras
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import categorical_crossentropy, mse
from tensorflow.keras.applications import VGG16

import numpy as np

l2_lambda = 0.0002
DropP = 0.3


# Enforce some Keras backend settings that we need
tensorflow.keras.backend.set_image_data_format("channels_first")
tensorflow.keras.backend.set_floatx("float32")
from data import (
    center_crop_volume,
    get_cross_slices_from_cube,
)


def clip_and_scale(
    data: np.ndarray,
    min_value: float = -1000.0,
    max_value: float = 400.0,
) -> np.ndarray:
    data = (data - min_value) / (max_value - min_value)
    data[data > 1] = 1.0
    data[data < 0] = 0.0
    return data


class Nodule_classifier:
    def __init__(self):

        self.input_size = 224
        self.input_spacing = 0.2

        # load malignancy model
        self.model_malignancy = VGG16(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=2,
            classifier_activation="softmax",
        )
        self.model_malignancy.load_weights(
            "/opt/algorithm/models/vgg16_malignancy_best_val_accuracy.h5",
            by_name=True,
            skip_mismatch=True,
        )

        # load texture model
        self.model_nodule_type = VGG16(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=3,
            classifier_activation="softmax",
        )
        self.model_nodule_type.load_weights(
            "/opt/algorithm/models/vgg16_noduletype_best_val_accuracy.h5",
            by_name=True,
            skip_mismatch=True,
        )

        print("Models initialized")

    def load_image(self) -> SimpleITK.Image:

        ct_image_path = list(Path("/input/images/ct/").glob("*"))[0]
        image = SimpleITK.ReadImage(str(ct_image_path))

        return image

    def preprocess(
        self,
        img: SimpleITK.Image,
    ) -> SimpleITK.Image:

        # Resample image
        original_spacing_mm = img.GetSpacing()
        original_size = img.GetSize()
        new_spacing = (self.input_spacing, self.input_spacing, self.input_spacing)
        new_size = [
            int(round(osz * ospc / nspc))
            for osz, ospc, nspc in zip(
                original_size,
                original_spacing_mm,
                new_spacing,
            )
        ]
        resampled_img = SimpleITK.Resample(
            img,
            new_size,
            SimpleITK.Transform(),
            SimpleITK.sitkLinear,
            img.GetOrigin(),
            new_spacing,
            img.GetDirection(),
            0,
            img.GetPixelID(),
        )

        # Return image data as a numpy array
        return SimpleITK.GetArrayFromImage(resampled_img)

    def predict(self, input_image: SimpleITK.Image) -> Dict:

        print(f"Processing image of size: {input_image.GetSize()}")

        nodule_data = self.preprocess(input_image)

        # Crop a volume of 50 mm^3 around the nodule
        nodule_data = center_crop_volume(
            volume=nodule_data,
            crop_size=np.array(
                (
                    self.input_size,
                    self.input_size,
                    self.input_size,
                )
            ),
            pad_if_too_small=True,
            pad_value=-1024,
        )

        # Extract the axial/coronal/sagittal center slices of the 50 mm^3 cube
        nodule_data = get_cross_slices_from_cube(volume=nodule_data)
        nodule_data = clip_and_scale(nodule_data)

        malignancy = self.model_malignancy(nodule_data[None]).numpy()[0, 1]
        texture = np.argmax(self.model_nodule_type(nodule_data[None]).numpy())

        result = dict(
            malignancy_risk=round(float(malignancy), 3),
            texture=int(texture),
        )

        return result

    def write_outputs(self, outputs: dict):

        with open("/output/lung-nodule-malignancy-risk.json", "w") as f:
            json.dump(outputs["malignancy_risk"], f)

        with open("/output/lung-nodule-type.json", "w") as f:
            json.dump(outputs["texture"], f)

    def process(self):

        image = self.load_image()
        result = self.predict(image)
        self.write_outputs(result)


"""
Densenet code
"""
def dense_block(conv):
    x = tensorflow.keras.layers.Conv3D(filters=12, kernel_size=(3, 3, 3), activation='relu', padding='same',
                                       kernel_regularizer=tensorflow.keras.regularizers.l2(l2_lambda))(conv)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    y = tensorflow.keras.layers.Conv3D(filters=12, kernel_size=(3, 3, 3), activation='relu', padding='same',
                                       kernel_regularizer=tensorflow.keras.regularizers.l2(l2_lambda))(x)
    y = tensorflow.keras.layers.BatchNormalization()(y)
    merge = tensorflow.keras.layers.Concatenate(axis=1)([x, y])
    x = tensorflow.keras.layers.Conv3D(filters=12, kernel_size=(3, 3, 3), activation='relu', padding='same',
                                       kernel_regularizer=tensorflow.keras.regularizers.l2(l2_lambda))(merge)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    merge = tensorflow.keras.layers.Concatenate(axis=1)([merge, x])
    x = tensorflow.keras.layers.Conv3D(filters=12, kernel_size=(3, 3, 3), activation='relu', padding='same',
                                       kernel_regularizer=tensorflow.keras.regularizers.l2(l2_lambda))(merge)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    return x


def denser_block(conv, layers, filters):
    x = tensorflow.keras.layers.Conv3D(filters=filters, kernel_size=(3, 3, 3), activation='relu', padding='same',
                                       kernel_regularizer=tensorflow.keras.regularizers.l2(l2_lambda))(conv)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    y = tensorflow.keras.layers.Conv3D(filters=filters, kernel_size=(3, 3, 3), activation='relu', padding='same',
                                       kernel_regularizer=tensorflow.keras.regularizers.l2(l2_lambda))(x)
    y = tensorflow.keras.layers.BatchNormalization()(y)
    merge = tensorflow.keras.layers.Concatenate(axis=1)([x, y])
    for z in range(layers - 2):
        x = tensorflow.keras.layers.Conv3D(filters=filters, kernel_size=(3, 3, 3), activation='relu',
                                           padding='same',
                                           kernel_regularizer=tensorflow.keras.regularizers.l2(l2_lambda))(merge)
        x = tensorflow.keras.layers.BatchNormalization()(x)
        merge = tensorflow.keras.layers.Concatenate(axis=1)([merge, x])
    return x


def multi_dense_model(input):
    input_layer = tensorflow.keras.layers.Input(shape=input)
    prepool = tensorflow.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same')(input_layer)

    pool1 = tensorflow.keras.layers.MaxPooling3D(pool_size=(2, 2, 1))(dense_block(prepool))
    pool1 = tensorflow.keras.layers.Dropout(DropP)(pool1)
    flatten1 = tensorflow.keras.layers.Flatten()(pool1)

    pool2 = tensorflow.keras.layers.MaxPooling3D(pool_size=(2, 2, 1))(denser_block(pool1, 10, 12))
    pool2 = tensorflow.keras.layers.Dropout(DropP)(pool2)
    flatten2 = tensorflow.keras.layers.Flatten()(pool2)

    pool3 = tensorflow.keras.layers.MaxPooling3D(pool_size=(2, 2, 1))(denser_block(pool2, 20, 12))
    pool3 = tensorflow.keras.layers.Dropout(DropP)(pool3)
    flatten3 = tensorflow.keras.layers.Flatten()(pool3)

    pool4 = tensorflow.keras.layers.MaxPooling3D(pool_size=(2, 2, 1))(denser_block(pool3, 20, 24))
    pool4 = tensorflow.keras.layers.Dropout(DropP)(pool4)
    flatten4 = tensorflow.keras.layers.Flatten()(pool4)

    pool5 = denser_block(pool4, 20, 48)
    flatten5 = tensorflow.keras.layers.Flatten()(pool5)

    final_merge = tensorflow.keras.layers.Concatenate(axis=1)([flatten1, flatten2, flatten3, flatten4, flatten5])
    output = tensorflow.keras.layers.Dense(1, activation='sigmoid', name='output')(final_merge)
    output_malignancy = tensorflow.keras.layers.Dense(2, activation='softmax', name='malignancy_regression')(output)
    output_type = tensorflow.keras.layers.Dense(3, activation='softmax', name='type_classification')(output)
    final_model = tensorflow.keras.models.Model(inputs=[input_layer], outputs=[output_malignancy, output_type])
    return final_model

if __name__ == "__main__":
    Nodule_classifier().process()

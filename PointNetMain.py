import os
import glob
from tensorflow.python.keras.utils import layer_utils
import trimesh
import pandas as pd
import open3d as o3d
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

DATASET_PATH = "." + os.path.sep + "L515_DataSet" + os.path.sep
DATASET_LABEL_PATH = "." + os.path.sep + "labels.xlsx"

def parse_dataset(labels):
  dataset = {}
  dataset_names = []
  for pcd in os.listdir(DATASET_PATH):
    if ".pcd" in pcd:
      dataset_names.append(pcd)
      dataset[pcd] = np.asarray(o3d.io.read_point_cloud(DATASET_PATH + pcd).points)
  return (
      dataset_names,
      dataset
  )

def configure_sets(dataset_names, dataset, labels):
  train_set = []
  train_labels = []
  for pcd in dataset_names:
    train_set.append(dataset[pcd])
    train_labels.append(labels[pcd])
    
  return (
    train_set,
    train_labels
  )
  
pcd_list = glob.glob("./L515_DataSet/*.pcd")
points = []
min_points = -1
for cloud in pcd_list:
  points.append(len(o3d.io.read_point_cloud(cloud).points))
min_points = min(points)

for cloud in pcd_list:
  pcd = o3d.io.read_point_cloud(cloud)
  pcd_len = len(pcd.points)
  difference = pcd_len - min_points
  p_difference = (((pcd_len - difference) * 100)/pcd_len)/100
  o3d.io.write_point_cloud(cloud, o3d.geometry.PointCloud.random_down_sample(pcd, p_difference))

column_names = ["sample", "label"]
labels = pd.read_excel(DATASET_LABEL_PATH, names = column_names)

labels = labels.set_index('sample')['label'].to_dict()
dataset_names, dataset_points = parse_dataset(labels)

train_dataset, test_dataset = train_test_split(dataset_names, test_size=0.2, random_state=25)

train_points, train_labels = configure_sets(train_dataset, dataset_points, labels)
test_points, test_labels = configure_sets(test_dataset, dataset_points, labels)



def augment(points, label):
  # jitter points
  points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
  # shuffle points
  points = tf.random.shuffle(points)
  return points, label


train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(45)

test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))
test_dataset = test_dataset.shuffle(len(test_points)).map(augment).batch(45)


def conv_bn(x, filters):
  x = layers.Conv1D(filters, kernel_size=3, padding="valid")(x)
  x = layers.BatchNormalization(momentum=0.0)(x)
  return layers.Activation("tanh")(x)


def dense_bn(x, filters):
  x = layers.Dense(filters)(x)
  x = layers.BatchNormalization(momentum=0.0)(x)
  return layers.Activation("tanh")(x)


class OrthogonalRegularizer(keras.regularizers.Regularizer):
  def __init__(self, num_features, l2reg=0.001):
    self.num_features = num_features
    self.l2reg = l2reg
    self.eye = tf.eye(num_features)

  def __call__(self, x):
    x = tf.reshape(x, (-1, self.num_features, self.num_features))
    xxt = tf.tensordot(x, x, axes=(2, 2))
    xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
    return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

def tnet(inputs, num_features):
  bias = keras.initializers.Constant(np.eye(num_features).flatten())
  reg = OrthogonalRegularizer(num_features)

  x = conv_bn(inputs, 32)
  x = conv_bn(x, 64)
  x = conv_bn(x, 512)
  x = layers.GlobalMaxPooling1D()(x)
  x = dense_bn(x, 256)
  x = dense_bn(x, 128)
  x = layers.Dense(
    num_features * num_features,
    kernel_initializer="zeros",
    bias_initializer=bias,
    activity_regularizer=reg,
  )(x)
  feat_T = layers.Reshape((num_features, num_features))(x)
  # Apply affine transformation to input features
  return layers.Dot(axes=(2, 1))([inputs, feat_T])

inputs = keras.Input(shape = (min_points, 3))

x = tnet(inputs, 3)
x = conv_bn(x, 32)
x = conv_bn(x, 32)
x = tnet(x, 32)
x = conv_bn(x, 32)
x = conv_bn(x, 64)
x = conv_bn(x, 512)
x = layers.GlobalMaxPooling1D()(x)
x = dense_bn(x, 256)
x = layers.Dropout(0.3)(x)
x = dense_bn(x, 128)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(2, activation = "softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
model.summary()


model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["sparse_categorical_accuracy"],
)

model.fit(train_dataset, epochs=30, validation_data=test_dataset)

model.save_weights("./PointNet")

inputs = keras.Input(shape = (min_points, 3))

x = tnet(inputs, 3)
x = conv_bn(x, 32)
x = conv_bn(x, 32)
x = tnet(x, 32)
x = conv_bn(x, 32)
x = conv_bn(x, 64)
x = conv_bn(x, 512)
x = layers.GlobalMaxPooling1D()(x)
x = dense_bn(x, 256)
x = layers.Dropout(0.3)(x)
x = dense_bn(x, 128)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(2, activation="sigmoid")(x)

new_model = keras.Model(inputs=inputs, outputs=outputs, name="new_PointNet")

new_model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["sparse_categorical_accuracy"],
)

new_model.load_weights("./PointNet")
new_model.summary()

def prepare(cloud_path):
  pcd = o3d.io.read_point_cloud(cloud_path)
  pcd_len = len(pcd.points)
  difference = pcd_len - min_points
  p_difference = (((pcd_len - difference) * 100)/pcd_len)/100
  pcd_downsample = pcd.random_down_sample(p_difference)
  return np.asarray(pcd_downsample.points)

CATEGORIES = ["NIJE KUTIJA", "KUTIJA"]

podatak = prepare("./L515_DataSet/out10.pcd")

prediction = new_model.predict(podatak[None,:])
print(prediction)

cathegory = ["NIJE KUTIJA", "KUTIJA"]

if(prediction[0][0] >= prediction[0][1]):
    pcd = o3d.io.read_point_cloud("./L515_DataSet/out0.pcd")
    o3d.visualization.draw_geometries([pcd], window_name='Nije kutija ' +  str((prediction[0][0]*100)), width=1366, height=768)
else:
    pcd = o3d.io.read_point_cloud("./L515_DataSet/out0.pcd")
    o3d.visualization.draw_geometries([pcd], window_name='Jeste kutija ' +  str((prediction[0][1]*100)), width=1366, height=768)






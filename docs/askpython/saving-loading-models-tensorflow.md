# 使用 TensorFlow 2.0+保存和加载模型

> 原文：<https://www.askpython.com/python-modules/saving-loading-models-tensorflow>

在本文中，我们将讨论使用 TensorFlow 2.0+保存加载模型。这是一篇初级到中级水平的文章，面向刚刚开始使用 TensorFlow 进行深度学习项目的人。

## 为什么需要保存模型？

作为深度学习的初学者，人们最常犯的一个错误是不保存他们的模型。

在训练期间和训练之后保存深度学习模型是一个很好的做法。它可以节省您的时间，并提高模型的可重复性。以下是您可能会考虑保存模型的几个原因:

*   用数百万个参数和庞大的数据集训练现代深度学习模型在计算和时间方面可能是昂贵的。而且，在不同的训练过程中，你可以得到不同的结果/准确度。因此，使用保存的模型来展示您的结果总是一个好主意，而不是在现场进行培训。
*   保存同一个模型的不同版本可以让你检查和理解模型的工作。
*   您可以在支持 TensorFlow 的不同语言和平台中使用相同的编译模型，例如:TensorFlow Lite 和 TensorFlow JS，而无需转换任何代码。

TensorFlow 恰好提供了许多保存模型的方法。我们将在接下来的几节中详细讨论它们。

### 培训时如何保存模型？

有时在模型训练期间保存模型权重是很重要的。如果在某个时期后结果出现异常，使用检查点可以更容易地检查模型以前的状态，甚至恢复它们。

使用`Model.train()`函数训练张量流模型。我们需要使用`tf.keras.callbacks.ModelCheckpoint()`定义一个模型检查点回调，告诉编译器在特定的时期间隔保存模型权重。

回调听起来很难，但就用法而言并不难。这里有一个使用它的例子。

```py
# This is the initialization block of code
# Not important for understanding the saving
# But to execute the next cells containing the code
# for saving and loading

import tensorflow as tf
from tensorflow import keras

# We define a dummy sequential model.
# This function to create a model will be used throughout the article

def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
  ])

  model.compile(optimizer='adam',
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.metrics.SparseCategoricalAccuracy()])

  return model

# Create a basic model instance
model = create_model()

# Get the dataset

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

```

```py
# Create a new model using the function
model = create_model()

# Specify the checkpoint file 
# We use the str.format() for naming files according to epoch
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"

# Get the directory of checkpoint
checkpoint_dir = os.path.dirname(checkpoint_path)

# Define the batch size
batch_size = 32

# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=5*batch_size)

# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))

# Train the model with the the checkpoint callback
model.fit(train_images, train_labels,
          epochs=50, 
          batch_size=batch_size, 
          callbacks=[cp_callback],
          verbose=0)

```

### 从检查点加载

如果您想要恢复您创建的检查点，您可以使用模型，您可以使用`model.load_weights()`功能。

下面是加载权重的语法和示例。

```py
# Syntax

model.load_weights("<path the checkpoint file(*.cpt)>")

# Example 

# Finds the latest checkpoint
latest = tf.train.latest_checkpoint(checkpoint_dir)

# Create a new model
model = create_model()

# Load the weights of the latest checkpoint
model.load_weights(latest)

```

### 保存已训练模型的权重

模型也可以在训练后保存。该过程相对来说比训练期间的检查点简单得多。

为了在模型被训练之后保存权重文件，我们使用 Model.save_weights()函数。使用它的示例如下:

```py
# Save the weights
model.save_weights('./checkpoints/my_checkpoint')

# Create a new model instance
model = create_model()

# Restore the weights
model.load_weights('./checkpoints/my_checkpoint')

```

### 加载已训练模型的权重

要从权重加载模型，我们可以像加载检查点权重一样使用`Model.load_weights()`。事实上，权重存储为一个检查点文件。

```py
# Restore the weights
model.load_weights('./checkpoints/my_checkpoint')

```

### 保存和加载整个模型

在上一节中，我们看到了如何保存模型的权重。这有一定的问题。在将模型权重加载到模型之前，必须定义模型。实际模型和要加载权重的模型之间的任何结构差异都会导致错误。

此外，当我们想要跨不同平台使用模型时，这种节省权重的方法变得很困难。例如，您希望在使用 TensorFlow JS 的浏览器中使用 python 中训练的模型。

在这种情况下，您可能需要保存整个模型，即结构和重量。TensorFlow 允许您使用函数`Model.save()`保存模型。这里有一个这样做的例子。

```py
# Save the whole model in SaveModel format

model.save('my_model')

```

TensorFlow 还允许用户使用 HDF5 格式保存模型。要以 HDF5 格式保存模型，只需使用 hdf5 扩展名提及文件名。

```py
# Save the model in hdf5 format

# The .h5 extension indicates that the model is to be saved in the hdf5 extension.
model.save('my_model.h5')

```

*注:HDF5 在 TensorFlow 中成为主流之前，最初由 Keras 使用。TensorFlow 使用 SaveModel 格式，并且总是建议使用推荐的较新格式。*

您可以使用`tf.keras.models.load_model()`加载这些保存的模型。该函数自动截取模型是以 SaveModel 格式保存还是以 hdf5 格式保存。下面是一个这样做的示例:

```py
# For both hdf5 format and SaveModel format use the appropriate path to the file

# SaveModel Format
loaded_model = tf.keras.models.load_model('my_model')

# HDF5 format
loaded_model = tf.keras.models.load_model('my_model.h5')

```

## 结论

这就把我们带到了教程的结尾。希望您现在可以在培训过程中保存和加载模型。敬请关注，了解更多关于深度学习框架的信息，如 PyTorch、TensorFlow 和 JAX。
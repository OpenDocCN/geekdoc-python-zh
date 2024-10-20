# 使用 Keras TensorFlow 预测莎士比亚文本

> 原文：<https://www.askpython.com/python/examples/predict-shakespearean-text>

嘿伙计们！在本教程中，我们将了解如何使用 Python 中的 Keras TensorFlow API 创建递归神经网络模型来预测莎士比亚文本。

***也读作:[股价预测使用 Python](https://www.askpython.com/python/examples/stock-price-prediction-python)***

为了产生新的文本，我们将使用定制的 RNN 模型来训练 GitHub 莎士比亚文本数据集 。

* * *

## **第一步:导入库**

我们利用了一些最流行的深度学习库。Sweetviz 是一个新的软件包，可自动进行探索性数据分析，尤其有利于分析我们的训练数据集。

```py
pip install sweetviz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import sweetviz as sw
import seaborn as sns
sns.set()

```

## **第二步:加载数据集**

```py
shakespeare_url='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
filepath=keras.utils.get_file('shakespeare.txt',shakespeare_url)
with open(filepath) as f:
    shakespeare_text=f.read()

```

```py
Downloading data from https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
1122304/1115394 [==============================] - 0s 0us/step
1130496/1115394 [==============================] - 0s 0us/step

```

既然我们已经将数据集下载到 Python 笔记本中，我们需要在利用它进行训练之前对其进行预处理。

## **第三步:预处理数据集**

标记化是将冗长的文本字符串分成较小部分或标记的过程。较大的文本块可以被标记成句子，然后变成单词。

预处理还包括从生成的标记中删除标点符号。

```py
tokenizer=keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts(shakespeare_text)

max_id=len(tokenizer.word_index)
dataset_size=tokenizer.document_count
[encoded]=np.array(tokenizer.texts_to_sequences([shakespeare_text]))-1

```

## **步骤 4:准备数据集**

我们将使用`tf.data.Dataset`,它通常对大量的元素有用，比如大量的文本数据。

`Dataset.repeat()`遍历数据集并重复数据集指定的次数。`window()`就像一个滑动窗口，每次滑动指定数量的窗口，进行反复迭代。

```py
train_size=dataset_size*90//100
dataset=tf.data.Dataset.from_tensor_slices(encoded[:train_size])

n_steps=100
window_length=n_steps+1
dataset=dataset.repeat().window(window_length,shift=1,drop_remainder=True)

dataset=dataset.flat_map(lambda window: window.batch(window_length))

batch_size=32
dataset=dataset.shuffle(10000).batch(batch_size)
dataset=dataset.map(lambda windows: (windows[:,:-1],windows[:,1:]))
dataset=dataset.map(lambda X_batch,Y_batch: (tf.one_hot(X_batch,depth=max_id),Y_batch))
dataset=dataset.prefetch(1)

```

## **第五步:建立模型**

模型构建非常简单。我们将创建一个顺序模型，并向该模型添加具有某些特征的层。

```py
model=keras.models.Sequential()
model.add(keras.layers.GRU(128,return_sequences=True,input_shape=[None,max_id]))
model.add(keras.layers.GRU(128,return_sequences=True))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(max_id,activation='softmax')))

```

接下来，我们将编译模型并在数据集上拟合模型。我们将使用`Adam`优化器，但是你也可以根据你的喜好使用其他可用的优化器。

```py
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam')
history=model.fit(dataset,steps_per_epoch=train_size // batch_size,epochs=1)

```

```py
31370/31370 [==============================] - 1598s 51ms/step - loss: 0.9528

```

## **第六步:测试模型**

我们在下面提到的代码片段中定义了一些函数。这些函数将根据我们定义的模型预处理和准备输入数据，并预测下一个字符，直到指定的字符数。

```py
def preprocess(texts):
    X=np.array(tokenizer.texts_to_sequences(texts))-1
    return tf.one_hot(X,max_id)

def next_char(text,temperature=1):
    X_new=preprocess([text])
    y_proba=model.predict(X_new)[0,-1:,:]
    rescaled_logits=tf.math.log(y_proba)/temperature
    char_id=tf.random.categorical(rescaled_logits,num_samples=1)+1
    return tokenizer.sequences_to_texts(char_id.numpy())[0]

def complete_text(text,n_chars=50,temperature=1):
    for _ in range(n_chars):
        text+=next_char(text,temperature)
    return text

```

让我们使用下面提到的代码来预测某个字母或单词的文本。

```py
print("Some predicted texts for letter 'D' are as follows:\n ")
for i in range(3):
  print(complete_text('d'))
  print()

```

```py
Some predicted texts for letter 'D' are as follows:

d, swalld tell you in mine,
the remeiviss if i shou

dima's for me, sir, to comes what this roguty.

dening to girl, ne'er i was deckong?
which never be

```

```py
print("Some predicted texts for word 'SHINE' are as follows:\n ")
for i in range(3):
  print(complete_text('shine'))
  print()

```

**输出:**

```py
Some predicted texts for word 'SHINE' are as follows:

shine on here is your viririno penaite the cursue,
i'll

shine yet it the become done to-k
make you his ocrowing

shine dises'-leck a word or my head
not oning,
so long 

```

* * *

## **结论**

恭喜你！您刚刚学习了如何使用 RNN 构建一个莎士比亚文本预测器。希望你喜欢它！😇

喜欢这个教程吗？无论如何，我建议你看一下下面提到的教程:

1.  [利用 Python 预测股价](https://www.askpython.com/python/examples/stock-price-prediction-python)
2.  [用 Python 进行加密价格预测](https://www.askpython.com/python/examples/crypto-price-prediction)
3.  [利用 Python 进行股票价格预测](https://www.askpython.com/python/examples/stock-price-prediction-python)
4.  [Python 中的票房收入预测——简单易行](https://www.askpython.com/python/examples/box-office-revenue-prediction)

感谢您抽出时间！希望你学到了新的东西！！😄

* * *
# 如何阅读？Python 中的数据文件？

> 原文：<https://www.askpython.com/python/examples/read-data-files-in-python>

在处理训练模型的数据输入和数据收集时，我们遇到了**。数据文件**。

这是一些软件用来存储数据的文件扩展名，其中一个例子是专门从事**统计分析**和**数据挖掘**的**分析工作室**。

与**一起工作。data** 文件扩展名非常简单，或多或少地识别数据的排序方式，然后使用 Python 命令相应地访问文件。

## 什么是. data 文件？

**。数据**文件是作为存储数据的一种手段而开发的。

很多时候，这种格式的数据要么是以**逗号分隔值**格式放置，要么是以**制表符分隔值**格式放置。

除此之外，该文件还可以是文本文件格式或二进制格式。在这种情况下，我们将需要以不同的方法访问它。

我们将和**一起工作。csv** 文件，但是让我们首先确定文件的内容是文本格式还是二进制格式。

## 识别里面的数据。*数据文件*

**。数据文件有两种不同的形式，文件本身要么是文本形式，要么是二进制形式。**

为了找出它属于哪一个，我们需要加载它并亲自测试。

我们开始吧！

### 1.测试:文本文件

。数据文件可能主要以文本文件的形式存在，在 Python 中访问文件非常简单。

作为 Python 中包含的一个特性，我们不需要导入任何模块来处理文件。

也就是说，在 Python 中打开、读取和写入文件的方式如下:

```py
# reading from the file
file = open("biscuits.data", "r")
file.read()
file.close()

# writing to the file
file = open("biscuits.data", "w")
file.write("Chocolate Chip")
file.close()

```

### 2.测试:二进制文件

的。数据文件也可以是二进制文件的形式。这意味着我们访问文件的方式也需要改变。

我们将使用二进制模式[读写文件](https://www.askpython.com/python/built-in-methods/python-write-file)，在这种情况下，模式是 **rb** ，或者*读取二进制*。

```py
# reading from the file
file = open("biscuits.data", "rb")
file.read()
file.close()

# writing to the file
file = open("biscuits.data", "wb")
file.write("Oreos")
file.close()

```

在 Python 中，文件操作相对容易理解，如果您希望了解不同的文件访问模式和访问方法，这是值得研究的。

这两种方法中的任何一种都应该有效，并且应该为您提供一种方法来检索关于存储在**中的内容的信息。数据**文件。

现在我们知道了文件的格式，我们可以使用 pandas 为 **csv** 文件创建一个数据帧。

### 3.用熊猫来阅读。*数据*文件

在检查提供的内容类型后，从这些文件中提取信息的一个简单方法是简单地使用 Pandas 提供的 [read_csv()](https://www.askpython.com/python-modules/pandas/python-pandas-module-tutorial#importing-data-from-csv-file-to-dataframe) 函数。

```py
import pandas as pd
# reading csv files
data =  pd.read_csv('file.data', sep=",")
print(data)

# reading tsv files
data = pd.read_csv('otherfile.data', sep="\t")
print(data)

```

该方法还自动将数据转换成数据帧。

下面使用的是一个[样本 csv 文件](https://www.stats.govt.nz/large-datasets/csv-files-for-download/)，它被重新格式化为一个**。数据**文件，并使用上面给出的相同代码进行访问。

```py
   Series reference                                        Description   Period  Previously published  Revised
0    PPIQ.SQU900000                 PPI output index - All industries   2020.06                  1183     1184
1    PPIQ.SQU900001         PPI output index - All industries excl OOD  2020.06                  1180     1181
2    PPIQ.SQUC76745  PPI published output commodity - Transport sup...  2020.06                  1400     1603
3    PPIQ.SQUCC3100  PPI output index level 3 - Wood product manufa...  2020.06                  1169     1170
4    PPIQ.SQUCC3110  PPI output index level 4 - Wood product manufa...  2020.06                  1169     1170
..              ...                                                ...      ...                   ...      ...
73   PPIQ.SQNMN2100  PPI input index level 3 - Administrative and s...  2020.06                  1194     1195
74   PPIQ.SQNRS211X     PPI input index level 4 - Repair & maintenance  2020.06                  1126     1127
75       FPIQ.SEC14  Farm expenses price index - Dairy farms - Freight  2020.06                  1102     1120
76       FPIQ.SEC99  Farm expenses price index - Dairy farms - All ...  2020.06                  1067     1068
77       FPIQ.SEH14    Farm expenses price index - All farms - Freight  2020.06                  1102     1110

[78 rows x 5 columns]

```

如你所见，它确实给了我们一个数据帧作为输出。

## 存储数据的其他格式有哪些？

有时候，存储数据的默认方法并不能解决问题。那么，使用文件存储的替代方法是什么呢？

### 1.JSON 文件

作为一种存储信息的方法， **JSON** 是一种非常好的数据结构，Python 对 [**JSON** 模块的巨大支持让集成看起来完美无瑕。](https://www.askpython.com/python-modules/python-json-module)

然而，为了在 Python 中使用它，您需要在脚本中导入`json`模块。

```py
import json

```

现在，在构建了一个 **JSON** 兼容结构之后，存储它的方法是一个简单的带有`json dumps`的文件操作。

```py
# dumping the structure in the form of a JSON object in the file.
with open("file.json", "w") as f:
    json.dumps(['foo', {'bar': ('baz', None, 1.0, 2)}], f)
# you can also sort the keys, and pretty print the input using this module
with open("file.json", "w") as f:
    json.dumps(['foo', {'bar': ('baz', None, 1.0, 2)}], f, indent=4,  sort_keys=True)

```

*注意，我们使用变量 **f** 将数据转储到文件中。*

从一个 **JSON** 文件中检索信息的等价函数叫做`load`。

```py
with open('file.json') as f:
    data = json.load(f)

```

这为我们提供了文件中的 **JSON** 对象的结构和信息。

### 2.泡菜

通常，当您存储信息时，信息以原始字符串格式存储，导致对象丢失其属性，我们需要通过 Python 从字符串重建对象。

pickle 模块是用来解决这个问题的，它是为序列化和反序列化 Python 对象结构而设计的，因此它可以存储在一个文件中。

这意味着您可以通过 pickle 存储一个列表，当下次 pickle 模块加载它时，您不会丢失 list 对象的任何属性。

为了使用它，我们需要导入`pickle`模块，没有必要安装它，因为它是标准 python 库的一部分。

```py
import pickle

```

让我们创建一个字典来处理到目前为止所有的文件操作。

```py
apple = {"name": "Apple", "price": 40}
banana = {"name": "Banana", "price": 60}
orange = {"name": "Orange", "price": 30}

fruitShop = {}
fruitShop["apple"] = apple
fruitShop["banana"] = banana
fruitShop["orange"] = orange

```

使用 pickle 模块就像使用 JSON 一样简单。

```py
file = open('fruitPickles', 'ab') 
# the 'ab' mode allows for us to append to the file  
# in a binary format

# the dump method appends to the file
# in a secure serialized format.
pickle.dump(fruitShop, file)                      
file.close()

file = open('fruitPickles', 'rb')
# now, we can read from the file through the loads function.
fruitShop = pickle.load(file)
file.close()

```

## 结论

你现在知道什么了。数据文件是什么，以及如何使用它们。除此之外，您还知道可以测试的其他选项，以便存储和检索数据。

查看我们的其他文章，获得关于这些模块的深入教程——[文件处理](https://www.askpython.com/python/python-file-handling)、[泡菜、](https://www.askpython.com/python-modules/pickle-module-python)和 [JSON](https://www.askpython.com/python/examples/read-a-json-file-in-python) 。

## 参考

*   StackOverflow 对。数据文件扩展名
*   [公文处理文档](https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files)
*   [官方 JSON 模块文档](https://docs.python.org/3/library/json.html)
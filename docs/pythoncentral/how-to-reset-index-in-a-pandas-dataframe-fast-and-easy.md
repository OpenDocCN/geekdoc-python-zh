# 如何在熊猫数据框架中重置索引(快速简单)

> 原文：<https://www.pythoncentral.io/how-to-reset-index-in-a-pandas-dataframe-fast-and-easy/>

Python pandas 库中的表格结构也称为数据帧。

这些结构用标签表示行和列。行标签称为索引，列标签称为列索引/标题。

通过过滤和操作大型数据集来创建数据帧。但是这些经过处理的数据帧与它们的原始数据集具有相同的索引。这需要重置数据帧的索引。

重置索引也很有帮助:

*   在预处理阶段，当丢弃缺失值或过滤数据时。除了使数据帧变小之外，它还使索引变得混乱。
*   当索引标签不能提供太多关于数据的信息时。
*   当索引需要被当作一个普通的 DataFrame 列时。

您可以在 pandas 中使用 reset_index()方法来重置数据帧中的索引。这样做可以将数据帧的原始索引转换为列。

在这篇文章中，我们将带你了解如何重置熊猫数据帧。

## **熊猫基础知识. reset_index**

除了将数据帧的索引重置为默认索引之外，reset_index()方法还有助于删除具有多索引的数据帧的一个或多个级别。

该方法的语法为:

```py
pandas.reset_index(level=None, drop=False, inplace=False, col_level=0, col_fill= ”)
```

其中参数具有以下含义:

| **参数** | **可接受的数据类型** | **默认值** | **描述** |
| 级别 | int，str，tuple，list | 无 | 默认删除所有级别。如果提到级别，它会删除这些级别。 |
| 下降 | bool | 假 | 默认情况下将旧索引添加到数据帧中。如果值更改为 True，则不会添加它。 |
| 原地 | 布尔 | 假 | 对当前数据帧对象进行修改 |
| col_level | int，str | 0 | 确定标签需要插入的层级(如果涉及多个层级)。默认情况下，标签插入到第一层(0)中。 |
| col_fill | 物体 |  | 如果列有多个级别，它确定如何命名级别。如果值为 None，则重复索引名称。 |

The。如果 inplace = True，reset_index()方法返回 None。否则，返回具有新索引的数据帧。

## **用重设索引。reset_index()**

使用。重置索引的 reset_index()方法与将该方法链接到 DataFrame 对象一样简单。

你必须从创建一个数据帧开始，就像这样:

```py
import pandas as pd
import numpy as np
import random
# We're making a DataFrame with an initial index. It represents marks out of 50. 

df = pd.DataFrame({
                    'Global Finance': [44, 29, 50, 17, 36],
                    'Politics': [31, 43, 21, 42, 17],
                    'Family Enterprise': [30, 30, 16, 46, 41]
                  }, index=['Leonard', 'Brayan', 'Wendy', 'Nathaniel', 'Edwin']
                 )
df
```

在表格形式中，数据帧看起来像这样:

|   | **全球金融** | **政治** | **家族企业** |
| 伦纳德 | 44 | 31 | 30 |
| 布雷安 | 29 | 43 | 30 |
| 温迪 | 50 | 21 | 16 |
| 纳撒尼尔 | 71 | 42 | 46 |
| 埃德温 | 36 | 17 | 41 |

重置指数现在只需打个电话就行了。reset_index()，像这样:

```py
df.reset_index()
```

索引将作为名为“index”的新列应用于数据帧它从零开始，沿着数据帧的长度继续。它看起来像这样:

|   | **索引** | **全球金融** | **政治** | **家族企业** |
| **0** | 伦纳德 | 44 | 31 | 30 |
| **1** | 布雷安 | 29 | 43 | 30 |
| **2** | 温迪 | 50 | 21 | 16 |
| **3** | 纳撒尼尔 | 71 | 42 | 46 |
| **4** | 埃德温 | 36 | 17 | 41 |

### **保持对数据帧的改变**

您在上面看到的输出表明数据帧的索引已经被更改。但是，如果您运行“df”，您将会看到更改没有持久化，并且输出没有索引。

如果您希望更改持续，您必须使用“inplace”参数，将其值设置为“True”下面是它运行的样子:

```py
df.reset_index(inplace=True)

df
```

这段代码的输出将是:

|   | **索引** | **全球金融** | **政治** | **家族企业** |
| **0** | 伦纳德 | 44 | 31 | 30 |
| **1** | 布雷安 | 29 | 43 | 30 |
| **2** | 温迪 | 50 | 21 | 16 |
| **3** | 纳撒尼尔 | 71 | 42 | 46 |
| **4** | 埃德温 | 36 | 17 | 41 |

## **用命名索引**重置数据帧中的索引

如果数据帧具有命名索引，即具有名称的索引，则重置索引将导致所讨论的命名索引成为数据帧中的列名。

让我们看看这是如何工作的，首先创建一个带有命名索引的数据帧:

```py
namedIndex = pd.Series(['Leonard', 'Brayan', 'Wendy', 'Nathaniel', 'Edwin'], name='initial_index') # Creating a series and giving it a name

df = pd.DataFrame({
                    'Global Finance': [44, 29, 50, 17, 36],
                    'Politics': [31, 43, 21, 42, 17],
                    'Family Enterprise': [30, 30, 16, 46, 41]
                  }, index=['Leonard', 'Brayan', 'Wendy', 'Nathaniel', 'Edwin']
                 ) # Creating the DataFrame, then passing the named series as the index

df
```

这个数据帧应该是这样的:

|   | **全球金融** | **政治** | **家族企业** |
| **初始 _ 索引** |   |   |   |
| 伦纳德 | 44 | 31 | 30 |
| 布雷安 | 29 | 43 | 30 |
| 温迪 | 50 | 21 | 16 |
| 纳撒尼尔 | 71 | 42 | 46 |
| 埃德温 | 36 | 17 | 41 |

执行 df.reset_index 会将表中的“initial_index”条目作为数据帧中的列名，如下:

|   | **初始 _ 索引** | **全球金融** | **政治** | **家族企业** |
| **0** | 伦纳德 | 44 | 31 | 30 |
| **1** | 布雷安 | 29 | 43 | 30 |
| **2** | 温迪 | 50 | 21 | 16 |
| **3** | 纳撒尼尔 | 71 | 42 | 46 |
| **4** | 埃德温 | 36 | 17 | 41 |

## **重置数据帧中的多级索引**

让我们来看看数据帧中的多级索引:

```py
# Creating a multi-level index
newIndex = pd.MultiIndex.from_tuples(
                                      [('BBA', 'Leonard'),
                                       ('BBA', 'Brayan'),
                                       ('MBA', 'Wendy'),
                                       ('MBA', 'Nathaniel'),
                                       ('BSC', 'Edwin')
                                      ],
                                  names= ['Branch', 'Name'])

# Creating multi-level columns
columns = pd.MultiIndex.from_tuples(
                                    [('subject1', 'Global Finance'),
                                     ('subject2', 'Politics'),
                                     ('subject3', 'Family Enterprise')
                                    ])

df = pd.DataFrame([
                    (45, 31, 30),
                    (29, 21, 30),
                    (50, 21, 16),
                    (17, 42, 46),
                    (36, 17, 41)      
                    ], index=newIndex, 
                    columns=columns)
df
```

其输出为:

|   |   | **主题 1** | **主题 2** | **主题 3** |
|  |   | **全球金融** | **政治** | **家族企业** |
| **分支** | **名称** |   |   |   |
| **BBA** | 伦纳德 | 44 | 31 | 30 |
| 布雷安 | 29 | 43 | 30 |
| **MBA** | 温迪 | 50 | 21 | 16 |
| 纳撒尼尔 | 71 | 42 | 46 |
| **BSC** | 埃德温 | 36 | 17 | 41 |

分支级别映射到多行，使其成为多级索引。应用。reset_index()函数将级别合并为数据帧中的列。

所以，运行“df.reset_index()”会这样:

|   |   |   | **主题 1** | **主题 2** | **主题 3** |
|   |   |   | **全球金融** | **政治** | **家族企业** |
|   | **分支** | **名称** |   |   |   |
| **0** | **BBA** | 伦纳德 | 44 | 31 | 30 |
| **1** | **BBA** | 布雷安 | 29 | 43 | 30 |
| **2** | **MBA** | 温迪 | 50 | 21 | 16 |
| **3** | **MBA** | 纳撒尼尔 | 71 | 42 | 46 |
| **4** | **BSC** | 埃德温 | 36 | 17 | 41 |

您也可以使用级别参数重置分支级别的索引，如下:

```py
df.reset_index(level='Branch')
```

产生输出:

|   |   | **主题 1** | **主题 2** | **主题 3** |
|   |   | **全球金融** | **政治** | **家族企业** |
| **名称** | **分支** |   |   |   |
| 伦纳德 | **BBA** | 44 | 31 | 30 |
| 布雷安 | **BBA** | 29 | 43 | 30 |
| 温迪 | **MBA** | 50 | 21 | 16 |
| 纳撒尼尔 | **MBA** | 71 | 42 | 46 |
| 埃德温 | **BSC** | 36 | 17 | 41 |
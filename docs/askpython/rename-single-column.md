# 重命名 Pandas 数据框架中的单个列

> 原文：<https://www.askpython.com/python-modules/pandas/rename-single-column>

在本文中，您将学习如何重命名 [pandas 数据帧](https://www.askpython.com/python-modules/pandas/dataframe-indexing)中的单个列。

***也读作:[用 Python 处理数据帧行和列](https://www.askpython.com/python-modules/pandas/dataframe-rows-and-columns)***

## 使用 rename()函数

要重命名列，我们使用 pandas DataFrame 的 rename()方法:

### rename()函数的参数

rename()函数支持以下参数:

*   **Mapper** :改变列名的函数字典。
*   **索引**:改变索引名称的字典或函数。
*   **列**:重命名列的字典或函数。
*   **轴**:定义目标轴，配合映射器使用。
*   **原地**:改变源数据帧。
*   **Errors** :如果发现任何错误的参数，则引发 KeyError。

**rename()函数要点:**

1.  甚至可以重命名多个列，以及单个列。
2.  用于明确说明意图。

### 如何重命名单个列？

让我们快速创建一个简单的数据框架，其中有几个名字和两列。您可以复制这个演示代码片段，或者使用您正在处理的 dataframe 来重命名单列。

```py
Import pandas as pd
d = {‘Name’ : [‘Raj’, ‘Neha’, ‘Virat’, ‘Deepika’], ‘Profession’ : [‘Artist’, ‘Singer’, ‘Cricketer’, ‘Actress’]}

df = pd.DataFrame(d)

print(df)

#Output: 
          Name          Profession
  0      Raj               Artist 
  1      Neha           Singer
  2      Virat            Cricketer
  3      Deepika       Actress

```

现在，让我们使用我们的 **rename()函数**来更改单个列的名称，而不编辑其中的数据。

```py
# rename single columns
df1 = df.rename(columns={‘Name’ : ‘PersonName’})
print(df1)

#output: 
          PersonName        Profession
  0      Raj                        Artist 
  1      Neha                     Singer
  2      Virat                      Cricketer
  3      Deepika                Actress

```

同样，我们可以更改另一个剩余列的名称:

```py
df2 = df1.rename(columns={‘Profession’ : ‘Prof’})
print(df2)

#output: 
          PersonName         Prof
  0      Raj                         Artist 
  1      Neha                     Singer
  2      Virat                      Cricketer
  3      Deepika                 Actress

```

## **结论**

我们希望这些解释和示例对您有所帮助，并且您可以在自己的项目中轻松使用它们。
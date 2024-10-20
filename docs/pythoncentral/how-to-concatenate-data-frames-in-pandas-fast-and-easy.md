# 如何在 Pandas 中连接数据帧(快速简单)

> 原文：<https://www.pythoncentral.io/how-to-concatenate-data-frames-in-pandas-fast-and-easy/>

使用熊猫合并()和。join()为您提供了一个混合了初始数据集行的数据集。这些行通常基于公共属性进行排列。

但是，如果父数据集的所有行之间没有任何匹配，那么结果数据集也有可能不包含这些行。

Pandas 提供了第三种处理数据集的方法:concat()函数。它沿着行或列轴将数据框缝合在一起。在本指南中，我们将带您了解如何使用函数来连接数据帧。

## **如何在 Pandas 中连接数据帧**

连接数据帧就是在第一个数据帧之后添加第二个数据帧。使用 concatenate 函数对两个数据帧执行此操作就像向它传递数据帧列表一样简单，就像这样:

```py
concatenation = pandas.concat([df1, df2])
```

请记住，上面的代码假设两个数据框中的列名相同。它沿着行连接数据框。

如果名称不同，并且代码被设置为沿行连接(被认为是轴 0)，则默认情况下也将添加列。另外，Python 会根据需要填充“NaN”值。

但是如果你想沿着列执行连接呢？concat()函数可以让您轻松实现这一点。

你可以像上面的例子一样调用这个函数——唯一的区别是你必须传递一个值为 1 或“列”的“轴”参数

下面是代码的样子:

```py
concatenation = pandas.concat([df1, df2], axis="columns")
```

在这个例子中，Python 假设数据帧之间的行是相同的。

但是，当您沿列进行连接并且行不同时，默认情况下，额外的行将被添加到结果数据框中。当然，如您所料，Python 会根据需要填充“NaN”值。

在下一节中，我们将看看 concat()可用的各种参数。

### **优化 concat()**

学习 concat()函数的基础知识可能已经向您展示了它是组合数据帧的最简单的方法之一。它通常用于创建一个大型集合，以便可以进行其他操作。

重要的是要记住，当 concat()被调用时，它会复制被连接的数据。因此，您必须仔细考虑是否需要多次 concat()调用。使用太多会降低程序速度。

如果打几个电话无法避免，可以考虑将复制参数设置为 False。

### **concat()中轴的作用**

您现在知道可以指定想要连接数据帧的轴。那么，当一个轴是首选时，另一个轴会发生什么呢？

由于串联函数总是默认产生一个集合并集——所有数据都被保留——另一个轴不会发生任何变化。

如果你用过。join()作为外部连接和 merge()之前，您可能已经注意到了这一点。您可以使用 join 参数强制实现这一点。

当使用 join 参数时，缺省为 outer。但是，内部选项也是可用的，允许您执行集合相交或内部联接。

但是请记住，在 concat()函数中以这种方式使用内部连接可能会导致少量的数据丢失，原因与常规内部连接发生数据丢失的原因相同。

只有轴标签匹配的行和列才会被保留。再次注意，join 参数只指示 pandas 处理你 *没有* 连接的轴。

### **concat()值得注意的参数**

下面快速浏览一下 concat()可以使用的一些最有用的参数:

*   **轴:** 该参数代表函数将连接的轴。默认情况下，它的值为 0，表示行数。但是您可以将该值设置为 1，以便沿着列进行连接。您还可以使用字符串“index”来表示行，使用“columns”来表示列。
*   **objs:** 它接受一个列表或你想要连接的数据帧或系列对象的任何序列。使用字典也是允许的，但是如果使用字典，Python 将使用键创建一个层次索引。
*   **ignore_index:** 它接受一个布尔值，默认为 False。设置为 True 时，创建的新数据框不会保留轴中的原始索引值，如轴参数所指定。因此，使用该参数可以提供新的索引值。
*   **键:** 它使你能够创建一个层次索引。最常见的使用方法是创建一个新的索引，同时保留原来的索引。这样，您就可以知道哪些行来自哪个数据框。
*   **复制:** 此参数说明是否要复制源数据。默认情况下，它的值为 True，但是如果设置为 False，Python 将不会复制数据。
*   **join:** 该参数的工作方式类似于 how 参数的工作方式，只是它只能接受值 inner 和 outer。默认情况下，它的值是 outer，它保存数据。但是，将其设置为 inner 会删除在其他数据集中没有匹配项的数据。

如果你想查看 concat()参数的详细列表，你可以在 [熊猫官方文档](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html#pandas.concat) 中找到。

## **正确使用 concat()的技巧**

下面是使用 concat()时需要记住的四件事，以及一些例子:

### **#1 小心使用索引和轴**

假设两个数据框保存着考试的结果，就像这样:

```py
firstDataFrame = pd.DataFrame({
    'name': ['I', 'J', 'K', 'L'],
    'science': [72,56,91, 83],
    'accounts': [67,95,80,77],
    'psychology': [81,71,87,86]
})secondDataFrame = pd.DataFrame({
    'name': ['M', 'N', 'O', 'P'],
    'science': [73,85,81,90],
    'accounts': [88,93,72,89],
    'psychology': [75,83,74,87]
})
```

现在，用 concat()方法连接的最简单的方法是向它传递一个数据帧列表。如您所知，默认情况下，该方法沿 0 轴垂直连接，并保留所有索引。

因此，使用简单连接的方法看起来应该是这样的:

```py
pd.concat([firstDataFrame, secondDataFrame])
```

您可能想要忽略预先存在的索引。在这种情况下，可以将 ignore_index 参数设置为 True。这样，得到的数据帧索引将被标记为从 0 到 n-1。

因此，对于这种情况，concat()方法应该这样调用:

```py
pd.concat([firstDataFrame, secondDataFrame], ignore_index=True)
```

您也可以选择水平连接数据帧。这就像将轴参数设置为 1 一样简单，就像这样:

```py
pd.concat([firstDataFrame, secondDataFrame], axis=1)
```

### **#2 避免重复索引**

如前所述，concat()函数按原样保存索引。但是您可能希望验证由 pd.concat()产生的索引没有重叠。

谢天谢地，这很容易做到。您必须将 verify_integrity 参数设置为 True 这样，如果有重复的索引，pandas 将引发异常。

我们来看一个相关的例子:

```py
try:
    pd.concat([firstDataFrame, secondDataFrame], verify_integrity=True)
except ValueError as e:
    print('ValueError', e)ValueError: Overlapping indices: Int64Index([0, 1, 2, 3], dtype='int64')
```

### **#3 使用分层索引简化数据分析**

将多级索引添加到连接的数据框中可使数据分析更加容易。继续考试结果数据框架示例，我们可以分别在 firstDataFrame 和 secondDataFrame 中添加学期 1 和学期 2 的指数。

实现这一点就像使用 keys 参数一样简单:

```py
res = pd.concat([firstDataFrame, secondDataFrame], keys=['Semester 1','Semester 2'])
res

```

要访问一组特定的值，您可以使用:

```py
res.loc['Semester 1']
```

您也可以使用 names 参数向分层索引中添加名称。让我们看看如何将名称“Class”添加到我们上面创建的索引中:

```py
pd.concat(
    [firstDataFrame, secondDataFrame], 
    keys=['Semester 1', 'Semester 2'],
    names=['Class', None],
)
```

也可以重置索引，然后将其转换为数据列。为此，您可以像这样使用 reset_index()方法:

```py
pd.concat(
    [firstDataFrame, secondDataFrame], 
    keys=['Semester 1', 'Semester 2'],
    names=['Class', None],
) .reset_index(level=0)
```

### **#4 考虑匹配和排序列**

concat()函数的一个优点是它可以对数据帧的列进行重新排序。默认情况下，该函数保持排序顺序与第一个数据框相同。

要按字母顺序对数据框进行排序，您可以将 sort 参数设置为 True，如下所示:

```py
pd.concat([firstDataFrame, secondDataFrame], sort=True)
```

你也可以像这样自定义排序参数:

```py
custom_sort = ['science', 'accounts', 'psychology', 'name']
res = pd.concat([firstDataFrame, secondDataFrame])
res[custom_sort]
```
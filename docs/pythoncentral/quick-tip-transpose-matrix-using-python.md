# 快速提示:如何使用 Python 转置矩阵

> 原文：<https://www.pythoncentral.io/quick-tip-transpose-matrix-using-python/>

你可能在数学课上记得这个，但是即使你不记得，也应该很容易理解。我们已经讨论了[矩阵](https://www.pythoncentral.io/multiply-matrices-python/)以及如何在 Python 中使用它们，今天我们将讨论如何快速简单地转置一个矩阵。当你转置一个矩阵时，你是在把它的列变成它的行。当你看到这样的例子时，就更容易理解了，所以看看下面的例子。

假设你的原始矩阵是这样的:

```py
x = [[1,2][3,4][5,6]]
```

在该矩阵中，有两列。第一个由 1、3 和 5 组成，第二个由 2、4 和 6 组成。当你转置矩阵时，列变成行。因此，上述矩阵的转置版本将如下所示:

```py
y = [[1,3,5][2,4,6]]
```

所以结果仍然是一个矩阵，但现在它的组织方式不同了，在不同的地方有不同的值。

在 Python 中自己转置一个矩阵实际上很容易。使用内置的压缩功能可以非常快速地完成这项工作。下面是它的样子:

```py
matrix = [[1,2][3.4][5,6]]
zip(*matrix)
```

上面代码的输出就是转置矩阵。超级简单。

你也可以使用 NumPy 转置一个矩阵，但是为了这样做，必须安装 NumPy，这是一种更为笨拙的方法，可以实现与 zip 函数同样的目标，而且非常快速简单。

既然您已经理解了什么是转置矩阵以及如何自己去做，那么在您自己的代码中尝试一下，看看它为您自己的定制函数和代码片段增加了什么类型的通用性和功能性。理解如何使用和操作矩阵真的可以为你的编码技能增加很多维度，这是一个放在你口袋里的好工具。
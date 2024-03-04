# Python 中的嵌套列表理解

> 原文：<https://www.pythonforbeginners.com/lists/nested-list-comprehensions-in-python>

Python 吸引程序员的一个方法是鼓励优雅、易读的代码。它通过各种功能做到这一点，包括列表理解。

编写更高效的代码有助于程序员节省时间和精力。[Python 中的列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)通过简化在 Python 中执行复杂语句所需的语法来实现这一目标。

嵌套列表理解更进一步，允许 Python 在一行代码中创建一个列表列表。这是一个强大而灵活的特性，通常用于生成矩阵。

## 为什么使用列表理解？

列表理解是一个有吸引力的特性，因为它可以节省程序员的时间和精力。简化语法意味着编码人员可以用更少的代码完成复杂的操作。

因为列表理解语句被简化了，它们通常更容易阅读。

使用列表理解的优势包括:

*   通常更容易阅读和维护
*   需要编写更少的代码
*   强大而灵活的功能
*   性能优于循环

然而，在每一种情况下，使用列表理解并不会使事情变得更容易。这就是为什么我们要深入研究一些何时以及如何使用这个流行的 Python 特性的例子。

## 如何使用嵌套列表理解

Python 中的列表理解使用*表示*，使用关键字中的*。这些关键字告诉 [Python 我们想要在一个 iterable 上循环](https://www.pythonforbeginners.com/loops/for-while-and-nested-loops-in-python)并对结果做一些事情。为了完成列表理解，我们需要一组方括号内的语句。*

**基本列表理解语法:**

```py
new_list = [expression for item in list]
```

**表达式**:该表达式用于修改报表中的各项。
**项:**可迭代
**列表中的元素:**可迭代对象。

**嵌套列表理解语法**:

```py
new_list = [[expression for item in list] for item in list] 
```

### 可重复的

使用 **range()** 函数生成 iterable 是 Python 中的一种常见技术。iterable 是一个可以迭代的 Python 对象，比如一个列表。

我们将使用 range()来构造 for 循环，我们可以用它来构建矩阵。

## 用 Python 构建矩阵

我们可以使用嵌套的方括号在 Python 中构建一个矩阵。在这个例子中，你可以看到我们正在创建一个列表列表。通过将三个不同的列表包装在另一个列表中，我们可以构建一个 Python 列表。

#### 示例 1:一个基本的 Python 矩阵

```py
# create a list of lists
matrix = [[0,1,2],[0,1,2],[0,1,2]]
print(matrix) 
```

**输出**

```py
[[0, 1, 2], [0, 1, 2], [0, 1, 2]]
```

或者，可以用一对嵌套的 for 循环和 **append()** 方法创建一个矩阵。

#### 示例 2:使用 for 循环创建矩阵

```py
matrix = []
for y in range(3):
    matrix.append([])
    for x in range(3):
        matrix[y].append(x)

print(matrix) 
```

**输出**

```py
[[0, 1, 2], [0, 1, 2], [0, 1, 2]]
```

最后，我们将使用 Python 列表理解创建一个矩阵。list comprehension 语句使用嵌套括号、 **range()** 函数以及关键字**表示**和中的**来构造语句。**

```py
matrix = [[x for x in range(3)] for y in range(3)]
print(matrix)
```

**输出**

```py
[[0, 1, 2], [0, 1, 2], [0, 1, 2]]
```

如您所见，list comprehension 语句比构造矩阵的 double for 循环方法占用更少的空间。

在每个示例中，输出都是相同的。每种技术都可以用来创建相同的 Python 列表。

然而，使用嵌套列表方法，我们只需要一行代码就可以获得想要的结果。我们使用的方法就像嵌套循环一样灵活，这种方法读写起来很麻烦。

## 运用列表理解的例子

让我们用列表理解来创建一个井字游戏板。大多数人都熟悉井字游戏。但如果你一直生活在岩石下，这是一个简单的领土游戏。

一个基本的井字游戏棋盘是一个 3×3 的正方形网格。我们可以用列表理解来创建游戏板，在每个方格中填入一个空格。

#### 示例 3:构建井字游戏棋盘

```py
tic_tac_toe_board = [[' ' for x in range(3)] for y in range(3)]

def PrintMatrix(matrix):
    for row in range(len(matrix)):
        print(matrix[row])

PrintMatrix(tic_tac_toe_board) 
```

**输出**

```py
[' ', ' ', ' ']
[' ', ' ', ' ']
[' ', ' ', ' '] 
```

我们可以使用列表符号在游戏板上放置一个“X”。

```py
tic_tac_toe_board[1][1] = 'X'
PrintMatrix(tic_tac_toe_board) 
```

**输出**

```py
[' ', ' ', ' ']
[' ', 'X', ' ']
[' ', ' ', ' '] 
```

## 从序列创建矩阵

使用列表理解可以将一系列数字转化为矩阵。通过操作列表理解语句的*表达式*部分，生成的矩阵中的每一项都将被修改。

我们将让矩阵存储一系列数字 0-8，均匀地分成三行。

#### 示例 4:使用表达式

```py
matrix = [[x+(y*3) for x in range(3)] for y in range(3)]

print(matrix) 
```

**输出**

```py
[[0, 1, 2], [3, 4, 5], [6, 7, 8]]
```

## 转置嵌套列表

我们可以使用 Python 列表理解来转置一个矩阵。要转置一个矩阵，我们需要把每一行变成一列。

#### 示例 5:转置矩阵

```py
matrix = [[1,2,3],
          [4,5,6],
          [7,8,9]]

transposed = [[x[y] for x in matrix] for y in range(len(matrix))]

print(transposed) 
```

**输出**

```py
[1, 4, 7]
[2, 5, 8]
[3, 6, 9] 
```

## 过滤嵌套列表理解语句

我们可以为列表理解提供一个条件，作为结果的过滤器。只有符合条件标准的项目才会被接受。

**条件:**选择满足条件标准的元素。

使用条件，我们可以告诉 Python 我们只对列表中的特定元素感兴趣。选择哪些元素将取决于所提供的条件。

例如，我们可以为列表理解提供一个只选择偶数的条件。

#### 示例 6:过滤偶数矩阵

```py
matrix = [x for x in range(1,10) if x%2 == 0]

print(matrix) 
```

**输出**

```py
[2, 4, 6, 8]]
```

此外，通过嵌套列表理解，我们可以创建一个偶数矩阵。

```py
matrix = [[x for x in range(1,10) if x%2 == 0] for y in range(2)]

print(matrix) 
```

**输出**

```py
[[2, 4, 6, 8], [2, 4, 6, 8]]
```

## 展平嵌套列表

也许我们需要将一个矩阵简化成一个列表。这在计算机图形和图像处理中有时是必要的。

我们可以在 Python 中使用列表理解来做到这一点。通过调整语句中的*表达式*，我们可以告诉 Python 展平原始的多维列表。

#### 例子 7:从二维走向一维

```py
matrix = [[1,2,3],
          [4,5,6],
          [7,8,9]]

# welcome to Flatland
flat = [x for row in matrix for x in row]

print(flat) 
```

**输出**

```py
[1, 2, 3, 4, 5, 6, 7, 8, 9]
```

让我们仔细看看这个语句:*flat =[x for row in matrix for x in row]。很困惑，对吧？虽然该语句是比 for 循环更有效的展平矩阵的方式，但这是列表理解可能比它的对应物更难理解的一种情况。*

下面是如何使用 for 循环和 **append()** 方法展平列表的方法。

#### 示例 8:平铺列表列表

```py
list = []
for row in matrix:
    for x in row:
        list.append(x)

print(list) 
```

**输出**

```py
[1, 2, 3, 4, 5, 6, 7, 8, 9]
```

作为一名程序员，决定什么时候使用列表理解取决于你。风格和技巧往往是主观的。重要的是你要有意识地努力改进你的代码，让它更容易理解。

## 用 Python 编写列表理解语句的技巧

Python 提供了列表理解，作为创建简洁、易读的代码的一种方式。为此，我们在 Python 中加入了一个简短的使用列表理解的技巧列表。

*   列表理解比使用标准 for 循环更有效。
*   以优雅代码的名义，我们应该保持列表理解语句简短。
*   使用条件增加列表理解表达式的灵活性。
*   列表理解非常适合创建矩阵。

什么时候使用列表理解没有固定的规则。您应该将其视为 Python 工具箱中的另一个工具，可以节省您的时间和精力。随着您的进步，学习何时使用列表理解来改进您的代码将变得更加明显。

## 结论

列表理解是 Python 的一个与众不同的特性，可以帮助编码人员编写优雅且易于理解的程序。使用列表理解不仅节省了代码行，而且通常比其他方法更容易阅读。

虽然列表理解通常被认为比其他方法更“Pythonic 化”，例如对于循环，但这不一定是真的。Python 代码是为灵活性和效率而设计的。Python 中的每个工具都有其优点和缺点。

为了充分理解 Python 列表，我们可以添加一个条件来过滤结果。使用过滤器增加了列表理解表达式的灵活性。

如果你渴望学习更多的 Python，我们也希望如此，这里有一些额外的链接，可以帮助你成为一名编程爱好者。

## 相关职位

*   了解如何使用 [Python 字符串串联](https://www.pythonforbeginners.com/concatenation/string-concatenation-and-formatting-in-python)制作更好的字符串。
*   如何使用 [Python split](https://www.pythonforbeginners.com/dictionary/python-split) ()将字符串拆分成单词？
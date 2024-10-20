# 用 Python 动态规划求解 0/1 背包

> 原文：<https://www.askpython.com/python/examples/knapsack-problem-dynamic-programming>

在本文中，我们将使用动态规划来解决 0/1 背包问题。

***动态规划**是一种算法技术，通过将其分解为更简单的子问题，并利用整体问题的最优解取决于其子问题的最优解的事实来解决优化问题* **。**

**0/1 背包**也许是动态规划下最流行的问题。为了掌握动态编程的诀窍，学习起来也是一个很大的问题。

在本教程中，我们将学习什么是 0/1 背包，以及如何使用动态编程在 Python 中解决它。

让我们开始吧。

## 0/1 背包的问题陈述

动态编程的问题陈述如下:

```py
Given weights and values of n items, put these items in a knapsack of capacity W to get the maximum total value in the knapsack.

```

首先，我们有一个权重数组，其中包含所有项目的权重。我们还有一个值数组，其中包含所有物品的值，我们还有一个背包的总重量。

根据这些信息，我们需要找到在重量限制范围内可以获得的最大值。

这个问题被称为 0/1 背包，因为我们既可以把一个项目作为一个整体包含进去，也可以把它排除在外。也就是说，我们不能拿一个项目的零头。

我们举个例子来了解一下。

取以下输入值。

```py
val = [50,100,150,200]
wt = [8,16,32,40]
W = 64

```

在这里，当我们包括项目 **1、2 和 4** 时，我们得到最大利润，总共得到 **200 + 50 + 100 = 350。**

因此，总利润为:

```py
350 

```

## 如何用动态规划求解 0/1 背包？

为了使用动态规划解决 0/1 背包问题，我们构建了一个具有以下维度的表。

```py
[n + 1][W + 1]

```

表格的行对应于从 **0 到 n** 的项目。

表中的列对应于从 **0 到 W.** 的重量限制

表格最后一个单元格的索引是:

```py
[n][W]

```

索引为[i][j]的单元格的值表示当考虑从 0 到 I 的项目并且总重量限制为 j 时可能的最大利润。

填完表格后，我们的答案会在表格的最后一个单元格中。

### 怎么填表？

让我们从将第 0 行和第 0 列设置为 0 开始。我们这样做是因为第 0 行意味着我们没有对象，第 0 列意味着可能的最大权重是 0。

现在对于每个单元格[i][j]，我们有两个选项:

1.  要么我们在最终选择中包含对象[i]。
2.  或者我们在最终选择中不包括对象[i]。

我们如何决定是否在选择中包含对象[i]?

包含对象[i]需要满足两个条件:

1.  包括物体[i]后的总重量**不应超过**的**重量限制。**
2.  包括对象[i]后的**利润**应该比不包括对象时的**大**。

让我们把对 0/1 背包的理解转换成 python 代码。

## 求解 0/1 背包的 Python 代码

让我们使用下面的[列表理解方法](https://www.askpython.com/python/list/python-list-comprehension)创建一个表格:

```py
table = [[0 for x in range(W + 1)] for x in range(n + 1)] 

```

我们将使用嵌套的 [for 循环](https://www.askpython.com/python/python-for-loop)遍历表格并填充每个单元格中的条目。

我们将以自下而上的方式填充表格。

```py
for i in range(n + 1): 
        for j in range(W + 1): 
            if i == 0 or j == 0: 
                table[i][j] = 0
            elif wt[i-1] <= j: 
                table[i][j] = max(val[i-1]  
+ table[i-1][j-wt[i-1]],  table[i-1][j]) 
            else: 
                table[i][j] = table[i-1][j] 

```

让我们一行一行地分解代码。

```py
  if i == 0 or j == 0: 
     table[i][j] = 0

```

这部分代码负责将第 0 行和第 0 列设置为 0。

```py
 elif wt[i-1] <= j: 

```

这行代码检查第 I 个对象的重量是否小于该单元格(j)允许的总重量。

```py
 table[i][j] = max(val[i-1]  
+ table[i-1][j-wt[i-1]],  table[i-1][j]) 

```

这行代码负责从我们可用的两个选项中选择最大值。我们可以包含该对象，也可以排除它。

这里的术语**table[I–1][j]**表示不包括第 I 项。术语**val[I–1]+table[I–1][j–wt[I–1]]**表示包含第 I 项。

```py
else:
  table[i][j] = table[i-1][j]

```

当第 I 个物体的重量大于允许极限(j)时，进入循环的这一部分。

当我们完成填充表格时，我们可以返回表格的最后一个单元格作为答案。

```py
return table[n][W]

```

### 背包求解函数的完整代码

求解背包问题的函数的完整代码如下所示:

```py
def knapSack(W, wt, val): 
    n=len(val)
    table = [[0 for x in range(W + 1)] for x in range(n + 1)] 

    for i in range(n + 1): 
        for j in range(W + 1): 
            if i == 0 or j == 0: 
                table[i][j] = 0
            elif wt[i-1] <= j: 
                table[i][j] = max(val[i-1]  
+ table[i-1][j-wt[i-1]],  table[i-1][j]) 
            else: 
                table[i][j] = table[i-1][j] 

    return table[n][W] 

```

让我们试着运行上面例子中的函数。

```py
val = [50,100,150,200]
wt = [8,16,32,40]
W = 64

print(knapSack(W, wt, val))

```

## 完全码

这是在您的系统上运行的完整代码。

```py
def knapSack(W, wt, val): 
    n=len(val)
    table = [[0 for x in range(W + 1)] for x in range(n + 1)] 

    for i in range(n + 1): 
        for j in range(W + 1): 
            if i == 0 or j == 0: 
                table[i][j] = 0
            elif wt[i-1] <= j: 
                table[i][j] = max(val[i-1]  
+ table[i-1][j-wt[i-1]],  table[i-1][j]) 
            else: 
                table[i][j] = table[i-1][j] 

    return table[n][W] 

val = [50,100,150,200]
wt = [8,16,32,40]
W = 64

print(knapSack(W, wt, val))

```

运行代码后，我们得到以下输出:

```py
350

```

## 结论

本教程是关于用 Python 动态编程解决 0/1 背包问题的。我们希望你和我们一起学习愉快！
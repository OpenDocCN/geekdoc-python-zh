# 用 Python 打印所有可能的子序列/子集

> 原文：<https://www.askpython.com/python/examples/possible-subsequences-subsets>

在本教程中，我们将理解一个非常有趣的问题，称为**打印特定字符串**的所有可能的子序列/子集。

* * *

## 概念解释

对于给定字符串中的每个元素，有**两种选择**:

*   包括子序列中的第一个元素，并查找剩余元素的子序列。
*   或者不包括第一个元素，而查找剩余元素的子序列。

这同样适用于每次递归调用，直到我们到达给定数组的最后一个索引。

在这种情况下，我们只需打印形成的子序列，然后返回查找下一个子序列。如果你想知道更多关于递归的知识，请阅读下面提到的教程。

***了解更多关于递归的知识:[Python 中的递归](https://www.askpython.com/python/python-recursion-function)***

* * *

## 代码实现

```py
def get_all_subsequence(n,output,i):       
    if (i==len(n)):
        if (len(output)!=0):
            print(output)
    else:
        # exclude first character
        get_all_subsequence(n,output,i+1)

        # include first character
        output+=n[i]
        get_all_subsequence(n,output,i+1)
    return

n = input()
get_all_subsequence(n,"",0)
print(n[0])

```

* * *

## 样本输出

通过上面的代码， **"abc"** 字符串的所有可能的子序列如下:

```py
c
b
bc
a
ac
ab
abc
a

```

* * *

我希望你通过递归理解了子序列或字符串子集的概念。

感谢您的阅读！快乐学习！😇

* * *
# Python 中如何生成平衡括号？

> 原文：<https://www.askpython.com/python/examples/generate-balanced-brackets>

在本教程中，我们将了解一个非常有趣的问题，称为**生成平衡括号**。平衡括号意味着左括号和右括号的数量完全相等。

* * *

## 理解生成平衡括号的概念

我们将处理许多变量，即 n 的值(由用户给定)、输出字符串、开括号和闭括号的计数以及迭代器。

在每个递归调用中，输出字符串将通过插入左括号或右括号来操作。并据此增加开括号和闭括号的计数，递归调用函数。

我们不断检查每个递归调用中括号的平衡。

***了解更多关于递归的知识:[Python 中的递归](https://www.askpython.com/python/python-recursion-function)***

* * *

## 在 Python 中生成平衡括号

```py
def all_balanced(n,output,itr,count_open,count_close):

    # base case
    if(itr == 2*n):
        print(output)
        return

    # Insert open curly bracket    
    if(count_open<n):
        output = output[:itr] + '{' + output[itr+1:]
        all_balanced(n,output,itr+1,count_open+1,count_close)

    # Insert closing curly brackwt
    if(count_open>count_close):
        output = output[:itr] + '}' + output[itr+1:]
        all_balanced(n,output,itr+1,count_open,count_close+1)

    return

n= int(input())
all_balanced(n,"",0,0,0)

```

* * *

## 抽样输出

下面的输出是当 n 的值等于 4 时的结果。这意味着将有 4 个左括号和 4 个右括号。

```py
{{{{}}}}
{{{}{}}}
{{{}}{}}
{{{}}}{}
{{}{{}}}
{{}{}{}}
{{}{}}{}
{{}}{{}}
{{}}{}{}
{}{{{}}}
{}{{}{}}
{}{{}}{}
{}{}{{}}

```

* * *

我希望你清楚平衡括号问题的概念、问题和代码实现。

感谢您的阅读！快乐学习！🙂

* * *
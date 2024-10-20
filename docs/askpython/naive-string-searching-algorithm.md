# Python 中的朴素文本搜索算法

> 原文：<https://www.askpython.com/python/examples/naive-string-searching-algorithm>

在本教程中，我们将研究如何识别文本中的模式。主内容之外还会有一个子串。目的是确定子字符串在文本中出现的次数和位置。

当存在大量文本，并且我们需要定位特定关键字或术语的出现时，这种模式查找方法非常有用。

在这一节中，我们将讨论最基本的 Python 中的朴素字符串匹配算法’以及如何通过更好更短的代码来改进它。

* * *

## 朴素算法简介

顾名思义，朴素算法是非常基础且易于实现的算法。这些算法使用最基本和最明显的策略来完成任务，就像孩子一样。

对于初学者来说，在学习更高效和复杂的算法之前，这些方法是一个很好的起点。其中之一是基本的字符串搜索算法。在字符串匹配/模式发现算法中，它是最基本的。

该过程从逐字母匹配字符串开始。它在主文本和子字符串中搜索第一个字符。如果匹配，则继续到两个字符串中的下一个字符。

如果这些字符在循环中的任何地方都不匹配，则循环被中断，并且循环从主文本字符串中的下一个字符重新开始。

* * *

## 实现简单的字符串搜索

```py
def naive(txt,wrd):
    lt=len(txt)#length of the string
    lw=len(wrd)/3length of the substring(pattern)
    for i in range(lt-lw+1):
        j=0
        while(j<lw):
            if txt[i+j]==wrd[j]:
                j+=1
            else:
                break
        else:
            print('found at position',i)

```

上面代码中的“naive”方法有两个参数:txt(从中搜索模式的主字符串)和 ward(要搜索的模式)。

因为至少应该保留子串的长度以匹配到末尾，所以采用从 0 到(字符串长度-子串长度+1)的循环。“for”循环从字符串(text[I])中提取每个字符。

然后有一个内部 while 循环，它将该字符与子串中的下一个字符进行比较，直到整个子串匹配为止。如果没有被发现，循环就被中断，下一次迭代，就像下一个字符一样，被从进程中删除。

当发现完整的子字符串时，while 条件被破坏，否则部分运行，并显示位置。另一个在循环中，只有当条件为假时才运行，而另一个在 while 循环条件为假时执行。

让我们看看以下输入的输出:

```py
naive("AABAACAADAABAABA","AABA")

```

输出结果如下:

```py
found at position 0
found at position 9
found at position 12

```

* * *

## 结论

恭喜你！您刚刚学习了如何实现简单的字符串搜索算法。希望你喜欢它！😇

喜欢这个教程吗？无论如何，我建议你看一下下面提到的教程:

1.  [找出没有连续 1 的可能字符串的数量](https://www.askpython.com/python/examples/number-of-possible-strings)
2.  [如何在 Python 中将字典转换成字符串？](https://www.askpython.com/python/string/dictionary-to-a-string)
3.  [在 Python 中把元组转换成字符串【一步一步】](https://www.askpython.com/python/string/convert-tuple-to-a-string)

感谢您抽出时间！希望你学到了新的东西！！😄

* * *
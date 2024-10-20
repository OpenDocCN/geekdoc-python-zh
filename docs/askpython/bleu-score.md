# Python 中的 BLEU score–初学者概述

> 原文：<https://www.askpython.com/python/bleu-score>

读者朋友们，你们好！在本文中，我们将关注 Python 中 BLEU score 的**实现。**

所以，让我们开始吧！🙂

***也读作:[在 Python 中自定义数据集](https://www.askpython.com/python-modules/pytorch-custom-datasets)***

* * *

## 什么是 BLEU 评分？

在机器学习建模、深度学习和自然语言处理的领域中，我们需要特定的错误度量，以使我们能够评估字符串输入上构建的模型。

BLEU 评分就是这样一种度量，它使我们能够估计机器翻译模型或系统的效率。今天，这已经被自然语言处理模型和应用程序广泛使用。

在幕后，BLEU 术语评分将候选句子与参考句子进行比较，然后评估候选句子与参考句子的融合程度。这样，它分别在 0-1 的范围内对分数进行评级。

* * *

## Python 中 BLEU 分数的计算

为了实现 BLEU 分数，我们将使用由 sentence_bleu()函数组成的 [NLTK 模块](https://www.askpython.com/python-modules/tokenization-in-python-using-nltk)。它使我们能够传递参考句子和候选句子。然后，它对照参考句子检查候选句子。

如果找到完全匹配，它返回 1 作为 BLEU 分数。如果完全不匹配，则返回 0。对于部分匹配，蓝色分数将在 0 和 1 之间。

## **实现 BLEU 评分**

在下面的例子中，

1.  我们已经导入了 NLTK 库和 sentence_bleu 子模块。
2.  此外，我们生成一个引用语句列表，并通过对象 **ref** 指向它们。
3.  然后我们创建一个**测试**句子，并使用 sentence_bleu()来测试它与 **ref** 的对比。
4.  因此，它给出了一个大约为 1 的输出。
5.  下一次，我们创建一个 **test01** 语句并将其传递给函数。
6.  由于语句由**和**组成，后者是引用语句的一部分，但不完全匹配引用语句，因此它返回一个接近 0 的近似值。

```py
from nltk.translate.bleu_score import sentence_bleu
ref = [
    'this is moonlight'.split(),
    'Look, this is moonlight'.split(),
    'moonlight it is'.split()
]
test = 'it is moonlight'.split()
print('BLEU score for test-> {}'.format(sentence_bleu(ref, test)))

test01 = 'it is cat and moonlight'.split()
print('BLEU score for test01-> {}'.format(sentence_bleu(ref, test01)))

```

**输出—**

```py
BLEU score for test-> 1.491668146240062e-154
BLEU score for test01-> 9.283142785759642e-155

```

* * *

## 用 Python 实现 N-gram score

如上所述，默认情况下，sentence_bleu()函数在引用语句中搜索 1 个单词进行匹配。我们可以根据引用语句在队列中搜索多个单词。这就是所谓的 N-gram。

*   **1 克:1 个字**
*   **2-gram:成对的单词**
*   **三个字母:三个一组**等等

同样，我们可以将以下参数传递给 sentence_bleu()函数来实现 N-gram:

```py
1-gram: (1, 0, 0, 0)
2-gram: (0, 1, 0, 0) 
3-gram: (1, 0, 1, 0)
4-gram: (0, 0, 0, 1)

```

**举例**:

在下面的示例中，我们使用下面提到的引用语句 **ref** 使用 sentence_bleu()函数计算了候选句子 **test01** 的 2-gram BLEU 得分，传递了 2-gram 得分的权重，即(0，1，0，0)。

```py
from nltk.translate.bleu_score import sentence_bleu
ref = [
    'this is moonlight'.split(),
    'Look, this is moonlight'.split(),
    'moonlight it is'.split()
]
test01 = 'it is cat and moonlight'.split()
print('2-gram:' sentence_bleu(ref, test01, weights=(0, 1, 0, 0)))

```

**输出**:

```py
2-gram: 0.25

```

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何问题，欢迎在下面评论。

更多与 Python 编程相关的帖子，请继续关注我们。

在那之前，学习愉快！！🙂
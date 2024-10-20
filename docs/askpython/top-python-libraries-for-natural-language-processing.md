# 自然语言处理的 5 大 Python 库

> 原文：<https://www.askpython.com/python/top-python-libraries-for-natural-language-processing>

**自然语言处理(NLP)** 是 **[数据科学](https://www.askpython.com/python/data-analytics-vs-data-science)** 和 **[人工智能](https://www.askpython.com/python/top-5-jobs-that-use-python)** 的交叉领域。它旨在理解人类语言的语义和内涵。

它的主要重点是从文本中找到有意义的信息，下一步是根据获得的见解训练数据模型。NLP 函数广泛用于文本挖掘、文本分类、文本分析、情感分析、语音识别和机器翻译。

这一切之所以成为可能，只是因为 Python 中提供了大量的 NLP 库。所有库的基本目标是正确地将自由文本句子转换成结构化特征，并且它必须能够有效地实现最新的算法和模型。

因此，在本文中，我们将介绍在解决现实世界问题时非常有用的 5 大自然语言库和工具。

***也读:[顶级机器学习库](https://www.askpython.com/python-modules/top-best-machine-learning-libraries)***

## Python 的顶级自然语言处理库

现在让我们来探索一下 Python 可用的 5 个不同的 NLP 库，它们可用于文本生成和训练。您甚至可以使用这些来创建 Python 中的聊天机器人。

### 1.[自然语言工具包(NLTK)](https://www.nltk.org/)

它是构建 Python 程序的重要库之一，使我们能够处理人类语言数据并从中获得洞察力。

它为 50 多个语料库(用于语言研究的大量书面或口头文本)和词汇资产(如 WordNet)提供了简单的接口。

NLTK 还帮助建立文本预处理库，用于 NLP 库和主动对话讨论的标记、解析、分类、词干、标记化和语义推理包装器。

NLTK 是免费和开源的。Windows、Mac OS 和 Linux 都可以轻松访问它。由于功能范围很广，所以速度很慢，有时很难满足生产使用的需求。

**NLTK**的特点包括词性标注、实体抽取、[分词](https://www.askpython.com/python-modules/tokenization-in-python-using-nltk)、[解析](https://www.askpython.com/python/examples/dependency-parsing-in-python)、语义推理、[词干](https://www.askpython.com/python/examples/stemming-and-lemmatization)、[文本分类](https://www.askpython.com/python/examples/email-spam-classification)。

**安装**

```py
pip install nltk

```

## 2. [Gensim](https://pypi.org/project/gensim/)

Gensim 是一个非常流行的自然语言处理作品库。它具有通过使用向量空间建模来识别两个文档之间的语义相似性的特殊特征。它的算法与内存无关，这意味着我们可以很容易地处理大于 RAM 的输入。

它是为“大型语料库(用于语言研究的大量书面或口头文本的集合)的主题建模、文档索引和相似性检索”而设计的。它广泛用于数据分析、文本生成应用和语义搜索应用。它给了我们一套在自然语言作品中非常重要的算法。

gensim 的一些算法是分层狄利克雷过程(HDP)、随机投影(RP)、潜在狄利克雷分配(LDA)、潜在语义分析(LSA/SVD/LSI)或 word2vec 深度学习。

**安装**

```py
pip install — upgrade gensim

```

## 3.[酷睿 NLP](https://stanfordnlp.github.io/CoreNLP/)

Standford CoreNLP 包含一组人类语言技术工具。CoreNLP 旨在使使用语义分析工具对一段文本的分析变得简单而熟练。在 CoreNLP 的帮助下，你可以提取所有类型的文本属性(如命名实体识别、词性标注等。)只用了几行代码。

由于 CoreNLP 是用 java 编写的，所以需要在你的设备上安装 Java。然而，它确实提供了许多流行编程语言的编程接口，包括 Python。它整合了许多 Standford 的 NLP 工具，如解析器、情感分析、引导模式学习、命名实体识别器(NER)和共指解析系统、词性标记器等等。

此外，CoreNLP 支持除英语、中文、德语、法语、西班牙语和阿拉伯语之外的四种语言。

**安装**

```py
pip install stanfordnlp

```

## 4.[空间](https://spacy.io/)

SpaCy 是一个开源的 Python 自然语言处理库。它是专门为解决现实世界中的问题而设计的，它有助于处理大量的文本数据。它配备了预先训练的统计模型和词向量，并且 SpaCy 是在 Cython 中用 python 编写的(Cython 语言是 Python 语言的超集)，这就是为什么它在处理大量文本数据时更快更有效。

SpaCy 的主要特点是:

*   它提供了像伯特一样的训练有素的变形金刚。
*   提供超过 49 种语言的标记化。
*   提供文本分类、句子分割、词汇化、词性标注、命名实体识别等功能。
*   它比其他库快得多。
*   它可以对文本进行预处理，用于深度学习。
*   它拥有超过 17 种语言的 55 条训练有素的管道。

**安装(以及依赖关系)**

```py
pip install –U setuptools wheel
pip install –U spacy
python -m spacy download en_core_web_sm

```

## 5.[图案](https://github.com/clips/pattern)

**Pattern** 是 Python 中一个非常有用的库，可以用来实现自然语言处理任务。它是开源的，任何人都可以免费使用。它可以用于 NLP、文本挖掘、web 挖掘、网络分析和机器学习。

它附带了一系列用于数据挖掘的工具(谷歌、维基百科 API、网络爬虫和 HTML DOM 解析器)、NLP (n-gram 搜索、情感分析、WordNet、词性标签)、ML(向量空间模型、聚类、SVM)以及具有图形中心性和可视化的网络分析。

对于科学和非科学的观众来说，这是一个非常强大的工具。它的语法非常简单明了，最棒的是函数名和参数的选择方式使得命令一目了然，它还可以作为 web 开发人员的快速开发框架。

**安装**

```py
pip install pattern

```

## 结论

在本文中，我们浏览了自然语言处理中最常用的 5 个 python 库，并讨论了根据我们的需求何时必须使用哪个库。我希望你能从这个博客中学到一些东西，这对你的项目来说是最好的。

## 参考

[https://medium . com/nl planet/awesome-NLP-21-popular-NLP-libraries-of-2022-2e 07a 914248 b](https://medium.com/nlplanet/awesome-nlp-21-popular-nlp-libraries-of-2022-2e07a914248b)

## 进一步阅读

[Python 中的潜在狄利克雷分配(LDA)算法](https://www.askpython.com/python/examples/latent-dirichlet-allocation-lda)
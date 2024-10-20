# Python 中的潜在狄利克雷分配算法

> 原文：<https://www.askpython.com/python/examples/latent-dirichlet-allocation-lda>

你好读者，在这篇文章中我们将试图理解什么是 LDA 算法。它是如何工作的以及如何在 python 中实现的。潜在狄利克雷分配是一种主要属于自然语言处理(NLP)领域的算法。

它用于主题建模。主题建模是对文本数据执行的机器学习技术，以分析文本数据并在文档集合中找到抽象的相似主题。

***也读作:[深度优先迭代深化(DFID)算法 Python 中的](https://www.askpython.com/python/examples/depth-first-iterative-deepening-dfid)***

## 什么是 LDA？

LDA 是一种专门为文本数据设计的主题建模算法。这种技术将每个文档视为算法作为最终结果产生的一些主题的混合物。主题是出现在数据集中所有文档集中的单词的概率分布。

预处理数据的结果将提供一组关键字或标记，LDA 算法将把这些预处理数据作为输入，并基于这些关键字的概率分布来尝试发现隐藏/潜在的主题。最初，该算法将把文档中的每个单词分配给' *n'* 个主题中的一个随机主题。

例如，考虑以下文本数据

*   文字 1:为 IPL 感到兴奋，今年让我们回到板球场，享受比赛。
*   文字 2:今年八月我们可能会面临第四波 Covid！
*   文本 3:尽早接种疫苗，现在正是时候。
*   文本 4:欧盟预算增加了今年的体育项目配额，这都要归功于今年的奥运冠军。

理论上，让我们考虑算法要处理的两个主题 Sports 和 Covid。该算法可以为主题 2 的 Covid 分配表示“IPL”的第一个单词。我们知道这种分配是错误的，但是该算法将基于两个因素在未来的迭代中尝试纠正这一点，这两个因素是主题在文档中出现的频率和单词在主题中出现的频率。由于在文本 1 中没有很多与 Covid 相关的术语，并且单词“IPL”在主题 2 Covid 中不会出现很多次，所以算法可以将单词“IPL”分配给新主题，即主题 1(体育)。通过多次这样的迭代，该算法将实现主题识别和跨主题的单词分布的稳定性。最后，每个文档可以表示为确定主题的混合。

***也读作:[Python 中的双向搜索](https://www.askpython.com/python/examples/bidirectional-search-in-python)***

## LDA 是如何工作的？

在 LDA 中执行以下步骤，为每个文档分配主题:

1)对于每个文档，将每个单词随机初始化为 K 个主题中的一个主题，其中 K 是预定义主题的数量。

2)对于每个文档 d:

对于文档中的每个单词 w，计算:

*   p(主题 t|文档 d):文档 d 中分配给主题 t 的单词的比例
*   P(word w| topic t):来自 w 的单词在所有文档中分配给主题 t 的比例

3)考虑所有其他单词及其主题分配，以概率 p(t'|d)*p(w|t ')将主题 T '重新分配给单词 w

最后一步重复多次，直到我们达到一个稳定的状态，主题分配不再发生进一步的变化。然后从这些主题分配中确定每个文档的主题比例。

**LDA 的示例:**

假设我们有以下 4 个文档作为语料库，我们希望对这些文档进行主题建模。

*   **文献 1** :我们在 YouTube 上看很多视频。
*   **文献 2** : YouTube 视频信息量很大。
*   **文献 3** :看技术博客让我很容易理解事情。
*   **文档 4** :比起 YouTube 视频，我更喜欢博客。

LDA 建模帮助我们发现上述语料库中的主题，并为每个文档分配主题混合。例如，该模型可能会输出如下所示的内容:

话题 1: 40%的视频，60%的 YouTube

话题 2: 95%的博客，5%的 YouTube

文档 1 和 2 将 100%属于主题 1。文档 3 将 100%属于主题 2。文档 4 的 80%属于主题 2，20%属于主题 1

## 如何用 Python 实现 LDA？

以下是实现 LDA 算法的步骤:

1.  收集数据并作为输入提供
2.  预处理数据(删除不必要的数据)
3.  修改 LDA 分析的数据
4.  建立和训练 LDA 模型
5.  分析 LDA 模型结果

这里，我们有从 Twitter 收集的输入数据，并将其转换为 CSV 文件，因为社交媒体上的数据是多种多样的，我们可以建立一个有效的模型。

## 导入 LDA 所需的库

```py
import numpy as np
import pandas as pd 
import re

import gensim
from gensim import corpora, models, similarities
from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import stopwords

```

## 清理数据

### 规范化空白

```py
def normalize_whitespace(tweet):
    tweet = re.sub('[\s]+', ' ', tweet)
    return tweet

text = "         We        are the students    of    Science. "
print("Text Before: ",text)
text = normalize_whitespace(text)
print("Text After: ",text)

```

**输出:**

```py
 Text Before:    We        are the students    of    Science. 

```

我们是理科学生。

### 删除停用词

```py
import nltk
nltk.download('stopwords')
import gensim
from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import stopwords

stop_words = stopwords.words('english')

def remove_stopwords(text):
  final_s=""
  text_arr= text.split(" ")                              #splits sentence when space occurs
  print(text_arr)
  for word in text_arr:                             
    if word not in stop_words:                     # if word is not in stopword then append(join) it to string 
      final_s= final_s + word + " "

  return final_s 

```

### 词干化和标记化

```py
import nltk
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer, SnowballStemmer, PorterStemmer

stemmer = PorterStemmer()

def tokenize_stemming(text):
    text = re.sub(r'[^\w\s]','',text)
    #replace multiple spaces with one space
    text = re.sub(r'[\s]+',' ',text)
    #transfer text to lowercase
    text = text.lower() 
    # tokenize text
    tokens = re.split(" ", text)

    # Remove stop words 
    result = []
    for token in tokens :
        if token not in stop_words and len(token) > 1:
            result.append(stemmer.stem(token))

    return result

```

***也读作:[在 Python 中使用 NLTK 进行标记化](https://www.askpython.com/python-modules/tokenization-in-python-using-nltk)***

## 术语频率(TF-IDF)

它是术语频率-逆文档频率的缩写，是一种数字统计，旨在反映一个词对集合或语料库中的文档有多重要。它经常被用作加权因子。

```py
corpus_doc2bow_vectors = [dictionary.doc2bow(tok_doc) for tok_doc in tokens]
print("# Term Frequency : ")
corpus_doc2bow_vectors[:5]

tfidf_model = models.TfidfModel(corpus_doc2bow_vectors, id2word=dictionary, normalize=False)
corpus_tfidf_vectors = tfidf_model[corpus_doc2bow_vectors]

print("\n# TF_IDF: ")
print(corpus_tfidf_vectors[5])

```

***也读作:[用 Python 从零开始创建 TF-IDF 模型](https://www.askpython.com/python/examples/tf-idf-model-from-scratch)***

## 使用单词包运行 LDA

```py
lda_model = gensim.models.LdaMulticore(corpus_doc2bow_vectors, num_topics=10, id2word=dictionary, passes=2, workers=2)

```

***也读作:[用 python 从零开始创建包字模型](https://www.askpython.com/python/examples/bag-of-words-model-from-scratch)***

## 使用 TF-IDF 运行 LDA

```py
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf_vectors, num_topics=10, id2word=dictionary, passes=2, workers=4)
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))

```

## 主题的分类

性能评估通过使用 LDA 单词袋模型对样本文档进行分类，我们将检查我们的测试文档将被分类到哪里。

```py
for index, score in sorted(lda_model[corpus_doc2bow_vectors[1]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))

```

***也读:[用 Python 分类新闻标题——机器学习](https://www.askpython.com/python/examples/classify-news-headlines-in-python)***

### 使用 LDA TF-IDF 模型对样本文档进行分类的性能评估。

```py
for index, score in sorted(lda_model_tfidf[corpus_doc2bow_vectors[1]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))

```

## 结论

在本文中，我们试图理解自然语言处理领域中最常用的算法。LDA 是主题建模的基础——一种统计建模和数据挖掘。

## 参考资料:

[https://en.wikipedia.org/wiki/Tf%E2%80%93idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
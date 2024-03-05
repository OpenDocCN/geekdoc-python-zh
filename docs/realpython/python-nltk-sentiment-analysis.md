# 情感分析:使用 Python 的 NLTK 库的第一步

> 原文：<https://realpython.com/python-nltk-sentiment-analysis/>

一旦你理解了 Python 的[基础，让自己熟悉它最流行的包不仅会提高你对这门语言的掌握，还会迅速增加你的通用性。在本教程中，您将学习自然语言工具包(NLTK)处理和分析文本的惊人能力，从基本的](https://realpython.com/learning-paths/python-basics-book/)[功能](https://realpython.com/defining-your-own-python-function/)到由[机器学习](https://realpython.com/learning-paths/machine-learning-python/)驱动的[情感分析](https://realpython.com/sentiment-analysis-python/)！

**情感分析**可以帮助你确定对某个特定话题的积极参与和消极参与的比例。您可以分析文本主体，如评论、推文和产品评论，以从您的受众那里获得洞察力。在本教程中，您将了解 NLTK 处理文本数据的重要特性，以及可以用来对数据执行情感分析的不同方法。

**本教程结束时，您将能够:**

*   **拆分**和**过滤**文本数据以备分析
*   分析**词频**
*   用不同的方法找出**的一致**和**的搭配**
*   使用 NLTK 的内置分类器执行快速**情感分析**
*   为**自定义分类**定义特征
*   使用并比较用于 NLTK 情感分析的**分类器**

**免费奖励:** ，它向您展示 Python 3 的基础知识，如使用数据类型、字典、列表和 Python 函数。

## NLTK 入门

NLTK 库包含各种实用程序，允许您有效地操作和分析语言数据。在它的高级特性中有**文本分类器**，你可以使用它进行多种分类，包括情感分析。

**情感分析**是利用算法将相关文本的各种样本分类为整体的正面和负面类别的实践。使用 NLTK，您可以通过强大的内置机器学习操作来使用这些算法，以从语言数据中获得洞察力。

[*Remove ads*](/account/join/)

### 安装和导入

您将从安装一些先决条件开始，包括 NLTK 本身以及贯穿本教程所需的特定资源。

首先，使用 [`pip`](https://realpython.com/what-is-pip/) 安装 NLTK:

```py
$ python3 -m pip install nltk
```

虽然这将安装 NLTK 模块，但是您仍然需要获得一些额外的资源。其中一些是文本样本，另一些是某些 NLTK 函数需要的数据模型。

要获得您需要的资源，请使用`nltk.download()`:

```py
import nltk

nltk.download()
```

NLTK 将显示一个下载管理器，显示所有可用的和已安装的资源。以下是您在本教程中需要下载的内容:

*   **`names` :** 马克·坎特罗威茨编撰的常用英文名列表
*   **`stopwords` :** 非常常见的单词列表，如冠词、代词、介词和连词
*   **`state_union` :** 不同美国总统的[国情咨文](https://en.wikipedia.org/wiki/State_of_the_Union)演讲样本，由凯瑟琳·阿伦斯编译
*   **`twitter_samples` :** 发布到推特上的社交媒体短语列表
*   **`movie_reviews` :** [两千条影评](http://www.cs.cornell.edu/people/pabo/movie-review-data/)按庞博和莉莲·李分类
*   **`averaged_perceptron_tagger` :** 一种数据模型，NLTK 使用它将单词分类到它们的[词性](https://en.wikipedia.org/wiki/Part_of_speech)
*   **`vader_lexicon`:**NLTK 在执行情感分析时引用的单词和行话的评分列表，由 C.J .休顿和 Eric Gilbert 创建
*   **`punkt`:**Jan Strunk 创建的一个数据模型，NLTK 使用它将全文拆分成单词列表

**注意:**在本教程中，你会发现许多关于**文集**及其复数形式**文集**的参考资料。语料库是大量相关文本样本的集合。在 NLTK 的上下文中，使用用于[自然语言处理(NLP)](https://en.wikipedia.org/wiki/Natural_language_processing) 的特征来编译语料库，例如特定特征的类别和数字分数。

直接从控制台下载特定资源的一种快速方法是将一个[列表](https://realpython.com/python-lists-tuples/)传递给`nltk.download()`:

>>>

```py
>>> import nltk

>>> nltk.download([
...     "names",
...     "stopwords",
...     "state_union",
...     "twitter_samples",
...     "movie_reviews",
...     "averaged_perceptron_tagger",
...     "vader_lexicon",
...     "punkt",
... ])
[nltk_data] Downloading package names to /home/user/nltk_data...
[nltk_data]   Unzipping corpora/names.zip.
[nltk_data] Downloading package stopwords to /home/user/nltk_data...
[nltk_data]   Unzipping corpora/stopwords.zip.
[nltk_data] Downloading package state_union to
[nltk_data]     /home/user/nltk_data...
[nltk_data]   Unzipping corpora/state_union.zip.
[nltk_data] Downloading package twitter_samples to
[nltk_data]     /home/user/nltk_data...
[nltk_data]   Unzipping corpora/twitter_samples.zip.
[nltk_data] Downloading package movie_reviews to
[nltk_data]     /home/user/nltk_data...
[nltk_data]   Unzipping corpora/movie_reviews.zip.
[nltk_data] Downloading package averaged_perceptron_tagger to
[nltk_data]     /home/user/nltk_data...
[nltk_data]   Unzipping taggers/averaged_perceptron_tagger.zip.
[nltk_data] Downloading package vader_lexicon to
[nltk_data]     /home/user/nltk_data...
[nltk_data] Downloading package punkt to /home/user/nltk_data...
[nltk_data]   Unzipping tokenizers/punkt.zip.
True
```

这将告诉 NLTK 根据标识符来查找和下载每个资源。

如果 NLTK 需要您尚未安装的额外资源，您将会看到一个有用的`LookupError`,其中包含下载该资源的详细信息和说明:

>>>

```py
>>> import nltk

>>> w = nltk.corpus.shakespeare.words()
...
LookupError:
**********************************************************************
 Resource shakespeare not found.
 Please use the NLTK Downloader to obtain the resource:

 >>> import nltk
 >>> nltk.download('shakespeare')
...
```

`LookupError`指定哪个资源是所请求的操作所必需的，以及使用其标识符下载它的指令。

### 编译数据

NLTK 提供了许多函数，您可以使用很少的参数或不使用参数来调用这些函数，这些函数将帮助您在接触它的机器学习功能之前对文本进行有意义的分析。NLTK 的许多实用程序有助于为更高级的分析准备数据。

很快，你将学习频率分布，一致性和搭配。但首先，你需要一些数据。

首先加载您之前下载的国情咨文:

```py
words = [w for w in nltk.corpus.state_union.words() if w.isalpha()]
```

注意，您使用语料库的`.words()`方法构建了一个单词列表，但是您使用`str.isalpha()`仅包含由字母组成的单词。否则，你的单词表可能会以仅仅是标点符号的“单词”结束。

看看你的清单。你会注意到很多像“of”、“a”、“the”之类的小词。这些常用词被称为**停用词**，它们会对你的分析产生负面影响，因为它们在文本中出现得太频繁了。幸运的是，有一种简便的方法可以过滤掉它们。

NLTK 提供了一个小型的停用词语料库，您可以将它加载到一个列表中:

```py
stopwords = nltk.corpus.stopwords.words("english")
```

确保将`english`指定为所需语言，因为该语料库包含各种语言的停用词。

现在，您可以从原始单词列表中删除停用单词:

```py
words = [w for w in words if w.lower() not in stopwords]
```

因为`stopwords`列表中的所有单词都是小写的，而原始列表中的单词可能不是，所以使用`str.lower()`来说明任何差异。否则，你的列表中可能会出现混合大小写或大写的停用词。

虽然您将在本教程中使用 NLTK 提供的语料库，但是也可以从任何来源构建您自己的文本语料库。建立一个语料库可以像加载一些纯文本一样简单，也可以像对每个句子进行标记和分类一样复杂。参考 NLTK 的文档，了解更多关于如何使用语料库阅读器的信息。

对于一些快速分析来说，创建一个语料库可能是多余的。如果你需要的只是一个单词表，有更简单的方法来实现这个目标。除了 Python 自己的字符串操作方法之外，NLTK 还提供了`nltk.word_tokenize()`，一个将原始文本分割成单个单词的函数。虽然**标记化**本身是一个更大的主题(并且很可能是创建自定义语料库时要采取的步骤之一)，但是这个标记化器非常好地提供了简单的单词列表。

要使用它，用您想要分割的原始文本调用`word_tokenize()`:

>>>

```py
>>> from pprint import pprint

>>> text = """
... For some quick analysis, creating a corpus could be overkill.
... If all you need is a word list,
... there are simpler ways to achieve that goal."""
>>> pprint(nltk.word_tokenize(text), width=79, compact=True)
['For', 'some', 'quick', 'analysis', ',', 'creating', 'a', 'corpus', 'could',
 'be', 'overkill', '.', 'If', 'all', 'you', 'need', 'is', 'a', 'word', 'list',
 ',', 'there', 'are', 'simpler', 'ways', 'to', 'achieve', 'that', 'goal', '.']
```

现在你有一个可行的单词表了！记住标点符号会被算作单个单词，所以后面用`str.isalpha()`过滤掉。

[*Remove ads*](/account/join/)

### 创建频率分布

现在你已经为**频率分布**做好准备了。频率分布本质上是一个表格，它告诉你每个单词在给定文本中出现的次数。在 NLTK 中，频率分布是一种特定的对象类型，作为一个名为`FreqDist`的独特类来实现。这个类为词频分析提供了有用的操作。

要用 NLTK 构建频率分布，用单词列表构建`nltk.FreqDist`类:

```py
words: list[str] = nltk.word_tokenize(text)
fd = nltk.FreqDist(words)
```

这将创建一个类似于 [Python 字典](https://realpython.com/python-dicts/)的频率分布对象，但是增加了一些特性。

**注意:**你在上面的`words: list[str] = ...`中看到的泛型类型提示是 Python 3.9 中的一个[新特性！](https://realpython.com/python39-new-features/#type-hint-lists-and-dictionaries-directly)

构建完对象后，您可以使用类似于`.most_common()`和`.tabulate()`的方法开始可视化信息:

>>>

```py
>>> fd.most_common(3)
[('must', 1568), ('people', 1291), ('world', 1128)]
>>> fd.tabulate(3)
 must people  world
 1568   1291   1128
```

这些方法允许您快速确定样品中的常用词。使用`.most_common()`，您可以获得包含每个单词的元组列表，以及它在您的文本中出现的次数。您可以使用`.tabulate()`以更易读的格式获得相同的信息。

除了这两种方法，您还可以使用频率分布来查询特定的单词。您还可以将它们用作迭代器，对 word 属性执行一些自定义分析。

例如，要发现大小写的差异，您可以查询同一个单词的不同变体:

>>>

```py
>>> fd["America"]
1076
>>> fd["america"]  # Note this doesn't result in a KeyError
0
>>> fd["AMERICA"]
3
```

这些返回值指示每个单词按照给定的精确值出现的次数。

因为频率分布对象是[可迭代的](https://realpython.com/python-for-loop/#iterables)，你可以在[列表理解](https://realpython.com/list-comprehension-python/)中使用它们来创建初始分布的子集。您可以将这些子集集中在对您自己的分析有用的属性上。

尝试创建一个新的频率分布，它基于最初的频率分布，但将所有单词规范化为小写:

```py
lower_fd = nltk.FreqDist([w.lower() for w in fd])
```

现在，无论大小写如何，您都可以更准确地表达单词的用法了。

想想这些可能性:你可以创建单词的频率分布，以一个特定的字母开始，或一个特定的长度，或包含某些字母。你的想象力是极限！

### 提取索引和搭配

在 NLP 的上下文中，**索引**是单词位置及其上下文的集合。您可以使用索引来查找:

1.  一个单词出现多少次
2.  每次出现的位置
3.  每个事件周围有哪些单词

在 NLTK 中，可以通过调用`.concordance()`来实现这一点。要使用它，您需要一个`nltk.Text`类的实例，它也可以用一个单词列表来构造。

在调用`.concordance()`之前，从原始的语料库文本构建一个新的单词列表，这样所有的上下文，甚至停用的单词都将存在:

>>>

```py
>>> text = nltk.Text(nltk.corpus.state_union.words())
>>> text.concordance("america", lines=5)
Displaying 5 of 1079 matches:
 would want us to do . That is what America will do . So much blood has already
ay , the entire world is looking to America for enlightened leadership to peace
beyond any shadow of a doubt , that America will continue the fight for freedom
 to make complete victory certain , America will never become a party to any pl
nly in law and in justice . Here in America , we have labored long and hard to
```

注意`.concordance()`已经忽略了大小写，允许您按照出现的顺序查看一个单词的所有大小写变体的上下文。还要注意，这个函数不会显示文本中每个单词的位置。

此外，由于`.concordance()`只将信息打印到控制台，它对于数据操作来说并不理想。要获得一个有用的列表，该列表还将为您提供每个事件的位置信息，请使用`.concordance_list()`:

>>>

```py
>>> concordance_list = text.concordance_list("america", lines=2)
>>> for entry in concordance_list:
...     print(entry.line)
...
 would want us to do . That is what America will do . So much blood has already
ay , the entire world is looking to America for enlightened leadership to peace
```

`.concordance_list()`给出了一个`ConcordanceLine`对象的列表，其中包含了每个单词出现的位置信息以及一些值得探索的属性。该列表也按出现的顺序排序。

类本身还有一些其他有趣的特性。其中一个是`.vocab()`，值得一提，因为它为给定的文本创建了一个频率分布。

再次访问`nltk.word_tokenize()`，看看您可以多快地创建一个定制的`nltk.Text`实例和一个伴随的频率分布:

>>>

```py
>>> words: list[str] = nltk.word_tokenize(
...     """Beautiful is better than ugly.
...     Explicit is better than implicit.
...     Simple is better than complex."""
... )
>>> text = nltk.Text(words)
>>> fd = text.vocab()  # Equivalent to fd = nltk.FreqDist(words)
>>> fd.tabulate(3)
 is better   than
 3      3      3
```

`.vocab()`本质上是从`nltk.Text`的实例创建频率分布的快捷方式。这样，你就不必单独调用实例化一个新的`nltk.FreqDist`对象。

NLTK 的另一个强大特性是它能够通过简单的函数调用快速找到**搭配**。搭配是在给定文本中经常一起出现的一系列单词。例如，在国情咨文语料库中，你会发现*联合*和*州*这两个词经常出现在一起。这两个词一起出现是一种搭配。

搭配可以由两个或更多的单词组成。NLTK 提供了处理几种搭配类型的类:

*   **二元组:**频繁出现的两个词的组合
*   **三元组:**频繁出现的三字组合
*   **四字格:**频繁出现的四字组合

NLTK 为您提供了特定的类来查找文本中的搭配。按照您到目前为止看到的模式，这些类也是由单词列表构建的:

```py
words = [w for w in nltk.corpus.state_union.words() if w.isalpha()]
finder = nltk.collocations.TrigramCollocationFinder.from_words(words)
```

`TrigramCollocationFinder`实例将专门搜索三元模型。您可能已经猜到，NLTK 也有分别用于二元模型和四元模型的`BigramCollocationFinder`和`QuadgramCollocationFinder`类。所有这些类都有许多实用程序来提供所有已识别搭配的信息。

他们最有用的工具之一是`ngram_fd`属性。该属性保存为每个搭配而不是为单个单词构建的频率分布。

使用`ngram_fd`，您可以在提供的文本中找到最常见的搭配:

>>>

```py
>>> finder.ngram_fd.most_common(2)
[(('the', 'United', 'States'), 294), (('the', 'American', 'people'), 185)]
>>> finder.ngram_fd.tabulate(2)
 ('the', 'United', 'States') ('the', 'American', 'people')
 294                           185
```

您甚至不必创建频率分布，因为它已经是 collocation finder 实例的一个属性。

现在，您已经了解了 NLTK 的一些最有用的工具，是时候投入情感分析了！

[*Remove ads*](/account/join/)

## 使用 NLTK 预先训练的情感分析器

NLTK 已经有了一个内置的、预训练的情感分析器，名为 VADER(**V**alence**A**ware**D**ictionary 和 s**E**entiment**R**easoner)。

由于 VADER 经过预训练，您可以比许多其他分析仪更快地获得结果。然而，VADER 最适合社交媒体中使用的语言，比如含有俚语和缩写的短句。当评价较长的结构化句子时，它不太准确，但它通常是一个很好的切入点。

要使用 VADER，首先创建一个`nltk.sentiment.SentimentIntensityAnalyzer`的实例，然后在一个原始的[字符串](https://realpython.com/python-strings/)上使用`.polarity_scores()`:

>>>

```py
>>> from nltk.sentiment import SentimentIntensityAnalyzer
>>> sia = SentimentIntensityAnalyzer()
>>> sia.polarity_scores("Wow, NLTK is really powerful!")
{'neg': 0.0, 'neu': 0.295, 'pos': 0.705, 'compound': 0.8012}
```

你会得到一本不同分数的字典。负的、中性的和正的分数是相关的:它们加起来都是 1，不能是负的。复合得分的计算方式不同。它不仅仅是一个平均值，它的范围可以从-1 到 1。

现在，您将使用两个不同的语料库对真实数据进行测试。首先，将`twitter_samples`语料库加载到一个字符串列表中，替换成不活动的 URL，以避免意外点击:

```py
tweets = [t.replace("://", "//") for t in nltk.corpus.twitter_samples.strings()]
```

注意，您使用了不同的语料库方法`.strings()`，而不是`.words()`。这会给你一个字符串形式的原始 tweets 列表。

不同的语料库有不同的特性，所以你可能需要使用 Python 的`help()`，就像在`help(nltk.corpus.tweet_samples)`中一样，或者查阅 NLTK 的文档来学习如何使用给定的语料库。

现在使用您的`SentimentIntensityAnalyzer`实例的`.polarity_scores()`函数对 tweets 进行分类:

```py
from random import shuffle

def is_positive(tweet: str) -> bool:
    """True if tweet has positive compound sentiment, False otherwise."""
    return sia.polarity_scores(tweet)["compound"] > 0

shuffle(tweets)
for tweet in tweets[:10]:
    print(">", is_positive(tweet), tweet)
```

在这种情况下，`is_positive()`仅使用复合得分的正性来进行呼叫。你可以选择 VADER 分数的任意组合来根据你的需要调整分类。

现在来看看第二部文集`movie_reviews`。顾名思义，这是一个影评集。这部文集的特别之处在于它已经被分类了。因此，你可以用它来判断你在给相似文本评分时所选择的算法的准确性。

请记住，VADER 可能更擅长给推特评分，而不是给长篇电影评论评分。为了获得更好的结果，您将设置 VADER 来评价评论中的单个句子，而不是整个文本。

由于 VADER 的评级需要原始数据，你不能像以前那样使用`.words()`。相反，列出语料库使用的文件 id，您可以稍后使用它们来引用单个评论:

```py
positive_review_ids = nltk.corpus.movie_reviews.fileids(categories=["pos"])
negative_review_ids = nltk.corpus.movie_reviews.fileids(categories=["neg"])
all_review_ids = positive_review_ids + negative_review_ids
```

存在于大多数(如果不是全部)语料库中。在`movie_reviews`的情况下，每个文件对应一个单独的审查。还要注意，您可以通过指定类别来过滤文件 id 列表。这种分类是这个语料库和其他同类型语料库特有的特征。

接下来，重新定义`is_positive()`来处理整个评审。您需要使用其文件 ID 获得该特定评论，然后在评级前将其分成句子:

```py
from statistics import mean

def is_positive(review_id: str) -> bool:
    """True if the average of all sentence compound scores is positive."""
    text = nltk.corpus.movie_reviews.raw(review_id)
    scores = [
        sia.polarity_scores(sentence)["compound"]
        for sentence in nltk.sent_tokenize(text)
    ]
    return mean(scores) > 0
```

`.raw()`是另一种存在于大多数语料库中的方法。通过指定一个文件 ID 或文件 ID 列表，您可以从语料库中获取特定的数据。在这里，您获得一条评论，然后使用`nltk.sent_tokenize()`从评论中获得一个句子列表。最后，`is_positive()`计算所有句子的平均复合得分，并将正面结果与正面评论相关联。

你可以借此机会对所有评论进行评分，看看 VADER 在这个设置中有多准确:

>>>

```py
>>> shuffle(all_review_ids)
>>> correct = 0
>>> for review_id in all_review_ids:
...     if is_positive(review_id):
...         if review_id in positive_review_ids:
...             correct += 1
...     else:
...         if review_id in negative_review_ids:
...             correct += 1
...
>>> print(F"{correct / len(all_review_ids):.2%} correct")
64.00% correct
```

在对所有评论进行评级后，你可以看到只有 64%被 VADER 使用`is_positive()`中定义的逻辑正确分类。

64%的准确率并不算高，但这是一个开始。稍微调整一下`is_positive()`,看看你是否能提高精确度。

在下一节中，您将构建一个自定义分类器，该分类器允许您使用额外的特征进行分类，并最终将其准确度提高到可接受的水平。

[*Remove ads*](/account/join/)

## 定制 NLTK 的情感分析

NLTK 提供了一些内置的分类器，适用于各种类型的分析，包括情感分析。诀窍是找出数据集的哪些属性在将每一段数据分类到您想要的类别中是有用的。

在机器学习的世界中，这些数据属性被称为**特征**，当您处理数据时，必须揭示和选择这些特征。虽然本教程不会深入探究[特征选择](https://en.wikipedia.org/wiki/Feature_selection)和[特征工程](https://en.wikipedia.org/wiki/Feature_engineering)，但是您将能够看到它们对分类器准确性的影响。

### 选择有用的功能

既然你已经学会了如何使用频率分布，为什么不把它们作为一个额外特性的起点呢？

通过使用`movie_reviews`语料库中预定义的类别，您可以创建正面和负面词汇集，然后确定哪些词汇在每个集合中出现频率最高。首先排除不需要的单词并建立初始类别组:

```py
 1unwanted = nltk.corpus.stopwords.words("english")
 2unwanted.extend([w.lower() for w in nltk.corpus.names.words()])
 3
 4def skip_unwanted(pos_tuple):
 5    word, tag = pos_tuple
 6    if not word.isalpha() or word in unwanted:
 7        return False
 8    if tag.startswith("NN"):
 9        return False
10    return True
11
12positive_words = [word for word, tag in filter(
13    skip_unwanted,
14    nltk.pos_tag(nltk.corpus.movie_reviews.words(categories=["pos"]))
15)]
16negative_words = [word for word, tag in filter(
17    skip_unwanted,
18    nltk.pos_tag(nltk.corpus.movie_reviews.words(categories=["neg"]))
19)]
```

这一次，您还将来自`names`语料库的单词添加到第 2 行的`unwanted`列表中，因为电影评论可能有许多演员的名字，这不应该是您的特征集的一部分。注意第 14 行和第 18 行的`pos_tag()`,它根据词类来标记单词。

在过滤你的单词列表之前调用`pos_tag()` *是很重要的，这样 NLTK 可以更准确地标记所有的单词。根据 NLTK 的[默认标签集](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)，在第 4 行定义的`skip_unwanted()`使用这些标签来排除名词。*

现在您已经准备好为您的定制特征创建频率分布了。由于许多单词同时出现在正集合和负集合中，因此首先要找到公共集合，这样就可以将它从分布对象中移除:

```py
positive_fd = nltk.FreqDist(positive_words)
negative_fd = nltk.FreqDist(negative_words)

common_set = set(positive_fd).intersection(negative_fd)

for word in common_set:
    del positive_fd[word]
    del negative_fd[word]

top_100_positive = {word for word, count in positive_fd.most_common(100)}
top_100_negative = {word for word, count in negative_fd.most_common(100)}
```

一旦在每个频率分布对象中有了唯一的正面和负面单词，您就可以最终从每个分布中最常见的单词构建集合。每组中的单词量是你可以调整的，以确定它对情感分析的影响。

这是可以从数据中提取的特征的一个例子，它还远非完美。仔细观察这些集合，你会注意到一些不常见的名字和单词，它们不一定是正面或负面的。此外，到目前为止，您已经学习的其他 NLTK 工具对于构建更多功能非常有用。一种可能是利用带有积极意义的搭配，比如 bigram“竖起大拇指！”

以下是如何设置正负二元模型查找器的方法:

```py
unwanted = nltk.corpus.stopwords.words("english")
unwanted.extend([w.lower() for w in nltk.corpus.names.words()])

positive_bigram_finder = nltk.collocations.BigramCollocationFinder.from_words([
    w for w in nltk.corpus.movie_reviews.words(categories=["pos"])
    if w.isalpha() and w not in unwanted
])
negative_bigram_finder = nltk.collocations.BigramCollocationFinder.from_words([
    w for w in nltk.corpus.movie_reviews.words(categories=["neg"])
    if w.isalpha() and w not in unwanted
])
```

剩下的就看你自己了！尝试不同的功能组合，想办法使用负 VADER 分数，创建比率，完善频率分布。可能性是无限的！

### 训练和使用分类器

新的特征集准备就绪后，训练分类器的第一个先决条件是定义一个从给定数据中提取特征的函数。

既然你在寻找积极的电影评论，那就把注意力放在积极的特征上，包括 VADER 评分:

```py
def extract_features(text):
    features = dict()
    wordcount = 0
    compound_scores = list()
    positive_scores = list()

    for sentence in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sentence):
            if word.lower() in top_100_positive:
                wordcount += 1
        compound_scores.append(sia.polarity_scores(sentence)["compound"])
        positive_scores.append(sia.polarity_scores(sentence)["pos"])

    # Adding 1 to the final compound score to always have positive numbers
    # since some classifiers you'll use later don't work with negative numbers.
    features["mean_compound"] = mean(compound_scores) + 1
    features["mean_positive"] = mean(positive_scores)
    features["wordcount"] = wordcount

    return features
```

`extract_features()`应该返回一个字典，它将为每段文本创建三个特征:

1.  平均复合得分
2.  平均正面分数
3.  文本中所有正面评论中前 100 个单词中的单词量

为了训练和评估分类器，您需要为要分析的每个文本建立一个特征列表:

```py
features = [
    (extract_features(nltk.corpus.movie_reviews.raw(review)), "pos")
    for review in nltk.corpus.movie_reviews.fileids(categories=["pos"])
]
features.extend([
    (extract_features(nltk.corpus.movie_reviews.raw(review)), "neg")
    for review in nltk.corpus.movie_reviews.fileids(categories=["neg"])
])
```

这个特性列表中的每一项都需要是一个元组，它的第一项是由`extract_features`返回的字典，第二项是文本的预定义类别。在最初用一些已经被分类的数据(比如`movie_reviews`语料库)训练分类器之后，您将能够对新数据进行分类。

训练分类器包括分割特征集，以便一部分用于训练，另一部分用于评估，然后调用`.train()`:

>>>

```py
>>> # Use 1/4 of the set for training
>>> train_count = len(features) // 4
>>> shuffle(features)
>>> classifier = nltk.NaiveBayesClassifier.train(features[:train_count])
>>> classifier.show_most_informative_features(10)
Most Informative Features
 wordcount = 2                 pos : neg    =      4.1 : 1.0
 wordcount = 3                 pos : neg    =      3.8 : 1.0
 wordcount = 0                 neg : pos    =      1.6 : 1.0
 wordcount = 1                 pos : neg    =      1.5 : 1.0
>>> nltk.classify.accuracy(classifier, features[train_count:])
0.668
```

因为你在重组特性列表，每次运行都会给你不同的结果。事实上，调整列表以避免在列表的第一个季度意外地将相似的分类评论分组是很重要的。

增加一个单一的特征略微提高了 VADER 的初始准确度，从 64%提高到 67%。更多的功能可能会有所帮助，只要它们真正表明一篇评论有多正面。您可以使用`classifier.show_most_informative_features()`来确定哪些特征最能代表特定的属性。

要对新数据进行分类，在某处找到一个电影评论，并将其传递给`classifier.classify()`。你也可以用`extract_features()`告诉你到底是怎么评分的:

>>>

```py
>>> new_review = ...
>>> classifier.classify(new_review)
>>> extract_features(new_review)
```

这是正确的吗？根据来自`extract_features()`的评分输出，您可以改进什么？

特征工程是提高给定算法准确性的重要部分，但不是全部。另一个策略是使用和比较不同的分类器。

[*Remove ads*](/account/join/)

## 比较附加分类器

NLTK 提供了一个类，可以使用流行的机器学习框架 [scikit-learn](https://scikit-learn.org/stable/) 中的大多数分类器。

scikit-learn 提供的许多分类器都可以快速实例化，因为它们的缺省值通常都很好。在这一节中，您将学习如何将它们集成到 NLTK 中来对语言数据进行分类。

### 安装和导入 scikit-learn

像 NLTK 一样，scikit-learn 是第三方 Python 库，所以您必须用`pip`安装它:

```py
$ python3 -m pip install scikit-learn
```

安装 scikit-learn 之后，您将能够直接在 NLTK 中使用它的分类器。

以下分类器是您可以使用的所有分类器的子集。这些将在 NLTK 中用于情感分析:

```py
from sklearn.naive_bayes import (
    BernoulliNB,
    ComplementNB,
    MultinomialNB,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
```

导入这些分类器后，首先必须实例化每个分类器。谢天谢地，所有这些都有很好的默认值，不需要太多的调整。

为了帮助评估准确性，有一个分类器名称及其实例的映射是很有帮助的:

```py
classifiers = {
    "BernoulliNB": BernoulliNB(),
    "ComplementNB": ComplementNB(),
    "MultinomialNB": MultinomialNB(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier(),
    "LogisticRegression": LogisticRegression(),
    "MLPClassifier": MLPClassifier(max_iter=1000),
    "AdaBoostClassifier": AdaBoostClassifier(),
}
```

现在，您可以使用这些实例进行训练和准确性评估。

### 通过 NLTK 使用 scikit-learn 分类器

因为 NLTK 允许您将 scikit-learn 分类器直接集成到它自己的分类器类中，所以训练和分类过程将使用您已经看到的相同方法，`.train()`和`.classify()`。

您还可以利用之前通过`extract_features()`构建的同一个`features`列表。为了提醒你，下面是你如何建立`features`名单的:

```py
features = [
    (extract_features(nltk.corpus.movie_reviews.raw(review)), "pos")
    for review in nltk.corpus.movie_reviews.fileids(categories=["pos"])
]
features.extend([
    (extract_features(nltk.corpus.movie_reviews.raw(review)), "neg")
    for review in nltk.corpus.movie_reviews.fileids(categories=["neg"])
])
```

`features`列表包含元组，其第一项是由`extract_features()`给出的一组特征，其第二项是来自`movie_reviews`语料库中预分类数据的分类标签。

由于列表的前半部分只包含正面评论，因此首先对其进行洗牌，然后遍历所有分类器来训练和评估每个分类器:

>>>

```py
>>> # Use 1/4 of the set for training
>>> train_count = len(features) // 4
>>> shuffle(features)
>>> for name, sklearn_classifier in classifiers.items():
...     classifier = nltk.classify.SklearnClassifier(sklearn_classifier)
...     classifier.train(features[:train_count])
...     accuracy = nltk.classify.accuracy(classifier, features[train_count:])
...     print(F"{accuracy:.2%} - {name}")
...
67.00% - BernoulliNB
66.80% - ComplementNB
66.33% - MultinomialNB
69.07% - KNeighborsClassifier
62.73% - DecisionTreeClassifier
66.60% - RandomForestClassifier
72.20% - LogisticRegression
73.13% - MLPClassifier
69.40% - AdaBoostClassifier
```

对于每个 scikit-learn 分类器，调用`nltk.classify.SklearnClassifier`来创建一个可用的 NLTK 分类器，可以像您之前看到的那样使用`nltk.NaiveBayesClassifier`及其其他内置分类器对其进行训练和评估。`.train()`和`.accuracy()`方法应该接收相同特性列表的不同部分。

现在，在添加第二个功能之前，您已经达到了超过 73%的准确率！虽然这并不意味着当你设计新特性时,`MLPClassifier`将继续是最好的，但是拥有额外的分类算法显然是有利的。

[*Remove ads*](/account/join/)

## 结论

您现在已经熟悉了 NTLK 的特性，它允许您将文本处理成可以过滤和操作的对象，这允许您分析文本数据以获得关于其属性的信息。您还可以使用不同的分类器对数据进行情感分析，并了解受众对内容的反应。

**在本教程中，您学习了如何:**

*   **拆分**和**过滤**文本数据以备分析
*   分析**词频**
*   用不同的方法找出**的一致**和**的搭配**
*   使用 NLTK 内置的 VADER 进行**快速情绪分析**
*   为**自定义分类**定义特征
*   使用和比较 scikit-learn 中的**分类器**,用于 NLTK 中的情感分析

有了这些工具，您就可以开始在自己的项目中使用 NLTK 了。为了获得一些灵感，看看一个[情绪分析可视化工具](https://realpython.com/twitter-sentiment-python-docker-elasticsearch-kibana/)，或者尝试在一个 [Python web 应用程序](https://realpython.com/flask-by-example-part-3-text-processing-with-requests-beautifulsoup-nltk/)中增加文本处理，同时了解其他流行的包！******
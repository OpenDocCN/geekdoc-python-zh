# Python 中的词干化和词汇化

> 原文：<https://www.askpython.com/python/examples/stemming-and-lemmatization>

在自然语言处理领域，词干化和词汇化是用于准备文本、文档以供进一步分析的文本规范化技术。

## 理解词干化和词汇化

在处理语言数据时，我们需要承认这样一个事实，即像“care”和“care”这样的词具有相同的意思，但以不同的时态形式使用。在这里，我们利用词干化和词汇化将单词简化为基本形式。

在本文中，我们将使用 [NLTK 库](https://www.askpython.com/python-modules/tokenization-in-python-using-nltk)和 SpaCy 库来执行词干化和词汇化。

## 什么是词干？

对单词进行词干分析的计算机程序或子程序可以称为词干分析程序、词干分析算法或词干分析器。 ( [维基](https://en.wikipedia.org/wiki/Stemming))

词干分析用于预处理文本数据。英语中一个单词有很多变体，所以为了减少机器学习算法学习的歧义，过滤这些单词并将其简化为基本形式是非常重要的。

NLTK 提供了对单词进行词干分析的类。使用最广泛的词干算法有 **PorterStemmer** 、 **SnowballStemmer** 等。

### 与波特斯特默一起创作斯特梅尔

让我们试试这个词干法。

```py
#Importing required modules
from nltk.stem.porter import PorterStemmer

#Creating the class object
stemmer = PorterStemmer()

#words to stem
words = ['rain','raining','faith','faithful','are','is','care','caring']

#Stemming the words
for word in words:
    print(word+' -> '+ stemmer.stem(word))

```

**输出:**

```py
rain --> rain
raining --> rain
faith --> faith
faithful --> faith
are --> are
is --> is
care --> care
caring --> care

```

PorterStemmer 类有`.stem`方法，该方法将一个单词作为输入参数，并返回简化为其根形式的单词。

### 用雪球斯特梅尔创造一个斯特梅尔

它也被称为 Porter2 词干算法，因为它倾向于修复波特斯特梅尔的一些缺点。让我们看看如何使用它。

```py
#Importing the class
from nltk.stem.snowball import SnowballStemmer

#words to stem
words = ['rain','raining','faith','faithful','are','is','care','caring']

#Creating the Class object
snow_stemmer = SnowballStemmer(language='english')

#Stemming the words
for word in words:
    print(word+' -> '+snow_stemmer.stem(word))

```

**输出:**

```py
rain --> rain
raining --> rain
faith --> faith
faithful --> faith
are --> are
is --> is
care --> care
caring --> care

```

两个词干分析器的输出看起来很相似，因为我们在演示中使用了有限的文本语料库。随意试验不同的单词，比较两者的输出。

## 什么是词汇化？

词条化是查找单词词条的算法过程——这意味着不同于词干化，词干化可能导致不正确的单词归约，词条化总是根据单词的含义来归约单词。

它有助于返回单词的基本形式或字典形式，这就是所谓的词条。

起初，词干化和词汇化看起来可能是一样的，但实际上它们是非常不同的。在下一节中，我们将看到它们之间的区别。

现在让我们来看看如何对文本数据执行词汇化。

### 使用 Python Spacy 创建 Lemmatizer

**注:** python -m spacy 下载 en_core_web_sm

为了下载所需的文件来执行词汇化，必须运行上面的代码行

```py
#Importing required modules
import spacy

#Loading the Lemmatization dictionary
nlp = spacy.load('en_core_web_sm')

#Applying lemmatization
doc = nlp("Apples and oranges are similar. Boots and hippos aren't.")

#Getting the output
for token in doc:
    print(str(token) + ' --> '+ str(token.lemma_))

```

**输出:**

```py
Apples --> apple
and --> and
oranges --> orange
are --> be
similar --> similar
. --> .
Boots --> boot
and --> and
hippos --> hippos
are --> be
n't --> not
. --> .

```

上面的代码返回一个`spacy.doc`对象类型的迭代器，它是输入单词的符号化形式。我们可以使用`.lemma_`属性访问词汇化的单词。

看看它是如何为我们自动标记句子的。

### 用 Python NLTK 创建一个 Lemmatizer

NLTK 使用 wordnet。NLTK 词汇化方法基于 WorldNet 内置的 morph 函数。

让我们看看如何使用它。

```py
import nltk
nltk.download('wordnet') #First download the required data

```

```py
#Importing the module
from nltk.stem import WordNetLemmatizer 

#Create the class object
lemmatizer = WordNetLemmatizer()

# Define the sentence to be lemmatized
sentence = "Apples and oranges are similar. Boots and hippos aren't."

# Tokenize the sentence
word_list = nltk.word_tokenize(sentence)
print(word_list)

# Lemmatize list of words and join
lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
print(lemmatized_output)

```

**输出:**

```py
['Apples', 'and', 'oranges', 'are', 'similar', '.', 'Boots', 'and', 'hippos', 'are', "n't", '.']
Apples and orange are similar . Boots and hippo are n't .

```

## 词汇化与词干化

我明白了。一开始，在词干化和词元化之间做出选择可能会令人困惑，但词元化肯定比词干化更有效。

我们看到这两种技术都将每个单词简化为它的词根。在词干分析中，这可能只是目标词的简化形式，而词汇化则简化为真正的英语词根，因为词汇化需要在 WordNet 语料库中交叉引用目标词。

词干化 vs .词汇化？这是一个速度和细节之间权衡的问题。词干化通常比词汇化更快，但可能不准确。然而，如果我们需要我们的模型尽可能的详细和精确，那么就应该优先选择引理化。

## 结论

在本文中，我们看到了词干化和词汇化的含义。我们看到了实现词干化和词汇化的各种方法。

我们还比较了词干化和词干化，以展示这两个过程之间的差异。快乐学习！🙂
# 用 Python 从字符串中提取特定的单词

> 原文：<https://www.pythonforbeginners.com/strings/extract-a-specific-word-from-a-string-in-python>

在处理文本数据时，有时我们必须搜索文本中特定单词的出现并提取特定单词。在本教程中，我们将学习在 python 中使用内置的字符串方法和正则表达式从字符串中提取特定单词的不同方法。所以，让我们深入研究一下。

## 使用 python 中的字符串切片从字符串中提取特定的单词

如果我们知道要从字符串中提取的单词的确切位置，我们可以对字符串执行切片操作，以从字符串中提取所需的单词，如下所示。

```py
 search_string= "I am a python programmer and I am writing this code for pythonforbeginners.com"
print("String from which word has to be searched is:")
print(search_string)
print("word to be extracted from string:")
word="writing"
print(word)
#calculate length of the word
lword=len(word)
#suppose we already know the starting index of word writing i.e. 34
extracted_string= search_string[34:34+lword]
print("Extracted word is:")
print(extracted_string)
```

输出:

```py
String from which word has to be searched is:
I am a python programmer and I am writing this code for pythonforbeginners.com
word to be extracted from string:
writing
Extracted word is:
writing
```

## 使用 find()方法从字符串中提取特定的单词。

如果我们想从字符串中提取一个特定的单词，并且我们不知道这个单词的确切位置，我们可以首先使用`find()` 方法找到这个单词的位置，然后我们可以使用字符串切片来提取这个单词。

当对任何字符串调用`find()`方法时，该方法将被搜索的字符串作为参数，并将被搜索的输入字符串第一次出现的位置作为输出。如果要搜索的字符串不存在，`find()`方法返回-1。

在用`find()`方法找到要提取的单词的位置后，我们可以简单地用 slice 操作提取它，如下。

```py
search_string= "I am a python programmer and I am writing this code for pythonforbeginners.com"
print("String from which word has to be searched is:")
print(search_string)
print("word to be extracted from string:")
word="writing"
print(word)
#calculate length of the word
lword=len(word)
start_index=search_string.find(word)
print("start index of the word in string is:")
print(start_index)
extracted_string= search_string[start_index:start_index+lword]
print("Extracted word is:")
print(extracted_string)
```

输出:

```py
String from which word has to be searched is:
I am a python programmer and I am writing this code for pythonforbeginners.com
word to be extracted from string:
writing
start index of the word in string is:
34
Extracted word is:
writing 
```

## 使用 index()方法。

如果我们不知道要提取的单词的确切位置，我们也可以使用 string `index()` 方法找出单词的确切位置，然后我们可以使用切片来提取单词。

当对任何字符串调用`index()` 方法时，该方法将被搜索的字符串作为参数，并将被搜索的输入字符串第一次出现的位置作为输出。如果要搜索的字符串不存在，`index()`抛出异常。为此，我们将不得不使用 [python try except](https://www.pythonforbeginners.com/error-handling/python-try-and-except) 来处理异常，同时使用`index()`方法。

我们可以使用 index()方法和字符串切片从 python 中的字符串中提取一个特定的单词，如下所示。

```py
 search_string= "I am a python programmer and I am writing this code for pythonforbeginners.com"
print("String from which word has to be searched is:")
print(search_string)
print("word to be extracted from string:")
word="writing"
print(word)
#calculate length of the word
lword=len(word)
try:

    start_index=search_string.index(word)
    print("start index of the word in string is:")
    print(start_index)
    extracted_string= search_string[start_index:start_index+lword]
    print("Extracted word is:")
    print(extracted_string)
except:
    print("word not found") 
```

输出:

```py
String from which word has to be searched is:
I am a python programmer and I am writing this code for pythonforbeginners.com
word to be extracted from string:
writing
start index of the word in string is:
34
Extracted word is:
writing
```

## 使用正则表达式提取任何特定的单词

我们可以使用 python 中的正则表达式从字符串中提取特定的单词。我们可以使用来自`re`模块的`search()`方法来找到该单词的第一个出现，然后我们可以使用切片来获得该单词。

`re.search()`方法将以正则表达式形式提取的单词和字符串作为输入 and，并返回一个包含单词起始和结束索引的`re.MatchObject`。如果没有找到给定的单词，`re.search()`将返回`None`。在获得要提取的单词的索引后，我们可以使用字符串切片来提取它，如下所示。

```py
import re
search_string= "I am a python programmer and I am writing this code for pythonforbeginners.com"
print("String from which word has to be searched is:")
print(search_string)
print("word to be extracted from string:")
word=r"writing"
print(word)
#calculate length of the word
lword=len(word)
start_index=re.search(word,search_string).start()
print("start index of the word in string is:")
print(start_index)
extracted_string= search_string[start_index:start_index+lword]
print("Extracted word is:")
print(extracted_string)
```

输出:

```py
 String from which word has to be searched is:
I am a python programmer and I am writing this code for pythonforbeginners.com
word to be extracted from string:
writing
start index of the word in string is:
34
Extracted word is:
writing
```

## 结论

在本文中，我们看到了如何使用不同的字符串方法和正则表达式找到字符串中的任何特定单词，然后使用 python 中的字符串切片打印出这个单词。我们也可以使用 [python string split](https://www.pythonforbeginners.com/dictionary/python-split) 操作来搜索单词是否存在，因为单词是用空格分隔的。请继续关注更多内容丰富的文章。
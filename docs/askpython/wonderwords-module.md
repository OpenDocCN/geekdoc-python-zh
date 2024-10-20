# Python Wonderwords 模块–简介

> 原文：<https://www.askpython.com/python-modules/wonderwords-module>

你好，学习伙伴！今天我们将学习 Python 中一个鲜为人知的新功能，叫做 Wonderwords 模块！

## Wonderwords 模块简介

`Wonderwords`是一个 Python 库，包含各种用于生成随机单词和句子的函数。该库的功能包括:

*   各种类别中的随机单词和句子生成
*   获取您自己的自定义正则表达式
*   附带了一个惊人的命令行界面来实现这个库
*   它也是开源的！

## Wonderwords 库的实现

现在让我们直接进入 wonderwords 模块的实现。

### 1.生成随机单词

首先，我们需要导入 wonderwords 库，为了获得随机单词，我们将导入子模块 RandomWord。下一步是创建一个随机单词对象。相同的代码如下所示。

```py
from wonderwords import RandomWord
R_word_obj = RandomWord()

```

要生成随机词，我们需要对创建的随机词对象使用`word`函数。下面的代码在一个循环的帮助下生成 5 个随机单词。

```py
for i in range(5):
    x = R_word_obj.word()
    print("Word "+str(i+1)+" : ",x)

```

代码的输出生成五个随机单词，如下所示。

```py
Word 1 :  irrigation
Word 2 :  porcupine
Word 3 :  lightning
Word 4 :  award
Word 5 :  small

```

我们还可以生成特定类别的单词，或者生成具有特定开头或结尾的单词，甚至两者都有。让我们在一个代码块中生成所有这些类型的单词。

下面的代码使用相同的`R_word_obj`显示不同类别中的随机单词。同样的输出显示在代码的正下方。

```py
print("Words starting with 'w' and end with 'er'")
for i in range(5):
    x = R_word_obj.word(starts_with="w",ends_with="er")
    print("Word "+str(i+1)+" : ",x)

print("\nGenerate random Adjectives")
for i in range(5):
    x = R_word_obj.word(include_parts_of_speech=["adjectives"])
    print("Word "+str(i+1)+" : ",x)

print("\nGenerate random Verbs")
for i in range(5):
    x = R_word_obj.word(include_parts_of_speech=["verbs"])
    print("Word "+str(i+1)+" : ",x)    

print("\nGenerate random words having length between 10 and 20")
for i in range(5):
    x = R_word_obj.word(word_min_length=10,word_max_length=20)
    print("Word "+str(i+1)+" : ",x)

```

```py
Words starting with 'w' and end with 'er'
Word 1 :  winter
Word 2 :  wrestler
Word 3 :  wafer
Word 4 :  wrestler
Word 5 :  winter

Generate random Adjectives
Word 1 :  beautiful
Word 2 :  orange
Word 3 :  old-fashioned
Word 4 :  ruthless
Word 5 :  lopsided

Generate random Verbs
Word 1 :  enlist
Word 2 :  tickle
Word 3 :  study
Word 4 :  delight
Word 5 :  whine

Generate random words having length between 10 and 20
Word 1 :  sensitivity
Word 2 :  precedence
Word 3 :  recapitulation
Word 4 :  co-producer
Word 5 :  willingness

```

我们还可以利用`random_words`函数生成一串单词，而不需要每次都使用 for 循环，并把单词的数量作为一个参数。相同的代码如下所示。

```py
l1 = R_word_obj.random_words(10,include_parts_of_speech=["verbs"])
print("Random Verbs: ",l1)
print("\n")
l2 = R_word_obj.random_words(30,include_parts_of_speech=["adjectives"])
print("Random Adjectives: ",l2)

```

```py
Random Verbs:  ['manipulate', 'dive', 'shave', 'talk', 'design', 'obtain', 'wreck', 'juggle', 'challenge', 'spill']

Random Adjectives:  ['enchanting', 'berserk', 'tight', 'utter', 'staking', 'calm', 'wakeful', 'nostalgic', 'juicy', 'bumpy', 'unbiased', 'shiny', 'small', 'verdant', 'wanting', 'telling', 'famous', 'orange', 'quack', 'absent', 'devilish', 'overconfident', 'boundless', 'faded', 'cloudy', 'goofy', 'encouraging', 'guarded', 'vigorous', 'null']

```

### 2.生成随机句子

为了生成随机句子，我们需要从 Wonderwords 库中导入 RandomSentence 子模块。然后我们创建一个随机句子对象来生成随机句子。代码如下所示。

```py
from wonderwords import RandomSentence
R_sent_obj = RandomSentence()
for i in range(5):
    x = R_sent_obj.sentence()
    print("Sentence "+str(i+1)+" : ",x)

```

上面的代码将生成 5 个简单的随机句子，其输出如下所示。

```py
Sentence 1 :  The onerous dogwood twists invoice.
Sentence 2 :  The erect chauvinist kills mail.
Sentence 3 :  The noxious meet ties terminology.
Sentence 4 :  The accurate trail suggests bustle.
Sentence 5 :  The racial theism accomplishes hostel.

```

我们也可以使用下面的代码生成包含形容词的句子。输出也与代码一起显示。

```py
print("Generate sentences with adjectives")
for i in range(5):
    x = R_sent_obj.bare_bone_with_adjective()
    print("Sentence "+str(i+1)+" : ",x)

```

```py
Generate sentences with adjectives
Sentence 1 :  The ritzy sunroom mixes.
Sentence 2 :  The goofy back assembles.
Sentence 3 :  The abusive tiara offends.
Sentence 4 :  The wakeful mix mixes.
Sentence 5 :  The itchy submitter bids.

```

## 结论

恭喜你！今天，您了解了一个全新的 Python 库，名为 Wonderworld。敬请关注，了解更多信息！感谢您的阅读！
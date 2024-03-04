# Python 中的正则表达式

> 原文：<https://www.pythonforbeginners.com/regex/regular-expressions-in-python>

## 什么是正则表达式？

这是一个用紧凑语法编写的字符串模式，允许我们快速检查
给定的字符串是否匹配或包含给定的模式。

正则表达式的强大之处在于它们可以指定模式，而不仅仅是
固定字符。这篇文章中的许多例子可以在:
[谷歌 Python 课程](https://developers.google.com/edu/python/regular-expressions "google_regex")上找到

## 基本模式

**a，X，9**
普通人物正好和自己完全匹配。

**。^ $ * + ?{ [ ] | ( )**
具有特殊含义的元字符(见下文)

**。**
(句点)匹配除换行符' n '以外的任何单个字符

**w**
匹配一个“单词”字符:一个字母或数字或下划线[a-zA-Z0-9_]。
它只匹配单个字符，而不是整个单词。

**W**
匹配任何非单词字符。

**w+**
匹配一个或多个单词/字符

**b**
词与非词的界限

**s**
匹配单个空白字符、空格、换行符、回车符、制表符、表单

**S**
匹配任何非空白字符。

**t，n，r**
制表符、换行符、回车

**D**
匹配除了数字以外的任何东西

**d**
匹配一个十进制数字[0-9]

**d{1，5}**
匹配长度在 1 到 5 之间的数字。

**{n} d{5}**
连续匹配 5 位数

 **匹配字符串的开始**

****$**
匹配字符串结束的**

*******
匹配 0 个或更多重复**

****？**
匹配其前面的 0 或 1 个字符**

**使用。匹配句点或匹配斜杠。**

**如果您不确定某个字符是否有特殊含义，例如' @ '，您可以
在它前面加上一条斜线@以确保它被视为一个字符。**

## **重新发现**

**findall()可能是 re 模块
中最强大的函数，我们将在这个脚本中使用这个函数。**

**在下面的例子中，我们创建了一个包含许多电子邮件地址的字符串。**

**然后我们创建一个变量(emails ),它将包含所有找到的
电子邮件字符串的列表。**

**最后，我们使用一个 for 循环，我们可以对找到的每个电子邮件字符串
做一些事情。**

```py
`str = 'purple [[email protected]](/cdn-cgi/l/email-protection), blah monkey [[email protected]](/cdn-cgi/l/email-protection) blah dishwasher'

## Here re.findall() returns a list of all the found email strings
emails = re.findall(r'[w.-][[email protected]](/cdn-cgi/l/email-protection)[w.-]+', str) ## ['[[email protected]](/cdn-cgi/l/email-protection)', '[[email protected]](/cdn-cgi/l/email-protection)']

for email in emails:
    # do something with each found email string
    print email` 
```

**我们也可以将此应用于文件。如果您有一个文件，并且想要遍历该文件的所有行，只需将它输入 findall()并让它在一个步骤中返回一个包含所有匹配的列表**

**read()在单个字符串中返回文件的全部文本。**

**(如果你想了解更多关于 Python 中的文件处理，我们写了一个
‘备忘单’，你可以在这里找到)**

```py
`# Open file
f = open('test.txt', 'r')

# Feed the file text into findall(); it returns a list of all the found strings
strings = re.findall(r'some pattern', f.read())` 
```

## **重新搜索**

**re.search()方法接受一个正则表达式模式和一个字符串，然后
在字符串中搜索该模式。**

**语法是 re.search(模式，字符串)。**

****其中:**
**模式**
要匹配的正则表达式。**

****字符串**
将被搜索以匹配字符串中任何位置的模式的字符串。**

**它在带有可选标志的字符串中搜索第一个出现的 RE 模式。**

**如果搜索成功，search()返回一个匹配对象，否则返回 None。**

**因此，搜索之后通常会立即跟随一个 if 语句来测试
搜索是否成功。**

**在模式字符串的开头使用“r”是很常见的，这表示
一个 python“原始”字符串，它通过反斜杠而不改变，这对于正则表达式来说
非常方便。**

**此示例搜索模式“word:”后跟一个 3 个字母的单词。**

**代码 match = re.search(pat，str)将搜索结果存储在名为“match”的变量
中。**

**然后，if 语句测试匹配，如果为真，则搜索成功，并且
match.group()是匹配的文本(例如‘word:cat’)。**

**如果匹配为 false，则搜索没有成功，并且没有匹配的文本。**

```py
`str = 'an example word:cat!!'

match = re.search(r'word:www', str)

# If-statement after search() tests if it succeeded
  if match:                      
    print 'found', match.group() ## 'found word:cat'

  else:
    print 'did not find'` 
```

**正如您在下面的例子中看到的，我使用了|操作符，它搜索我指定的模式。**

```py
`import re
programming = ["Python", "Perl", "PHP", "C++"]

pat = "^B|^P|i$|H$"

for lang in programming:

    if re.search(pat,lang,re.IGNORECASE):
        print lang , "FOUND"

    else:
        print lang, "NOT FOUND"` 
```

****上述脚本的输出将是:****

**Python 找到了
Perl 找到了
PHP 找到了
C++没有找到**

## **再接**

**re 模块中的 re.sub()函数可用于替换子字符串。**

**re.sub()的语法是 re.sub(pattern，repl，string)。**

**这将用 repl 替换字符串中的匹配项。**

**在这个例子中，我将用 repl ("good ")替换 string (text)中所有出现的 re 模式(" cool")
。**

```py
`import re
text = "Python for beginner is a very cool website"
pattern = re.sub("cool", "good", text)
print text2` 
```

**这是另一个例子(取自[谷歌的 Python 类](https://code.google.com/edu/languages/google-python-class/regular-expressions.html "Google"))，它搜索所有
的电子邮件地址，并更改它们以保留用户(1)但将
yo-yo-dyne.com 作为主机。**

```py
`str = 'purple [[email protected]](/cdn-cgi/l/email-protection), blah monkey [[email protected]](/cdn-cgi/l/email-protection) blah dishwasher' 

## re.sub(pat, replacement, str) -- returns new string with all replacements,

## 1 is group(1), 2 group(2) in the replacement

print re.sub(r'([w.-]+)@([w.-]+)', r'[[email protected]](/cdn-cgi/l/email-protection)', str)

## purple [[email protected]](/cdn-cgi/l/email-protection), blah monkey [[email protected]](/cdn-cgi/l/email-protection) blah dishwasher` 
```

## **重新编译**

**使用 re.compile()函数，我们可以将模式编译成模式对象，
，它有各种操作的方法，比如搜索模式匹配
或执行字符串替换。**

**让我们看两个例子，使用 re.compile()函数。**

**第一个例子检查用户的输入是否只包含字母、
空格或。(无数字)**

**不允许使用任何其他字符。**

```py
`import re

name_check = re.compile(r"[^A-Za-zs.]")

name = raw_input ("Please, enter your name: ")

while name_check.search(name):
    print "Please enter your name correctly!"
    name = raw_input ("Please, enter your name: ")` 
```

**第二个例子检查用户的输入是否只包含数字、
括号、空格或连字符(没有字母)**

**不允许使用任何其他字符**

```py
`import re

phone_check = re.compile(r"[^0-9s-()]")

phone = raw_input ("Please, enter your phone: ")

while phone_check.search(phone):
    print "Please enter your phone correctly!"
    phone = raw_input ("Please, enter your phone: ")` 
```

****上述脚本的输出将是:****

**请输入您的电话号码**

**请正确输入您的电话号码！**

**它将继续询问，直到你只输入数字。**

## **在地址中查找电子邮件域**

**让我们用我在 [stackoverflow](https://stackoverflow.com/questions/5629907/python-regular-expressions-find-email-domain-in-address "so-regular-expressions") 上找到的一个简洁的脚本来结束这篇关于 Python 中正则表达式的文章。**

****@**
扫描直到看到这个字符**

****【w .】**
潜在匹配的一组字符，所以 w 是全字母数字字符，
和尾随句点。添加到该字符集。**

****+**+
先前设定的一个或多个。**

**因为这个正则表达式匹配句点字符和@后面的每一个字母数字
,所以它甚至会匹配句子中间的电子邮件域。**

```py
`import re

s = 'My name is Conrad, and [[email protected]](/cdn-cgi/l/email-protection) is my email.'

domain = re.search("@[w.]+", s)

print domain.group()` 
```

**输出:
@gmail.com**

##### **更多阅读**

**[https://developers . Google . com/edu/python/regular-expressions](https://developers.google.com/edu/python/regular-expressions "regular-expressions_google")
[http://www.doughellmann.com/PyMOTW/re/](http://www.doughellmann.com/PyMOTW/re/ "Re PYMOTW")
[http://www.daniweb.com/](https://www.daniweb.com/software-development/python/tutorials/238544/simple-regex-tutorial# "simple-regex-tutorial")**
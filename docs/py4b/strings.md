# Python 中的字符串是什么？

> 原文：<https://www.pythonforbeginners.com/basics/strings>

## 什么是字符串？

字符串是按顺序排列的字符列表。

字符是你可以在键盘上一次击键输入的任何东西，比如一个字母、一个数字或一个反斜杠。

字符串可以有空格:“hello world”。

空字符串是包含 0 个字符的字符串。

Python 字符串是不可变的

Python 将所有由引号(“”或“)分隔的内容都识别为字符串。

## 访问字符串

```py
 **Use [ ] to access characters in a string:**
word = "computer"
letter = word[0] 

**Use [ # :#] to get set of letters**
word= word[0:3]

**To pick from beginning to a set point:**
word = [:4]

**To pick from set point to end:**
word = [3:]

**To pick starting from the end:**
word = [-1] 
```

## 引用

```py
 **Strings can be enclosed in single quotes**
print 'Hello World in single quotes'

**Strings can also be enclosed in double quotes**
print "Hello World in double quotes"

**Strings can also be enclosed in triple quotes**
print """ This is a multiline string
Such strings are also possible in Groovy.
These strings can contain ' as well as " quotes """

**Strings can be continued on the next line**
print "This string is continued on the
next line (in the source) but a newline is not added" 
```

## Raw Strings

原始字符串的存在是为了让您可以更方便地表达将被转义序列处理修改的字符串。

这在写出[正则表达式](https://www.pythonforbeginners.com/regex/regular-expressions-in-python)或其他字符串形式的代码时尤其有用。

## 连接字符串

在 Python 中，有几种不同的方法来将字符串[连接起来](/concatenation/string-concatenation-and-formatting-in-python/)。串联将两个(或多个)字符串组合成一个新的字符串对象。

你可以使用+运算符，就像这样:

```py
 print "You can concatenate two " + "strings with the '+' operator." 
```

```py
 str1 = "Hello"
str2 = "World"
str1 + str2    	# concatenation: a new string 
```

**字符串可以用空格连接**

```py
word = 'left' "right" 'left' 
```

**任何字符串表达式都可以用+** 连接

```py
 word = wordA + "
" + wordB 
```

## 反向字符串

```py
 string = "Hello World"

print ' '.join(reversed(string))
>>Output:
d l r o W o l l e H 
```

## 更改大小写字符串

```py
 string = "Hello World"

print string.lower()

print string.upper()

print string.title() 
```

## 替换字符串

```py
 string = "Hello World" 
```

string.replace("你好"、"再见")

## 重复字符串

```py
 print "."* 10 # prints ten dots( print string * n ; prints the string n times) 
```

## 拆分字符串

Python 有一个非常简洁的函数，可以将字符串分解成更小的字符串。

split 函数使用定义的分隔符
将单个字符串拆分成一个字符串数组。如果没有定义分隔符，则使用空白。

```py
x = 'blue,red,green'
x.split(",")
['blue', 'red', 'green']

word = "This is some random text"
words2 = word.split(" ")
print words
['This', 'is', 'some', 'random', 'text'] 
```

## 开始/结束

```py
 Checking if a string starts or ends with a substring:
s = "hello world"

s.startswith("hello")
True

s.endswith("rld")
True 
```

## 条状字符串

Python 字符串具有 strip()、lstrip()、rstrip()方法，用于移除字符串两端的任何字符。

如果没有指定要删除的字符，则空白将被删除。

```py
string = "Hello World"

#Strip off newline characters from end of the string
print string.strip('
')

strip() 	#removes from both ends
lstrip() 	#removes leading characters (Left-strip)
rstrip() 	#removes trailing characters (Right-strip)

spacious = "   xyz   "
print spacious.strip()

spacious = "   xyz   "
print spacious.lstrip()

spacious =  "xyz   "
print spacious.rstrip() 
```

## 分割字符串

字符串是有索引的，所以我们可以用对应的索引来引用字符串的每一个位置。请记住，python 和许多其他语言一样，是从 0 开始计数的！！

```py
print string[1]	        #get one char of the word
print string[1:2]       #get one char of the word (same as above)
print string[1:3]       #get the first three char
print string[:3]        #get the first three char
print string[-3:]       #get the last three char
print string[3:]        #get all but the three first char
print string[:-3]       #get all but the three last character

x = "my string"

x[start:end] 			# items start through end-1
x[start:]    			# items start through the rest of the list
x[:end]      			# items from the beginning through end-1
x[:]         			# a copy of the whole list 
```

## 格式化字符串

下面是 python 中[字符串格式化](/basics/strings-formatting/)的几个例子。

### 带%的字符串格式

百分比“%”字符标记说明符的开始。

%s #用于字符串
%d #用于数字
%f #用于浮点

x = 'apple'
y = 'lemon'
z = "篮子里的物品是%s 和%s" % (x，y)

注意:确保对值使用元组。

### 使用{ }设置字符串格式

成对的空花括号{}充当我们要放在字符串中的变量的占位符。

然后，我们将这些变量按顺序作为 strings format()方法的输入传递到我们的字符串中。

有了这个方法，我们不需要先把整数转换成字符串类型，format 方法会自动完成。

```py
 fname = "Joe"
lname = "Who"
age = "24"

#Example of how to use the format() method:
print "{} {} is {} years ".format(fname, lname, age)

#Another really cool thing is that we don't have to provide the inputs in the 
#same order, if we number the place-holders.

print "{0} {1} is {2} years".format(fname, lname, age) 
```

## 连接字符串

这个方法接受一个字符串列表，并用每个元素之间的调用字符串将它们连接在一起。

```py
 >>> ' '.join(['the', 'cat', 'sat', 'on', 'the', 'mat'])
'the cat sat on the mat'

#Let's look at one more example of using the Join method:
#creating a new list
>>> music = ["Abba","Rolling Stones","Black Sabbath","Metallica"]

#Join a list with an empty space
>>> print ' '.join(music)

#Join a list with a new line
>>> print "
".join(music) 
```

## 测试字符串

Python 中的字符串可以测试真值。返回类型将是布尔值(真或假)

```py
my_string = "Hello World"

my_string.isalnum()			#check if all char are numbers
my_string.isalpha()			#check if all char in the string are alphabetic
my_string.isdigit()			#test if string contains digits
my_string.istitle()			#test if string contains title words
my_string.isupper()			#test if string contains upper case
my_string.islower()			#test if string contains lower case
my_string.isspace()			#test if string contains spaces
my_string.endswith('d')		#test if string endswith a d
my_string.startswith('H')	#test if string startswith H 
```

## 内置字符串方法

String 方法在调用它的字符串上工作！这些是[内置的字符串方法](/basics/strings-built-in-methods/)。(使用点符号)

```py
>>string.string_method()

string = "Hello World" 
```

为了操作字符串，我们可以使用 Pythons 的一些内置方法

```py
string.upper() 				#get all-letters in uppercase
string.lower() 				#get all-letters in lowercase
string.capitalize() 		#capitalize the first letter
string.title() 				#capitalze the first letter of words
string.swapcase() 			#converts uppercase and lowercase
string.strip() 				#remove all white spaces
string.lstrip() 			#removes whitespace from left
string.rstrip() 			#removes whitespace from right
string.split() 				#splitting words
string.split(',') 			#split words by comma
string.count('l') 			#count how many times l is in the string
string.find('Wo') 			#find the word Wo in the string
string.index("Wo") 			#find the letters Wo in the string
":".join(string) 			#add a : between every char
" ".join(string) 			#add a whitespace between every char
len(string) 				#find the length of the string
string.replace('World', 'Tomorrow') #replace string World with Tomorrow 
```

**来源**
[https://github.com/adaptives/python-examples](https://github.com/adaptives/python-examples "python-examples")
[http://en.wikibooks.org/wiki/Python_Programming](https://en.wikibooks.org/wiki/Python_Programming "python_programming")
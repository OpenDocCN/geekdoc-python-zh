# 在 Python 中从字符串中删除字符的 5 种方法

> 原文：<https://www.askpython.com/python/string/remove-character-from-string-python>

以下方法用于在 Python 中移除字符串中的特定字符。

1.  通过使用`Naive`方法
2.  通过使用`replace()`功能
3.  通过使用`slice`和`concatenation`
4.  通过使用`join()`和`list comprehension`
5.  通过使用`translate()`方法

注意，这个字符串在 Python 中是不可变的。因此原始字符串保持不变，这些方法返回一个新的字符串。

* * *

## 1.使用 Naive 方法从字符串中移除字符

在这个方法中，我们必须运行一个循环，追加字符，并从现有的字符构建一个新的字符串，除非索引是 n(其中 n 是要删除的字符的索引)

```py
input_str = "DivasDwivedi"

# Printing original string  
print ("Original string: " + input_str) 

result_str = "" 

for i in range(0, len(input_str)): 
    if i != 3: 
        result_str = result_str + input_str[i] 

# Printing string after removal   
print ("String after removal of i'th character : " + result_str)

```

**输出**:

原始字符串:DivasDwivedi
删除第 I 个字符后的字符串:DivsDwivedi

* * *

## 2.使用 replace()方法从字符串中删除字符

```py
str = "Engineering"

print ("Original string: " + str) 

res_str = str.replace('e', '') 

# removes all occurrences of 'e' 
print ("The string after removal of character: " + res_str) 

# Removing 1st occurrence of e 

res_str = str.replace('e', '', 1) 

print ("The string after removal of character: " + res_str) 

```

**输出**:

原始字符串:工程
删除字符后的字符串:工程
删除字符后的字符串:工程

* * *

## 3.使用切分和连接从字符串中删除字符

```py
str = "Engineering"

print ("Original string: " + str) 

# Removing char at pos 3 
# using slice + concatenation 
res_str = str[:2] +  str[3:] 

print ("String after removal of character: " + res_str) 

```

**输出**:

原始字符串:删除字符后的工程
字符串:工程

* * *

## 4.使用 join()方法和列表理解从字符串中删除字符

在这种技术中，字符串中的每一个元素都被转换成一个等价的列表元素，之后它们中的每一个都被连接起来形成一个字符串，其中不包括要删除的特定字符。

```py
str = "Engineering"

print ("Original string: " + str) 

# Removing char at index 2 
# using join() + list comprehension 
res_str = ''.join([str[i] for i in range(len(str)) if i != 2]) 

print ("String after removal of character: " + res_str) 

```

**输出**:

原始字符串:删除字符后的工程
字符串:工程

* * *

## 5.使用 translate()方法从字符串中删除字符

```py
str = 'Engineer123Discipline'

print(str.translate({ord(i): None for i in '123'}))

```

**输出**:

工程学科

* * *

## 参考

*   [Python 字符串](https://docs.python.org/2.4/lib/string-methods.html)
*   Python 从字符串中删除字符
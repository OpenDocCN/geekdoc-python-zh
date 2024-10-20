# Python 字符串 split()函数

> 原文：<https://www.askpython.com/python/string/python-string-split-function>

Python string split()函数允许用户拆分字符串列表。当我们处理 CSV 数据时，这非常有用。

## String split()函数语法

```py
string.split(separator, maxsplit)
```

*   **分隔符**:它基本上是作为一个分隔符，在指定的分隔符值处分割字符串。
*   **maxsplit** :这是字符串可以拆分的上限

* * *

### **举例:** split()函数

```py
input= 'Engineering comprises of many courses.'

# splits at space
print(input.split())

```

**输出:**

```py
['Engineering', 'comprises', 'of', 'many', 'courses.']
```

* * *

### **示例:**使用“，”作为分隔符

```py
input = "hello, Engineering and Medical, are two different disciplines"

result = input.split(",")

print(result)

```

**输出:**

```py
['hello', 'Engineering and Medical', 'are two different disciplines']
```

* * *

### **示例:**设置 maxsplit = value

```py
input = "hello, Engineering and Medical, are two different disciplines"

# maxsplit = 1, returns a list with 2 elements..
i = input.split(",", 1)

print(i)

```

**输出:**

```py
['hello', ' Engineering and Medical, are two different disciplines']
```

* * *

### 多行字符串分割()函数

```py
input = 'Engineering discipline\nCommerce and Science\nYes and No'
result = input.split('\n')
for x in result:
    print(x)

```

**输出:**

```py
Engineering discipline
Commerce and Science
Yes and No
```

* * *

### split()函数中的多字符分隔符

```py
input = 'Engineering||Science||Commerce'
result = input.split('||')
print(result)

```

**输出:**

```py
['Engineering', 'Science', 'Commerce']
```

* * *

### str.split()函数

Python 字符串 split()函数也可以与类引用一起使用。我们必须将源字符串传递给 split。

```py
print(str.split('SAFA', sep='A'))
print(str.split('AMURA', sep='A', maxsplit=3))

```

* * *

### CSV-字符串 split()函数

```py
csv_input = input('Enter CSV Data\n')
csv_output1 = csv_input.split(sep=',')

print('\nList of inputs =', csv_output1)

```

**输出:**

```py
Enter CSV Data
Android, Kotlin, Perl, Go

List of inputs = ['Android', ' Kotlin', ' Perl', ' Go']
```

* * *

## 结论

Python string split()函数在将基于分隔符的值拆分成字符串列表时非常有用。

* * *

## 参考

*   Python split()函数
*   [字符串函数文档](https://docs.python.org/3.8/library/string.html)
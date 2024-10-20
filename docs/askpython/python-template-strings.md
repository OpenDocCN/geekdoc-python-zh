# Python 模板字符串

> 原文：<https://www.askpython.com/python/string/python-template-strings>

Python 的**模板字符串** **类**提供了一种简单的字符串替换方式，其中模板字段被替换为用户提供的合适的替换字符串。

有时，最好使用更容易替换的字符串，而不是使用其他格式的字符串进行替换。模板字符串正是为了这个目的而使用的，可以很容易地用最少的麻烦毫无例外地替换字符串。

* * *

## 字符串模板规则

模板字符串支持基于`**$**`的替换，这符合以下规则:

*   `**$$**` - >这是单个`$`符号的转义序列，否则它会被归类为修饰符。
*   `$**identifier**` - >这是一个替换占位符。
*   `$**{identifier}**`——>相当于`$**identifier**`。当有效字符出现在占位符之后但不是占位符的一部分时，使用。
*   任何其他`$`的出现都会引发一个`**ValueError**`异常。

下面是一个演示基本模板替换的示例:

```py
from string import Template

# Create a template with 'name' as the placeholder
template = Template('Hello $name!')

student = 'Amit'

# Perform the template substitution and print the result
print(template.substitute(name=student))

```

输出

```py
Hello Amit!

```

下面是演示模板替换的其他规则的另一个片段:

```py
from string import Template

# Create a template with $CODE as the placeholder
# The $$ is to escape the dollar sign
template = Template('The generated Code is ${CODE}-$$100')

code = 'H875'

# Perform the template substitution and print the result
print(template.substitute(CODE=code))

```

输出

```py
The generated Code is H875-$100

```

* * *

## 字符串模板类方法

### 1.模板构造函数

我们在之前的 snipper 中已经遇到过这种情况，我们使用`Template(template_string)`创建了字符串模板对象。

格式:`template_object = Template(template_string)`

### 2.替代(映射，**kwargs)

这也是我们之前代码片段的一部分，它执行了从`*mapping*`到关键字参数`*kwargs*`的模板替换。

第二个参数是一个`**kwargs`，因为我们将关键字参数作为占位符进行替换。因此，它作为字典传递，用于模板替换。

为了说明这一点，我们展示了如何将字典传递到模板字符串中。

```py
from string import Template

template = Template('The shares of $company have $state. This is $reaction.')

# Perform the template substitution and print the result
print(template.substitute(state = 'dropped', company='Google', reaction='bad'))

# Perform substitution by passing a Dictionary
dct = {'state': 'risen', 'company': 'Apple', 'reaction': 'good'}

print(template.substitute(**dct))

# Incomplete substitution results in a KeyError
try:
    template.substitute(state = 'dropped')
except KeyError:
    print('Incomplete substitution resulted in KeyError!')

```

输出

```py
The shares of Google have dropped. This is bad.
The shares of Apple have risen. This is good.
Incomplete substitution resulted in KeyError!

```

### 3.安全 _ 替代(映射，**kwargs)

这类似于`substitute()`，除了如果占位符从*映射*和 *kwargs* 中丢失，而不是引发 [`KeyError`](https://docs.python.org/3/library/exceptions.html#KeyError) 异常，原始占位符将完整地出现在结果字符串中。

```py
from string import Template

template = Template('The shares of $company have $state. This is $reaction.')

print(template.safe_substitute(company='Google'))

```

输出

```py
The shares of Google have $state. This is $reaction.

```

如您所见，没有`KeyError`，导致了不完整但无错误的替换。这就是为什么替代是“安全”的。

* * *

## 模板类属性

模板对象有`template`属性，它返回模板字符串。虽然可以修改，但最好不要更改该属性值。

```py
from string import Template

t = Template('Hello $name, are you $cond?')
print(t.template)

```

输出

```py
Hello $name, are you $cond?

```

* * *

## 结论

在本文中，我们学习了 String Template 类，以及它的一些用于模板字符串的常规和安全替换的方法。我们还看到了如何使用它们进行简单的字符串替换。

* * *

## 参考

*   [Python 模板字符串](https://docs.python.org/3/library/string.html#template-strings)
*   关于模板字符串的 JournalDev 文章
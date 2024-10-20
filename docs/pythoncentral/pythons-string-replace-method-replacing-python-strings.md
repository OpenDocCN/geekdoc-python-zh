# Python 的 string.replace()方法–替换 Python 字符串

> 原文：<https://www.pythoncentral.io/pythons-string-replace-method-replacing-python-strings/>

## 替换 Python 字符串

通常你会有一个字符串(`str` object)，在这里你会想要通过用另一段文本替换*一段文本来修改内容。在 Python 中，一切都是对象——包括字符串。这包括`str`对象。幸运的是，Python 的`string`模块附带了一个`replace()`方法。`replace()`方法是`string`模块的一部分，既可以从`str`对象调用，也可以单独从`string`模块调用。*

## Python 的 string.replace()原型

`string.replace()`方法的原型如下:

> `string.replace(s, old, new[, maxreplace])`

##### 函数参数

*   `s`: Find and replace the string.
*   **Old `: the old substring you want to replace.`**
`*   **New `: The new substring that you want to place in the position of the old substring.`***   `**[max replace] `: the maximum number of times you want to replace a substring.`**``

## 例子

### 从字符串模块直接导入字符串

```py
our_str = 'Hello World'

import string

new_str = string.replace(our_str, 'World', 'Jackson')
print(new_str)

new_str = string.replace(our_str, 'Hello', 'Hello,')
print(new_str)

our_str = 'Hello you, you and you!'
new_str = string.replace(our_str, 'you', 'me', 1)
print(new_str)
new_str = string.replace(our_str, 'you', 'me', 2)
print(new_str)
new_str = string.replace(our_str, 'you', 'me', 3)
print(new_str)
```

这给了我们以下输出:

```py
Hello Jackson
Hello, World
Hello me, you and you!
Hello me, me and you!
Hello me, me and me!
```


并使用来自`str`对象的`string.replace()`方法:

```py
our_str = 'Hello World'

new_str = our_str.replace('World', 'Jackson')
print(new_str)

new_str = our_str.replace('Hello', 'Hello,')
print(new_str)

our_str = 'Hello you, you and you!'
new_str = our_str.replace('you', 'me', 1)
print(new_str)
new_str = our_str.replace('you', 'me', 2)
print(new_str)
new_str = our_str.replace('you', 'me', 3)
print(new_str)
```

这给了我们:

```py
Hello Jackson
Hello, World
Hello me, you and you!
Hello me, me and you!
Hello me, me and me!
```

令人震惊的是，我们得到了相同的输出。

现在你知道了！Python 的`string.replace()`。
# 如何在 Python 中生成随机字符串

> 原文：<https://www.askpython.com/python/examples/generate-random-strings-in-python>

在本文中，我们将看看如何在 Python 中生成随机字符串。顾名思义，我们需要生成一个随机的字符序列，它适用于`random`模块。

这里有各种方法，所以我们将从最直观的方法开始；使用随机整数。

* * *

## 从随机整数序列构建一个字符串

如您所知，`chr(integer)`将整数映射到字符，假设它在 ASCII 限制内。(本文中取为`255`)

我们可以使用这个映射将任意整数缩放到 ASCII 字符级别，使用`chr(x)`，其中`x`是随机生成的。

```py
import random

# The limit for the extended ASCII Character set
MAX_LIMIT = 255

random_string = ''

for _ in range(10):
    random_integer = random.randint(0, MAX_LIMIT)
    # Keep appending random characters using chr(x)
    random_string += (chr(random_integer))

print(random_string, len(random_string))

```

**样本输出**

```py
ð|ÒR:
     Rè 10

```

这里，虽然字符串的长度似乎是 10 个字符，但是我们会看到一些奇怪的字符以及换行符、空格等。

这是因为我们考虑了整个 ASCII 字符集。

如果我们只想处理英文字母，我们可以使用它们的 ASCII 值。

```py
import random

random_string = ''

for _ in range(10):
    # Considering only upper and lowercase letters
    random_integer = random.randint(97, 97 + 26 - 1)
    flip_bit = random.randint(0, 1)
    # Convert to lowercase if the flip bit is on
    random_integer = random_integer - 32 if flip_bit == 1 else random_integer
    # Keep appending random characters using chr(x)
    random_string += (chr(random_integer))

print(random_string, len(random_string))

```

**样本输出**

```py
wxnhvYDuKm 10

```

如你所见，现在我们只有大写和小写字母。

但是我们可以避免所有这些麻烦，让 Python 为我们做这些工作。Python 为此给了我们`string`模块！

让我们看看如何使用几行代码就能做到这一点！

## 使用字符串模块在 Python 中生成随机字符串

这里定义了 [Python 字符串](https://www.askpython.com/python/string/python-string-functions)使用的字符列表，我们可以在这些字符组中挑选。

然后我们将使用`random.choice()`方法随机选择字符，而不是像以前那样使用整数。

让我们定义一个函数`random_string_generator()`，它为我们做所有这些工作。这将生成一个随机字符串，给定字符串的长度，以及允许从中采样的字符集。

```py
import random
import string

def random_string_generator(str_size, allowed_chars):
    return ''.join(random.choice(allowed_chars) for x in range(str_size))

chars = string.ascii_letters + string.punctuation
size = 12

print(chars)
print('Random String of length 12 =', random_string_generator(size, chars))

```

这里，我们将允许的字符列表指定为`string.ascii_letters`(大写和小写字母)，以及`string.punctuation`(所有标点符号)。

现在，我们的主函数只有 2 行，我们可以使用`random.choice(set)`随机选择一个字符。

**样本输出**

```py
abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>[email protected][\]^_`{|}~
Random String of length 12 = d[$Om{;#cjue

```

我们确实生成了一个随机字符串，并且`string`模块允许在字符集之间进行简单的操作！

### 使随机生成更加安全

虽然上面的随机生成方法是可行的，但是如果你想让你的函数在加密上更加安全，那么使用`random.SystemRandom()`函数。

随机生成器函数示例如下所示:

```py
import random
import string

output_string = ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(10))

print(output_string)

```

**输出**

```py
iNsuInwmS8

```

这确保了您的字符串生成在加密方面是安全的。

* * *

## 随机 UUID 生成

如果你想生成一个随机的 [UUID](https://en.wikipedia.org/wiki/Universally_unique_identifier) 字符串，`uuid`模块对这个目的很有帮助。

```py
import uuid

# Generate a random UUID
print('Generated a random UUID from uuid1():', uuid.uuid1())
print('Generated a random UUID from uuid4():', uuid.uuid4())

```

**样本输出**

```py
Generated a random UUID from uuid1(): af5d2f80-6470-11ea-b6cd-a73d7e4e7bfe
Generated a random UUID from uuid4(): 5d365f9b-14c1-49e7-ad64-328b61c0d8a7

```

* * *

## 结论

在本文中，我们学习了如何在`random`和`string`模块的帮助下，在 Python 中生成随机字符串。

## 参考

*   关于生成随机字符串的 JournalDev 文章
*   [关于随机字符串生成的 StackOverflow 问题](https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits)

* * *
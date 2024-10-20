# Python 的 encode()和 decode()函数

> 原文：<https://www.askpython.com/python/string/python-encode-and-decode-functions>

Python 的`encode`和`decode`方法用于使用给定的编码对输入字符串进行编码和解码。让我们在本文中详细看看这两个函数。

* * *

## 给定字符串编码

我们对输入字符串使用`encode()`方法，每个字符串对象都有。

**格式**:

```py
input_string.encode(encoding, errors)

```

这使用`encoding`对`input_string`进行编码，其中`errors`决定在字符串编码失败时要遵循的行为。

`encode()`将导致一系列的`bytes`。

```py
inp_string = 'Hello'
bytes_encoded = inp_string.encode()
print(type(bytes_encoded))

```

正如所料，这将产生一个对象`<class 'bytes'>`:

```py
<class 'bytes'>

```

要遵循的编码类型由 `encoding`参数显示。有各种类型的字符编码方案，其中方案 **UTF-8** 在 Python 中默认使用。

让我们用一个例子来看一下`encoding`参数。

```py
a = 'This is a simple sentence.'

print('Original string:', a)

# Decodes to utf-8 by default
a_utf = a.encode()

print('Encoded string:', a_utf)

```

**输出**

```py
Original string: This is a simple sentence.
Encoded string: b'This is a simple sentence.'

```

**注意**:正如您所看到的，我们已经将输入字符串编码为 UTF-8 格式。虽然没有太大的区别，但是您可以观察到该字符串带有前缀`b`。这意味着字符串被转换成字节流，这就是它在任何计算机上的存储方式。作为字节！

这实际上不是人类可读的，只是为了可读性，以原始字符串表示，加上前缀`b`，表示它不是一个字符串，而是一个字节序列。

* * *

### 处理错误

`errors`有多种类型，下面提到其中一些:

| **错误类型** | **行为** |
| `strict` | **默认**行为，失败时引发`UnicodeDecodeError`。 |
| `ignore` | **从结果中忽略**不可编码的 Unicode。 |
| `replace` | **用问号(`?`)替换** *所有*无法编码的 Unicode 字符 |
| `backslashreplace` | **插入**一个反斜杠转义序列(`\uNNNN`)代替不可编码的 Unicode 字符。 |

让我们用一个简单的例子来看看上面的概念。我们将考虑并非所有字符都是可编码的输入字符串(例如`ö`)，

```py
a = 'This is a bit möre cömplex sentence.'

print('Original string:', a)

print('Encoding with errors=ignore:', a.encode(encoding='ascii', errors='ignore'))
print('Encoding with errors=replace:', a.encode(encoding='ascii', errors='replace'))

```

**输出**

```py
Original string: This is a möre cömplex sentence.
Encoding with errors=ignore: b'This is a bit mre cmplex sentence.'
Encoding with errors=replace: b'This is a bit m?re c?mplex sentence.'

```

* * *

## 解码字节流

类似于编码一个字符串，我们可以使用`decode()`函数将一个字节流解码成一个字符串对象。

格式:

```py
encoded = input_string.encode()
# Using decode()
decoded = encoded.decode(decoding, errors)

```

因为`encode()`将一个字符串转换成字节，`decode()`只是做相反的事情。

```py
byte_seq = b'Hello'
decoded_string = byte_seq.decode()
print(type(decoded_string))
print(decoded_string)

```

**输出**

```py
<class 'str'>
Hello

```

这表明`decode()`将字节转换为 Python 字符串。

与`encode()`类似，`decoding`参数决定解码字节序列的编码类型。`errors`参数表示解码失败时的行为，与`encode()`的值相同。

* * *

## 编码的重要性

由于对输入字符串的编码和解码取决于格式，所以在编码/解码时我们必须小心。如果我们使用了错误的格式，它将导致错误的输出，并可能导致错误。

下面的片段显示了编码和解码的重要性。

第一次解码是不正确的，因为它试图解码以 UTF-8 格式编码的输入字符串。第二个是正确的，因为编码和解码格式是相同的。

```py
a = 'This is a bit möre cömplex sentence.'

print('Original string:', a)

# Encoding in UTF-8
encoded_bytes = a.encode('utf-8', 'replace')

# Trying to decode via ASCII, which is incorrect
decoded_incorrect = encoded_bytes.decode('ascii', 'replace')
decoded_correct = encoded_bytes.decode('utf-8', 'replace')

print('Incorrectly Decoded string:', decoded_incorrect)
print('Correctly Decoded string:', decoded_correct)

```

**输出**

```py
Original string: This is a bit möre cömplex sentence.
Incorrectly Decoded string: This is a bit m��re c��mplex sentence.
Correctly Decoded string: This is a bit möre cömplex sentence.

```

* * *

## 结论

在本文中，我们学习了如何使用`encode()`和`decode()`方法对输入字符串进行编码，并对编码的字节序列进行解码。

我们还学习了它如何通过`errors`参数处理编码/解码中的错误。这对于加密和解密非常有用，例如在本地缓存加密的密码并解码以备后用。

## 参考

*   JournalDev 关于编码-解码的文章

* * *
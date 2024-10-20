# Python 的新秘密模块

> 原文：<https://www.blog.pythonlibrary.org/2017/02/16/pythons-new-secrets-module/>

Python 3.6 增加了一个名为 **secrets** 的新模块，该模块被设计为*“提供一种显而易见的方式来可靠地生成适合于管理秘密的加密性强的伪随机值，如帐户认证、令牌等”*。Python 的**随机**模块从来不是为加密用途而设计的，而是为建模和模拟而设计的。当然，您总是可以使用 Python 操作系统模块中的 **urandom()** 函数:

```py

>>> import os
>>> os.urandom(8)
'\x9c\xc2WCzS\x95\xc8'

```

但是现在我们有了 secrets 模块，我们可以创建自己的“加密的强伪随机值”。这里有一个简单的例子:

```py

>>> import secrets
>>> import string
>>> characters = string.ascii_letters + string.digits
>>> bad_password = ''.join(secrets.choice(characters) for i in range(8))
>>> bad_password
'SRvM54Z1'

```

在这个例子中，我们导入了**秘密**和**字符串**模块。接下来，我们创建一个大写字母和整数的字符串。最后，我们使用 secrets 模块的 choice()方法随机选择字符来生成一个错误的密码。我称之为错误密码的原因是因为我们没有在密码中添加符号。与许多人使用的相比，这实际上是一个相当不错的选择。我的一位读者指出，这可能被认为不好的另一个原因是，用户可能只是把它写在一张纸上。虽然这可能是真的，但使用字典单词通常是非常不鼓励的，所以你应该学会像这样使用密码或投资一个安全的密码管理器。

* * *

### 生成带有秘密的令牌

secrets 模块还提供了几种生成令牌的方法。以下是一些例子:

```py

>>>: secrets.token_bytes()
b'\xd1Od\xe0\xe4\xf8Rn\x8cO\xa7XV\x1cb\xd6\x11\xa0\xcaK'

>>> secrets.token_bytes(8)
b'\xfc,9y\xbe]\x0e\xfb'

>>> secrets.token_hex(16)
'6cf3baf51c12ebfcbe26d08b6bbe1ac0'

>>> secrets.token_urlsafe(16)
'5t_jLGlV8yp2Q5tolvBesQ'

```

**token_bytes** 函数将返回一个包含 nbytes 字节数的随机字节串。在第一个例子中，我没有提供字节数，所以 Python 为我选择了一个合理的数字。然后我又试着调用了一次，要了 8 个字节。我们尝试的下一个函数是 **token_hex** ，它将返回一个十六进制的随机字符串。最后一个函数是 **token_urlsafe** ，它将返回一个随机的 URL 安全文本字符串。文本也是 Base64 编码的！请注意，在实践中，您可能应该为您的令牌使用至少 32 个字节，以防止强力攻击( [source](https://docs.python.org/3.6/library/secrets.html#how-many-bytes-should-tokens-use) )。

* * *

### 包扎

secrets 模块是 Python 的一个有价值的补充。坦率地说，我认为像这样的东西早就应该添加进去了。但至少现在我们有了它，我们可以安全地生成加密的强令牌和密码。花点时间看看这个模块的文档，因为它有一些有趣的食谱可以玩。

* * *

### 相关阅读

*   秘密模块[文档](https://docs.python.org/3.6/library/secrets.html)
*   Python 3.6 的新特性:[秘密模块](https://docs.python.org/3.6/library/secrets.html#module-secrets)
*   PEP 506 - [向标准库添加秘密模块](https://www.python.org/dev/peps/pep-0506/)
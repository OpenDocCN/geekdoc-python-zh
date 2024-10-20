# Python 中密码学的简单介绍

> 原文：<https://www.askpython.com/python-modules/cryptography-module>

***密码术*** 被定义为通过将重要信息转换成人类无法直接理解的东西以保持实际消息安全并远离坏人之手来保持重要信息安全的过程。

在现代技术时代，每个人都应该加密发送的数据，因为这不仅是一个很好的做法，而且可以保证个人和官方信息的安全。

还必须有一个强大的加密算法，以确保加密的文本更难破解，你的信息更安全，不会落入坏人之手。

## 为什么密码学很重要？

加密很重要，原因如下:

*   **保护重要信息和通信信息**,防止未经授权的人员获取信息。
*   拥有**数字签名**，有助于保护重要信息不被伪造。
*   保持信息的完整性也很重要。

## 用 Python 实现加密

既然我们已经学到了很多关于密码学的知识。现在让我们学习如何使用 Python 编程语言自己实现它。

### 1.导入模块

为了执行加密，我们将使用`cryptography`模块和`Fernet`对象。

```py
from cryptography.fernet import Fernet

```

### 2.实现加密

为了实现加密，我们将生成一个 Fernet 密钥(称为“秘密密钥”)，然后我们使用该密钥创建一个 Fernet 对象。

这把钥匙非常重要，需要妥善保管！如果有人发现了你的密钥，他/她可以解密你所有的秘密信息，如果你丢失了它，你就不能再解密你自己的信息了。

```py
key = Fernet.generate_key()
Fernet_obj= Fernet(key)

```

下一步是使用 encrypt 函数对文本进行加密，并将消息传递给该函数。该函数将返回加密的消息。

除此之外，我们还可以使用`decrypt`函数存储来自加密消息的解密消息，并传递加密消息。

```py
Encry_text = Fernet_obj.encrypt(b"I am a secret! I will get encrypted into something you will never understand")
Org_text= Fernet_obj.decrypt(Encry_text)

```

### 3.打印结果

现在让我们用[打印](https://www.askpython.com/python/built-in-methods/python-print-function)我们获得的加密和解密的消息。

```py
print("The Encrypted text is: ", cipher_text)
print("\nThe Decrypted text is: ",plain_text)

```

输出如下所示。

```py
The Encrypted text is:  b'gAAAAABgsSrnZRaDQbApvKL_xiXfCXHV_70u5eXZKDqYIkMKwxochYNy0lmVrvPFtQWya22MZh92rimscuA5VBuoN-B5YfCRHvpBYhKsbIiuPuz-CklJ-EFyZtZ_S7TRe-b9VSoee03Z8jkxwQpR8FatZ1XWA7xZvm5WpGSQFZkN8w7Ix8riyOo='

The Decrypted text is:  b'I am a secret! I will get encrypted into something you will never understand'

```

## 结论

恭喜你！今天，您了解了加密技术以及如何自己实现加密技术。自己尝试同样的方法，对外界保密你的信息！编码快乐！
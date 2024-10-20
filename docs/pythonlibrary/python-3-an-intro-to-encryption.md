# Python 3:加密介绍

> 原文：<https://www.blog.pythonlibrary.org/2016/05/18/python-3-an-intro-to-encryption/>

Python 3 的标准库中没有太多处理加密的内容。相反，你得到的是哈希库。我们将在本章中简要地看一下这些，但是主要的焦点将放在下面的第三方包上:PyCrypto 和 cryptography。我们将学习如何用这两个库来加密和解密字符串。

* * *

### 散列法

如果您需要安全散列或消息摘要算法，那么 Python 的标准库已经在 **hashlib** 模块中涵盖了您。它包括 FIPS 安全哈希算法 SHA1、SHA224、SHA256、SHA384 和 SHA512 以及 RSA 的 MD5 算法。Python 也支持 adler32 和 crc32 哈希函数，但这些都在 **zlib** 模块中。

哈希最常见的用途之一是存储密码的哈希而不是密码本身。当然，散列必须是一个好的散列或者它可以被解密。散列的另一个流行用例是散列一个文件，然后分别发送该文件及其散列。然后，接收文件的人可以对该文件运行散列，以查看它是否与发送的散列相匹配。如果是的话，那就意味着没有人在传输过程中更改过文件。

让我们试着创建一个 md5 散列:

```py

>>> import hashlib
>>> md5 = hashlib.md5()
>>> md5.update('Python rocks!')
Traceback (most recent call last):
  File "", line 1, in <module>md5.update('Python rocks!')
TypeError: Unicode-objects must be encoded before hashing
>>> md5.update(b'Python rocks!')
>>> md5.digest()
b'\x14\x82\xec\x1b#d\xf6N}\x16*+[\x16\xf4w'
```

让我们花点时间来分解一下。首先，我们导入 **hashlib** ，然后创建一个 md5 散列对象的实例。接下来，我们将一些文本添加到 hash 对象中，得到一个回溯。事实证明，要使用 md5 散列，您必须向它传递一个字节字符串，而不是一个常规字符串。所以我们尝试了一下，然后调用它的 **digest** 方法来获取我们的散列。如果你喜欢十六进制摘要，我们也可以这样做:

```py

>>> md5.hexdigest()
'1482ec1b2364f64e7d162a2b5b16f477'

```

实际上有一种创建散列的快捷方法，所以我们接下来在创建 sha512 散列时会看到:

```py

>>> sha = hashlib.sha1(b'Hello Python').hexdigest()
>>> sha
'422fbfbc67fe17c86642c5eaaa48f8b670cbed1b'

```

如您所见，我们可以创建我们的 hash 实例，同时调用它的 digest 方法。然后我们打印出散列来看看它是什么。我选择使用 sha1 散列，因为它有一个很好的短散列，更适合页面。但是它也不太安全，所以请随意尝试其他产品。

* * *

### 密钥派生

Python 对内置于标准库中的密钥派生的支持非常有限。事实上，hashlib 提供的唯一方法是 **pbkdf2_hmac** 方法，这是 PKCS#5 基于密码的密钥派生函数 2。它使用 HMAC 作为它的伪随机函数。您可以使用类似这样的东西来散列您的密码，因为它支持 salt 和迭代。例如，如果您要使用 SHA-256，您将需要至少 16 字节的 salt 和最少 100，000 次迭代。

顺便提一句，salt 只是一个随机数据，你可以把它作为额外的输入加入到你的 hash 中，使你的密码更难“解密”。基本上，它保护您的密码免受字典攻击和预先计算的彩虹表。

让我们看一个简单的例子:

```py

>>> import binascii
>>> dk = hashlib.pbkdf2_hmac(hash_name='sha256',
        password=b'bad_password34', 
        salt=b'bad_salt', 
        iterations=100000)
>>> binascii.hexlify(dk)
b'6e97bad21f6200f9087036a71e7ca9fa01a59e1d697f7e0284cd7f9b897d7c02'

```

在这里，我们使用一个糟糕的 salt 在一个密码上创建一个 SHA256 散列，但是有 100，000 次迭代。当然，实际上并不建议使用 SHA 来创建密码的密钥。相反，你应该使用类似于 **scrypt** 的东西。另一个不错的选择是第三方包 bcrypt。它是专门为密码哈希而设计的。

* * *

### PyCryptodome

PyCrypto 包可能是 Python 中最著名的第三方加密包。可悲的是 PyCrypto 的开发在 2012 年停止。其他人继续发布 PyCryto 的最新版本，所以如果你不介意使用第三方的二进制文件，你仍然可以获得 Python 3.5 的版本。比如我在 Github 上找到了一些 PyCrypto 的二进制 Python 3.5 轮子(https://Github . com/SF Bahr/py crypto-Wheels)。

幸运的是，这个项目有一个名为 PyCrytodome 的分支，它是 PyCrypto 的替代产品。要为 Linux 安装它，您可以使用以下 pip 命令:

```py

pip install pycryptodome

```

Windows 有点不同:

```py

pip install pycryptodomex

```

如果您遇到问题，可能是因为您没有安装正确的依赖项，或者您需要一个 Windows 编译器。查看 PyCryptodome [网站](http://pycryptodome.readthedocs.io/en/latest/)获取更多安装帮助或联系支持。

同样值得注意的是，PyCryptodome 在 PyCrypto 的上一个版本上有许多增强。这是非常值得你花时间去访问他们的主页，看看有什么新功能存在。

#### 加密字符串

一旦你看完了他们的网站，我们可以继续看一些例子。对于我们的第一个技巧，我们将使用 DES 加密一个字符串:

```py

>>> from Crypto.Cipher import DES
>>> key = 'abcdefgh'
>>> def pad(text):
        while len(text) % 8 != 0:
            text += ' '
        return text
>>> des = DES.new(key, DES.MODE_ECB)
>>> text = 'Python rocks!'
>>> padded_text = pad(text)
>>> encrypted_text = des.encrypt(text)
Traceback (most recent call last):
  File "", line 1, in <module>encrypted_text = des.encrypt(text)
  File "C:\Programs\Python\Python35-32\lib\site-packages\Crypto\Cipher\blockalgo.py", line 244, in encrypt
    return self._cipher.encrypt(plaintext)
ValueError: Input strings must be a multiple of 8 in length
>>> encrypted_text = des.encrypt(padded_text)
>>> encrypted_text
b'>\xfc\x1f\x16x\x87\xb2\x93\x0e\xfcH\x02\xd59VQ'
```

这段代码有点混乱，所以让我们花点时间来分解它。首先，应该注意 DES 加密的密钥大小是 8 个字节，这就是为什么我们将密钥变量设置为大小字母字符串。我们要加密的字符串长度必须是 8 的倍数，所以我们创建了一个名为 **pad** 的函数，它可以用空格填充任何字符串，直到它是 8 的倍数。接下来，我们创建一个 DES 实例和一些想要加密的文本。我们还创建了文本的填充版本。只是为了好玩，我们尝试加密字符串的原始未填充变量，这将引发一个**值错误**。在这里，我们了解到，我们毕竟需要填充字符串，所以我们把它传入。如你所见，我们现在有了一个加密的字符串！

当然，如果我们不知道如何解密我们的字符串，这个例子是不完整的:

```py

>>> des.decrypt(encrypted_text)
b'Python rocks!   '

```

幸运的是，这很容易实现，因为我们所需要做的就是调用 des 对象上的**decrypt**方法来获取解密后的字节字符串。我们的下一个任务是学习如何使用 RSA 用 PyCrypto 加密和解密文件。但是首先我们需要创建一些 RSA 密钥！

#### 创建一个 RSA 密钥

如果你想用 RSA 加密你的数据，那么你要么需要一个公开的/私有的 RSA 密钥对，要么你需要自己生成一个。对于这个例子，我们将只生成我们自己的。由于这很容易做到，我们将在 Python 的解释器中完成:

```py

>>> from Crypto.PublicKey import RSA
>>> code = 'nooneknows'
>>> key = RSA.generate(2048)
>>> encrypted_key = key.exportKey(passphrase=code, pkcs=8, 
        protection="scryptAndAES128-CBC")
>>> with open('/path_to_private_key/my_private_rsa_key.bin', 'wb') as f:
        f.write(encrypted_key)
>>> with open('/path_to_public_key/my_rsa_public.pem', 'wb') as f:
        f.write(key.publickey().exportKey())

```

首先我们从 **Crypto 导入 **RSA** 。公钥**。然后我们创造一个愚蠢的密码。接下来，我们生成一个 2048 位的 RSA 密钥。现在我们来看看好东西。要生成私钥，我们需要调用 RSA key 实例的 **exportKey** 方法，并给它我们的密码，使用哪个 PKCS 标准以及使用哪个加密方案来保护我们的私钥。然后我们把文件写到磁盘上。

接下来，我们通过 RSA key 实例的 **publickey** 方法创建我们的公钥。我们在这段代码中使用了一个快捷方式，通过将对 **exportKey** 的调用与 publickey 方法调用链接起来，也将它写入磁盘。

#### 加密文件

现在我们有了一个私钥和一个公钥，我们可以加密一些数据并将其写入文件。这是一个非常标准的例子:

```py

from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
from Crypto.Cipher import AES, PKCS1_OAEP

with open('/path/to/encrypted_data.bin', 'wb') as out_file:
    recipient_key = RSA.import_key(
        open('/path_to_public_key/my_rsa_public.pem').read())
    session_key = get_random_bytes(16)

    cipher_rsa = PKCS1_OAEP.new(recipient_key)
    out_file.write(cipher_rsa.encrypt(session_key))

    cipher_aes = AES.new(session_key, AES.MODE_EAX)
    data = b'blah blah blah Python blah blah'
    ciphertext, tag = cipher_aes.encrypt_and_digest(data)

    out_file.write(cipher_aes.nonce)
    out_file.write(tag)
    out_file.write(ciphertext)

```

前三行包括我们从 PyCryptodome 的进口。接下来，我们打开一个要写入的文件。然后，我们将公钥导入到一个变量中，并创建一个 16 字节的会话密钥。对于本例，我们将使用混合加密方法，因此我们使用 PKCS#1 OAEP，这是最佳的非对称加密填充。这允许我们向文件中写入任意长度的数据。然后我们创建我们的 AES 密码，创建一些数据和加密数据。这将返回加密的文本和 MAC。最后，我们写出随机数、MAC(或标签)和加密文本。

顺便说一下，随机数是一个仅用于加密通信的任意数字。它们通常是随机数或伪随机数。对于 AES，它的长度必须至少为 16 个字节。请随意尝试在您最喜欢的文本编辑器中打开加密文件。你应该只看到胡言乱语。

现在让我们学习如何解密我们的数据:

```py

from Crypto.PublicKey import RSA
from Crypto.Cipher import AES, PKCS1_OAEP

code = 'nooneknows'

with open('/path/to/encrypted_data.bin', 'rb') as fobj:
    private_key = RSA.import_key(
        open('/path_to_private_key/my_rsa_key.pem').read(),
        passphrase=code)

    enc_session_key, nonce, tag, ciphertext = [ fobj.read(x) 
                                                for x in (private_key.size_in_bytes(), 
                                                16, 16, -1) ]

    cipher_rsa = PKCS1_OAEP.new(private_key)
    session_key = cipher_rsa.decrypt(enc_session_key)

    cipher_aes = AES.new(session_key, AES.MODE_EAX, nonce)
    data = cipher_aes.decrypt_and_verify(ciphertext, tag)

print(data)

```

如果您遵循前面的例子，这段代码应该很容易解析。在本例中，我们以二进制模式打开加密文件进行读取。然后我们导入我们的私钥。请注意，当您导入私钥时，必须提供您的密码。否则你会得到一个错误。接下来我们读取我们的文件。您会注意到，我们首先读入私钥，然后读入随机数的 16 个字节，接下来的 16 个字节是标签，最后是文件的其余部分，也就是我们的数据。

然后我们需要解密我们的会话密钥，重新创建我们的 AES 密钥并解密数据。

您可以使用 PyCryptodome 做更多的事情。然而，我们需要继续前进，看看在 Python 中还能使用什么来满足我们的加密需求。

* * *

### 密码术包

**密码术**包旨在成为“人类的密码术”，就像**请求**库是“人类的 HTTP”。这个想法是，你将能够创建简单的安全易用的密码配方。如果需要的话，您可以使用低级加密原语，这需要您知道自己在做什么，否则您可能会创建一些不太安全的东西。

如果您使用的是 Python 3.5，可以用 pip 安装，如下所示:

```py

pip install cryptography

```

您将会看到加密技术安装了一些依赖项。假设它们都成功完成，我们可以尝试加密一些文本。我们来试试 **Fernet** 模块。Fernet 模块实现了一个易于使用的身份验证方案，该方案使用对称加密算法，该算法可以保证在没有您定义的密钥的情况下，您用它加密的任何消息都不能被操纵或读取。Fernet 模块还支持通过**multipernet**进行密钥轮换。让我们看一个简单的例子:

```py

>>> from cryptography.fernet import Fernet
>>> cipher_key = Fernet.generate_key()
>>> cipher_key
b'APM1JDVgT8WDGOWBgQv6EIhvxl4vDYvUnVdg-Vjdt0o='
>>> cipher = Fernet(cipher_key)
>>> text = b'My super secret message'
>>> encrypted_text = cipher.encrypt(text)
>>> encrypted_text
(b'gAAAAABXOnV86aeUGADA6mTe9xEL92y_m0_TlC9vcqaF6NzHqRKkjEqh4d21PInEP3C9HuiUkS9f'
 b'6bdHsSlRiCNWbSkPuRd_62zfEv3eaZjJvLAm3omnya8=')
>>> decrypted_text = cipher.decrypt(encrypted_text)
>>> decrypted_text
b'My super secret message'

```

首先，我们需要导入 Fernet。接下来，我们生成一个密钥。我们打印出密钥，看看它看起来像什么。如你所见，这是一个随机的字节串。如果你愿意，你可以试着运行几次 **generate_key** 方法。结果总是不同的。接下来，我们使用我们的密钥创建我们的 Fernet 密码实例。

现在我们有了一个可以用来加密和解密信息的密码。下一步是创建一个值得加密的消息，然后使用 **encrypt** 方法加密它。我继续打印我们的加密文本，所以你可以看到，你不能再阅读文本。要解密我们的超级秘密消息，我们只需调用我们的密码上的**解密**，并传递加密文本给它。结果是我们得到了我们的消息的一个纯文本字节串。

* * *

### 包扎

这一章仅仅触及了 PyCryptodome 和加密软件包的皮毛。然而，它确实给了你一个关于用 Python 加密和解密字符串和文件的很好的概述。请务必阅读文档并开始试验，看看还能做些什么！

* * *

### 相关阅读

*   github 上 Python 3 的 PyCrypto Wheels
*   PyCryptodome [文档](http://pycryptodome.readthedocs.io/en/latest/src/introduction.html)
*   Python 的加密[服务](https://docs.python.org/3/library/crypto.html)
*   加密软件包的[网站](https://cryptography.io/en/latest/)
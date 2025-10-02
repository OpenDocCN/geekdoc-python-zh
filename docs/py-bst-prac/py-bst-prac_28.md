# 密码学

## Cryptography

[Cryptography](https://cryptography.io/en/latest/) [https://cryptography.io/en/latest/] 是一个开发活跃的库，它提供 了加密方法（recipes）和基元（primitives），支持 Python 2.6-2.7、Python 3.2+ 和 PyPy。

Cryptography 分为两个层，方法（recipes）层和危险底层（hazardous materials，简称 hazmat）。 方法层提供用于适当的对称加密，hazmat 层提供底层的加密基元。

### 安装

```py
$ pip install cryptography 
```

### 例子

示例代码使用了高层的对称加密方法：

```py
from cryptography.fernet import Fernet
key = Fernet.generate_key()
cipher_suite = Fernet(key)
cipher_text = cipher_suite.encrypt(b"A really secret message. Not for prying eyes.")
plain_text = cipher_suite.decrypt(cipher_text) 
```

## PyCrypto

[PyCrypto](https://www.dlitz.net/software/pycrypto/) [https://www.dlitz.net/software/pycrypto/] 是另一个密码库，它提供 安全的哈希函数和各种加密算法，支持 Python 2.1 到 3.3。

### 安装

```py
$ pip install pycrypto 
```

### 例子

```py
from Crypto.Cipher import AES
# Encryption
encryption_suite = AES.new('This is a key123', AES.MODE_CBC, 'This is an IV456')
cipher_text = encryption_suite.encrypt("A really secret message. Not for prying eyes.")

# Decryption
decryption_suite = AES.new('This is a key123', AES.MODE_CBC, 'This is an IV456')
plain_text = decryption_suite.decrypt(cipher_text) 
```

© 版权所有 2014\. A <a href="http://kennethreitz.com/pages/open-projects.html">Kenneth Reitz</a> 工程。 <a href="http://creativecommons.org/licenses/by-nc-sa/3.0/"> Creative Commons Share-Alike 3.0</a>.
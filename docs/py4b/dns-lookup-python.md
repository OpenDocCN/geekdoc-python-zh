# 使用 Python 进行 DNS 查找

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/dns-lookup-python>

套接字模块提供了一种简单的方法来查找主机名的 ip 地址。

```py
 import socket

addr1 = socket.gethostbyname('google.com')
addr2 = socket.gethostbyname('yahoo.com')

print(addr1, addr2) 
```

这将输出以下 ip 地址:

```py
 173.194.121.9 98.138.253.109 
```
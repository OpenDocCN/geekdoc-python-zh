# Python 中的 ipaddress 模块[简单示例]

> 原文：<https://www.askpython.com/python-modules/ipaddress-module>

嘿伙计们！今天我们将学习 Python 中的 ipaddress 模块。所以事不宜迟，让我们开始吧。

## 什么是 IP 地址？

IP 代表互联网协议。它用于识别任何网络上的节点。因此，任何连接到互联网的设备都需要拥有一个 IP 地址。

IP 地址有两种版本:IPv4 和 IPv6。目前使用的是 IPv4，而由于与网络上的设备相比，IPv4 地址的短缺，主要网络提供商正在缓慢地采用 IPv6。

要了解更多关于 IP 地址的信息，请点击查看 [Wiki 页面。](https://en.wikipedia.org/wiki/IP_address)

## Python 中的 ipaddress 模块怎么用？

现在让我们从使用 ipaddressmodule 开始。要设置主机地址，我们使用 *ipaddress.ip_address( )* 。

该函数根据传递的值自动确定使用哪个版本。它要么返回 IPv4 地址，要么返回 IPv6 地址。

### 1.如何创建有效的 IPv4 地址？

IPv4 验证 0 到 255 范围内的值。适合 32 位的整数代表地址中的一个二进制八位数。一个长度为 4 的被打包到字节对象中的整数。

```py
import ipaddress
ipaddress.ip_address('199.138.0.1')

```

输出:

```py
IPv4Address('199.138.0.1')

```

### 2.如何创建有效的 IPv6 地址？

IPv6 验证范围从 0 到 ffff 的值。适合 128 位的整数。一个长度为 16 的整数，被打包到一个字节对象中。

```py
import ipaddress
ipaddress.ip_address('2011:cb0::')
ipaddress.ip_address('FFFF:9999:2:FDE:257:0:2FAE:112D')

```

输出:

```py
IPv6Address('2011:cb0::')
IPv6Address('ffff:9999:2:fde:257:0:2fae:112d')

```

## 使用 ipaddress 模块在 Python 中处理 IP 地址

IP 地址伴随着一套规则。IP 地址的范围被分配了不同的功能。

例如，127.0.0.1 是分配给计算机网络模块的环回地址。当你向这个 IP 地址发送 ping 数据包时，你实际上是在 ping 你自己的计算机。

### 1.基本 IP 功能

让我们看看如何使用 Python 中的 ipaddress 模块来验证哪些地址是回送地址、多播地址、本地链路地址或保留地址

```py
import ipaddress

ipa = ipaddress.ip_address('199.138.0.1')
print(ipa.is_private) # Checks if address is private
print(ipa.is_global)  # Checks if address is global

#If address is a loopback address
print(ipaddress.ip_address("127.0.0.1").is_loopback) 

#If address is reserved for multiclass use
print(ipaddress.ip_address("229.100.0.23").is_multicast) 

#If address is reserved for link local usage
print(ipaddress.ip_address("169.254.0.100").is_link_local)

#True if the address is otherwise IETF reserved.
print(ipaddress.ip_address("240.10.0.1").is_reserved)

```

**输出:**

```py
False
True
True
True
True
True

```

### 2.反向 IP 查找

反向指针函数请求 DNS 解析此处作为参数添加的 IP 地址。如果 DNS 能够解析 IP，您将收到一个带有指定名称的输出。

如果您 ping 一个分配给某个域名的 IP，您很可能会得到该域名所在的服务器的名称。但是，这可能会根据防火墙的设置而改变。

```py
ipaddress.ip_address("199.138.0.1").reverse_pointer

```

输出:

```py
'1.0.138.199.in-addr.arpa'

```

# 使用 IP 地址模块处理 IP 网络

IP 网络和 IPv6 网络可以帮助用户定义和检查 IP 网络定义。

我们不需要编写自定义代码就可以得到我们需要的格式的 IP 网络。

1.  *前缀/ < nbits >* 表示网络掩码中设置的高位位数。
2.  2.网络掩码是一个 IP 地址，包含许多高位位。
3.  3.主机掩码是*网络掩码*的逻辑逆，用于 Cisco 访问控制列表。

```py
ipn = ipaddress.ip_network("10.0.0.0/16")
print(ipn.with_prefixlen)
print(ipn.with_hostmask)
print(ipn.with_netmask)

```

输出:

```py
10.0.0.0/16
10.0.0.0/0.0.255.255
10.0.0.0/255.255.0.0

```

### 1.检查 IP 地址是 IPv4 还是 IPv6

*ipaddress.ip_network( )* 函数用于返回网络类型的地址。它确认 IP 是在 IP4 网络还是 IP6 网络中。

```py
import ipaddress
ipaddress.ip_network('199.138.0.1')
ipaddress.ip_network('FFFF:9999:2:FDE:257:0:2FAE:112D')

```

输出:

```py
IPv4Network('199.138.0.1/32')
IPv6Network('ffff:9999:2:fde:257:0:2fae:112d/128')

```

### 2.识别 IP 网络上的主机

主机是属于网络的所有 IP 地址，除了网络地址和网络广播地址。

*host( )* 返回网络中可用主机的迭代器。

掩码长度为 31 的网络，网络地址和网络广播地址也包括在结果中，掩码长度为 32 的网络返回单个主机地址的返回列表。

```py
ipn= ipaddress.ip_network('192.0.2.0/29')
list(ipn.hosts())

```

输出:

```py
[IPv4Address('192.0.2.1'),
 IPv4Address('192.0.2.2'),
 IPv4Address('192.0.2.3'),
 IPv4Address('192.0.2.4'),
 IPv4Address('192.0.2.5'),
 IPv4Address('192.0.2.6')]

```

### 3.识别网络的广播地址

使用 broadcast_address，我们可以请求 DNS 服务器使用网络上的广播地址进行响应。

```py
ipn= ipaddress.ip_network('199.1.8.0/29')
ipn.broadcast_address

```

输出:

```py
IPv4Address('199.1.8.7')

```

### 4.识别 IP 网络重叠

这个函数告诉我们，如果一个网络部分或全部包含在另一个网络中。它返回 true 或 false。

```py
ipn1 = ipaddress.ip_network("10.10.1.32/29")
ipn2 = ipaddress.ip_network("10.10.1.32/27")
ipn3 = ipaddress.ip_network("10.10.1.48/29")
print(ipn1.overlaps(ipn2))
print(ipn1.overlaps(ipn3))
print(ipn3.overlaps(ipn2))

```

输出:

```py
True
False
True

```

### 5.IP 网络上的子网

它返回网络对象的一个[迭代器](https://www.askpython.com/python/built-in-methods/python-iterator)。prefixlen_diff 是应该增加的前缀长度，new_prefix 是子网的新前缀，大于我们的前缀。

```py
ipn1 = ipaddress.ip_network("10.10.1.32/29")
print(list(ipn1.subnets()))
print(list(ipn1.subnets(prefixlen_diff=2)))
print(list(ipn1.subnets(new_prefix=30))) 

```

输出:

```py
[IPv4Network('10.10.1.32/30'), IPv4Network('10.10.1.36/30')]
[IPv4Network('10.10.1.32/31'), IPv4Network('10.10.1.34/31'), IPv4Network('10.10.1.36/31'), IPv4Network('10.10.1.38/31')]
[IPv4Network('10.10.1.32/30'), IPv4Network('10.10.1.36/30')]

```

### 6.使用 ipaddress 模块创建超网

超网是一个或多个子网的组合。你可以[在这里](https://en.wikipedia.org/wiki/Supernetwork)了解更多关于超网的信息。使用 ipaddress 模块中的超网方法，您可以根据需要指定信息来创建子网。

*   前缀长度应该增加多少
*   *new_prefix* 是子网的所需新前缀，应该大于我们的前缀。

```py
ipnn = ipaddress.ip_network("172.10.15.160/29")
print(ipnn.supernet(prefixlen_diff=3))
print(ipnn.supernet(new_prefix=20))

```

输出:

```py
172.10.15.128/26
172.10.0.0/20

```

### 7.检查一个 IP 网络是否是另一个 IP 网络的超网/子网

如果一个网络是另一个网络的子网，或者如果一个网络是另一个网络的超网，则返回 true。返回真或假。

```py
a = ipaddress.ip_network("192.168.1.0/24")
b = ipaddress.ip_network("192.168.1.128/30")

print(b.subnet_of(a))
print(a.supernet_of(b))

```

输出:

```py
True
True

```

### 8.使用 with 接口对象

接口对象可以用作字典中的键，因为它们是可散列的。

IPv4Interface 继承了 IPv4Address 的所有属性，因为 IPv4Interface 是 IPv4Address 的子类。

这里，*199.167.1.6*的 IP 地址在网络 *199.167.1.0/24*

```py
from ipaddress import IPv4Interface
ifc = IPv4Interface("199.167.1.6/24")
print(ifc.ip)
print(ifc.network)

```

输出:

```py
199.167.1.6
199.167.1.0/24

```

我们可以用前缀表示法将网络接口表示为网络掩码和主机掩码。

```py
interface = IPv4Interface('192.0.2.5/24')
print(interface.with_prefixlen)
print(interface.with_netmask)
print(interface.with_hostmask)

```

输出:

```py
192.0.2.5/24
192.0.2.5/255.255.255.0
192.0.2.5/0.0.0.255

```

## 使用 IP 地址的杂项操作

使用 Python 中的[比较运算符，你可以检查一个 IP 地址与另一个的比较情况。看看下面的例子。](https://www.askpython.com/python/python-comparison-operators)

```py
ipa1=ipaddress.ip_address("127.0.0.2")
ipa2=ipaddress.ip_address("127.0.0.1")
print(ipa1>ipa2)
print(ipa1==ipa2)
print(ipa1!=ipa2)

```

输出:

```py
True
False
True

```

我们可以从 IP 地址对象中加减整数。

```py
ipa = ipaddress.ip_address("10.10.1.0")
print( ipa + 9)

```

输出:

```py
10.10.1.9

```

**通过使用内置函数 *str( )* 和 *int()，可以将地址转换成字符串或整数。***

```py
str(ipaddress.IPv4Address('199.138.0.1'))
int(ipaddress.IPv4Address('192.198.0.1'))

```

输出:

```py
'199.138.0.1'
3234201601

```

IPv6 地址被转换成不带区域 ID 的字符串。

```py
str(ipaddress.IPv6Address('::8'))
int(ipaddress.IPv6Address('::100'))

```

输出:

```py
'::8'
256

```

## 结论

在本教程中，我们学习了 IPv4 和 IPv6 地址、网络和接口。更多此类内容，敬请关注。快乐学习！🙂

## 参考

[IP 地址模块正式文档](https://docs.python.org/3/library/ipaddress.html#ipaddress.IPv4Network)
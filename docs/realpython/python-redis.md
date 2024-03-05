# 如何在 Python 中使用 Redis

> 原文:# t0]https://realython . com/python-redis/

在本教程中，您将学习如何将 Python 与 Redis(读作 [RED-iss](https://redis.io/topics/faq) ，或者可能是 [REE-diss](https://groups.google.com/forum/#!topic/redis-db/MtwjZC5gCeE) 或 [Red-DEES](https://en.wikipedia.org/wiki/Talk:Redis#Pronounciation) ，这取决于您问的是谁)一起使用，Redis 是一个闪电般快速的内存中键值存储，可用于从 A 到 z 的任何内容。下面是关于数据库的畅销书*七周七个数据库*对 Redis 的评论:

> 它不只是简单易用；这是一种快乐。如果 API 是程序员的 UX，那么 Redis 应该和 Mac Cube 一起放在现代艺术博物馆里。
> 
> …
> 
> 而且说到速度，Redis 很难被打败。读取速度很快，写入速度更快，根据一些基准测试，每秒可处理超过 100，000 次`SET`操作。([来源](https://realpython.com/asins/1680502530/))

好奇吗？本教程是为没有或很少有 Redis 经验的 Python 程序员编写的。我们将同时处理两个工具，并介绍 Redis 本身以及它的一个 Python 客户端库， [`redis-py`](https://github.com/andymccurdy/redis-py) 。

`redis-py`(你[将](https://realpython.com/absolute-vs-relative-python-imports/)作为`redis`导入的)是 Redis 的众多 Python 客户端之一，但它的特点是被 Redis 开发者自己宣传为[“当前 Python 的发展方向”](https://redis.io/clients#python)。它允许您从 Python 调用 Redis 命令，并返回熟悉的 Python 对象。

在本教程中，您将学习:

*   从源代码安装 Redis 并理解生成的二进制文件的用途
*   学习 Redis 本身，包括它的语法、协议和设计
*   掌握`redis-py`的同时也看到了它是如何实现 Redis 协议的
*   设置 Amazon ElastiCache Redis 服务器实例并与之通信

**免费下载:** [从 Python 技巧中获取一个示例章节:这本书](https://realpython.com/bonus/python-tricks-sample-pdf/)用简单的例子向您展示了 Python 的最佳实践，您可以立即应用它来编写更漂亮的+Python 代码。

## 从源安装 Redis】

正如我的曾曾祖父所说，没有什么比从源头安装更能磨练意志了。本节将引导您下载、制作和安装 Redis。我保证这一点也不疼！

**注意**:本节主要针对 Mac OS X 或 Linux 上的安装。如果你使用的是 Windows，有一个微软的 Redis】分支可以作为 Windows 服务安装。我只想说，Redis 作为一个程序，在 Linux 上运行最舒适，在 Windows 上的安装和使用可能很挑剔。

首先，下载 Redis 源代码作为一个 tarball:

```py
$ redisurl="http://download.redis.io/redis-stable.tar.gz"
$ curl -s -o redis-stable.tar.gz $redisurl
```

接下来，切换到`root`并将档案的源代码提取到`/usr/local/lib/`:

```py
$ sudo su root
$ mkdir -p /usr/local/lib/
$ chmod a+w /usr/local/lib/
$ tar -C /usr/local/lib/ -xzf redis-stable.tar.gz
```

或者，您现在可以删除归档本身:

```py
$ rm redis-stable.tar.gz
```

这将在`/usr/local/lib/redis-stable/`为您留下一个源代码库。Redis 是用 C 编写的，所以你需要用 [`make`](https://www.gnu.org/software/make/) 实用程序编译、链接和安装:

```py
$ cd /usr/local/lib/redis-stable/
$ make && make install
```

使用`make install`做两个动作:

1.  第一个`make`命令编译并链接源代码。

2.  `make install`部分获取二进制文件并将其复制到`/usr/local/bin/`，这样您就可以从任何地方运行它们(假设`/usr/local/bin/`在`PATH`中)。

以下是到目前为止的所有步骤:

```py
$ redisurl="http://download.redis.io/redis-stable.tar.gz"
$ curl -s -o redis-stable.tar.gz $redisurl
$ sudo su root
$ mkdir -p /usr/local/lib/
$ chmod a+w /usr/local/lib/
$ tar -C /usr/local/lib/ -xzf redis-stable.tar.gz
$ rm redis-stable.tar.gz
$ cd /usr/local/lib/redis-stable/
$ make && make install
```

此时，花点时间确认 Redis 在您的 [`PATH`](https://realpython.com/add-python-to-path/) 中，并检查它的版本:

```py
$ redis-cli --version
redis-cli 5.0.3
```

如果您的 shell 找不到`redis-cli`，检查以确保`/usr/local/bin/`在您的`PATH`环境变量中，如果没有，添加它。

除了`redis-cli`，`make install`实际上导致一些不同的可执行文件(和一个符号链接)被放置在`/usr/local/bin/`:

```py
$ # A snapshot of executables that come bundled with Redis
$ ls -hFG /usr/local/bin/redis-* | sort
/usr/local/bin/redis-benchmark*
/usr/local/bin/redis-check-aof*
/usr/local/bin/redis-check-rdb*
/usr/local/bin/redis-cli*
/usr/local/bin/redis-sentinel@
/usr/local/bin/redis-server*
```

虽然所有这些都有一些预期的用途，但您可能最关心的两个是`redis-cli`和`redis-server`，我们将简要介绍一下。但是在我们开始之前，先设置一些基线配置。

[*Remove ads*](/account/join/)

## 配置 Redis

Redis 是高度可配置的。虽然它开箱即可运行，但让我们花点时间来设置一些与数据库持久性和基本安全性相关的基本配置选项:

```py
$ sudo su root
$ mkdir -p /etc/redis/
$ touch /etc/redis/6379.conf
```

现在，把下面的内容写到`/etc/redis/6379.conf`。我们将在整个教程中逐步介绍其中大部分的含义:

```py
# /etc/redis/6379.conf

port              6379
daemonize         yes
save              60 1
bind              127.0.0.1
tcp-keepalive     300
dbfilename        dump.rdb
dir               ./
rdbcompression    yes
```

Redis 配置是自文档化的，为了方便阅读，在 Redis 源代码中有一个[样本`redis.conf`文件](http://download.redis.io/redis-stable/redis.conf)。如果您在生产系统中使用 Redis，排除所有干扰，花时间完整阅读这个示例文件，熟悉 Redis 的来龙去脉，并调整您的设置是值得的。

一些教程，包括 Redis 的部分文档，也可能建议运行位于 [`redis/utils/install_server.sh`](http://download.redis.io/redis-stable/utils/install_server.sh) 的 Shell 脚本`install_server.sh`。无论如何欢迎你运行这个作为上面的一个更全面的选择，但是注意关于`install_server.sh`的几个更好的点:

*   它不能在 Mac OS X 上运行——只能在 Debian 和 Ubuntu Linux 上运行。
*   它将为`/etc/redis/6379.conf`注入一组更完整的配置选项。
*   它会写一个系统 V [`init`脚本](https://bash.cyberciti.biz/guide//etc/init.d)到`/etc/init.d/redis_6379`让你做`sudo service redis_6379 start`。

Redis 快速入门指南还包含一个关于[更合适的 Redis 设置](https://redis.io/topics/quickstart#installing-redis-more-properly)的章节，但是上面的配置选项对于本教程和入门来说应该完全足够了。

**安全提示:**几年前，Redis 的作者指出了早期版本的 Redis 在没有设置配置的情况下存在的安全漏洞。Redis 3.2(截至 2019 年 3 月的当前版本 5.0.3)采取措施防止这种入侵，默认情况下将`protected-mode`选项设置为`yes`。

我们显式设置`bind 127.0.0.1`让 Redis 只监听来自本地主机接口的连接，尽管您需要在实际的生产服务器中扩展这个白名单。如果您没有在`bind`选项下指定任何内容，那么`protected-mode`的作用是作为一种安全措施来模拟这种绑定到本地主机的行为。

解决了这个问题，我们现在可以开始使用 Redis 本身了。

## 10 分钟左右到 Redis

本节将为您提供足够的 Redis 知识，概述它的设计和基本用法。

### 开始使用

Redis 有一个客户端-服务器架构，使用 T2 请求-响应模型。这意味着您(客户端)通过 TCP 连接连接到 Redis 服务器，默认情况下是在端口 6379 上。你请求一些动作(比如某种形式的读、写、获取、设置或更新)，服务器*服务*给你一个响应。

可以有许多客户机与同一个服务器对话，这正是 Redis 或任何客户机-服务器应用程序的真正意义所在。每个客户端在套接字上进行一次读取(通常是阻塞式的),等待服务器响应。

`redis-cli`中的`cli`代表**命令行接口**，`redis-server`中的`server`是用来，嗯，运行服务器的。与您在命令行运行`python`的方式相同，您可以运行`redis-cli`跳转到交互式 REPL (Read Eval Print Loop ),在这里您可以直接从 shell 运行客户端命令。

然而，首先，您需要启动`redis-server`以便有一个正在运行的 Redis 服务器与之对话。在开发中，这样做的一个常见方法是在 [localhost](https://en.wikipedia.org/wiki/Localhost) (IPv4 地址`127.0.0.1`)启动一个服务器，这是默认设置，除非您告诉 Redis。您还可以向`redis-server`传递您的配置文件的名称，这类似于将它的所有键值对指定为[命令行参数](https://realpython.com/python-command-line-arguments/):

```py
$ redis-server /etc/redis/6379.conf
31829:C 07 Mar 2019 08:45:04.030 # oO0OoO0OoO0Oo Redis is starting oO0OoO0OoO0Oo
31829:C 07 Mar 2019 08:45:04.030 # Redis version=5.0.3, bits=64, commit=00000000, modified=0, pid=31829, just started
31829:C 07 Mar 2019 08:45:04.030 # Configuration loaded
```

我们将`daemonize`配置选项设置为`yes`，因此服务器在后台运行。(否则，使用`--daemonize yes`作为`redis-server`的选项。)

现在，您已经准备好启动 Redis REPL。在命令行上输入`redis-cli`。您将看到服务器的*主机:端口*对，后跟一个`>`提示符:

```py
127.0.0.1:6379>
```

下面是一个最简单的 Redis 命令， [`PING`](https://redis.io/commands/ping) ，它只是测试与服务器的连接，如果一切正常，就返回`"PONG"`:

```py
127.0.0.1:6379> PING
PONG
```

Redis 命令不区分大小写，尽管它们的 Python 对应物绝对不区分大小写。

**注意:**作为另一个健全性检查，您可以使用`pgrep`搜索 Redis 服务器的进程 ID:

```py
$ pgrep redis-server
26983
```

要终止服务器，从命令行使用`pkill redis-server`。在 Mac OS X 上，你也可以使用`redis-cli shutdown`。

接下来，我们将使用一些常见的 Redis 命令，并将它们与纯 Python 中的命令进行比较。

[*Remove ads*](/account/join/)

### Redis 作为 Python 字典

Redis 代表**远程词典服务**。

“你是说，像 Python [字典](https://realpython.com/python-dicts/)？”你可能会问。

是的。概括地说，在 Python 字典(或通用的[散列表](https://realpython.com/python-hash-table/))和 Redis 是什么以及做什么之间有许多相似之处:

*   Redis 数据库保存*键:值*对，并支持`GET`、`SET`和`DEL`等命令，以及[几百个](https://redis.io/commands)附加命令。

*   Redis **键**永远是[弦](https://realpython.com/python-strings/)。

*   Redis **值**可以是多种不同的数据类型。我们将在本教程中介绍一些更重要的数值数据类型:`string`、`list`、`hashes`和`sets`。一些高级类型包括[地理空间项目](https://redis.io/commands#geo)和新的[流](https://redis.io/commands#stream)类型。

*   许多 Redis 命令以常数 O(1)时间运行，就像从 Python `dict`或任何哈希表中检索值一样。

Redis 的创建者 Salvatore Sanfilippo 可能不喜欢将 Redis 数据库比作普通的 Python。他将该项目称为“数据结构服务器”(而不是键值存储，如 [memcached](https://www.memcached.org/) )，因为值得称赞的是，Redis 支持存储除 *string:string* 之外的其他类型的 *key:value* 数据类型。但是对于我们这里的目的，如果您熟悉 Python 的 dictionary 对象，这是一个有用的比较。

让我们跳进来，通过例子来学习。我们的第一个玩具数据库(ID 为 0)将是一个*国家:首都*的映射，其中我们使用 [`SET`](https://redis.io/commands/set) 来设置键-值对:

```py
127.0.0.1:6379> SET Bahamas Nassau
OK
127.0.0.1:6379> SET Croatia Zagreb
OK
127.0.0.1:6379> GET Croatia
"Zagreb"
127.0.0.1:6379> GET Japan
(nil)
```

纯 Python 中相应的语句序列如下所示:

>>>

```py
>>> capitals = {}
>>> capitals["Bahamas"] = "Nassau"
>>> capitals["Croatia"] = "Zagreb"
>>> capitals.get("Croatia")
'Zagreb'
>>> capitals.get("Japan")  # None
```

我们用`capitals.get("Japan")`而不是`capitals["Japan"]`是因为 Redis 在找不到键的时候会返回`nil`而不是错误，类似于 Python 的 [`None`](https://realpython.com/null-in-python/) 。

Redis 还允许在一个命令中设置和获取多个键值对，分别是 [`MSET`](https://redis.io/commands/mset) 和 [`MGET`](https://redis.io/commands/mget) :

```py
127.0.0.1:6379> MSET Lebanon Beirut Norway Oslo France Paris
OK
127.0.0.1:6379> MGET Lebanon Norway Bahamas
1) "Beirut"
2) "Oslo"
3) "Nassau"
```

Python 中最接近的是`dict.update()`:

>>>

```py
>>> capitals.update({
...     "Lebanon": "Beirut",
...     "Norway": "Oslo",
...     "France": "Paris",
... })
>>> [capitals.get(k) for k in ("Lebanon", "Norway", "Bahamas")]
['Beirut', 'Oslo', 'Nassau']
```

我们使用`.get()`而不是`.__getitem__()`来模拟 Redis 在没有找到键时返回类似 null 值的行为。

作为第三个例子， [`EXISTS`](https://redis.io/commands/exists) 命令就像它听起来那样，检查一个键是否存在:

```py
127.0.0.1:6379> EXISTS Norway
(integer) 1
127.0.0.1:6379> EXISTS Sweden
(integer) 0
```

Python 有 [`in`关键字](https://realpython.com/python-keywords/#the-in-keyword)来测试同一个东西，哪个路由到`dict.__contains__(key)`:

>>>

```py
>>> "Norway" in capitals
True
>>> "Sweden" in capitals
False
```

这几个例子旨在使用原生 Python 展示一些常见的 Redis 命令在高层次上发生了什么。Python 示例中没有客户机-服务器组件，而且`redis-py`还没有出现。这只是为了举例说明 Redis 的功能。

下面是您见过的几个 Redis 命令及其 Python 功能等效物的总结:



```py
capitals["Bahamas"] = "Nassau"
```



```py
capitals.get("Croatia")
```



```py
capitals.update(
    {
        "Lebanon": "Beirut",
        "Norway": "Oslo",
        "France": "Paris",
    }
)
```



```py
[capitals[k] for k in ("Lebanon", "Norway", "Bahamas")]
```



```py
"Norway" in capitals
```

Python Redis 客户端库`redis-py`(您将在本文中深入研究)的工作方式有所不同。它封装了到 Redis 服务器的实际 TCP 连接，并向服务器发送原始命令，这些命令是使用 [REdis 序列化协议](https://redis.io/topics/protocol) (RESP)序列化的字节。然后，它获取原始回复，并将其解析回一个 Python 对象，如`bytes`、`int`，甚至是`datetime.datetime`。

**注意**:到目前为止，你一直通过交互式`redis-cli` REPL 与 Redis 服务器对话。你也可以[直接](https://redis.io/topics/rediscli)发布命令，就像你将一个脚本的名字传递给`python`可执行文件一样，比如`python myscript.py`。

到目前为止，您已经看到了 Redis 的一些基本数据类型，它们是 *string:string* 的映射。虽然这种键-值对在大多数键-值存储中很常见，但是 Redis 提供了许多其他可能的值类型，您将在下面看到。

[*Remove ads*](/account/join/)

### Python 与 Redis 中的更多数据类型

在启动`redis-py` Python 客户端之前，对一些 Redis 数据类型有一个基本的了解也是有帮助的。需要明确的是，所有 Redis 键都是字符串。到目前为止，除了示例中使用的字符串值之外，它还是可以采用数据类型(或结构)的值。

一个**散列**是一个*字符串:字符串*的映射，称为**字段-值**对，位于一个顶级键下:

```py
127.0.0.1:6379> HSET realpython url "https://realpython.com/"
(integer) 1
127.0.0.1:6379> HSET realpython github realpython
(integer) 1
127.0.0.1:6379> HSET realpython fullname "Real Python"
(integer) 1
```

这为一个**键**、`"realpython"`设置了三个字段-值对。如果您习惯于 Python 的术语和对象，这可能会令人困惑。Redis 散列大致类似于嵌套一层的 Python `dict`:

```py
data = {
    "realpython": {
        "url": "https://realpython.com/",
        "github": "realpython",
        "fullname": "Real Python",
    }
}
```

Redis 字段类似于上面内部字典中每个嵌套的键-值对的 Python 键。Redis 将术语 **key** 保留给保存散列结构本身的顶级数据库键。

就像基本的*字符串有`MSET`一样:字符串有*键-值对，散列也有 [`HMSET`](https://redis.io/commands/hmset) 在散列值对象中设置多个对*:*

```py
127.0.0.1:6379> HMSET pypa url "https://www.pypa.io/" github pypa fullname "Python Packaging Authority"
OK
127.0.0.1:6379> HGETALL pypa
1) "url"
2) "https://www.pypa.io/"
3) "github"
4) "pypa"
5) "fullname"
6) "Python Packaging Authority"
```

使用`HMSET`可能更类似于我们将`data`赋给上面的嵌套字典的方式，而不是像使用`HSET`那样设置每个嵌套对。

另外两个值类型是 [**列表**](https://redis.io/topics/data-types-intro#redis-lists) 和 [**集合**](https://redis.io/topics/data-types-intro#redis-sets) ，它们可以代替 hash 或 string 作为 Redis 值。它们很大程度上是它们听起来的样子，所以我不会用额外的例子来占用你的时间。散列、列表和集合每个都有一些特定于给定数据类型的命令，在某些情况下由它们的首字母表示:

*   **哈希:**对哈希进行操作的命令以`H`开头，比如`HSET`、`HGET`或者`HMSET`。

*   **集合:**对集合进行操作的命令以一个`S`开始，比如`SCARD`，它获取一个给定键对应的集合值的元素个数。

*   **列表:**操作列表的命令以`L`或`R`开始。例子包括`LPOP`和`RPUSH`。`L`或`R`指的是对单子的哪一面进行操作。一些列表命令也以`B`开头，这意味着**阻塞**。一个阻塞操作不会让其他操作在它执行的时候打断它。例如，`BLPOP`在一个列表结构上执行一个阻塞的左弹出。

**注意:**Redis 列表类型的一个值得注意的特点是它是一个[链表](https://realpython.com/linked-lists-python/)而不是数组。这意味着追加是 O(1 ),而在任意索引号索引是 O(N)。

下面是 Redis 中特定于字符串、散列、列表和集合数据类型的命令的快速列表:

| 类型 | 命令 |
| --- | --- |
| 设置 | `SADD`、`SCARD`、`SDIFF`、`SDIFFSTORE`、`SINTER`、`SINTERSTORE`、`SISMEMBER`、`SMEMBERS`、`SMOVE`、`SPOP`、`SRANDMEMBER`、`SREM`、`SSCAN`、`SUNION`、`SUNIONSTORE` |
| 混杂 | `HDEL`、`HEXISTS`、`HGET`、`HGETALL`、`HINCRBY`、`HINCRBYFLOAT`、`HKEYS`、`HLEN`、`HMGET`、`HMSET`、`HSCAN`、`HSET`、`HSETNX`、`HSTRLEN`、`HVALS` |
| 列表 | `BLPOP`、`BRPOP`、`BRPOPLPUSH`、`LINDEX`、`LINSERT`、`LLEN`、`LPOP`、`LPUSH`、`LPUSHX`、`LRANGE`、`LREM`、`LSET`、`LTRIM`、`RPOP`、`RPOPLPUSH`、`RPUSH`、`RPUSHX` |
| 用线串 | `APPEND`，`BITCOUNT`，`BITFIELD`，`BITOP`，`BITPOS`，`DECR`，`DECRBY`，`GET`，`GETBIT`，`GETRANGE`，`GETSET`，`INCR`，`INCRBY`，`INCRBYFLOAT`，`MGET`，`MSET`，`MSETNX`，`PSETEX`，`SET`，`SETBIT`，`SETEX`，`SETNX`，`SETRANGE`，`STRLEN` |

这个表并不是 Redis 命令和类型的完整描述。还有更高级数据类型的大杂烩，比如[地理空间项目](https://redis.io/commands#geo)、[排序集](https://redis.io/commands#sorted_set)和[超级日志](https://redis.io/commands#hyperloglog)。在 Redis [commands](https://redis.io/commands) 页面，您可以按数据结构组进行过滤。还有[数据类型总结](https://redis.io/topics/data-types)和[Redis 数据类型介绍](https://redis.io/topics/data-types-intro)。

既然我们要切换到用 Python 做事，你现在可以用 [`FLUSHDB`](https://redis.io/commands/flushdb) 清空你的玩具数据库，退出`redis-cli` REPL:

```py
127.0.0.1:6379> FLUSHDB
OK
127.0.0.1:6379> QUIT
```

这将把您带回您的 shell 提示符。您可以让`redis-server`在后台运行，因为您在本教程的剩余部分也需要它。

## 在 Python 中使用`redis-py`:Redis

现在您已经掌握了 Redis 的一些基础知识，是时候进入`redis-py`了，Python 客户端允许您从用户友好的 Python API 与 Redis 对话。

[*Remove ads*](/account/join/)

### 第一步

[`redis-py`](https://github.com/andymccurdy/redis-py) 是一个完善的 Python 客户端库，允许您通过 Python 调用直接与 Redis 服务器对话:

```py
$ python -m pip install redis
```

接下来，确保您的 Redis 服务器仍然在后台运行。您可以使用`pgrep redis-server`进行检查，如果您空手而归，那么使用`redis-server /etc/redis/6379.conf`重新启动一个本地服务器。

现在，让我们进入以 Python 为中心的部分。下面是`redis-py`的“hello world”:

>>>

```py
 1>>> import redis
 2>>> r = redis.Redis()
 3>>> r.mset({"Croatia": "Zagreb", "Bahamas": "Nassau"})
 4True
 5>>> r.get("Bahamas")
 6b'Nassau'
```

第 2 行中使用的`Redis`是包的中心类，是执行(几乎)任何 Redis 命令的主要工具。TCP 套接字连接和重用是在后台完成的，您可以使用类实例`r`上的方法调用 Redis 命令。

还要注意，第 6 行中返回对象的类型`b'Nassau'`是 Python 的 [`bytes`](https://docs.python.org/3/library/stdtypes.html#bytes) 类型，而不是`str`。在`redis-py`中最常见的返回类型是`bytes`而不是`str`，所以你可能需要调用`r.get("Bahamas").decode("utf-8")`，这取决于你想对返回的字节字符串做什么。

上面的代码看着眼熟吗？几乎所有情况下的方法都与 Redis 命令的名称相匹配，Redis 命令执行相同的操作。这里调用了`r.mset()`和`r.get()`，分别对应于原生 Redis API 中的`MSET`和`GET`。

这也意味着`HGETALL`变成了`r.hgetall()` , `PING`变成了`r.ping()`，以此类推。有几个[例外，但是这个规则适用于大多数命令。](https://github.com/andymccurdy/redis-py#api-reference)

虽然 Redis 命令参数通常会转换成外观相似的方法签名，但它们采用 Python 对象。例如，上例中对`r.mset()`的调用使用 Python `dict`作为第一个参数，而不是一系列字节串。

我们构建了没有参数的`Redis`实例`r`，但是如果你需要的话，它附带了许多[参数](https://github.com/andymccurdy/redis-py/blob/b940d073de4c13f8dfb08728965c6ac7c183c935/redis/client.py#L605):

```py
# From redis/client.py
class Redis(object):
    def __init__(self, host='localhost', port=6379,
                 db=0, password=None, socket_timeout=None,
                 # ...
```

您可以看到默认的*主机名:端口*对是`localhost:6379`，这正是我们在本地保存的`redis-server`实例中所需要的。

`db`参数是数据库号。您可以在 Redis 中一次管理多个数据库，每个数据库由一个整数标识。默认情况下，数据库的最大数量是 16。

当您只从命令行运行`redis-cli`时，这会从数据库 0 开始。使用`-n`标志启动一个新的数据库，就像在`redis-cli -n 5`中一样。

### 允许的密钥类型

值得知道的一件事是，`redis-py`要求你传递给它的键是`bytes`、`str`、`int`或`float`。(在将它们发送到服务器之前，它会将最后 3 种类型转换为`bytes`。)

考虑这样一种情况，您希望使用日历日期作为键:

>>>

```py
>>> import datetime
>>> today = datetime.date.today()
>>> visitors = {"dan", "jon", "alex"}
>>> r.sadd(today, *visitors)
Traceback (most recent call last):
# ...
redis.exceptions.DataError: Invalid input of type: 'date'.
Convert to a byte, string or number first.
```

您需要显式地将 Python `date`对象转换成`str`，这可以通过`.isoformat()`来实现:

>>>

```py
>>> stoday = today.isoformat()  # Python 3.7+, or use str(today)
>>> stoday
'2019-03-10'
>>> r.sadd(stoday, *visitors)  # sadd: set-add
3
>>> r.smembers(stoday)
{b'dan', b'alex', b'jon'}
>>> r.scard(today.isoformat())
3
```

概括地说，Redis 本身只允许字符串作为键。`redis-py`在接受何种 Python 类型方面更自由一些，尽管它最终会在将数据发送到 Redis 服务器之前将其转换为字节。

[*Remove ads*](/account/join/)

### 例子:PyHats.com

是时候拿出一个更完整的例子了。让我们假设我们已经决定建立一个利润丰厚的网站，PyHats.com，向任何愿意购买的人出售价格高得离谱的帽子，并雇佣你来建立这个网站。

您将使用 Redis 来处理 PyHats.com 的一些产品目录、库存和 bot 流量检测。

今天是网站的第一天，我们将出售三顶限量版的帽子。每个 hat 保存在字段-值对的 Redis 散列中，该散列有一个作为前缀的随机整数的键，例如`hat:56854717`。使用`hat:`前缀是 Redis 在 Redis 数据库中创建一种[名称空间](https://realpython.com/python-namespaces-scope/)的惯例:

```py
import random

random.seed(444)
hats = {f"hat:{random.getrandbits(32)}": i for i in (
    {
        "color": "black",
        "price": 49.99,
        "style": "fitted",
        "quantity": 1000,
        "npurchased": 0,
    },
    {
        "color": "maroon",
        "price": 59.99,
        "style": "hipster",
        "quantity": 500,
        "npurchased": 0,
    },
    {
        "color": "green",
        "price": 99.99,
        "style": "baseball",
        "quantity": 200,
        "npurchased": 0,
    })
}
```

让我们从数据库`1`开始，因为我们在前面的例子中使用了数据库`0`:

>>>

```py
>>> r = redis.Redis(db=1)
```

要将这些数据初始写入 Redis，我们可以使用`.hmset()` (hash multi-set)，为每个字典调用它。“multi”是对设置多个字段-值对的引用，这里的“field”对应于`hats`中任何嵌套字典的一个键:

```py
 1>>> with r.pipeline() as pipe:
 2...    for h_id, hat in hats.items():
 3...        pipe.hmset(h_id, hat)
 4...    pipe.execute()
 5Pipeline<ConnectionPool<Connection<host=localhost,port=6379,db=1>>>
 6Pipeline<ConnectionPool<Connection<host=localhost,port=6379,db=1>>>
 7Pipeline<ConnectionPool<Connection<host=localhost,port=6379,db=1>>>
 8[True, True, True]
 9
10>>> r.bgsave()
11True
```

上面的代码块还引入了 Redis [**管道**](https://redis.io/topics/pipelining) 的概念，这是一种减少从 Redis 服务器读写数据所需的往返事务数量的方法。如果您刚刚调用了三次`r.hmset()`，那么这将需要对写入的每一行进行一次往返操作。

通过管道，所有的命令都在客户端进行缓冲，然后使用第 3 行中的`pipe.hmset()`一次性发送出去。这就是当您在第 4 行调用`pipe.execute()`时，三个`True`响应同时返回的原因。您将很快看到一个更高级的管道用例。

**注意**:Redis 文档提供了一个[的例子](https://redis.io/topics/mass-insert)用`redis-cli`做同样的事情，你可以通过管道把本地文件的内容进行批量插入。

让我们快速检查一下 Redis 数据库中的所有内容:

>>>

```py
>>> pprint(r.hgetall("hat:56854717"))
{b'color': b'green',
 b'npurchased': b'0',
 b'price': b'99.99',
 b'quantity': b'200',
 b'style': b'baseball'}

>>> r.keys()  # Careful on a big DB. keys() is O(N)
[b'56854717', b'1236154736', b'1326692461']
```

我们首先要模拟的是当用户点击*购买*时会发生什么。如果该物品有库存，则将其`npurchased`增加 1，并将其`quantity`(库存)减少 1。你可以使用`.hincrby()`来做到这一点:

>>>

```py
>>> r.hincrby("hat:56854717", "quantity", -1)
199
>>> r.hget("hat:56854717", "quantity")
b'199'
>>> r.hincrby("hat:56854717", "npurchased", 1)
1
```

**注意** : `HINCRBY`仍然对一个字符串哈希值进行操作，但是它试图将该字符串解释为一个以 10 为基数的 64 位有符号整数来执行操作。

这适用于与其他数据结构的递增和递减相关的其他命令，即`INCR`、`INCRBY`、`INCRBYFLOAT`、`ZINCRBY`和`HINCRBYFLOAT`。如果值处的字符串不能用整数表示，就会出现错误。

然而，事情并没有那么简单。在两行代码中更改`quantity`和`npurchased`隐藏了点击、购买和支付所包含的更多内容。我们需要多做一些检查，以确保我们不会给某人留下一个较轻的钱包和一顶帽子:

*   **步骤 1:** 检查商品是否有货，否则在后端引发异常。
*   **第二步:**如果有货，则执行交易，减少`quantity`字段，增加`npurchased`字段。
*   **第三步:**警惕前两步之间任何改变库存的变化(一个[竞争条件](https://realpython.com/python-concurrency/#threading-version))。

第 1 步相对简单:它包括一个`.hget()`来检查可用数量。

第二步稍微复杂一点。这对增加和减少操作需要被原子地执行**:要么两个都应该成功完成，要么都不应该(在至少一个失败的情况下)。*

*对于客户机-服务器框架，关注原子性并注意在多个客户机试图同时与服务器对话的情况下会出现什么问题总是至关重要的。Redis 对此的回答是使用一个 [**事务**](https://redis.io/topics/transactions) 块，这意味着要么两个命令都通过，要么都不通过。

在`redis-py`中，`Pipeline`默认是一个**事务管道**类。这意味着，即使这个类实际上是以别的东西命名的(管道)，它也可以用来创建一个事务块。

在 Redis 中，交易以`MULTI`开始，以`EXEC`结束:

```py
 1127.0.0.1:6379> MULTI
 2127.0.0.1:6379> HINCRBY 56854717 quantity -1
 3127.0.0.1:6379> HINCRBY 56854717 npurchased 1
 4127.0.0.1:6379> EXEC
```

`MULTI`(第 1 行)标志交易开始，`EXEC`(第 4 行)标志结束。两者之间的一切都作为一个全有或全无的缓冲命令序列来执行。这意味着不可能减少`quantity`(第 2 行)，但是平衡`npurchased`增加操作失败(第 3 行)。

让我们回到第 3 步:我们需要注意在前两步之间任何改变库存的变化。

第三步是最棘手的。假设我们的库存中只剩下一顶孤零零的帽子。在用户 A 检查剩余的帽子数量和实际处理他们的交易之间，用户 B 也检查库存，并且同样发现库存中列出了一顶帽子。两个用户都将被允许购买帽子，但我们有 1 顶帽子要卖，而不是 2 顶，所以我们陷入了困境，一个用户的钱用完了。不太好。

Redis 对步骤 3 中的困境有一个聪明的答案:它被称为[](https://en.wikipedia.org/wiki/Optimistic_concurrency_control)**，并且不同于典型的锁定在 RDBMS(如 PostgreSQL)中的工作方式。简而言之，乐观锁定意味着调用函数(客户端)不获取锁，而是在它持有锁的时间内监视它正在写入*的数据的变化。如果在此期间出现冲突，调用函数会再次尝试整个过程。**

*您可以通过使用`WATCH`命令(`redis-py`中的`.watch()`)来实现乐观锁定，该命令提供了一个 [**检查并设置**](https://redis.io/topics/transactions#cas) 行为。

让我们引入一大块代码，然后一步一步地浏览它。你可以想象当用户点击*立即购买*或*购买*按钮时`buyitem()`被调用。其目的是确认商品是否有货，并根据结果采取行动，所有这些都以安全的方式进行，即寻找竞争条件并在检测到竞争条件时重试:

```py
 1import logging
 2import redis
 3
 4logging.basicConfig()
 5
 6class OutOfStockError(Exception):
 7    """Raised when PyHats.com is all out of today's hottest hat"""
 8
 9def buyitem(r: redis.Redis, itemid: int) -> None:
10    with r.pipeline() as pipe:
11        error_count = 0
12        while True:
13            try:
14                # Get available inventory, watching for changes
15                # related to this itemid before the transaction
16                pipe.watch(itemid)
17                nleft: bytes = r.hget(itemid, "quantity")
18                if nleft > b"0":
19                    pipe.multi()
20                    pipe.hincrby(itemid, "quantity", -1)
21                    pipe.hincrby(itemid, "npurchased", 1)
22                    pipe.execute()
23                    break
24                else:
25                    # Stop watching the itemid and raise to break out
26                    pipe.unwatch()
27                    raise OutOfStockError(
28                        f"Sorry, {itemid} is out of stock!"
29                    )
30            except redis.WatchError:
31                # Log total num. of errors by this user to buy this item,
32                # then try the same process again of WATCH/HGET/MULTI/EXEC
33                error_count += 1
34                logging.warning(
35                    "WatchError #%d: %s; retrying",
36                    error_count, itemid
37                )
38    return None
```

关键行出现在第 16 行的`pipe.watch(itemid)`，它告诉 Redis 监控给定的`itemid`的值的任何变化。该程序通过调用第 17 行中的`r.hget(itemid, "quantity")`来检查库存:

```py
16pipe.watch(itemid)
17nleft: bytes = r.hget(itemid, "quantity")
18if nleft > b"0":
19    # Item in stock. Proceed with transaction.
```

如果在用户检查商品库存并试图购买它的这段短暂时间内，库存被触动，那么 Redis 将返回一个错误，`redis-py`将引发一个`WatchError`(第 30 行)。也就是说，如果在第 20 行和第 21 行的`.hget()`调用之后，但在后续的`.hincrby()`调用之前，`itemid`指向的任何散列发生了变化，那么我们将在`while True`循环的另一次迭代中重新运行整个过程。

这是锁定的“乐观”部分:我们没有让客户机通过获取和设置操作对数据库进行耗时的完全锁定，而是让 Redis 仅在需要重试库存检查的情况下通知客户机和用户。

这里的一个关键是理解**客户端**和**服务器端**操作之间的区别:

```py
nleft = r.hget(itemid, "quantity")
```

这个 Python 赋值带来了客户端`r.hget()`的结果。相反，您在`pipe`上调用的方法有效地将所有命令缓冲成一个，然后在一个请求中将它们发送给服务器:

```py
16pipe.multi()
17pipe.hincrby(itemid, "quantity", -1)
18pipe.hincrby(itemid, "npurchased", 1)
19pipe.execute()
```

在事务管道的中间，没有数据返回到客户端。您需要调用`.execute()`(第 19 行)来一次获得结果序列。

尽管这个块包含两个命令，但它只包含一个从客户端到服务器的往返操作。

这意味着客户端不能立即*使用第 20 行`pipe.hincrby(itemid, "quantity", -1)`的结果*，因为`Pipeline`上的方法返回的只是`pipe`实例本身。此时，我们还没有向服务器请求任何东西。虽然通常`.hincrby()`会返回结果值，但是在整个事务完成之前，您不能在客户端立即引用它。

这里有一个第 22 条军规:这也是为什么不能将对`.hget()`的调用放入事务块。如果您这样做了，那么您将无法知道是否要增加`npurchased`字段，因为您无法从插入到事务管道中的命令中获得实时结果。

最后，如果库存为零，那么我们`UNWATCH`商品 ID 并产生一个`OutOfStockError`(第 27 行)，最终显示令人垂涎的*售罄*页面，这将使我们的帽子购买者不顾一切地想以更奇怪的价格购买更多的帽子:

```py
24else:
25    # Stop watching the itemid and raise to break out
26    pipe.unwatch()
27    raise OutOfStockError(
28        f"Sorry, {itemid} is out of stock!"
29    )
```

这里有一个例子。请记住，我们的起始数量是 hat 56854717 的`199`，因为我们在上面调用了`.hincrby()`。让我们模拟 3 次购买，这将修改`quantity`和`npurchased`字段:

>>>

```py
>>> buyitem(r, "hat:56854717")
>>> buyitem(r, "hat:56854717")
>>> buyitem(r, "hat:56854717")
>>> r.hmget("hat:56854717", "quantity", "npurchased")  # Hash multi-get
[b'196', b'4']
```

现在，我们可以快进更多的购买，模拟一连串的购买，直到股票耗尽为零。同样，想象这些来自一大堆不同的客户端，而不仅仅是一个`Redis`实例:

>>>

```py
>>> # Buy remaining 196 hats for item 56854717 and deplete stock to 0
>>> for _ in range(196):
...     buyitem(r, "hat:56854717")
>>> r.hmget("hat:56854717", "quantity", "npurchased")
[b'0', b'200']
```

现在，当一些可怜的用户在游戏中迟到时，他们应该会遇到一个`OutOfStockError`,告诉我们的应用程序在前端呈现一个错误消息页面:

>>>

```py
>>> buyitem(r, "hat:56854717")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 20, in buyitem
__main__.OutOfStockError: Sorry, hat:56854717 is out of stock!
```

看来是时候进货了。

[*Remove ads*](/account/join/)

### 使用密钥到期

下面介绍一下 **key expiry** ，这是 Redis 中的另一个特色。当您的 [**密钥**](https://redis.io/commands/expire) 到期时，该密钥及其对应的值将在一定的秒数后或在某个时间戳自动从数据库中删除。

在`redis-py`中，您可以通过`.setex()`来实现这一点，它允许您设置一个基本的*字符串:带有有效期的字符串*键值对:

>>>

```py
 1>>> from datetime import timedelta
 2
 3>>> # setex: "SET" with expiration
 4>>> r.setex(
 5...     "runner",
 6...     timedelta(minutes=1),
 7...     value="now you see me, now you don't"
 8... )
 9True
```

您可以将第二个参数指定为一个以秒为单位的数字或一个`timedelta`对象，如上面的第 6 行所示。我喜欢后者，因为它看起来不那么暧昧，更刻意。

还有一些方法(当然还有相应的 Redis 命令)可以获得您设置为过期的密钥的剩余寿命(**生存时间**):

>>>

```py
>>> r.ttl("runner")  # "Time To Live", in seconds
58
>>> r.pttl("runner")  # Like ttl, but milliseconds
54368
```

下面，你可以加速窗口直到过期，然后看着密钥过期，之后`r.get()`会返回`None`，`.exists()`会返回`0`:

>>>

```py
>>> r.get("runner")  # Not expired yet
b"now you see me, now you don't"

>>> r.expire("runner", timedelta(seconds=3))  # Set new expire window
True
>>> # Pause for a few seconds
>>> r.get("runner")
>>> r.exists("runner")  # Key & value are both gone (expired)
0
```

下表总结了与键值过期相关的命令，包括上面提到的命令。解释直接取自`redis-py`方法[文档字符串](https://realpython.com/documenting-python-code/#documenting-your-python-code-base-using-docstrings):

| 签名 | 目的 |
| --- | --- |
| `r.setex(name, time, value)` | 将密钥`name`的值设置为在`time`秒后到期的`value`，其中`time`可以由一个`int`或一个 Python `timedelta`对象表示 |
| `r.psetex(name, time_ms, value)` | 将 key `name`的值设置为`value`，该值在`time_ms`毫秒后到期，其中`time_ms`可以由一个`int`或一个 Python `timedelta`对象表示 |
| `r.expire(name, time)` | 在键`name`上设置一个过期标志`time`秒，其中`time`可以用一个`int`或一个 Python `timedelta`对象来表示 |
| `r.expireat(name, when)` | 在键`name`上设置一个 expire 标志，其中`when`可以表示为一个表示 Unix 时间的`int`或一个 [Python `datetime`](https://realpython.com/python-datetime/) 对象 |
| `r.persist(name)` | 移除`name`的到期时间 |
| `r.pexpire(name, time)` | 在键`name`上设置一个过期标志`time`毫秒，`time`可以用一个`int`或一个 Python `timedelta`对象来表示 |
| `r.pexpireat(name, when)` | 在键`name`上设置一个过期标志，其中`when`可以表示为一个以毫秒为单位表示 Unix 时间的`int`(Unix time * 1000)或一个 Python `datetime`对象 |
| `r.pttl(name)` | 返回密钥`name`到期前的毫秒数 |
| `r.ttl(name)` | 返回密钥`name`到期前的秒数 |

### PyHats.com 第二部

首次亮相几天后，PyHats.com 吸引了如此多的炒作，以至于一些有事业心的用户正在创建机器人，在几秒钟内购买数百件商品，你已经决定这对你的帽子业务的长期健康不利。

现在您已经看到了如何使密钥过期，让我们在 PyHats.com 的后端使用它。

我们将创建一个新的 Redis 客户端，充当消费者(或观察者)并处理传入的 IP 地址流，这些地址可能来自到网站服务器的多个 HTTPS 连接。

观察器的目标是监视来自多个来源的 IP 地址流，留意在可疑的短时间内来自单个地址的大量请求。

网站服务器上的一些中间件用`.lpush()`将所有传入的 IP 地址推送到 Redis 列表中。这里有一个简单的方法来模仿一些入侵的 IP，使用一个新的 Redis 数据库:

>>>

```py
>>> r = redis.Redis(db=5)
>>> r.lpush("ips", "51.218.112.236")
1
>>> r.lpush("ips", "90.213.45.98")
2
>>> r.lpush("ips", "115.215.230.176")
3
>>> r.lpush("ips", "51.218.112.236")
4
```

可以看到，`.lpush()`在推送操作成功后返回列表的长度。每次调用`.lpush()`都会将 IP 放在 Redis 列表的开头，这个列表由字符串`"ips"`作为关键字。

在这个简化的模拟中，从技术上讲，请求都来自同一个客户机，但是您可以认为它们可能来自许多不同的客户机，并且都被推送到同一个 Redis 服务器上的同一个数据库。

现在，打开一个新的 shell 选项卡或窗口，启动一个新的 Python REPL。在这个 shell 中，您将创建一个新的客户端，它的用途与其他客户端完全不同，它位于一个无限的`while True`循环中，并在`ips`列表上执行一个阻塞的左弹出 [`BLPOP`](https://redis.io/commands/blpop) 调用，处理每个地址:

```py
 1# New shell window or tab
 2
 3import datetime
 4import ipaddress
 5
 6import redis
 7
 8# Where we put all the bad egg IP addresses
 9blacklist = set()
10MAXVISITS = 15
11
12ipwatcher = redis.Redis(db=5)
13
14while True:
15    _, addr = ipwatcher.blpop("ips")
16    addr = ipaddress.ip_address(addr.decode("utf-8"))
17    now = datetime.datetime.utcnow()
18    addrts = f"{addr}:{now.minute}"
19    n = ipwatcher.incrby(addrts, 1)
20    if n >= MAXVISITS:
21        print(f"Hat bot detected!: {addr}")
22        blacklist.add(addr)
23    else:
24        print(f"{now}:  saw {addr}")
25    _ = ipwatcher.expire(addrts, 60)
```

让我们看一下几个重要的概念。

`ipwatcher`就像一个[消费者](https://realpython.com/intro-to-python-threading/#producer-consumer-threading)，无所事事，等待新的 IP 被推上`"ips"` Redis 列表。它以`bytes`的形式接收它们，如 b“51 . 218 . 112 . 236”，并用 [`ipaddress`模块](https://realpython.com/python-ipaddress-module/)将它们变成更合适的[地址对象](https://docs.python.org/3/library/ipaddress.html#address-objects):

```py
15_, addr = ipwatcher.blpop("ips")
16addr = ipaddress.ip_address(addr.decode("utf-8"))
```

然后，使用地址和`ipwatcher`看到地址时的分钟形成 Redis 字符串键，将相应的计数增加`1`，并在此过程中获得新的计数:

```py
17now = datetime.datetime.utcnow()
18addrts = f"{addr}:{now.minute}"
19n = ipwatcher.incrby(addrts, 1)
```

如果这个地址被浏览的次数超过了`MAXVISITS`，那么看起来就好像我们手上有一个 PyHats.com 的网页抓取器试图创造下一个[郁金香泡沫](https://en.wikipedia.org/wiki/Tulip_mania)。唉，我们别无选择，只能给这个用户返回类似可怕的 403 状态码的东西。

我们使用`ipwatcher.expire(addrts, 60)`来终止*(地址分钟)*组合，从它最后一次被看到起 60 秒。这是为了防止我们的数据库被陈旧的一次性页面查看器堵塞。

如果您在新的 shell 中执行这个代码块，您应该会立即看到以下输出:

```py
2019-03-11 15:10:41.489214:  saw 51.218.112.236
2019-03-11 15:10:41.490298:  saw 115.215.230.176
2019-03-11 15:10:41.490839:  saw 90.213.45.98
2019-03-11 15:10:41.491387:  saw 51.218.112.236
```

输出立即出现，因为这四个 IP 位于由`"ips"`键入的队列式列表中，等待由我们的`ipwatcher`取出。使用`.blpop()`(或`BLPOP`命令)将阻塞，直到列表中有一个项目可用，然后弹出它。它的行为类似于 Python 的 [`Queue.get()`](https://docs.python.org/3/library/queue.html#queue.Queue.get) ，也是阻塞直到一个项目可用。

除了提供 IP 地址，我们的`ipwatcher`还有第二份工作。对于一个小时中给定的一分钟(第 1 分钟到第 60 分钟)，`ipwatcher`会将一个 IP 地址分类为 hat-bot，如果它在该分钟内发送了 15 个或更多的`GET`请求。

切换回您的第一个 shell，模拟一个页面抓取器，在几毫秒内用 20 个请求将站点炸开:

```py
for _ in range(20):
    r.lpush("ips", "104.174.118.18")
```

最后，切换回包含`ipwatcher`的第二个 shell，您应该会看到如下输出:

```py
2019-03-11 15:15:43.041363:  saw 104.174.118.18
2019-03-11 15:15:43.042027:  saw 104.174.118.18
2019-03-11 15:15:43.042598:  saw 104.174.118.18
2019-03-11 15:15:43.043143:  saw 104.174.118.18
2019-03-11 15:15:43.043725:  saw 104.174.118.18
2019-03-11 15:15:43.044244:  saw 104.174.118.18
2019-03-11 15:15:43.044760:  saw 104.174.118.18
2019-03-11 15:15:43.045288:  saw 104.174.118.18
2019-03-11 15:15:43.045806:  saw 104.174.118.18
2019-03-11 15:15:43.046318:  saw 104.174.118.18
2019-03-11 15:15:43.046829:  saw 104.174.118.18
2019-03-11 15:15:43.047392:  saw 104.174.118.18
2019-03-11 15:15:43.047966:  saw 104.174.118.18
2019-03-11 15:15:43.048479:  saw 104.174.118.18
Hat bot detected!:  104.174.118.18
Hat bot detected!:  104.174.118.18
Hat bot detected!:  104.174.118.18
Hat bot detected!:  104.174.118.18
Hat bot detected!:  104.174.118.18
Hat bot detected!:  104.174.118.18
```

现在， `Ctrl` + `C` 退出`while True`循环，您会看到该违规 IP 已被添加到您的黑名单中:

>>>

```py
>>> blacklist
{IPv4Address('104.174.118.18')}
```

你能发现这个检测系统的缺陷吗？过滤器检查分钟为`.minute`而不是*最后 60 秒*(一个滚动分钟)。实现滚动检查来监控用户在过去 60 秒内被查看了多少次将会更加棘手。有一个巧妙的解决方案，在 [ClassDojo](https://engineering.classdojo.com/blog/2015/02/06/rolling-rate-limiter/) 使用 Redis 的排序集合。Josiah Carlson 的 [*Redis in Action*](https://realpython.com/asins/1617290858/) 还使用 IP-to-location 缓存表给出了这一部分的一个更详细的通用示例。

[*Remove ads*](/account/join/)

### 持久性和快照

Redis 的读写速度如此之快的原因之一是数据库保存在服务器的内存(RAM)中。然而，Redis 数据库也可以在一个叫做[快照](https://redis.io/topics/persistence#snapshotting)的过程中被存储(持久化)到磁盘。这背后的要点是以二进制格式保存物理备份，以便在需要时(比如在服务器启动时)可以重建数据并将其放回内存。

当您在本教程开始时使用`save`选项设置基本配置时，您已经在不知情的情况下启用了快照:

```py
# /etc/redis/6379.conf

port              6379
daemonize         yes
save              60 1 bind              127.0.0.1
tcp-keepalive     300
dbfilename        dump.rdb
dir               ./
rdbcompression    yes
```

格式为`save <seconds> <changes>`。这告诉 Redis，如果发生了给定秒数和数量的数据库写操作，就将数据库保存到磁盘。在这种情况下，我们告诉 Redis 每 60 秒将数据库保存到磁盘，如果在这 60 秒内至少发生了一次修改写操作。相对于[示例 Redis 配置文件](http://download.redis.io/redis-stable/redis.conf)，这是一个相当激进的设置，它使用以下三个`save`指令:

```py
# Default redis/redis.conf
save 900 1
save 300 10
save 60 10000
```

**RDB 快照**是数据库的完整(而非增量)时间点捕获。(RDB 指的是 Redis 数据库文件。)我们还指定了写入的结果数据文件的目录和文件名:

```py
# /etc/redis/6379.conf

port              6379
daemonize         yes
save              60 1
bind              127.0.0.1
tcp-keepalive     300
dbfilename        dump.rdb dir               ./ rdbcompression    yes
```

这将指示 Redis 保存到一个名为`dump.rdb`的二进制数据文件中，该文件位于执行`redis-server`的当前工作目录下:

```py
$ file -b dump.rdb
data
```

您也可以使用 Redis 命令 [`BGSAVE`](https://redis.io/commands/bgsave) 手动调用保存:

```py
127.0.0.1:6379> BGSAVE
Background saving started
```

`BGSAVE`中的“BG”表示保存在后台进行。该选项在`redis-py`方法中也可用:

>>>

```py
>>> r.lastsave()  # Redis command: LASTSAVE
datetime.datetime(2019, 3, 10, 21, 56, 50)
>>> r.bgsave()
True
>>> r.lastsave()
datetime.datetime(2019, 3, 10, 22, 4, 2)
```

这个例子介绍了另一个新的命令和方法`.lastsave()`。在 Redis 中，它返回最后一次 DB 保存的 Unix 时间戳，Python 将其作为一个`datetime`对象返回给您。上面，你可以看到`r.lastsave()`结果由于`r.bgsave()`而改变。

如果使用`save`配置选项启用自动快照，则`r.lastsave()`也会改变。

换句话说，有两种方法可以启用快照:

1.  显式地，通过 Redis 命令`BGSAVE`或`redis-py`方法`.bgsave()`
2.  隐式地，通过`save`配置选项(也可以在`redis-py`中用`.config_set()`设置)

RDB 快照的速度很快，因为父进程使用 [`fork()`](http://man7.org/linux/man-pages/man2/fork.2.html) 系统调用将耗时的磁盘写入任务传递给子进程，以便父进程可以继续执行。这就是`BGSAVE`中的*背景*所指的。

还有[`SAVE`](https://redis.io/commands/save)(`redis-py`中的`.save()`)，但是这是同步(阻塞)保存而不是使用`fork()`，所以没有特定的原因你不应该使用它。

尽管`.bgsave()`发生在后台，但这也不是没有代价的。如果 Redis 数据库首先足够大，那么`fork()`本身发生的时间实际上可能相当长。

如果这是一个问题，或者如果您不能因为 RDB 快照的周期性而丢失哪怕一丁点数据，那么您应该研究一下作为快照替代方案的[仅附加文件](https://redis.io/topics/persistence#append-only-file) (AOF)策略。AOF 将 Redis 命令实时复制到磁盘，允许您通过重放这些命令来进行基于命令的重建。

[*Remove ads*](/account/join/)

### 序列化变通办法

让我们回到谈论 Redis 数据结构。借助其散列数据结构，Redis 实际上支持一级嵌套:

```py
127.0.0.1:6379> hset mykey field1 value1
```

Python 客户端的等效内容如下所示:

```py
r.hset("mykey", "field1", "value1")
```

在这里，您可以将`"field1": "value1"`视为 Python 字典`{"field1": "value1"}`的键值对，而`mykey`是顶级键:

| 重复命令 | 纯 Python 等价物 |
| --- | --- |
| `r.set("key", "value")` | `r = {"key": "value"}` |
| `r.hset("key", "field", "value")` | `r = {"key": {"field": "value"}}` |

但是，如果您希望这个字典的值(Redis hash)包含字符串以外的内容，比如以字符串为值的`list`或嵌套字典，该怎么办呢？

这里有一个例子，使用一些类似于 [JSON](https://realpython.com/python-json/) 的数据来使区别更加清晰:

```py
restaurant_484272 = {
    "name": "Ravagh",
    "type": "Persian",
    "address": {
        "street": {
            "line1": "11 E 30th St",
            "line2": "APT 1",
        },
        "city": "New York",
        "state": "NY",
        "zip": 10016,
    }
}
```

假设我们想要设置一个 Redis 散列，其中的键`484272`和字段-值对对应于来自`restaurant_484272`的键-值对。Redis 不直接支持这个，因为`restaurant_484272`是嵌套的:

>>>

```py
>>> r.hmset(484272, restaurant_484272)
Traceback (most recent call last):
# ...
redis.exceptions.DataError: Invalid input of type: 'dict'.
Convert to a byte, string or number first.
```

事实上，你可以用 Redis 来实现这一点。在`redis-py`和 Redis 中有两种不同的模拟嵌套数据的方法:

1.  用类似`json.dumps()`的代码将值序列化成一个字符串
2.  在键字符串中使用分隔符来模拟值中的嵌套

让我们来看一个例子。

**选项 1:将值序列化为字符串**

您可以使用`json.dumps()`将`dict`序列化为 JSON 格式的字符串:

>>>

```py
>>> import json
>>> r.set(484272, json.dumps(restaurant_484272))
True
```

如果调用`.get()`，得到的值将是一个`bytes`对象，所以不要忘了反序列化它以得到原来的对象。`json.dumps()`和`json.loads()`互为反码，分别用于序列化和反序列化数据:

>>>

```py
>>> from pprint import pprint
>>> pprint(json.loads(r.get(484272)))
{'address': {'city': 'New York',
 'state': 'NY',
 'street': '11 E 30th St',
 'zip': 10016},
 'name': 'Ravagh',
 'type': 'Persian'}
```

这适用于任何序列化协议，另一个常见的选择是 [`yaml`](https://github.com/yaml/pyyaml) :

>>>

```py
>>> import yaml  # python -m pip install PyYAML
>>> yaml.dump(restaurant_484272)
'address: {city: New York, state: NY, street: 11 E 30th St, zip: 10016}\nname: Ravagh\ntype: Persian\n'
```

无论您选择使用哪种序列化协议，概念都是相同的:您获取一个 Python 特有的对象，并将其转换为可跨多种语言识别和交换的字节串。

**选项 2:在关键字串中使用分隔符**

还有第二种选择，通过在 Python `dict`中串联多层键来模仿“嵌套”。这包括通过[递归](https://realpython.com/python-thinking-recursively/)来展平嵌套字典，这样每个键都是一个串联的键串，并且值是原始字典中嵌套最深的值。考虑我们的字典对象`restaurant_484272`:

```py
restaurant_484272 = {
    "name": "Ravagh",
    "type": "Persian",
    "address": {
        "street": {
            "line1": "11 E 30th St",
            "line2": "APT 1",
        },
        "city": "New York",
        "state": "NY",
        "zip": 10016,
    }
}
```

我们想把它做成这样的形式:

```py
{
    "484272:name":                     "Ravagh",
    "484272:type":                     "Persian",
    "484272:address:street:line1":     "11 E 30th St",
    "484272:address:street:line2":     "APT 1",
    "484272:address:city":             "New York",
    "484272:address:state":            "NY",
    "484272:address:zip":              "10016",
}
```

这就是下面的`setflat_skeys()`所做的，增加的特性是它在`Redis`实例本身上执行`.set()`操作，而不是返回输入字典的副本:

```py
 1from collections.abc import MutableMapping
 2
 3def setflat_skeys(
 4    r: redis.Redis,
 5    obj: dict,
 6    prefix: str,
 7    delim: str = ":",
 8    *,
 9    _autopfix=""
10) -> None:
11    """Flatten `obj` and set resulting field-value pairs into `r`.
12
13 Calls `.set()` to write to Redis instance inplace and returns None.
14
15 `prefix` is an optional str that prefixes all keys.
16 `delim` is the delimiter that separates the joined, flattened keys.
17 `_autopfix` is used in recursive calls to created de-nested keys.
18
19 The deepest-nested keys must be str, bytes, float, or int.
20 Otherwise a TypeError is raised.
21 """
22    allowed_vtypes = (str, bytes, float, int)
23    for key, value in obj.items():
24        key = _autopfix + key
25        if isinstance(value, allowed_vtypes):
26            r.set(f"{prefix}{delim}{key}", value)
27        elif isinstance(value, MutableMapping):
28            setflat_skeys(
29                r, value, prefix, delim, _autopfix=f"{key}{delim}"
30            )
31        else:
32            raise TypeError(f"Unsupported value type: {type(value)}")
```

该函数遍历`obj`的键-值对，首先检查值的类型(第 25 行),看它是否应该停止进一步递归并设置该键-值对。否则，如果值看起来像一个`dict`(第 27 行)，那么它递归到那个映射中，添加以前看到的键作为键前缀(第 28 行)。

让我们看看它是如何工作的:

>>>

```py
>>> r.flushdb()  # Flush database: clear old entries
>>> setflat_skeys(r, restaurant_484272, 484272)

>>> for key in sorted(r.keys("484272*")):  # Filter to this pattern
...     print(f"{repr(key):35}{repr(r.get(key)):15}")
...
b'484272:address:city'             b'New York'
b'484272:address:state'            b'NY'
b'484272:address:street:line1'     b'11 E 30th St'
b'484272:address:street:line2'     b'APT 1'
b'484272:address:zip'              b'10016'
b'484272:name'                     b'Ravagh'
b'484272:type'                     b'Persian'

>>> r.get("484272:address:street:line1")
b'11 E 30th St'
```

上面的最后一个循环使用了`r.keys("484272*")`，其中`"484272*"`被解释为一个模式，匹配数据库中所有以`"484272"`开头的键。

还要注意`setflat_skeys()`如何只调用`.set()`而不是`.hset()`，因为我们正在使用普通的*字符串:字符串*字段-值对，并且 484272 ID 键被添加到每个字段字符串的前面。

[*Remove ads*](/account/join/)

### 加密

另一个帮助你晚上睡得好的技巧是在发送任何东西到 Redis 服务器之前添加对称加密。把这看作是安全性的一个附加组件，您应该通过在您的 [Redis 配置](#configuring-redis)中设置适当的值来确保安全性。下面的例子使用了 [`cryptography`](https://github.com/pyca/cryptography/) 包:

```py
$ python -m pip install cryptography
```

举例来说，假设您有一些敏感的持卡人数据(CD ),无论如何，您都不希望这些数据以明文形式存放在任何服务器上。在 Redis 中缓存它之前，您可以序列化数据，然后使用 [Fernet](https://cryptography.io/en/latest/fernet/) 对序列化的字符串进行加密:

>>>

```py
>>> import json
>>> from cryptography.fernet import Fernet

>>> cipher = Fernet(Fernet.generate_key())
>>> info = {
...     "cardnum": 2211849528391929,
...     "exp": [2020, 9],
...     "cv2": 842,
... }

>>> r.set(
...     "user:1000",
...     cipher.encrypt(json.dumps(info).encode("utf-8"))
... )

>>> r.get("user:1000")
b'gAAAAABcg8-LfQw9TeFZ1eXbi'  # ... [truncated]

>>> cipher.decrypt(r.get("user:1000"))
b'{"cardnum": 2211849528391929, "exp": [2020, 9], "cv2": 842}'

>>> json.loads(cipher.decrypt(r.get("user:1000")))
{'cardnum': 2211849528391929, 'exp': [2020, 9], 'cv2': 842}
```

因为`info`包含的值是一个`list`，您需要将它序列化成 Redis 可以接受的字符串。(您可以使用`json`、`yaml`或任何其他序列化方式来实现这个目的。)接下来，使用`cipher`对象加密和解密该字符串。您需要使用`json.loads()`对解密的字节进行反序列化，这样您就可以将结果恢复为初始输入的类型，即`dict`。

**注** : [Fernet](https://github.com/fernet/spec/blob/master/Spec.md#token-format) 在 CBC 模式下使用 AES 128 加密。有关使用 AES 256 的示例，请参见 [`cryptography`文档](https://cryptography.io/en/latest/hazmat/primitives/symmetric-encryption/)。无论您选择做什么，都使用`cryptography`，而不是`pycrypto`(作为`Crypto`导入)，后者不再被主动维护。

如果安全性至关重要，那么在字符串通过网络连接之前对其进行加密绝对不是一个坏主意。

### 压缩

最后一个快速优化是压缩。如果带宽是一个问题，或者您对成本很敏感，那么当您从 Redis 发送和接收数据时，您可以实现无损压缩和解压缩方案。下面是一个使用 bzip2 压缩算法的示例，在这种极端情况下，该算法将通过连接发送的字节数减少了 2000 多倍:

>>>

```py
 1>>> import bz2
 2
 3>>> blob = "i have a lot to talk about" * 10000
 4>>> len(blob.encode("utf-8"))
 5260000
 6
 7>>> # Set the compressed string as value
 8>>> r.set("msg:500", bz2.compress(blob.encode("utf-8")))
 9>>> r.get("msg:500")
10b'BZh91AY&SY\xdaM\x1eu\x01\x11o\x91\x80@\x002l\x87\'  # ... [truncated]
11>>> len(r.get("msg:500"))
12122
13>>> 260_000 / 122  # Magnitude of savings
142131.1475409836066
15
16>>> # Get and decompress the value, then confirm it's equal to the original
17>>> rblob = bz2.decompress(r.get("msg:500")).decode("utf-8")
18>>> rblob == blob
19True
```

序列化、加密和压缩在这里的关联方式是它们都发生在客户端。您在客户端对原始对象进行一些操作，一旦您将字符串发送到服务器，这些操作最终会更有效地利用 Redis。当您请求最初发送给服务器的内容时，客户端会再次执行相反的操作。

## 使用 Hiredis

对于像`redis-py`这样的客户端库来说，遵循**协议**来构建它是很常见的。在这种情况下，`redis-py`实现了 [REdis 序列化协议](https://redis.io/topics/protocol)，即 RESP。

实现该协议的一部分包括转换原始字节串中的一些 Python 对象，将其发送到 Redis 服务器，并将响应解析回可理解的 Python 对象。

例如，字符串响应“OK”将作为`"+OK\r\n"`返回，而整数响应 1000 将作为`":1000\r\n"`返回。对于其他数据类型，如 [RESP 数组](https://redis.io/topics/protocol#resp-arrays)，这可能会变得更加复杂。

一个**解析器**是请求-响应循环中的一个工具，它解释这个原始响应并把它加工成客户机可识别的东西。`redis-py`自带解析器类`PythonParser`，它用纯 Python 进行解析。(见 [`.read_response()`](https://github.com/andymccurdy/redis-py/blob/cfa2bc9/redis/connection.py#L289) 如果你好奇的话。)

然而，还有一个 C 库， [Hiredis](https://github.com/redis/hiredis) ，它包含一个快速解析器，可以为一些 redis 命令提供显著的加速，比如`LRANGE`。你可以把 Hiredis 看作是一个可选的加速器，在特殊情况下使用它没有坏处。

要使`redis-py`能够使用 Hiredis 解析器，您所要做的就是在与`redis-py`相同的环境中安装 Python 绑定:

```py
$ python -m pip install hiredis
```

你在这里实际安装的是 [`hiredis-py`](https://github.com/redis/hiredis-py) ，它是 [`hiredis`](https://github.com/redis/hiredis) C 库的一部分的 Python 包装器。

好的一面是，你真的不需要亲自打电话给`hiredis`。只要`pip install`它，这将让`redis-py`看到它是可用的，并使用它的`HiredisParser`而不是`PythonParser`。

在内部，`redis-py`将尝试[导入](https://realpython.com/python-import/) `hiredis`，并使用一个`HiredisParser`类来匹配它，但将回退到它的`PythonParser`，这在某些情况下可能会慢一些:

```py
# redis/utils.py
try:
    import hiredis
    HIREDIS_AVAILABLE = True
except ImportError:
    HIREDIS_AVAILABLE = False

# redis/connection.py
if HIREDIS_AVAILABLE:
    DefaultParser = HiredisParser
else:
    DefaultParser = PythonParser
```

[*Remove ads*](/account/join/)

## 使用企业 Redis 应用程序

虽然 Redis 本身是开源的和免费的，但一些托管服务已经出现，它们提供以 Redis 为核心的数据存储，并在开源的 Redis 服务器上构建一些附加功能:

*   [**Amazon elastic cache for Redis**](https://docs.aws.amazon.com/AmazonElastiCache/latest/red-ug/WhatIs.html)**:**这是一个 web 服务，它让您在云中托管 Redis 服务器，您可以从 Amazon EC2 实例连接到该服务器。关于完整的设置说明，你可以浏览亚马逊的[elastic cache for Redis](https://docs.aws.amazon.com/AmazonElastiCache/latest/red-ug/GettingStarted.CreateCluster.html)启动页面。

*   [**微软的 Azure Cache for Redis**](https://azure.microsoft.com/en-us/services/cache/)**:**这是另一项强大的企业级服务，让您可以在云中建立一个可定制的、安全的 Redis 实例。

两者的设计有一些共性。您通常为您的缓存指定一个自定义名称，该名称作为 DNS 名称的一部分嵌入，例如`demo.abcdef.xz.0009.use1.cache.amazonaws.com` (AWS)或`demo.redis.cache.windows.net` (Azure)。

设置完成后，这里有一些关于如何连接的快速提示。

从命令行来看，这与我们前面的例子基本相同，但是您需要用`h`标志指定一个主机，而不是使用默认的 localhost。对于 **Amazon AWS** ，从您的实例 shell 执行以下命令:

```py
$ export REDIS_ENDPOINT="demo.abcdef.xz.0009.use1.cache.amazonaws.com"
$ redis-cli -h $REDIS_ENDPOINT
```

对于**微软 Azure** ，可以使用类似的调用。Azure Cache for Redis [默认使用 SSL](https://docs.microsoft.com/en-us/azure/azure-cache-for-redis/cache-how-to-redis-cli-tool) (端口 6380)而不是端口 6379，允许进出 Redis 的加密通信，TCP 就不能这么说了。除此之外，您需要提供的只是一个非默认端口和访问密钥:

```py
$ export REDIS_ENDPOINT="demo.redis.cache.windows.net"
$ redis-cli -h $REDIS_ENDPOINT -p 6380 -a <primary-access-key>
```

`-h`标志指定了一个主机，如您所见，默认情况下是`127.0.0.1` (localhost)。

当你在 Python 中使用`redis-py`时，最好不要在 Python 脚本中使用敏感变量，并且要小心你对这些文件的读写权限。Python 版本如下所示:

>>>

```py
>>> import os
>>> import redis

>>> # Specify a DNS endpoint instead of the default localhost
>>> os.environ["REDIS_ENDPOINT"]
'demo.abcdef.xz.0009.use1.cache.amazonaws.com'
>>> r = redis.Redis(host=os.environ["REDIS_ENDPOINT"])
```

这就是全部了。除了指定不同的`host`，您现在可以像平常一样调用命令相关的方法，比如`r.get()`。

**注意**:如果你想单独使用`redis-py`和 AWS 或 Azure Redis 实例的组合，那么你真的不需要在你的机器上本地安装和制作 Redis 本身，因为你既不需要`redis-cli`也不需要`redis-server`。

如果你正在部署一个中型到大型的生产应用程序，Redis 在其中起着关键作用，那么使用 AWS 或 Azure 的服务解决方案可能是一种可扩展的、经济高效的、有安全意识的操作方式。

## 总结

这就结束了我们通过 Python 访问 Redis 的旋风之旅，包括安装和使用连接到 Redis 服务器的 Redis REPL，以及在实际例子中使用`redis-py`。以下是你学到的一些东西:

*   通过直观的 Python API，您可以(几乎)完成使用 Redis CLI 所能完成的一切。
*   掌握持久性、序列化、加密和压缩等主题可以让您充分发挥 Redis 的潜力。
*   在更复杂的情况下，Redis 事务和管道是库的基本部分。
*   企业级 Redis 服务可以帮助您在生产中顺利使用 Redis。

Redis 有一系列广泛的特性，其中一些我们在这里没有真正涉及到，包括[服务器端 Lua 脚本](https://redis.io/commands/eval)、[分片](https://redis.io/topics/partitioning)和[主从复制](https://redis.io/topics/replication)。如果你认为 Redis 是你的拿手好戏，那么请确保关注它的发展，因为它实现了一个更新的协议。

## 延伸阅读

这里有一些资源，您可以查看以了解更多信息。

书籍:

*   **西亚卡尔森:** [*雷迪斯在行动*](https://realpython.com/asins/1617290858/)
*   **卡尔如下:** [*小背书*](https://www.openmymind.net/2012/1/23/The-Little-Redis-Book/)
*   吕克·帕金斯等人。艾尔。: [*七周七个数据库*](https://realpython.com/asins/1680502530/)

正在使用的重定向:

*   **Twitter:**[Twitter 上的实时交付架构](https://www.infoq.com/presentations/Real-Time-Delivery-Twitter)
*   **Spool:** [Redis 位图——快速、简单、实时的指标](https://blog.getspool.com/2011/11/29/fast-easy-realtime-metrics-using-redis-bitmaps/)
*   **3scale:** [享受亚马逊和 Rackspace 之间 Redis 复制的乐趣](http://tech.3scale.net/2012/07/25/fun-with-redis-replication)
*   **Instagram:** [在 Redis 中存储上亿个简单的键值对](https://instagram-engineering.com/storing-hundreds-of-millions-of-simple-key-value-pairs-in-redis-1091ae80f74c)
*   **Craigslist:**[Redis sharing at Craigslist](https://blog.zawodny.com/2011/02/26/redis-sharding-at-craigslist/)
*   **圆盘:t1】T2【圆盘上的铆钉】**

其他:

*   **数字海洋:** [如何保护你的 Redis 安装](https://www.digitalocean.com/community/tutorials/how-to-install-and-secure-redis-on-ubuntu-18-04)
*   **AWS:**[riz 用户指南的弹性缓存](https://docs.aws.amazon.com/AmazonElastiCache/latest/red-ug/WhatIs.html)
*   **微软:** [用于 Redis 的 Azure 缓存](https://azure.microsoft.com/en-us/services/cache/)
*   **速查表:** [速查表](https://www.cheatography.com/tasjaevan/cheat-sheets/redis/)
*   **ClassDojo:** [用 Redis 排序集进行更好的速率限制](https://engineering.classdojo.com/blog/2015/02/06/rolling-rate-limiter/)
*   **抗雷(Salvatore sanfilippo):**[【redis 坚持不懈】](http://oldblog.antirez.com/post/redis-persistence-demystified.html)
*   **马丁·克莱普曼:** [如何做分布式锁定](https://martin.kleppmann.com/2016/02/08/how-to-do-distributed-locking.html)
*   **高可扩展性:**[Redis 解决的 11 个常见 Web 用例](http://highscalability.com/blog/2011/7/6/11-common-web-use-cases-solved-in-redis.html)**************
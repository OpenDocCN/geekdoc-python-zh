# 使用 Boto 在 AWS 上介绍 Python

> 原文：<https://www.pythoncentral.io/introduction-to-python-on-aws-with-boto/>

亚马逊网络服务为我们提供了廉价、便捷的云计算基础设施。那么我们如何在上面运行 Python 呢？

## 在 Amazon EC2 上设置 Python

EC2 是亚马逊的弹性计算云。它是用于在 AWS 上创建和操作虚拟机的服务。您可以使用 SSH 与这些机器进行交互，但是使用设置为 web 应用程序的 IPython HTML Notebook 要好得多。

您可以手动设置 IPython 笔记本服务器，但是有几个更简单的选项。

*   [NotebookCloud](https://notebookcloud.appspot.com/docs "Notebook Cloud") 是一个简单的 web 应用程序，使您能够从浏览器创建 IPython 笔记本服务器。它真的很容易使用，而且免费。
*   来自麻省理工学院的 StarCluster 是一个与亚马逊合作的更强大的库，它使用配置文件来简化创建、复制和共享云配置。它支持 IPython 开箱即用，并且在线提供了额外的配置文件。

两个选项都是开源的。

您不需要使用 IPython 或 Amazon EC2 来使用 AWS，但是这样做有很多好处。

无论您是在 EC2 上运行您的机器，还是只想使用普通机器上的一些服务，您通常都需要一种方法让您的程序与 AWS 服务器进行对话。

# Python Boto 库

AWS 有一个广泛的 API，允许你编程访问每一个服务。有很多库可以使用这个 API，对于 Python，我们有 boto。

Boto 为几乎所有的亚马逊网络服务以及其他一些服务提供了一个 Python 接口，比如 Google Storage。Boto 是成熟的、有据可查的、易于使用的。

要使用 Boto，您需要提供您的 AWS 凭证，特别是您的访问密钥和秘密密钥。这些可以在每次连接时手动提供，但是将它们添加到 boto 配置文件更容易，这样 boto 就可以自动提供密钥。

如果您希望为您的 boto 设置使用一个配置，您需要在~/.boto 下创建一个文件。如果您希望在整个系统范围内使用这个配置，您应该在/etc/boto.cfg 下创建这个文件。ini 格式，至少应该包含一个凭据部分，如下所示:

 `[Credentials]
aws_access_key_id = <your access key>
aws_secret_access_key = <your secret key>` 

您可以通过创建连接对象来使用 boto，这些对象表示到服务的连接，然后与这些连接对象进行交互。

```py

from boto.ec2 import EC2Connection

conn = EC2Connection()

```

注意，如果没有在配置文件中设置 AWS 键，您需要将它们传递给任何连接构造函数。

```py

conn = EC2Connection(access_key, secret_key)

```

## 创建您的第一台 Amazon EC2 机器

现在，您有了一个连接，您可以使用它来创建一个新机器。您首先需要创建一个安全组，允许您访问您在该组中创建的任何机器。

```py

group_name  = 'python_central'

description = 'Python Central: Test Security Group.'
group = conn . create _ security _ group(
group _ name，description 
)
group.authorize('tcp '，8888，8888，'[0 . 0 . 0 . 0/0【T1 ')](http://0.0.0.0/0) 
```

现在您有了一个组，您可以使用它创建一个虚拟机。为此，您需要一个 AMI，一个 Amazon 机器映像，这是一个基于云的软件发行版，您的机器将使用它作为操作系统和堆栈。我们将使用 NotebookCloud 的 AMI，因为它是可用的，并且已经用一些 Python 好东西进行了设置。

我们需要一些随机数据来为该服务器创建自签名证书，这样我们就可以使用 HTTPS 来访问它。

```py

import random

from string import ascii_lowercase as letters
#以正确的格式创建随机数据
 data = random.choice(('UK '，' US')) 
对于范围(4)中的 a:
data+= ' | '
对于范围(8)中的 b:
data+= random . choice(letters)
```

我们还需要创建一个散列密码来登录服务器。

```py

import hashlib
#您选择的密码放在这里
 password = 'password '
h = hashlib . new(' sha1 ')
salt =(' % 0 '+str(12)+' x ')% random . getrandbits(48)
h . update(password+salt)
密码= ':'。join(('sha1 '，salt，h.hexdigest()))
```

现在，我们将散列密码添加到数据字符串的末尾。当我们创建一个新的虚拟机时，我们将把这个数据字符串传递给 AWS。机器将使用字符串中的数据创建一个自签名证书和一个包含您的散列密码的配置文件。

```py

data += '|' + password

```

现在，您可以创建服务器了。

```py

# NotebookCloud AMI

AMI = 'ami-affe51c6'
conn.run_instances( 
 AMI，
 instance_type = 't1.micro '，
security _ groups =[' python _ central ']，
 user_data = data，
 max_count = 1 
 ) 

```

要在线查找服务器，您需要一个 URL。你的服务器需要一两分钟的时间来启动，所以这是一个休息和烧水的好时机。

要获得 URL，我们只需轮询 AWS，看看服务器是否已经分配了公共 DNS 名称。以下代码假设您刚刚创建的实例是您的 AWS 帐户上的唯一实例。

```py

import time
while True:
inst =[
I for r in conn . get _ all _ instances()
for I in r . instances
][0]
dns = inst。__dict__['公共域名']
如果 dns: 
 #我们希望这个实例 id 用于以后的
instance _ id = I . _ _ dict _ _[' id ']
中断
time.sleep(5) 

```

现在，将 dns 名称转换成正确的 URL，并让浏览器指向它。

```py

print('https://{}:8888'.format(dns))

```

现在，您应该自豪地拥有了一台全新的 IPython HTML 笔记本服务器。请记住，您需要上面提供的密码才能登录。

如果您想终止实例，您可以使用下面的代码行轻松完成。请注意，要做到这一点，您需要实例 id。如果遇到问题，您可以随时访问 AWS 控制台，并在那里控制您的所有实例。

```py

conn.terminate_instances(instance_ids=[instance_id])

```

玩得开心！
# 系统管理

## Fabric

[Fabric](http://docs.fabfile.org) [http://docs.fabfile.org] 是一个简化系统管理任务的库。Chef 和 Puppet 倾向于关注管理服务器和系统库，而 Fabric 更加关注应用级别的任务，比如说部署。

安装 Fabric:

```py
$ pip install fabric 
```

下面的代码将会创建我们可以使用的两个任务： `memory_usage` 和 `deploy`。前者将会在每台机器上输出内存使用情况。后者将会 ssh 到每台服务器，cd 到我们的工程目录，激活虚拟环境，拉取最新的代码库，以及重启应用服务器。

```py
from fabric.api import cd, env, prefix, run, task

env.hosts = ['my_server1', 'my_server2']

@task
def memory_usage():
    run('free -m')

@task
def deploy():
    with cd('/var/www/project-env/project'):
        with prefix('. ../bin/activate'):
            run('git pull')
            run('touch app.wsgi') 
```

将上述代码保存到文件 `fabfile.py` 中，我们可以这样检查内存的使用：

```py
$ fab memory_usage
[my_server1] Executing task 'memory'
[my_server1] run: free -m
[my_server1] out:              total     used     free   shared  buffers   cached
[my_server1] out: Mem:          6964     1897     5067        0      166      222
[my_server1] out: -/+ buffers/cache:     1509     5455
[my_server1] out: Swap:            0        0        0

[my_server2] Executing task 'memory'
[my_server2] run: free -m
[my_server2] out:              total     used     free   shared  buffers   cached
[my_server2] out: Mem:          1666      902      764        0      180      572
[my_server2] out: -/+ buffers/cache:      148     1517
[my_server2] out: Swap:          895        1      894 
```

and we can deploy with:

```py
$ fab deploy 
```

额外的特性包括并行执行、和远程程序交互、以及主机分组。

> [Fabric 文档](http://docs.fabfile.org) [http://docs.fabfile.org]

## Salt

[Salt](http://saltstack.org/) [http://saltstack.org/] 是一个开源的基础管理工具。它支持从中心节点（主要的主机）到多个主机（指从机）的远程命令执行。它也支持系统语句，能够使用简单的模板文件配置多台服务器。

Salt 支持 Python 2.6 和 2.7，并能通过 pip 安装：

```py
$ pip install salt 
```

在配置好一台主服务器和任意数量的从机后，我们可以在从机上使用任意的 shell 命令或者预制的复杂命令的模块。

下面的命令使用 ping 模块列出所有可用的从机：

```py
$ salt '*' test.ping 
```

主机过滤是通过匹配从机 id 或者使用颗粒系统（grains system）。 [颗粒（grains）](http://docs.saltstack.com/en/latest/topics/targeting/grains.html) [http://docs.saltstack.com/en/latest/topics/targeting/grains.html] 系统使用静态的主机信息，比如操作系统版本或者 CPU 架构，来为 Salt 模块提供主机分类内容。

下列命令行使用颗粒系统列举了所有可用的运行 CentOS 的从机：

```py
$ salt -G 'os:CentOS' test.ping 
```

Salt 也提供状态系统。状态能够用来配置从机。

例如，当一个从机接受读取下列状态文件的指令，他将会安装和启动 Apache 服务器：

```py
apache:
  pkg:
    - installed
  service:
    - running
    - enable: True
    - require:
      - pkg: apache 
```

状态文件可以使用 YAML、Jinja2 模板系统或者纯 Python 编写。

> [Salt 文档](http://docs.saltstack.com) [http://docs.saltstack.com]

## Psutil

[Psutil](https://code.google.com/p/psutil/) [https://code.google.com/p/psutil/] 是获取不同系统信息（比如 CPU、内存、硬盘、网络、用户、进程）的接口。

下面是一个关注一些服务器过载的例子。如果任意一个测试（网络、CPU）失败，它将会发送一封邮件。

```py
# 获取系统变量的函数:
from psutil import cpu_percent, net_io_counters
# 休眠函数:
from time import sleep
# 用于 email 服务的包:
import smtplib
import string
MAX_NET_USAGE = 400000
MAX_ATTACKS = 4
attack = 0
counter = 0
while attack <= MAX_ATTACKS:
    sleep(4)
    counter = counter + 1
    # Check the cpu usage
    if cpu_percent(interval = 1) > 70:
        attack = attack + 1
    # Check the net usage
    neti1 = net_io_counters()[1]
    neto1 = net_io_counters()[0]
    sleep(1)
    neti2 = net_io_counters()[1]
    neto2 = net_io_counters()[0]
    # Calculate the bytes per second
    net = ((neti2+neto2) - (neti1+neto1))/2
    if net > MAX_NET_USAGE:
        attack = attack + 1
    if counter > 25:
        attack = 0
        counter = 0
# 如果 attack 大于 4，就编写一封十分重要的 email
TO = "you@your_email.com"
FROM = "webmaster@your_domain.com"
SUBJECT = "Your domain is out of system resources!"
text = "Go and fix your server!"
BODY = string.join(("From: %s" %FROM,"To: %s" %TO,"Subject: %s" %SUBJECT, "",text), "\r\n")
server = smtplib.SMTP('127.0.0.1')
server.sendmail(FROM, [TO], BODY)
server.quit() 
```

一个类似于基于 psutil 并广泛扩展的 top，并拥有客服端-服务端监控能力的完全终端应用叫做 [glance](https://github.com/nicolargo/glances/) [https://github.com/nicolargo/glances/] 。

## Ansible

[Ansible](http://ansible.com/) [http://ansible.com/] 是一个开源系统自动化工具。相比于 Puppet 或者 Chef 最大的优点是它不需要客户机上的代理。Playbooks 是 Ansible 的配置、部署和编制语言，它用 YAML 格式编写，使用 Jinja2 作为模板。

Ansible 支持 Python 2.6 和 2.7，并能使用 pip 安装：

```py
$ pip install ansible 
```

Ansible requires an inventory file that describes the hosts to which it has access. Below is an example of a host and playbook that will ping all the hosts in the inventory file. Ansible 需要一个清单文件，来描述主机经过何处。以下是一个主机和 playbook 的例子，在清单文件中将会 ping 所有主机。

清单文件示例如下： `hosts.yml`

```py
[server_name]
127.0.0.1 
```

playbook 示例如下： `ping.yml`

```py
---
- hosts: all

  tasks:
    - name: ping
      action: ping 
```

要运行 playbook：

```py
$ ansible-playbook ping.yml -i hosts.yml --ask-pass 
```

Ansible playbook 在 `hosts.yml` 中将会 ping 所有的服务器。你也可以选择成组的服务器使用 Ansible。了解更多关于 Ansible 的信息，请阅读 [Ansible Docs](http://docs.ansible.com/) [http://docs.ansible.com/] 。

[An Ansible tutorial](https://serversforhackers.com/an-ansible-tutorial/) [https://serversforhackers.com/an-ansible-tutorial/] 也是一个很棒的且详细的指引来开始熟悉 Ansible。

## Chef

[Chef](https://www.chef.io/chef/) [https://www.chef.io/chef/] 是一个系统的云基础设施自动化框架，它使部署服务器和应用到任何物理、虚拟或者云终端上变得简单。你可以选择进行配置管理，那将主要使用 Ruby 去编写你的基础设施代码。

Chef 客户端运行于组成你的基础设施的每台服务器上，这些客户端定期检查 Chef 服务器来确保系统是均衡并且处于设想的状态。由于每台服务器拥有它自己的独立的 Chef 客户端，每个服务器配置自己，这种分布式方法使得 Chef 成为一个可扩展的自动化平台。

Chef 通过使用定制的在 cookbook 中实现的食谱（配置元素）来工作。Cookbook 通常作为基础设施的选择项，作为包存放在 Chef 服务器中。请阅读 [Digital Ocean tutorial series](https://www.digitalocean.com/community/tutorials/how-to-install-a-chef-server-workstation-and-client-on-ubuntu-vps-instances) [https://www.digitalocean.com/community/tutorials/how-to-install-a-chef-server-workstation-and-client-on-ubuntu-vps-instances] 关于 Chef 的部分来学习如何创建一个简单的 Chef 服务器。

要创建一个简单的 cookbook，使用 [knife](https://docs.chef.io/knife.html) [https://docs.chef.io/knife.html] 命令：

```py
knife cookbook create cookbook_name 
```

[Getting started with Chef](http://gettingstartedwithchef.com/first-steps-with-chef.html) [http://gettingstartedwithchef.com/first-steps-with-chef.html] 对 Chef 初学者来说是一个好的开始点，许多社区维护着 cookbook，可以作为是一个好的参考。要服务自己的基础设施配置需求，请见 [Chef Supermarket](https://supermarket.chef.io/cookbooks) [https://supermarket.chef.io/cookbooks] 。

*   [Chef 文档](https://docs.chef.io/) [https://docs.chef.io/]

## Puppet

[Puppet](http://puppetlabs.com) [http://puppetlabs.com] 是来自 Puppet Labs 的 IT 自动化和配置管理软件，允许系统管理员定义他们的 IT 基础设施状态，这样就能够提供一种优雅的方式管理他们成群的物理和虚拟机器。

Puppet 均可作为开源版和企业版获取到。其模块是小的、可共享的代码单元，用以自动化或定义系统的状态。 [Puppet Forge](https://forge.puppetlabs.com/) [https://forge.puppetlabs.com/] 是一个模块仓库，它由社区编写，面向开源和企业版的 Puppet。

Puppet 代理安装于其状态需要被监控或者修改的节点上。作为特定服务器的 Puppet Master 负责组织代理节点。

代理节点发送系统的基本信息到 Puppet Master，比如说操作系统、内核、架构、ip 地址、主机名等。接着，Puppet Master 编译携带有节点生成信息的目录，告知每个节点应如何配置，并发送给代理。代理便会执行前述目录中的变化，并向 Puppet Master 发送回一份报告。

Facter 是一个有趣的工具，它用来传递 Puppet 获取到的基本系统信息。这些信息可以在编写 Puppet 模块的时候作为变量来引用。

```py
$ facter kernel
Linux 
```

```py
$ facter operatingsystem
Ubuntu 
```

在 Puppet 中编写模块十分直截了当。Puppet 清单（manifest）组成了 Puppet 模块。Puppet 清单以扩展名 `.pp` 结尾。下面是一个 Puppet 中 ‘Hello World’的例子。

```py
notify { 'This message is getting logged into the agent node': #As nothing is specified in the body the resource title
 #the notification message by default.
} 
```

这里是另一个基于系统的逻辑的例子。注意操纵系统信息是如何作为变量使用的，变量前加了前缀符号 `$` 。类似的，其他信息比如说主机名就能用 `$hostname` 来引用。

```py
notify{ 'Mac Warning':
    message => $operatingsystem ? {
        'Darwin' => 'This seems to be a Mac.',
        default  => 'I am a PC.',
    },
} 
```

Puppet 有多种资源类型，需要时可以使用包-文件-服务（package-file-service）范式来承担配置管理的主要任务。下面的 Puppet 代码确保了系统中安装了 OpenSSH-Server 包，并且在每次 sshd 配置文件改变时重启 sshd 服务。

```py
package { 'openssh-server':
    ensure => installed,
}

file { '/etc/ssh/sshd_config':
    source   => 'puppet:///modules/sshd/sshd_config',
    owner    => 'root',
    group    => 'root',
    mode     => '640',
    notify   =>  Service['sshd'], # sshd will restart
 # whenever you edit this
 # file
    require  => Package['openssh-server'],

}

service { 'sshd':
    ensure    => running,
    enable    => true,
    hasstatus => true,
    hasrestart=> true,
} 
```

了解更多信息，参考 [Puppet Labs 文档](http://docs.puppetlabs.com) [http://docs.puppetlabs.com] 。

## Blueprint

待处理

Write about Blueprint

## Buildout

[Buildout](http://www.buildout.org) [http://www.buildout.org] 是一个开源软件构件工具。Buildout 由 Python 编写。它实现了配置和构建脚本分离的原则。Buildout 主要用于下载和设置正在开发或部署软件的 Python egg 格式的依赖。在任何环境中构建任务的指南（recipe，原意为“食谱”，引申为“指南”）能被创建，许多早已可用。

© 版权所有 2014\. A <a href="http://kennethreitz.com/pages/open-projects.html">Kenneth Reitz</a> 工程。 <a href="http://creativecommons.org/licenses/by-nc-sa/3.0/"> Creative Commons Share-Alike 3.0</a>.
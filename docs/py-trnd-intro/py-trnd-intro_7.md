# 第八章：部署 Tornado

到目前为止，为了简单起见，在我们的例子中都是使用单一的 Tornado 进程运行的。这使得测试应用和快速变更非常简单，但是这不是一个合适的部署策略。部署一个应用到生产环境面临着新的挑战，既包括最优化性能，也包括管理独立进程。本章将介绍强化你的 Tornado 应用、增加请求吞吐量的策略，以及使得部署 Tornado 服务器更容易的工具。

*   8.1 运行多个 Tornado 实例的原因
*   8.2 使用 Nginx 作为反向代理
    *   8.2.1 Nginx 基本配置
    *   8.2.2 Nginx 的 SSL 解密
    *   8.3 使用 Supervisor 监控 Tornado 进程

## 8.1 运行多个 Tornado 实例的原因

在大多数情况下，组合一个网页不是一个特别的计算密集型处理。服务器需要解析请求，取得适当的数据，以及将多个组件组装起来进行响应。如果你的应用使用阻塞的调用查询数据库或访问文件系统，那么服务器将不会在等待调用完成时响应传入的请求。在这些情况下，服务器硬件有剩余的 CPU 时间来等待 I/O 操作完成。

鉴于响应一个 HTTP 请求的时间大部分都花费在 CPU 空闲状态下，我们希望利用这个停工时间，最大化给定时间内我们可以处理的请求数量。也就是说，我们希望服务器能够在处理已打开的请求等待数据的过程中接收尽可能多的新请求。

正如我们在第五章讨论的异步 HTTP 请求中所看到的，Tornado 的非阻塞架构在解决这类问题上大有帮助。回想一下，异步请求允许 Tornado 进程在等待出站请求返回时执行传入的请求。然而，我们碰到的问题是当同步函数调用块时。设想在一个 Tornado 执行的数据库查询或磁盘访问块中，进程不允许回应新的请求。这个问题最简单的解决方法是运行多个解释器的实例。通常情况下，你会使用一个反向代理，比如 Nginx，来非配多个 Tornado 实例的加载。

## 8.2 使用 Nginx 作为反向代理

一个代理服务器是一台中转客户端资源请求到适当的服务器的机器。一些网络安装使用代理服务器过滤或缓存本地网络机器到 Internet 的 HTTP 请求。因为我们将运行一些在不同 TCP 端口上的 Tornado 实例，因此我们将使用反向代理服务器：客户端通过 Internet 连接一个反向代理服务器，然后反向代理服务器发送请求到代理后端的 Tornado 服务器池中的任何一个主机。代理服务器被设置为对客户端透明的，但它会向上游的 Tornado 节点传递一些有用信息，比如原始客户端 IP 地址和 TCP 格式。

我们的服务器配置如图 8-1 所示。反向代理接收所有传入的 HTTP 请求，然后把它们分配给独立的 Tornado 实例。

![图 8-1](img/2015-09-04_55e96fc4934fc.jpg)

图 8-1 反向代理服务器后端的 Tornado 实例

### 8.2.1 Nginx 基本配置

代码清单 8-1 中的列表是一个 Nginx 配置的示例。Nginx 启动后监听来自 80 端口的连接，然后分配这些请求到配置文件中列出的上游主机。在这种情况下，我们假定上游主机监听来自他们自己的环回接口上的端口的连接。

代码清单 8-1 一个简单的 Nginx 代理配置

```py
user nginx;
worker_processes 5;

error_log /var/log/nginx/error.log;

pid /var/run/nginx.pid;

events {
    worker_connections 1024;
    use epoll;
}

proxy_next_upstream error;

upstream tornadoes {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
    server 127.0.0.1:8003;
}

server {
    listen 80;
    server_name www.example.org *.example.org;

    location /static/ {
        root /var/www/static;
        if ($query_string) {
            expires max;
        }
    }

    location / {
        proxy_pass_header Server;
        proxy_set_header Host $http_host;
        proxy_redirect off;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Scheme $scheme;
        proxy_pass http://tornadoes;
    }
}

```

这个配置示例假定你的系统使用了 epoll。在不同的 UNIX 发行版本中经常会有轻微的不同。一些系统可能使用了 poll、/dev/poll 或 kqueue 代替。

按顺序来看匹配 location /static/或 location /的请求可能会很有帮助。Nginx 把位于 location 指令中的字符串看作是一个以行起始锚点开始、任何字母重复结束的正则表达式。所以/被看作是表达式^/.*。当 Nginx 匹配这些字符串时，像/static 这样更加特殊的字符串在像/这样的更加的通用的字符串之前被检查。Nginx 文档中详细解释了匹配的顺序。

除了一些标准样板外，这个配置文件最重要的部分是 upstream 指令和服务器配置中的 proxy 指令。Nginx 服务器在 80 端口监听连接，然后分配这种请求给 upstream 服务器组中列出的 Tornado 实例。proxy_pass 指令指定接收转发请求的服务器 URI。你可以在 proxy_pass URI 中的主机部分引用 upstream 服务器组的名字。

Nginx 默认以循环的方式分配请求。此外，你也可以选择基于客户端的 IP 地址分配请求，这种情况下（除非连接中断）可以确保来自同一 IP 地址的请求总是被分配到同一个上游节点。你可以在[HTTPUpstreamModule 文档](http://wiki.nginx.org/HttpUpstreamModule)中了解更多关于这个选项的知识。

还需要注意的是 location /static/指令，它告诉 Nginx 直接提供静态目录的文件，而不再代理请求到 Tornado。Nginx 可以比 Tornado 更高效地提供静态文件，所以减少 Tornado 进程中不必要的加载是非常有意义的。

### 8.2.2 Nginx 的 SSL 解密

应用的开发者在浏览器和客户端之间传输个人信息时需要特别注意保护信息不要落入坏人之手。在不安全的 WiFi 接入中，用户很容易受到 cookie 劫持攻击，从而威胁他们在流行的社交网站上的账户。对此，大部分主要的社交网络应用都默认或作为用户可配置选项使用安全协议。同时，我们使用 Nginx 解密传入的 SSL 加密请求，然后把解码后的 HTTP 请求分配给上游服务器。

代码清单 8-2 展示了一个用于解密传入的 HTTPS 请求的 server 块，并使用代码清单 8-1 中我们使用过的代理指令转发解密后的通信。

代码清单 8-2 使用 SSL 的 server 块

```py
server {
    listen 443;
    ssl on;
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/cert.key;

    default_type application/octet-stream;

    location /static/ {
        root /var/www/static;
        if ($query_string) {
            expires max;
        }
    }

    location = /favicon.ico {
        rewrite (.*) /static/favicon.ico;
    }

    location / {
        proxy_pass_header Server;
        proxy_set_header Host $http_host;
        proxy_redirect off;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Scheme $scheme;
        proxy_pass http://tornadoes;
    }
}

```

这段代码和上面的配置非常相似，除了 Nginx 将在标准 HTTPS 的 443 端口监听安全 Web 请求外。如果你想强制使用 SSL 连接，你可以在 server 块中包含一个 rewrite 指令来监听 80 端口的 HTTP 连接。代码清单 8-3 是这种重定向的一个例子。

代码清单 8-3 用于重定向 HTTP 请求到安全渠道的 server 块

```py
server {
    listen 80;
    server_name example.com;

    rewrite /(.*) https://$http_host/$1 redirect;
}

```

Nginx 是一个非常鲁棒的工具，我们在这里仅仅接触到帮助 Tornado 部署的一些简单的配置选项。Nginx 的[wiki 文档](http://wiki.nginx.org/)是获得安装和配置这个强有力的工具额外信息的一个非常好的资源。

## 8.3 使用 Supervisor 监控 Tornado 进程

正如 8.2 节中埋下的伏笔，我们将在我们的 Tornado 应用中运行多个实例以充分利用现代的多处理器和多核服务器架构。开发团队大多传闻每个核运行一个 Tornado 进程。但是，正如我们所知道的，大量的传闻并不代表事实，所以你的结果可能不同。在本节中，我们将讨论在 UNIX 系统中管理多个 Tornado 实例的策略。

到目前为止，我们都是在命令行中运行 Tornado 服务器的，就像`$ python main.py --port=8000`。但是，再擦河南刮起的生产部署中，这是不可管理的。因为我们为每个 CPU 核心运行一个独立的 Tornado 进程，因此有很多进程需要监控和控制。supervisor 守护进程可以帮助我们完成这个任务。

Supervisor 的设计是每次开机时启动其配置文件中列出的进程。这里，我们将看到管理我们在 Nginx 配置文件中作为上游主机提到的四个 Tornado 实例的 Supervisor 配置。典型的 supervisord.conf 文件中包含了全局的配置指令，并加载 conf.d 目录下的其他配置文件。代码清单 8-4 展示了我们想启动的 Tornado 进程的配置文件。

代码清单 8-4 tornado.conf

```py
[group:tornadoes]
programs=tornado-8000,tornado-8001,tornado-8002,tornado-8003

[program:tornado-8000]
command=python /var/www/main.py --port=8000
directory=/var/www
user=www-data
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/tornado.log
loglevel=info

[program:tornado-8001]
command=python /var/www/main.py --port=8001
directory=/var/www
user=www-data
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/tornado.log
loglevel=info

[program:tornado-8002]
command=python /var/www/main.py --port=8002
directory=/var/www
user=www-data
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/tornado.log
loglevel=info

[program:tornado-8003]
command=python /var/www/main.py --port=8003
directory=/var/www
user=www-data
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/tornado.log
loglevel=info

```

为了 Supervisor 有意义，你需要至少包含一个 program 部分。在代码清单 8-4 中，我们定义了四个程序，分别命名为 tornado-8000 到 tornado-8003。program 部分定义了 Supervisor 将要运行的每个命令的参数。command 的值是必须的，通常是带有我们希望监听的 port 参数的 Tornado 应用。我们还为每个程序的工作目录、有效用户和日志文件定义了额外的设置；而把 autorestart 和 redirect_stderr 设置为 true 是非常有用的。

为了一起管理所有的 Tornado 进程，创建一个组是很有必要的。在这个例子的顶部，我们声明了一个叫作 tornadoes 的组，并在其中列出了每个程序。现在，当我们想要管理我们的 Tornado 应用时，我们可以通过带有通配符的组名引用所有的组成程序。比如，要重启应用时，我们只需要在 supervisorctl 工具中使用命令`restart tornadoes:*`。

一旦你安装和配置好 Supervisor，你就可以使用 supervisorctl 来管理 supervisord 进程。为了启动你的 Web 应用，你可以让 Supervisor 重新读取配置，然后任何配置改变的程序或程序组将被重启。你同样可以手动启动、停止和重启被管理的程序或检查整个系统的状态。

```py
supervisor> update
tornadoes: stopped
tornadoes: updated process group
supervisor> status
tornadoes:tornado-8000 RUNNING pid 32091, uptime 00:00:02
tornadoes:tornado-8001 RUNNING pid 32092, uptime 00:00:02
tornadoes:tornado-8002 RUNNING pid 32093, uptime 00:00:02
tornadoes:tornado-8003 RUNNING pid 32094, uptime 00:00:02

```

Supervisor 和你系统的初始化进程一起工作，并且它应该在系统启动时自动注册守护进程。当 supervisor 启动后，程序组会自动在线。默认情况下，Supervisor 会监控子进程，并在任何程序意外终止时重生。如果你想不管错误码，重启被管理的进程，你可以设置 autorestart 为 true。

Supervisor 不只可以使管理多个 Tornado 实例更容易，还能让你在 Tornado 服务器遇到意外的服务中断后重新上线时泰然处之。
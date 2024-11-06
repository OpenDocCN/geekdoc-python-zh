# 网站的部署

# 部署

根据和代码/系统的紧密程度，我们可以简单地把部署分为应用级和系统级。 应用级通常提供把 HTTP 请求递交给应用，而系统级和运维的关系更为紧密。

下面是一个非常常见的 Python 应用部署架构：

+   [nginx](https://www.rabbitmq.com/) ：静态文件服务；SSL 负载转移；反向代理；

+   [Memcached](http://memcached.org/) / [Redis](http://redis.io/) ：缓存；

+   [Celery](http://www.celeryproject.org/) ：运行后台任务；

+   [Redis](http://redis.io/) / [RabbitMQ](https://www.rabbitmq.com/) ：任务队列（通常对接 Celery）；

+   [uWSGI] / [Gunicorn] ：WSGI 服务器；

## web 应用服务器

### uWSGI

uWSGI 是一个主要以 C 语言实现的高性能服务器，性能优异、配置灵活，可以使用 C、C++ 甚至 Objective-C 编写插件。

uWSGI 并不仅仅是一个 WSGI 服务器，它也可以用作 Ruby Rack 应用或者 Perl PSGI 应用的后端服务器。

### Gunicorn

Gunicorn 是一个类 Unix 系统上一个 Python 实现的 WSGI 服务器，性能很高使用也很简单。

作为 WSGI 服务器，Guicorn 拥有很好的性能，但是它并不擅长静态文件处理，因此在实际部署中常常隐藏于 Nginx 之后，由 Nginx 提供静态文件服务，动态请求则通过反向代理发送给 Gunicorn。

### Tornado

### Nginx/Apache

## 服务器部署

Python 应用部署的常用工具有 [Fabric](http://fabric.readthedocs.org/)、[SaltStack](http://saltstack.com/)、[Ansible](http://www.ansible.com/) 和 [Puppet](https://puppetlabs.com/)，前三者都是 Python 应用，Puppet 这是 Ruby 便携的服务器部署工具，除 Fabric 是完全开源外，后三者背后都有商业公司， 提供应用的商业版本或者应用之外的企业服务。

### Fabric

Fabric 是一个基于 SSH 的命令式部署工具，通过编写 `fabfile` 来扩展和定制 Fabric 的功能。不得不说，对于复杂的部署环境来说，Fabric 的部署方式已经有些落后，但是对于少量、 简单的服务环境，Fabric 使用起来简单、方便。

### Ansible

Ansible 同样是基于 SSH 的服务器运维工具。

# uWSGI

# uWSGI

虽然 [uWSGI](https://uwsgi-docs.readthedocs.org/) 最常见的使用场景是当作 WSGI 服务器，但它其实还提供了更多的其它功能，[这篇博客](https://lincolnloop.com/blog/uwsgi-swiss-army-knife/) 中介绍了 uWSGI 如���瑞士军刀般强大��各项功能。

## 安装指南

官方提供了多种安装方式，比如使用 pip 安装 uWSGI 安装包： `pip install uwsgi`。

或者使用安装脚本：

```
curl http://uwsgi.it/install | bash -s default /tmp/uwsgi 
```

或者编译安装最新版本：

```
wget http://projects.unbit.it/downloads/uwsgi-latest.tar.gz
tar zxvf uwsgi-latest.tar.gz
cd <dir>
make 
```

## 简单使用

以一个简单的 uwsgi 命令为例介绍它的常见用法：

```
uwsgi --http :9090 --wsgi-file foobar.py --master --processes 4 --threads 2 --stats 127.0.0.1:9191 
```

+   `--http :9090` 是指监听端口 9090 的 HTTP 请求。

+   `--wsgi-file foobar.py` 是指从文件 `foobar.py` 中寻找 WSGI 应用对象，默认寻找 `application` 对象。

+   `--processes 4 --threads 2` 指 uWSGI 启动 4 个进程，每个进程启动 2 个线程。

+   `--stats 127.0.0.1:9191` 提供了 uWSGI 状态监视端口，可以安装 uwsgitop 来查看 uWGI 进程状态。

# Gunicorn

# Gunicorn

[Gunicorn](http://gunicorn.org/) 全称 Green Unicorn，是 Ruby 服务器 Unicorn 的 Python 版本，虽然完全是由 Python 实现，但是拥有作为 WSGI 服务器性能优秀，是现在最流行的 WSGI 服务之一。

## 安装指南

虽然在很多 Linux 操作系统上都有 Gunicorn 的安装包，但通常还是建议使用 Python 包管理工具 `pip` 来安装最新版本的 Gunicorn：

```
[sudo ]easy_install pip
pip install gunicorn 
```

## 简单使用

Gunicorn 的使用非常简单。

假设你的 WSGI 应用包叫做 `myapp`，包中有一个符合 WSGI Application 规范的 `app` 对象，就可以这样部署启动 Gunicorn 服务器来托管应用：

```
gunicorn -w 4 myapp:app 
```

上面命令的 `-w 4` 是指启动 4 个 worker 进程，用以提高服务器的负载能力。

### 常用参数

+   `-c CONFIG, --config=CONFIG` - 指定配置文件，格式可以是 `$(PATH)`、`file:$(PATH)` 或者 `python:$(MODULE_NAME)`。

+   `-b BIND, --bind=BIND` - 指定 socket 地址，可以是 `$(HOST)`、`$(HOST):$(PORT)` 或者 `unix:$(PATH)` 格式。`$(HOST)` 可以是单纯的 IP 地址。

+   `-w WORKERS, --workers=WORKERS` - worker 进程数。推荐数量为服务器 CPU 核数的 2-4 倍。

+   `-k WORKERCLASS, --worker-class=WORKERCLASS` - worker 进程的类型，详细介绍请查看产品文档。你可以在这里设置 `$(NAME)` ，其值可以是 `sync`、`eventlet`、`gevent` 或者 `tornado`、`gthread` `gaiohttp`。默认是 `sync`。

+   `-n APP_NAME, --name=APP_NAME` - 如果安装了 setproctitle，你可以使用该参数设置 Gunicorn 的进程名。

## 部署实例

Gunicorn 虽然是一个性能优秀的 WSGI 服务器，但是并不擅长静态资源管理，在生产环境下通常会配合 Nginx 使用。由 Nginx 作为服务器前端，将动态请求通过反向代理传递给 Gunicorn。Nginx 反向代理配置示例：

```
server {
  listen 80;
  server_name example.org;
  access_log  /var/log/nginx/example.log;

  location / {
      proxy_pass http://127.0.0.1:8000;
      proxy_set_header Host $host;
      proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
  }
} 
```

## 配置

参考：[`docs.gunicorn.org/en/latest/configure.html`](http://docs.gunicorn.org/en/latest/configure.html)

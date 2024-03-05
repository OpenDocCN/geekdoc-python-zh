# 在杜库部署姜戈

> 原文：<https://realpython.com/deploying-a-django-app-on-dokku/>

这篇文章最初是为 *[Gun.io](http://www.gun.io)* 写的，详细介绍了如何使用 Dokku 作为 Heroku 的替代品来部署你的 Django 应用。

## 什么是 Dokku？

几天前，我被指向了 Dokku 项目，这是一个“Docker powered mini-Heroku ”,你可以将其部署在你自己的服务器上，作为你自己的私有 PaaS。

你为什么想要自己的迷你 Heroku？

*   嗯， [Heroku 可以花*很多*的钱](http://joshsymonds.com/blog/2012/06/03/my-love-slash-hate-relationship-with-heroku/)；它托管在云中，您可能还不想让您的应用程序离开房间；而且你也没有 100%掌控平台的方方面面。
*   或者也许你就是那种喜欢自己动手的人。

不管怎样， [Jeff Lindsay](http://progrium.com/blog/) 用不到 100 行 bash 代码一起黑掉了 Dokku！

Dokku 将 [Docker](http://www.docker.io/) 与 [Gitreceive](https://github.com/progrium/gitreceive) 和 [Buildstep](https://github.com/progrium/buildstep) 捆绑到一个易于部署、分叉/破解和更新的包中。

[*Remove ads*](/account/join/)

## 您需要什么来开始

你可以在自己的私人网络上使用从 AWS 到电脑的任何东西。我决定使用[数字海洋](https://www.digitalocean.com/)作为这个小项目的云托管服务。

托管 Dokku 的要求很简单:

*   Ubuntu 14.10 x64
*   SSH 功能

> Digital Ocean 有一个[预配置的 Droplet](https://www.digitalocean.com/community/tutorials/how-to-use-the-digitalocean-dokku-application) ，您可以使用已经为 Dokku 环境提供的 Droplet。请随意使用这个。**我们将使用全新的服务器，因此你可以在任何服务器上重现这个过程，而不仅仅是在数字海洋**上。

我们开始吧！

1.  首先，在[数字海洋](https://www.digitalocean.com)上注册一个账户，确保给账户添加一个公钥。*如果您需要创建一个新的密钥，您可以按照本[指南](https://www.digitalocean.com/community/articles/how-to-set-up-ssh-keys--2)中的步骤一到三来帮助您进行设置。第四步以后会派上用场。*
2.  接下来，通过点击“创建液滴”创建一个“液滴”(旋转一个节点)。确保您选择“Ubuntu 14.10 x64”作为您的图像。我最初选择了 x32 版本，但杜库不愿意安装(见[https://github.com/progrium/dokku/issues/51](https://github.com/progrium/dokku/issues/51))。将您的 ssh 公钥添加到 droplet，这样您就可以通过 ssh 进入机器，而不必每次登录时都输入密码。数字海洋大约需要一分钟来启动您的机器。
3.  准备就绪后，Digital Ocean 会发电子邮件告诉您，并在邮件中包含机器的 IP 地址，或者机器会出现在您的 droplets 面板下。使用 IP 地址 SSH 进入机器，并遵循数字海洋的 [ssh 指南](https://www.digitalocean.com/community/articles/how-to-set-up-ssh-keys--2)中的步骤四。

## 安装 Dokku

现在我们的主机已经设置好了，是时候安装和配置 Dokku 了。SSH 回到您的主机并运行以下命令:

```py
$ wget -qO- https://raw.github.com/progrium/dokku/v0.2.3/bootstrap.sh | sudo DOKKU_TAG=v0.2.3 bash
```

> 无论是否以 root 用户身份登录，都要确保使用`sudo`。有关更多信息，请参见下面的“总结”部分。

安装可能需要 2 到 5 分钟。完成后，注销您的主机。

确保使用以下格式为用户上传公钥:

```py
$ cat ~/.ssh/id_rsa.pub | ssh root@<machine-address> "sudo sshcommand acl-add dokku <your-app-name> "
```

用主机的 IP 地址或域名替换`<machine-address>`,用 Django 项目的名称替换`<your-app-name>`。

例如:

```py
$ cat ~/.ssh/id_rsa.pub | ssh root@104.236.38.176 "sudo sshcommand acl-add dokku hellodjango"
```

## 将 Django 应用程序部署到 Dokku

对于本教程，让我们按照 Heroku 上的[Django 入门指南来设置一个初始的 Django 应用程序。](https://devcenter.heroku.com/articles/django)

> 同样，Dokku 使用 [Buildstep](https://github.com/progrium/buildstep) 来构建你的应用。它内置了 [Heroku Python Buildpack](https://github.com/heroku/heroku-buildpack-python) ，这足以运行一个 Django 或 Flask 应用程序，开箱即用。然而，如果你想添加一个定制的构建包[，你可以](https://github.com/progrium/buildstep#adding-buildpacks)。

创建一个 Django 项目并添加一个本地 Git repo:

```py
$ mkdir hellodjango && cd hellodjango
$ virtualenv venv
$ source venv/bin/activate
$ pip install django-toolbelt
$ django-admin.py startproject hellodjango .
$ echo "web: gunicorn hellodjango.wsgi" > Procfile
$ pip freeze > requirements.txt
$ echo "venv" > .gitignore
$ git init
$ git add .
$ git commit -m "First Commit HelloDjango"
```

我们必须在我们的主机上添加 Dokku 作为 Git remote:

```py
$ git remote add production dokku@<machine-address>:hellodjango
```

现在我们可以推出我们的代码了:

```py
$ git push production master
```

用您的主机的地址或域名替换`<machine-address>`。如果一切顺利，您应该会在终端中看到应用程序部署消息:

```py
=====> Application deployed:
 http://104.236.38.176:49153
```

接下来，访问`http://<machine-address>:49153`，你会看到熟悉的“欢迎来到 Django”页面。现在，您可以在本地开发您的应用程序，然后将其推送到您自己的 mini-heroku！。

[*Remove ads*](/account/join/)

## 总结

最初，我安装了没有“sudo”的 Dokku:

```py
$ wget -qO- https://raw.github.com/progrium/dokku/v0.2.3/bootstrap.sh | DOKKU_TAG=v0.2.3 bash
```

当我尝试推送到 Dokku 时，python 构建包在尝试下载/构建 python 时会失败。解决这个问题的方法是卸载 Dokku，然后使用 sudo:

```py
$ wget -qO- https://raw.github.com/progrium/dokku/v0.2.3/bootstrap.sh | sudo DOKKU_TAG=v0.2.3 bash
```

不幸的是，杜库没有赫罗库走得远。

例如，所有命令都需要直接在主机服务器上运行，因为 Dokku 没有像 Heroku 那样的客户端应用程序。

因此，为了运行这样的命令:

```py
$ heroku run python manage.py syncdb
```

您需要首先通过 SSH 进入服务器。最简单的方法是创建一个`dokku`命令:

```py
alias dokku="ssh -t root@<machine-address> dokku"
```

现在，您可以运行以下命令来同步数据库:

```py
$ dokku run hellodjango python manage.py syncdb
```

Dokku 允许您为每个应用程序单独配置环境变量。只需创建或编辑`/home/git/APP_NAME/ENV`，并在其中填入如下内容:

```py
export DATABASE_URL=somethinghere
```

Dokku 仍然是一个年轻的平台，所以希望它继续成长，变得更加有用。它也是开源的，所以如果你想投稿，可以在 [Github](https://github.com/progrium/dokku) 上提出一个请求或者发表一个问题。**
# 使用 Fabric 和 Ansible 自动化 Django 部署

> 原文：<https://realpython.com/automating-django-deployments-with-fabric-and-ansible/>

在上一篇[文章中，我们介绍了在一台服务器上成功开发和部署 Django 应用程序所需的所有步骤。**在本教程中，我们将使用**](https://realpython.com/development-and-deployment-of-cookiecutter-django-on-fedora/) **[Fabric](http://www.fabfile.org/) (v [1.12.0](http://docs.fabfile.org/en/1.12/) )和[ansi ble](https://github.com/ansible/ansible)(v[2 . 1 . 3](https://github.com/ansible/ansible/releases/tag/v2.1.3.0-1))**来自动化部署过程，以解决这些问题:

1.  **扩展**:当要扩展一个 web 应用程序来处理成千上万的日常请求时，依靠单个服务器并不是一个好方法。简而言之，当服务器接近最大 CPU 利用率时，它会导致缓慢的加载时间，最终导致服务器故障。为了克服这个问题，应用程序必须扩展到在多台服务器上运行，这样服务器就可以*累积*处理传入的并发请求。
2.  冗余:手动将一个 web 应用程序部署到一个新的服务器上意味着大量的重复工作，出现人为错误的可能性更大。自动化流程是关键。

具体来说，我们将自动化:

1.  添加新的非超级用户
2.  配置服务器
3.  从 GitHub repo 中提取 Django 应用程序代码
4.  安装[依赖项](https://realpython.com/courses/managing-python-dependencies/)
5.  将应用程序虚拟化

## 设置和配置

首先旋转一个新的[数字海洋](https://www.digitalocean.com/?refcode=d8f211a4b4c2)水滴，确保使用 Fedora 25 图像。不要设置预先配置的 SSH 密钥；我们将通过一个织物[脚本](https://realpython.com/run-python-scripts/)自动完成这个过程。因为部署过程应该是可伸缩的，所以创建一个单独的存储库来存放所有的部署脚本。在本地创建一个新的项目目录，并使用 Python 2.7x 创建和激活一个 virtualenv。

> 为什么选择 Python 2.7？Fabric 不支持 Python 3。不要担心:当我们提供服务器时，我们将使用 Python 3.5。

```py
$ mkdir automated-deployments
$ cd automated-deployments
$ virtualenv env
$ source env/bin/activate
```

[*Remove ads*](/account/join/)

## 织物设置

Fabric 是一个工具，用于通过 SSH 自动执行日常 shell 命令，我们将使用它来:

1.  设置 SSH 密钥
2.  [强化](https://en.wikipedia.org/wiki/Hardening_(computing%29))用户密码
3.  安装可转换的依赖项
4.  升级服务器

从安装结构开始:

```py
$ pip install fabric==1.12.0
```

创建一个名为“prod”的新文件夹，并向其中添加一个名为 *fabfile.py* 的新文件，以保存所有 Fabric 脚本:

```py
# prod/fabfile.py

import os
from fabric.contrib.files import sed
from fabric.api import env, local, run
from fabric.api import env

# initialize the base directory
abs_dir_path = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))

# declare environment global variables

# root user
env.user = 'root'

# list of remote IP addresses
env.hosts = ['<remote-server-ip>']

# password for the remote server
env.password = '<remote-server-password>'

# full name of the user
env.full_name_user = '<your-name>'

# user group
env.user_group = 'deployers'

# user for the above group
env.user_name = 'deployer'

# ssh key path
env.ssh_keys_dir = os.path.join(abs_dir_path, 'ssh-keys')
```

记下行内注释。确保将您的远程服务器的 IP 地址添加到`env.hosts`变量中。同样更新`env.full_name_user`。暂缓更新`env.password`；我们很快就会谈到这一点。查看所有的`env`变量——它们完全可以根据您的系统设置进行定制。

### 设置 SSH 密钥

将以下代码添加到 *fabfile.py* :

```py
def start_provision():
    """
 Start server provisioning
 """
    # Create a new directory for a new remote server
    env.ssh_keys_name = os.path.join(
        env.ssh_keys_dir, env.host_string + '_prod_key')
    local('ssh-keygen -t rsa -b 2048 -f {0}'.format(env.ssh_keys_name))
    local('cp {0}  {1}/authorized_keys'.format(
        env.ssh_keys_name + '.pub', env.ssh_keys_dir))
    # Prevent root SSHing into the remote server
    sed('/etc/ssh/sshd_config', '^UsePAM yes', 'UsePAM no')
    sed('/etc/ssh/sshd_config', '^PermitRootLogin yes',
        'PermitRootLogin no')
    sed('/etc/ssh/sshd_config', '^#PasswordAuthentication yes',
        'PasswordAuthentication no')

    install_ansible_dependencies()
    create_deployer_group()
    create_deployer_user()
    upload_keys()
    set_selinux_permissive()
    run('service sshd reload')
    upgrade_server()
```

该函数充当结构脚本的入口点。除了触发一系列功能(每个功能将在后续步骤中解释)之外，它还明确地-

*   在本地系统的指定位置生成一对新的 SSH 密钥
*   将公钥的内容复制到 *authorized_keys* 文件中
*   对远程 *sshd_config* 文件进行更改，以阻止 root 登录并禁用无密码身份验证

> 阻止 root 用户的 SSH 访问是一个可选步骤，但建议这样做，因为它可以确保没有人拥有超级用户权限。

在项目根目录下为您的 SSH 密钥创建一个目录:

```py
├── prod
│   └── fabfile.py
└── ssh-keys
```

### 强化用户密码

这一步包括添加三个不同的函数，每个函数依次执行以配置 SSH 密码强化…

#### 创建部署者组

```py
def create_deployer_group():
    """
 Create a user group for all project developers
 """
    run('groupadd {}'.format(env.user_group))
    run('mv /etc/sudoers /etc/sudoers-backup')
    run('(cat /etc/sudoers-backup; echo "%' +
        env.user_group + ' ALL=(ALL) ALL") > /etc/sudoers')
    run('chmod 440 /etc/sudoers')
```

这里，我们添加了一个名为`deployers`的新组，并授予它 sudo 权限，这样用户就可以使用 root 权限执行进程。

#### 创建用户

```py
def create_deployer_user():
    """
 Create a user for the user group
 """
    run('adduser -c "{}" -m -g {}  {}'.format(
        env.full_name_user, env.user_group, env.user_name))
    run('passwd {}'.format(env.user_name))
    run('usermod -a -G {}  {}'.format(env.user_group, env.user_name))
    run('mkdir /home/{}/.ssh'.format(env.user_name))
    run('chown -R {} /home/{}/.ssh'.format(env.user_name, env.user_name))
    run('chgrp -R {} /home/{}/.ssh'.format(
        env.user_group, env.user_name))
```

这个功能-

*   向`deployers`用户组添加一个新用户，这是我们在最后一个函数中定义的
*   设置保存 SSH 密钥对的 SSH 目录，并授予组和用户访问该目录的权限

#### 上传 SSH 密钥

```py
def upload_keys():
    """
 Upload the SSH public/private keys to the remote server via scp
 """
    scp_command = 'scp {}  {}/authorized_keys {}@{}:~/.ssh'.format(
        env.ssh_keys_name + '.pub',
        env.ssh_keys_dir,
        env.user_name,
        env.host_string
    )
    local(scp_command)
```

在这里，我们-

*   将本地生成的 SSH 密钥上传到远程服务器，以便非根用户可以通过 SSH 登录，而无需输入密码
*   将公钥和授权密钥复制到远程服务器新创建的 *ssh-keys* 目录中

[*Remove ads*](/account/join/)

### 安装可转换的依赖关系

添加以下函数来安装 Ansible 的依赖包:

```py
def install_ansible_dependencies():
    """
 Install the python-dnf module so that Ansible
 can communicate with Fedora's Package Manager
 """
    run('dnf install -y python-dnf')
```

> 请记住，这是特定于 Fedora Linux 发行版的，因为我们将使用 [DNF 模块](http://docs.ansible.com/ansible/dnf_module.html)来安装包，但它可能因发行版而异。

### 将 SELinux 设置为许可模式

下一个功能将 [SELinux](https://en.wikipedia.org/wiki/Security-Enhanced_Linux) 设置为[许可模式](https://wiki.gentoo.org/wiki/SELinux/Tutorials/Permissive_versus_enforcing#Permissive_versus_enforcing)。这样做是为了克服任何潜在的 Nginx 502 坏网关[错误](https://asdqwe.net/blog/solutions-502-bad-gateway-error-on-nginx/)。

```py
def set_selinux_permissive():
    """
 Set SELinux to Permissive/Disabled Mode
 """
    # for permissive
    run('sudo setenforce 0')
```

> 同样，这是特定于 Fedora Linux 发行版的。

### 升级服务器

最后，升级服务器:

```py
def upgrade_server():
    """
 Upgrade the server as a root user
 """
    run('dnf upgrade -y')
    # optional command (necessary for Fedora 25)
    run('dnf install -y python')
    run('reboot')
```

## 健全性检查

至此，我们完成了 Fabric 脚本。在运行它之前，确保您以 root 用户身份 SSH 到服务器，并更改密码:

```py
$ ssh root@<server-ip-address>
You are required to change your password immediately (root enforced)
Changing password for root.
(current) UNIX password:
New password:
Retype new password:
```

确保用新密码更新`env.password`。退出服务器，返回本地终端，然后执行 Fabric:

```py
$ fab -f ./prod/fabfile.py start_provision
```

如果一切顺利，将会生成新的 SSH 密钥，并且会要求您创建一个密码(一定要这样做！):

```py
Generating public/private rsa key pair.
Enter passphrase (empty for no passphrase):
Enter same passphrase again:
```

将运行许多任务。创建`deployer`用户后，系统会提示您为该用户添加密码-

```py
[104.236.66.172] out: Changing password for user deployer.
```

-上传 SSH 密钥时，您必须输入:

```py
deployer@104.236.66.172s password:
```

该脚本成功退出后，您将无法再以 root 用户身份登录远程服务器。相反，您将只能使用非根用户`deployer`。

尝试一下:

```py
$ ssh root@<server-ip-address>
Permission denied (publickey,gssapi-keyex,gssapi-with-mic).
```

这是意料之中的。然后，当你跑的时候-

```py
$ ssh -i ./ssh-keys/104.236.66.172_prod_key deployer@104.236.66.172
```

-您应该可以正常登录:

```py
[deployer@fedora-512mb-nyc2-01 ~]$
```

[*Remove ads*](/account/join/)

## 易解引物

Ansible 是一个配置管理和供应工具，用于通过 SSH 自动执行部署任务。

您可以从您的 shell 远程启动针对应用服务器的[单个任务](http://docs.ansible.com/ansible/intro_adhoc.html),并随时执行任务。任务也可以组合成[剧本](http://docs.ansible.com/ansible/playbooks_intro.html)——多个*剧本*的集合，其中每个剧本定义了部署过程中需要的某些特定任务。它们在部署过程中针对应用服务器执行。剧本是在 [YAML](http://docs.ansible.com/ansible/YAMLSyntax.html) 写成的。

### 剧本

行动手册由如下模块化架构组成:

1.  [Hosts](http://docs.ansible.com/ansible/playbooks_intro.html#hosts-and-users) 指定所有需要协调的远程服务器的 IP 地址或域名。行动手册总是在目标主机组上运行。
2.  [角色](http://docs.ansible.com/ansible/playbooks_roles.html)分为子部分。让我们来看一些角色示例:
    *   任务是需要在部署过程中执行的多个任务的集合。
    *   当模块对远程服务器进行更改时，处理程序提供了一种触发一组操作的方式(最好被认为是钩子)。
    *   在这种情况下，模板通常用于指定一些模块相关的配置文件——比如 nginx。
3.  [变量](http://docs.ansible.com/ansible/playbooks_variables.html)只是一个键值对列表，其中每个键(变量)都映射到一个值。这些变量可以在行动手册中作为占位符使用。

### 剧本样本

现在让我们来看一个单文件行动手册示例:

```py
--- # My Ansible playbook for configuring Nginx -  hosts:  all vars: http_port:  80 app_name:  django_bootstrap tasks: -  name:  Install nginx dnf:  name=nginx state=latest -  name:  Create nginx config file template:  src=django_bootstrap.conf dest=/etc/nginx/conf.d/{{ app_name }}.conf become:  yes notify: -  restart nginx handlers: -  name:  Restart nginx service:  name=nginx state=restarted enabled=yes become:  yes
```

在这里，我们定义了:

*   Hosts as `hosts: all`，这表示剧本将在 *[清单/hosts](http://docs.ansible.com/ansible/intro_inventory.html)* 文件中列出的所有服务器上运行
*   模板中使用的变量`http_port: 80`和`app_name: django_bootstrap`
*   任务为了安装 nginx，设置 nginx 配置(`become`表示我们需要 admin 权限)，并触发重启处理程序
*   处理程序来重新启动 nginx 服务

## 剧本设置

现在让我们为姜戈建立一个剧本。将一个 *deploy.yml* 文件添加到“prod”目录:

```py
## # This playbook deploys the whole app stack ## -  name:  apply common configuration to server hosts:  all user:  deployer roles: -  common
```

上面的代码片段将负责的主机、用户和角色结合在一起。

### 主机

将一个*主机*(纯文本格式)文件添加到“prod”目录中，并在它们各自的角色名称下列出服务器。我们在这里配置一台服务器:

```py
[common]
<server-ip-address>
```

在上面的代码片段中，`common`指的是角色名。在角色下，我们有一个需要配置的 IP 地址列表。确保添加您的远程服务器的 IP 地址来代替`<server-ip-address>`。

### 变量

现在我们定义角色将使用的变量。在“prod”中添加一个名为“group_vars”的新文件夹，然后在该文件夹中创建一个名为 *all* (纯文本格式)的新文件。在这里，首先指定以下变量:

```py
# App Name app_name:  django_bootstrap # Deployer User and Groups deployer_user:  deployer deployer_group:  deployers # SSH Keys Directory ssh_dir:  <path-to-your-ssh-keys>
```

确保更新`<path-to-your-ssh-keys>`。要获得正确的路径，请在项目根目录下运行:

```py
$ cd ssh-keys
$ pwd
/Users/michael.herman/repos/realpython/automated-deployments/ssh-keys
```

有了这些文件，我们现在就可以将我们的部署过程与需要在服务器上执行的所有角色协调起来了。

[*Remove ads*](/account/join/)

## 剧本角色

同样，行动手册只是不同行动的集合，所有这些行动都在特定的角色下运行。在“prod”中创建一个名为“roles”的新目录。

> 您是否捕捉到了 *deploy.yml* 文件中的角色名称？

然后在“roles”目录中添加一个名为“common”的新目录——角色。角色由“任务”、“处理程序”和“模板”组成。为每个目录添加一个新目录。

完成后，您的文件结构应该如下所示:

```py
├── prod
│   ├── deploy.yml
│   ├── fabfile.py
│   ├── group_vars
│   │   └── all
│   ├── hosts
│   └── roles
│       └── common
│           ├── handlers
│           ├── tasks
│           └── templates
└── ssh-keys
    ├── 104.236.66.172_prod_key
    ├── 104.236.66.172_prod_key.pub
    └── authorized_keys
```

所有的剧本都定义在一个“任务”目录中，从一个 *main.yml* 文件开始。这个文件作为*所有*剧本任务的入口点。它只是需要按顺序执行的多个 YAML 文件的列表。

现在在“tasks”目录下创建该文件，然后向其中添加以下内容:

```py
## # Configure the server for the Django app ## -  include:  01_server.yml -  include:  02_git.yml -  include:  03_postgres.yml -  include:  04_dependencies.yml -  include:  05_migrations.yml -  include:  06_nginx.yml -  include:  07_gunicorn.yml -  include:  08_systemd.yml # - include: 09_fix-502.yml
```

现在，让我们创建每个任务。一定要为每个任务在“tasks”目录中添加一个新文件，并为每个文件添加相应的代码。如果你迷路了，参考[回购](https://github.com/realpython/automated-deployments)。

### 01_server.yml*

```py
## # Update the DNF package cache and install packages as a root user ## -  name:  Install required packages dnf:  name={{item}} state=latest become:  yes with_items: -  vim -  fail2ban -  python3-devel -  python-virtualenv -  python3-virtualenv -  python-devel -  gcc -  libselinux-python -  redhat-rpm-config -  libtiff-devel -  libjpeg-devel -  libzip-devel -  freetype-devel -  lcms2-devel -  libwebp-devel -  tcl-devel -  tk-devel -  policycoreutils-devel
```

在这里，我们列出了所有需要安装的系统软件包。

### 02_git.yml*

```py
## # Clone and pull the repo ## -  name:  Set up git configuration dnf:  name=git state=latest become:  yes -  name:  Clone or pull the latest code git:  repo={{ code_repository_url }} dest={{ app_dir }}
```

将以下变量添加到 *group_vars/all* 文件中:

```py
# Github Code's Repo URL code_repository_url:  https://github.com/realpython/django-bootstrap # App Directory app_dir:  /home/{{ deployer_user }}/{{app_name}}
```

确保派生然后克隆 [django-bootstrap](https://github.com/realpython/django-bootstrap) repo，然后将`code_repository_url`变量更新为您的派生的 URL。

### 03 _ posters . yml*

```py
## # Set up and configure postgres ## -  name:  Install and configure db dnf:  name={{item}} state=latest become:  yes with_items: -  postgresql-server -  postgresql-contrib -  postgresql-devel -  python-psycopg2 -  name:  Run initdb command raw:  postgresql-setup initdb become:  yes -  name:  Start and enable postgres service:  name=postgresql enabled=yes state=started become:  yes -  name:  Create database postgresql_db:  name={{ app_name }} become_user:  postgres become:  yes -  name:  Configure a new postgresql user postgresql_user:  db={{ app_name }} name={{ db_user }} password={{ db_password }} priv=ALL role_attr_flags=NOSUPERUSER become:  yes become_user:  postgres notify: -  restart postgres
```

使用行动手册所需的数据库配置更新 *group_vars/all* :

```py
# DB Configuration db_url:  postgresql://{{deployer_user}}:{{db_password}}@localhost/{{app_name}} db_password:  thisissomeseucrepassword db_name:  "{{  app_name  }}" db_user:  "{{  deployer_user  }}"
```

用安全密码更新`db_password`变量。

您是否注意到我们在 *main.yml* 文件中重新启动了 postgres 服务，以便在配置数据库后应用更改？这是我们的第一个负责人。在“handlers”文件夹中创建一个名为 *main.yml* 的新文件，然后添加以下内容:

```py
-  name:  restart postgres service:  name=postgresql state=restarted become:  yes
```

[*Remove ads*](/account/join/)

### **04_dependencies.yml**

```py
## # Set up all the dependencies in a virtualenv required by the Django app ## -  name:  Create a virtualenv directory file:  path={{ venv_dir }} state=directory -  name:  Install dependencies pip:  requirements={{ app_dir }}/requirements.txt virtualenv={{ venv_dir }} virtualenv_python=python3.5 -  name:  Create the .env file for running ad-hoc python commands in our virtualenv template:  src=env.j2 dest={{ app_dir }}/.env become:  yes
```

更新 *group_vars/all* 如下:

```py
# Application Dependencies Setup
venv_dir: '/home/{{ deployer_user }}/envs/{{ app_name }}'
venv_python: '{{ venv_dir }}/bin/python3.5'
```

将名为 *env.j2* 的模板添加到“templates”文件夹中，并添加以下环境变量:

```py
#!/bin/bash
export DEBUG="True"
export DATABASE_URL="postgresql://deployer:thisissomeseucrepassword@localhost/django_bootstrap"
export DJANGO_SECRET_KEY="changeme"
export DJANGO_SETTINGS_MODULE="config.settings.production"
```

> 非常小心环境变量及其在 *env.j2* 中的值，因为它们用于启动和运行 Django 项目。

### 05 _ migration . yml*

```py
## # Run db migrations and get all static files ## -  name:  Make migrations shell:  ".  {{  app_dir  }}/.env;  {{  venv_python  }}  {{  app_dir  }}/manage.py  makemigrations  " become:  yes -  name:  Migrate database django_manage:  app_path={{ app_dir }} command=migrate virtualenv={{ venv_dir }} -  name:  Get all static files django_manage:  app_path={{ app_dir }} command=collectstatic virtualenv={{ venv_dir }} become:  yes
```

### 06_nginx.yml*

```py
## # Configure nginx web server ## -  name:  Set up nginx config dnf:  name=nginx state=latest become:  yes -  name:  Write nginx conf file template:  src=django_bootstrap.conf dest=/etc/nginx/conf.d/{{ app_name }}.conf become:  yes notify: -  restart nginx
```

将以下变量添加到 *group_vars/all* :

```py
# Remote Server Details server_ip:  <remote-server-ip> wsgi_server_port:  8000
```

别忘了更新`<remote-server-ip>`。然后将处理程序添加到 *handlers/main.yml* :

```py
-  name:  restart nginx service:  name=nginx state=restarted enabled=yes become:  yes
```

然后我们需要添加 *django_bootstrap.conf* 模板。在“templates”目录中创建该文件，然后添加代码:

```py
upstream app_server {
    server 127.0.0.1:{{ wsgi_server_port }} fail_timeout=0;
}

server {
    listen 80;
    server_name {{ server_ip }};
    access_log /var/log/nginx/{{ app_name }}-access.log;
    error_log /var/log/nginx/{{ app_name }}-error.log info;

    keepalive_timeout 5;

    # path for staticfiles
    location /static {
            autoindex on;
            alias {{ app_dir }}/staticfiles/;
    }

    location / {
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Host $http_host;
        proxy_redirect off;

        if (!-f $request_filename) {
            proxy_pass http://app_server;
            break;
        }
    }
}
```

### 07 _ guni corn . yml*T2】

```py
## # Set up Gunicorn and configure systemd to execute gunicorn_start script ## -  name:  Create a deploy directory file:  path={{ deploy_dir }} state=directory become:  yes -  name:  Create the gunicorn_start script for running our app from systemd service template:  src=gunicorn_start dest={{ deploy_dir }}/gunicorn_start become:  yes -  name:  Make the gunicorn_start script executable raw:  cd {{ deploy_dir }}; chmod +x gunicorn_start become:  yes
```

向 *groups_vars/all* 添加更多变量:

```py
# Deploy Dir in App Directory deploy_dir:  '{{  app_dir  }}/deploy' # WSGI Vars django_wsgi_module:  config.wsgi django_settings_module:  config.settings.production django_secret_key:  'changeme' database_url:  '{{  db_url  }}'
```

添加 *gunicorn_start* 模板:

```py
#!/bin/bash

### Define script variables

# Name of the app
NAME='{{ app_name }}'
# Path to virtualenv
VIRTUALENV='{{ venv_dir }}'
# Django Project Directory
DJANGODIR='{{ app_dir }}'
# The user to run as
USER={{ deployer_user }}
# The group to run as
GROUP={{deployer_group }}
# Number of worker processes Gunicorn should spawn
NUM_WORKERS=3
# Settings file that Gunicorn should use
DJANGO_SETTINGS_MODULE={{django_settings_module}}
# WSGI module name
DJANGO_WSGI_MODULE={{ django_wsgi_module }}

### Activate virtualenv and create environment variables

echo "Starting $NAME as `whoami`"
# Activate the virtual environment
cd $VIRTUALENV
source bin/activate
cd $DJANGODIR
# Defining the Environment Variables
export DJANGO_SECRET_KEY='{{ django_secret_key }}'
export DATABASE_URL='{{ db_url }}'
export DJANGO_SETTINGS_MODULE=$DJANGO_SETTINGS_MODULE
export PYTHONPATH=$DJANGODIR:$PYTHONPATH

### Start Gunicorn

exec gunicorn ${DJANGO_WSGI_MODULE}:application \
        --name $NAME \
        --workers $NUM_WORKERS \
        --user=$USER --group=$GROUP \
        --log-level=debug \
        --bind=127.0.0.1:8000
```

[*Remove ads*](/account/join/)

### **08_systemd.yml**

```py
## # Set up systemd for executing gunicorn_start script ## -  name:  write a systemd service file template:  src=django-bootstrap.service dest=/etc/systemd/system become:  yes notify: -  restart app -  restart nginx
```

添加模板-*django-bootstrap . service*:

```py
#!/bin/sh

[Unit]
Description=Django Web App
After=network.target

[Service]
PIDFile=/var/run/djangoBootstrap.pid
User={{ deployer_user }}
Group={{ deployer_group }}
ExecStart=/bin/sh {{ deploy_dir }}/gunicorn_start
Restart=on-abort

[Install]
WantedBy=multi-user.target
```

将以下内容添加到处理程序中:

```py
-  name:  restart app service:  name=django-bootstrap state=restarted enabled=yes become:  yes
```

### 09 _ fix-502 . yml*T2】

```py
## # Fix the 502 nginx error post deployment # -  name:  Fix nginx 502 error raw:  cd ~; cat /var/log/audit/audit.log | grep nginx | grep denied | audit2allow -M mynginx; semodule -i mynginx.pp become:  yes
```

## 健全性检查(最终)

激活 virtualenv 后，[在本地安装 Ansible】:](https://realpython.com/what-is-pip/)

```py
$ pip install ansible==2.1.3
```

在项目根目录下创建一个名为 *deploy_prod.sh* 的新文件来运行剧本，确保更新`<server-ip>`:

```py
#!/bin/bash

ansible-playbook ./prod/deploy.yml --private-key=./ssh_keys<server-ip>_prod_key -K -u deployer -i ./prod/hosts -vvv
```

然后运行以下命令来执行行动手册:

```py
$ sh deploy_prod.sh
```

如果出现任何错误，请向终端咨询如何纠正错误的信息。修复后，再次执行部署脚本。当脚本完成后，访问服务器的 IP 地址，以验证您的 Django web 应用程序正在运行！

[![](img/6dc3844ef04fdb417afe67ad7d9cde70.png)](https://files.realpython.com/media/django-live.cab4b480c1de.png)

如果您看到 502 错误，请确保取消注释*prod/roles/common/tasks/main . yml*中的这一行，这表明 nginx 和 Gunicorn 之间的通信存在问题:

```py
# - include: 09_fix-502.yml
```

然后再次执行剧本。

> 如果你多次执行剧本，确保注释掉在 *03_postgres.yml* 中的`Run initdb command`，因为它只需要运行一次。否则，它将在尝试重新初始化数据库服务器时抛出错误。

[*Remove ads*](/account/join/)

## 结论

这篇文章提供了如何用 Fabric 和 Ansible 自动配置服务器的基本理解。Ansible Playbooks 特别强大，因为您可以通过 YAML 文件在服务器上自动化几乎任何任务。希望您现在可以开始编写自己的行动手册，甚至在工作场所使用它们来配置生产就绪型服务器。

请在下面添加问题和评论。完整的代码可以在[自动化部署](https://github.com/realpython/automated-deployments)仓库中找到。*******
# REPL 环境

# Python 的 REPL 环境

安装完 Python 之后都会提供一个 REPL－Read-Eval-Print Loop，不过 Python 默认的 REPL 环境实在是简陋， 很多开发者就开发了更加强大的 REPL 工具，包括控制台和基于 web 的，其中比较有名的有：

+   IPython（IPython Notebook）

+   BPython

+   ptPython

# BPython

# BPython

相比 IPython，BPython 并不算得上功能强大，它的开发初衷仅在于部分增强原始 REPL 的交互性， 并没有向 IDE 发展的意思，以保持轻量、简单为目的。 根据[官网](http://www.bpython-interpreter.org/about.html)介绍，它有这些特性：

## 行内的语法高亮

## 自动提示

## 函数参数提示

## 时光倒流（Rewind）

这里需要注意一下，BPython 的 rewind 并不是撤销，而是将之前执行过的命令在全新的环境中重新执行到上一步。

## 历史输出

退出 BPython 时会将屏幕中所有的信息输出到 stdout，也就是说你可以在 shell 中看到之前看过的一切。

## 支持 Python 3

## 自定义配置

```
bpython --config=/path/to/a/bpython/config/file 
```

# IPython

# IPython

IPython 是一个增强了交互能力的 Python REPL 环境，IPython v1-3 版本安装同时会安装 IPython notebook，相当于一个网页版的 REPL 环境。 相对于原生的 Python REPL，IPython 主要提供了 Tab 自动补全、shell 命令交互、 快速文档查询等功能。 除了在 shell 中使用外，Django shell 以及 PyCharm IDE 都提供了 IPython 的支持—— 安装了 IPython 后会选择其作为默认 python shell。

## 启动 & 退出

启动 IPython：`ipython`

启动 IPython notebook：`ipython notebook`

启动 IPython Qt shell：`ipython qtshell`

退出：双击 Ctrl-D 或者 `exit` 或者 `quit` 或者 `exit()`。

## 特性

### Tab 补全

原生的 Python shell 可以通过 `rlcompleter` 和 `readline` 库获得 Tab 补全能力：

```
import rlcompleter, readline
readline.parse_and_bind('tab: complete') 
```

IPython 的支持更强大一些。不过 IPython 的自动补全依旧是循环式的，需要菜单式的可以试试 ptpython。

### 魔法指令（Magic Command）

魔法指令是 IPython 定义的命令，以提供 IPython 环境下与环境交互的能力，通常对应于 shell 命令。 你可以在 IPython 控制台中输入 `ls` 或者 `%ls` 来查看当前目录文件。

魔法指令通常以 `%` 开始，对于非 `%` 开始的魔法指令，IPython 会首先判断它是不是 Python 对象，如是对象则优先作为 Python 对象处理。

更全面的 magic 命令列表和功能介绍见[这里](https://damontallen.github.io/IPython-quick-ref-sheets/)。

魔法指令中还包括了很多 Shell 命令，但并不是全部，IPython 也支持直接执行 Shell 命令，只需要以感叹号 `!` 起始，例如：`!ls`。

### Plot

常见用法是配合 mathplotlib 等 plot 库和 IPython 插件，在终端、Qt console 或者 Notebook 中展示图表。

### 内省

#### 函数内省

IPython 提供了 `?` 和 `??` 命令用于内省，`?` 会显示函数简介，`??` 会连源代码一起显示 （如果可用的话）。

#### 输入历史

`hist` 魔法指令可以用来输出输入历史：

```
In [7]: hist
1: a = 1
2: b = 2
3: c = 3
4: d = {}
5: e = []
6: for i in range(20):
    e.append(i)
    d[i] = b 
```

要去掉历史记录中的序号，使用命令 `hist -n`：

```
In [8]: hist -n
a = 1
b = 2
c = 3
d = {}
e = []
for i in range(20):
    e.append(i)
    d[i] = b 
```

这样你就可以方便的将代码复制到一个文本编辑器中。

要在历史记录中搜索可以先输入一个匹配模型，然后按 `Ctrl-p` ，找到一个匹配后，继续按 `Ctrl-p` 会向后搜索再上一个匹配，`Ctrl-n` 则是向前搜索最近的匹配。

### 编辑

当在 Python 提示符下试验一个想法时，经常需要通过编辑器修改源代码（甚至是反复修改）。 在 IPython 下输入魔法指令 `edit` 就会根据环境变量 `$EDITOR` 调用相应的编辑器。 如果 `$EDITOR` 为空，则会调用 vi（Unix）或记事本（Windows）。 要回到 IPython 提示符，直接退出编辑器即可。

如果是保存并退出编辑器，输入编辑器的代码会在当前名字空间下被自动执行。 如果你不想这样，使用 `edit -x`。如果要再次编辑上次最后编辑的代码，使用 `edit -p`。 在上一个特性里，我提到使用 `hist -n` 可以很容易的将代码拷贝到编辑器。 一个更简单的方法是 `edit` 加 Python 列表的切片（slice）语法。假定 `hist` 输出如下:

```
In [29]: hist -n
a = 1
b = 2
c = 3
d = {}
e = []

for i in range(20):
    e.append(i)
    d[i] = b 
```

现在要将第 4、5、6 句代码导出到编辑器，只要输入：

```
edit 4:7 
```

### 交互调试

调试方面常用的魔术命令有 `ru`、`prun`、`time`、`timeit` 等等。更直接的方式是借助系统调试工具 pdb。

## 快捷键及配置

## 作为库使用

NotFinished，虽然 ipython -p pysh 提供了一个强大的 shell 替代品，但它缺少正确的 job 控制。在运行某个很耗时的任务时按下 Ctrl-z 将会停止 IPython session 而不是那个子进程。

## 问题和方法

虽然作为标准 Python Shell 的替代，IPython 总的来说很完美，但仍然存在一些问题。

### 粘贴代码缩进问题

默认情况下，IPython 会对粘贴的已排好缩进的代码重新缩进。例如下面的代码:

```
for i in range(10):
    for j in range(10):
        for k in range(10):
            pass 
```

会变成:

```
for i in range(10):
        for j in range(10):
                for k in range(10):
                        pass 
```

这是因为 `autoindent` 默认是启用状态，可以用 magic 命令 `autoindent` 来开关自动缩进， 就像在 Vim 中设置 `set paste` 一样。

或者，你也可以使用魔法指令 `%paste` 粘贴并执行剪贴板中的代码。

## Vim 模式

## Mac 下的快捷启动

IPython wiki 提供了一段 [AppleScript launcher](https://github.com/ipython/ipython/wiki/Cookbook:-Launching-IPython-on-OSX#using-iterm2-and-an-applescript-launcher) 脚本，把它写入 `~/bin/ipy` 在 Mac Spotlight 中输入 `ipy` 并回车就可以在新 iTerm 窗口中打开 IPython Shell。 下面是我稍微修改后的版本：

```
#!/usr/bin/osascript
-- Run ipython on a new iTerm
-- See http://www.iterm2.com/#/section/documentation/scripting

tell application "iTerm"
    activate
    set ipyterm to (make new terminal)
    tell ipyterm
        set ipysession to (make new session)
        tell ipysession
            set name to "IPython"
            exec command "${which ipython}"
        end tell
    end tell
end tell 
```

## IPython notebook

### 启动

在终端输入： `ipython notebook`

IPython notebook 的使用和 Vim 有些类似，分“命令模式”和“编辑模式”，切换方法同样为 `ESC` 键。 在命令模式下按 `h` 可以呼出帮助菜单，貌似不区分大小写。

# 设置 Django Shell

# Django Shell 增强

由于 Django 系统的特殊性，很难在 Shell 中直接导入 Django 应用，而只能使用它自己提供的 shell 命令。不过，并不是说就没有办法使用增强的第三方 Shell 了，[Django-Extensions](http://django-extensions.readthedocs.org/) 插件提供了切换默认 Shell 的能力。

## Django Extensions

[Django-Extensions](http://django-extensions.readthedocs.org/) 是一个 Django 第三方插件集，其中囊括了很多实用的 Django 插件， 当然也包括本文的主角 `shell_plus`。

## shell_plus

装上插件后只需要将 `django_extensions` 加入 `INSTALLED_APPS` 列表中即可使用 IPython 作为默认 Shell（如果已安装的话）。 需要注意的是，如果在 virtualenv 环境中使用 Django 的话，同样需要在 virtualenv 中安装 IPython。

如果你偏好的是 BPython 的话，[shell_plus](http://django-extensions.readthedocs.org/en/latest/shell_plus.html#interactive-python-shells) 同样提供了相关命令：`shell_plus`。使用方法：

```
./manage.py shell_plus --bpython 
```

该命令对 IPython 同样适用：

```
./manage.py shell_plus --ipython 
```

或者，有需要的话，也可以切换回原生 Python：

```
./manage.py shell_plus --plain 
```

默认 Shell 的选择顺序是 BPython、IPython、Python。

你同样可以在文件中配置默认 Shell：

```
SHELL_PLUS = "ipython" 
```

## 设置

默认情况下 shell_plus 会自动加载你的全部 Model，其中可能会出现重名或者其它问题， 你可以通过配置文件来修改它的加载机制。

注意：这里的设置只会作用于 shell_pull 而不会影响你的整改系统。

将名为 blog 的 app 中 的 Messages 模块以 blog_messages 名加载

```
SHELL_PLUS_MODEL_ALIASES = {'blog': {'Messages': 'blog_messages'},} 
```

不加载 app 'sites' 中的全部 model，以及 app 'blog' 中的 'pictures' Model。

```
SHELL_PLUS_DONT_LOAD = ['sites', 'blog.pictures'] 
```

`SHELL_PLUS_MODEL_ALIASES` 和 `SHELL_PLUS_DONT_LOAD` 可以同时使用。

也可以像这样在运行 manage.py 时再设定不自动加载的模块：

```
./manage.py shell_plus --dont-load app1 --dont-load app2.module1 
```

命令行和配置文件中的设置会合并后再生效。

## IPython Notebook

shell_plus 同时还支持 [IPython Notebook]：

```
./manage.py shell_plus --notebook 
```

Notebook 相关的设置参数有两个：`NOTEBOOK_ARGUMENTS` 和 `IPYTHON_ARGUMENTS`：

`NOTEBOOK_ARGUMENTS` 用于配置 IPython Notebook：

```
ipython notebook -h 
```

```
NOTEBOOK_ARGUMENTS = [
    '--ip=x.x.x.x',
    '--port=xx',
] 
```

`IPYTHON_ARGUMENTS` 用于设置 IPython：

```
ipython -h 
```

使用 IPython Notebook 时同样会自动加载 Django settings 模块和数据库 Model。 其实现依赖于定制的 IPython 插件，Django Extensions 默认会向 IPython Notebook 发送命令 `--ext django_extensions.management.notebook_extension` 开启该功能， 你可以在 Django 设置中修改 `IPYTHON_ARGUMENTS` 参数覆盖该设置：

```
IPYTHON_ARGUMENTS = [
    '--ext', 'django_extensions.management.notebook_extension',
    '--ext', 'myproject.notebook_extension',
    '--debug',
] 
```

开启自动加载可以通过引用 Django Extensions 默认 Notebook，或者复制它的自动加载代码到你自己的扩展中。

IPython Notebook 扩展目前并不支持 `--dont-load` 选项。

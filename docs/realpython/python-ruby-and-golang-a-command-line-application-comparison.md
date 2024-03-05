# Python、Ruby 和 Golang:命令行应用程序比较

> 原文：<https://realpython.com/python-ruby-and-golang-a-command-line-application-comparison/>

2014 年末，我开发了一个名为 [pymr](https://github.com/kpurdon/pymr) 的工具。我最近觉得有必要学习 golang 并更新我的 ruby 知识，所以我决定重温 pymr 的想法，并用多种语言构建它。**在这篇文章中，我将分解“Mr”(Merr)应用程序(pymr，gomr，rumr)并展示每种语言中特定部分的实现。我会在最后提供一个总体的个人偏好，但会把个人作品的比较留给你。**

> 对于那些想直接跳到代码的人，请查看 [repo](https://github.com/realpython/python-ruby-golang) 。

## 应用程序结构

这个应用程序的基本思想是，您有一些相关的目录，您希望在这些目录上执行一个命令。“mr”工具提供了一种注册目录的方法，以及一种在已注册目录组上运行命令的方法。该应用程序具有以下组件:

*   命令行界面
*   注册命令(用给定的标签写一个文件)
*   运行命令(在注册的目录上运行给定的命令)

[*Remove ads*](/account/join/)

## 命令行界面

“mr”工具的命令行界面是:

```py
$ pymr --help
Usage: pymr [OPTIONS] COMMAND [ARGS]...

Options:
 --help  Show this message and exit.

Commands:
 register  register a directory
 run       run a given command in matching...
```

为了比较命令行界面的构建，我们来看看每种语言中的 register 命令。

**Python (pymr)**

为了用 python 构建命令行界面，我选择使用 [click](http://click.pocoo.org/4/) 包。

```py
@pymr.command()
@click.option('--directory', '-d', default='./')
@click.option('--tag', '-t', multiple=True)
@click.option('--append', is_flag=True)
def register(directory, tag, append):
    ...
```

**红宝石(鲁姆)**

为了在 ruby 中构建命令行界面，我选择了使用 thor gem。

```py
desc 'register', 'Register a directory'
method_option :directory,
              aliases: '-d',
              type: :string,
              default: './',
              desc: 'Directory to register'
method_option :tag,
              aliases: '-t',
              type: :array,
              default: 'default',
              desc: 'Tag/s to register'
method_option :append,
              type: :boolean,
              desc: 'Append given tags to any existing tags?'
def register
  ...
```

**戈朗(gomr)**

为了在 Golang 中构建命令行界面，我选择使用 [cli.go](https://github.com/codegangsta/cli) 包。

```py
app.Commands  =  []cli.Command{ { Name:  "register", Usage:  "register a directory", Action:  register, Flags:  []cli.Flag{ cli.StringFlag{ Name:  "directory, d", Value:  "./", Usage:  "directory to tag", }, cli.StringFlag{ Name:  "tag, t", Value:  "default", Usage:  "tag to add for directory", }, cli.BoolFlag{ Name:  "append", Usage:  "append the tag to an existing registered directory", }, }, }, }
```

## 注册

注册逻辑如下:

1.  如果用户要求`--append`读取存在的`.[py|ru|go]mr`文件。
2.  将现有标签与给定标签合并。
3.  用新标签写一个新的`.[...]mr`文件。

这可以分解成几个小任务，我们可以用每种语言进行比较:

*   搜索和读取文件。
*   合并两个项目(仅保留唯一的集合)
*   编写文件

### 文件搜索

**Python (pymr)**

对于 python 来说，这涉及到 [os](https://docs.python.org/2/library/os.html) 模块。

```py
pymr_file = os.path.join(directory, '.pymr')
if os.path.exists(pymr_file):
    # ...
```

**红宝石(鲁姆)**

对于 ruby 来说，这涉及到了[文件](http://ruby-doc.org/core-1.9.3/File.html)类。

```py
rumr_file = File.join(directory, '.rumr')
if File.exist?(rumr_file)
    # ...
```

**戈朗(gomr)**

对于 golang，这涉及到[路径](http://golang.org/pkg/path/)包。

```py
fn  :=  path.Join(directory,  ".gomr") if  _,  err  :=  os.Stat(fn);  err  ==  nil  { // ... }
```

[*Remove ads*](/account/join/)

### 唯一合并

**Python (pymr)**

对于 python，这涉及到使用一个[集合](https://docs.python.org/2/library/sets.html)。

```py
# new_tags and cur_tags are tuples
new_tags = tuple(set(new_tags + cur_tags))
```

**红宝石(鲁姆)**

对于 ruby 来说，这涉及到[的使用。uniq](http://ruby-doc.org/core-2.2.0/Array.html#method-i-uniq) 数组方法。

```py
# Edited (5/31)
# old method:
#  new_tags = (new_tags + cur_tags).uniq

# new_tags and cur_tags are arrays
new_tags |= cur_tags
```

**戈朗(gomr)**

对于 golang，这涉及到自定义函数的使用。

```py
func  AppendIfMissing(slice  []string,  i  string)  []string  { for  _,  ele  :=  range  slice  { if  ele  ==  i  { return  slice } } return  append(slice,  i) } for  _,  tag  :=  range  strings.Split(curTags,  ",")  { newTags  =  AppendIfMissing(newTags,  tag) }
```

### 文件读/写

我试图选择每种语言中最简单的文件格式。

**Python (pymr)**

对于 python 来说，这涉及到使用 [`pickle`模块](https://realpython.com/python-pickle-module/)。

```py
# read
cur_tags = pickle.load(open(pymr_file))

# write
pickle.dump(new_tags, open(pymr_file, 'wb'))
```

**红宝石(鲁姆)**

对于 ruby 来说，这涉及到使用 [YAML](http://ruby-doc.org/stdlib-2.2.1/libdoc/yaml/rdoc/YAML.html) 模块。

```py
# read
cur_tags = YAML.load_file(rumr_file)

# write
# Edited (5/31)
# old method:
#  File.open(rumr_file, 'w') { |f| f.write new_tags.to_yaml }
IO.write(rumr_file, new_tags.to_yaml)
```

**戈朗(gomr)**

对于 golang，这涉及到使用[配置](https://github.com/robfig/config)包。

```py
// read cfg,  _  :=  config.ReadDefault(".gomr") // write outCfg.WriteFile(fn,  0644,  "gomr configuration file")
```

[*Remove ads*](/account/join/)

## 运行(命令执行)

运行逻辑如下:

1.  递归地从给定的基本路径开始搜索`.[...]mr`文件
2.  加载一个找到的文件，看看给定的标签是否在其中
3.  在匹配文件的目录中调用给定的命令。

这可以分解成几个小任务，我们可以用每种语言进行比较:

*   递归目录搜索
*   字符串比较
*   调用 Shell 命令

### 递归目录搜索

**Python (pymr)**

对于 python 来说，这涉及到 [os](https://docs.python.org/2/library/os.html) 模块和 [fnmatch](https://docs.python.org/2/library/fnmatch.html) 模块。

```py
for root, _, fns in os.walk(basepath):
    for fn in fnmatch.filter(fns, '.pymr'):
        # ...
```

**红宝石(鲁姆)**

对于 ruby，这涉及到[查找](http://ruby-doc.org/stdlib-2.2.0/libdoc/find/rdoc/Find.html)和[文件](http://ruby-doc.org/core-2.2.0/File.html)类。

```py
# Edited (5/31)
# old method:
#  Find.find(basepath) do |path|
#        next unless File.basename(path) == '.rumr'
Dir[File.join(options[:basepath], '**/.rumr')].each do |path|
    # ...
```

**戈朗(gomr)**

对于 golang，这需要 [filepath](http://golang.org/pkg/path/filepath/) 包和一个自定义回调函数。

```py
func  RunGomr(ctx  *cli.Context)  filepath.WalkFunc  { return  func(path  string,  f  os.FileInfo,  err  error)  error  { // ... if  strings.Contains(path,  ".gomr")  { // ... } } } filepath.Walk(root,  RunGomr(ctx))
```

### 字符串比较

**Python (pymr)**

对于这个任务，python 中不需要任何额外的东西。

```py
if tag in cur_tags:
    # ...
```

**红宝石(鲁姆)**

在 ruby 中，这个任务不需要额外的东西。

```py
if cur_tags.include? tag
    # ...
```

**戈朗(gomr)**

对于 golang，这需要[字符串](http://golang.org/pkg/strings/)包。

```py
if  strings.Contains(cur_tags,  tag)  { // ... }
```

[*Remove ads*](/account/join/)

### 调用外壳命令

**Python (pymr)**

对于 python，这需要 [os](https://docs.python.org/2/library/os.html) 模块和[子流程](https://realpython.com/python-subprocess/)模块。

```py
os.chdir(root)
subprocess.call(command, shell=True)
```

**红宝石(鲁姆)**

对于 ruby，这涉及到[内核](http://ruby-doc.org/core-2.2.0/Kernel.html#method-i-system)模块和反勾号语法。

```py
# Edited (5/31)
# old method
#  puts `bash -c "cd #{base_path} && #{command}"`
Dir.chdir(File.dirname(path)) { puts `#{command}` }
```

**戈朗(gomr)**

对于 golang 来说，这涉及到 [os](https://golang.org/pkg/os/) 包和 [os/exec](https://golang.org/pkg/os/exec/) 包。

```py
os.Chdir(filepath.Dir(path)) cmd  :=  exec.Command("bash",  "-c",  command) stdout,  err  :=  cmd.Output()
```

## 包装

该工具的理想分发模式是通过一个包。然后用户可以安装它`tool install [pymr,rumr,gomr]`,并在系统路径上执行一个新命令。我不想在这里介绍打包系统，我只想展示每种语言所需的基本配置文件。

**Python (pymr)**

对于 python 来说，需要一个`setup.py`。一旦创建并上传了包，就可以用`pip install pymr`进行安装。

```py
from setuptools import setup, find_packages

classifiers = [
    'Environment :: Console',
    'Operating System :: OS Independent',
    'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
    'Intended Audience :: Developers',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7'
]

setuptools_kwargs = {
    'install_requires': [
        'click>=4,<5'
    ],
    'entry_points': {
        'console_scripts': [
            'pymr = pymr.pymr:pymr',
        ]
    }
}

setup(
    name='pymr',
    description='A tool for executing ANY command in a set of tagged directories.',
    author='Kyle W Purdon',
    author_email='kylepurdon@gmail.com',
    url='https://github.com/kpurdon/pymr',
    download_url='https://github.com/kpurdon/pymr',
    version='2.0.1',
    packages=find_packages(),
    classifiers=classifiers,
    **setuptools_kwargs
)
```

**红宝石(鲁姆)**

对于 ruby，需要一个`rumr.gemspec`。一旦宝石被创建并上传，就可以安装`gem install rumr`。

```py
Gem::Specification.new do |s|
  s.name        = 'rumr'
  s.version     = '1.0.0'
  s.summary     = 'Run system commands in sets' \
                  ' of registered and tagged directories.'
  s.description = '[Ru]by [m]ulti-[r]epository Tool'
  s.authors     = ['Kyle W. Purdon']
  s.email       = 'kylepurdon@gmail.com'
  s.files       = ['lib/rumr.rb']
  s.homepage    = 'https://github.com/kpurdon/rumr'
  s.license     = 'GPLv3'
  s.executables << 'rumr'
  s.add_dependency('thor', ['~>0.19.1'])
end
```

**戈朗(gomr)**

对于 golang 来说，源代码只是被编译成可以重新分发的二进制文件。不需要额外的文件，当前也没有要推送的包存储库。

[*Remove ads*](/account/join/)

## 结论

对于这个工具，Golang 感觉是个错误的选择。我不需要它有很高的性能，我也没有利用 Golang 提供的本地并发性。这就给我留下了 Ruby 和 Python。对于大约 80%的逻辑，我个人的偏好是两者之间的一个掷硬币。以下是我觉得用一种语言写更好的作品:

### 命令行接口声明

Python 是这里的赢家。 [click](http://click.pocoo.org/) 库装饰风格声明简洁明了。请记住，我只尝试了红宝石[雷神](https://github.com/erikhuda/thor)宝石，所以红宝石可能有更好的解决方案。这也不是对任何一种语言的评论，而是我在 python 中使用的 CLI 库是我的首选。

### 递归目录搜索

鲁比是这里的赢家。我发现使用 ruby 的`Find.find()`尤其是`next unless`语法，这一整段代码更加清晰易读。

### 包装

鲁比是这里的赢家。文件要简单得多，建造和推销宝石的过程也简单得多。捆绑器工具也使得在半隔离环境中安装变得轻而易举。

## 最终决定

由于打包和递归目录搜索偏好，我会选择 Ruby 作为这个应用程序的工具。然而，偏好上的差异是如此之小，以至于 Python 也非常适合。然而，Golang 并不是这里的正确工具。

> 这篇文章最初发表在凯尔的个人博客上，并在 Reddit 上引起了热烈的讨论。*****
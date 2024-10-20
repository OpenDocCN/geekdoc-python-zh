# 如何在 Python 中递归复制文件夹(目录)

> 原文：<https://www.pythoncentral.io/how-to-recursively-copy-a-directory-folder-in-python/>

尝试过用 Python 复制目录/文件夹吗？失败过吗？没关系。再试一次！

如果你还没有读过，看看文章[如何用 shutil](https://www.pythoncentral.io/how-to-copy-a-file-in-python-with-shutil/ "How to Copy a File in Python with shutil") 在 Python 中复制一个文件，获得如何用`shutil`复制文件的解释。

## **用 Python 递归复制文件的目录/文件夹**

在上面提到的文章中，我们看到了如何用 Python 复制单个文件。我想你会同意，更有用的是将整个目录复制到其他目录的能力。

Python 的`shutil`模块又一次拯救了我们，它有一个名为`copytree`的函数来做我们想做的事情。

我们不需要如此大的改变，所以我将稍微修改上面的代码，以便能够复制目录，如下所示:

```py

import shutil
def copyDirectory(src，dest):
try:
shutil . copy tree(src，dest) 
 #目录相同
除了 shutil。错误为 e: 
打印('目录未复制。错误:%s' % e) 
 #任何表示目录不存在的错误
除了 OSError as e: 
 print('未复制目录。错误:%s' % e) 

```

好的，很好！几乎和我们复制单个文件的函数一样，但是它只允许复制目录而不是文件，抛出的异常有点不同。太棒了。

### **用 Python 复制文件和目录**

但是如果我们想要一个可靠的函数来复制文件和目录呢？没问题！我们可以写一个简单的，稍微复杂一点的函数。

观察:

```py

import errno
def copy(src，dest):
try:
shutil . copy tree(src，dest) 
除 OSError as e: 
 #如果错误是因为源不是目录引起的
如果 e.errno == errno。ENOTDIR: 
 shutil.copy(src，dest) 
 else: 
 print('目录未复制。错误:%s' % e) 

```

这个函数将复制文件和目录。首先，我们将我们的`copytree`函数放在一个`try`块中，以捕捉任何讨厌的异常。如果我们的异常是由于源目录/文件夹实际上是一个文件引起的，那么我们就复制这个文件。简单的东西。

需要注意的是，实际上不可能复制一个目录。您需要递归地遍历目录并基于旧目录创建目录结构，然后将每个子目录中的文件复制到目的地的正确目录中。如果你想自己实现，请看本文中关于如何用 Python 递归遍历目录的例子。

### **忽略文件和目录**

函数`shutil.copytree`接受一个参数，该参数允许您指定一个函数来返回应该被忽略的目录或文件的列表。
这样做的一个简单示例函数如下:
【python】
def ignore _ function(ignore):
def _ ignore _(path，names):
ignored _ names =[]
if ignore in names:
ignored _ names . append(ignore)
return set(ignored _ names)
return _ ignore _

好的，这个函数是做什么的？嗯，我们指定某种文件或目录名作为参数`ignore`，它充当`names`的过滤器。如果`ignore`在`names`中，那么我们将它添加到一个`ignored_names`列表中，该列表向`copytree`指定跳过哪些文件或目录。

我们将如何使用这个函数？请参见下面我们修改后的复制功能:

```py

def copy(src, dest):

try:

shutil.copytree(src, dest, ignore=ignore_function('specificfile.file'))

except OSError as e:

# If the error was caused because the source wasn't a directory

if e.errno == errno.ENOTDIR:

shutil.copy(src, dest)

else:

print('Directory not copied. Error: %s' % e)

```

但是如果我们想过滤掉不止一个文件呢？幸运的是，`shutil`再次为我们提供了掩护！它有一个名为`shutil.ignore_patterns`的功能，允许我们指定 glob 模式来过滤掉文件和目录。请看下面我们再次修改的复制功能:

```py

def copy(src, dest):

try:

shutil.copytree(src, dest, ignore=ignore_patterns('*.py', '*.sh', 'specificfile.file'))

except OSError as e:

# If the error was caused because the source wasn't a directory

if e.errno == errno.ENOTDIR:

shutil.copy(src, dest)

else:

print('Directory not copied. Error: %s' % e)

```

这将忽略所有 Python 文件、shell 文件和我们自己的特定文件。`ignore_patterns`接受指定要忽略的模式的参数，并返回一个`copytree`可以理解的函数，很像我们的自定义函数`ignore_function`所做的，但更健壮。

这就是我要和你分享的全部！

和平。
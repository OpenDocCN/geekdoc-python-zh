# Python 中的递归文件和目录操作(第 3 部分)

> 原文：<https://www.pythoncentral.io/recursive-file-and-directory-manipulation-in-python-part-3/>

在本系列的第 2 部分中，我们扩展了我们的文件搜索脚本，使之能够在一棵树下搜索多个文件扩展名，并将结果(找到的匹配扩展名的文件的所有路径)写入日志文件。现在我们已经到了本系列的最后一部分，我们将在脚本中添加更多的功能(以函数的形式),以便能够移动、复制甚至删除搜索结果。

在查看移动/复制/删除函数之前，我们将首先获取将结果记录到文件中的子例程，并将其封装在一个函数中。下面是我们脚本的这一部分之前的样子:

```py

# The header in our logfile

loghead = 'Search log from filefind for files in {}\n\n'.format(

    os.path.realpath(topdir))
#我们的日志文件的正文
 logbody = ' '
#在结果
中循环查找找到的搜索结果:
 #用扩展名“% s”>>" % search result
log body+= ' \ n \ n % s \ n \ n“% ' \ n '连接找到的字典
 logbody += " < <结果的结果。加入(找到[搜索结果])
#将结果写入日志文件
，打开(日志名，' w ')作为日志文件:
 logfile.write(日志头)
 logfile.write(日志体)

```

为了将它放入函数定义中，我们简单地将定义语句放在上面，添加适当的参数，并相应地缩进其余部分(注意`found`如何变成`results`以及`logname`如何变成`logpath`):

```py

# Logging results for findfile

def logres(logpath, results):

    # The header in our logfile

    loghead = 'Search log from filefind for files in {}\n\n'.format(

        os.path.realpath(topdir))
#我们的日志文件的正文
 logbody = ' '
#在结果中循环搜索结果的结果
:
#将结果字典
 logbody += " < <结果中的结果与扩展名“% s”>>" % search result
log body+= ' \ n \ n % s \ n \ n“% ' \ n '。加入(结果[搜索结果])
#将结果写入日志文件
，打开(日志路径，' w ')作为日志文件:
 logfile.write(日志头)
 logfile.write(日志体)

```

出于错误报告的目的，我们还将定义一个小函数来写一个错误日志，稍后我们将看到原因。下面是这个函数，它接受 3 个参数，并将字符串列表写入错误日志:

```py

def logerr(logpath, errlist, act):

    loghead = 'List of files that produced errors when attempting to %s:\n\n' % act

    logbody = '\n'.join(errlist)
以 open(logpath，' w ')作为日志:
log . write(loghead+log body)

```

定义了两个日志记录函数后，我们现在将编写函数来对文件搜索的结果列表执行批处理操作。我们将首先看看如何对从原始位置找到的文件执行批量移动到目标目录。我们将用来实际移动文件的函数是来自`shutil`模块的`move`函数(想象一下:P)，所以我们想把这个语句添加到脚本的开头:

```py

# We'll use copy2 later

from shutil import move, copy2

```

对于我们的函数定义，不是直接作用于脚本中的`found`变量，而是让我们的方法接受一个结果字典并作用于它。如果我们需要的话，它还需要将它们移动到的目录的路径和错误日志路径。它还需要一个变量来存储错误(文件的路径字符串列表):

```py

# Moving results

def batchmove(results, dest, errlog=None):

    # List of results that produce errors

    errors = []

```

在写函数定义的其余部分之前，关于`move`函数有一些重要的注意事项——首先，这个函数将把源参数移动到相同类型的目的地。这意味着如果源路径是一个目录，目标也将是一个目录，对于文件也是如此。其次，如果目标存在并且是一个文件，那么源*必须*是一个文件，否则功能将失败。换句话说，如果目的地是一个目录，那么源(无论是文件还是目录)都会被移动到目的地目录中，但是如果目的地是一个文件，那么源可能只是一个文件(我们不能将目录移动到文件中)。也就是说，我们需要做的就是确保`batchmove`的`dest`参数是一个现有的目录，所以我们将在测试后使用 try 语句:

```py

    # Make sure dest is a directory!

    if os.path.isfile(dest):

        print("The move destination '%s' already exists as a file!" % dest)

        exit(input('Press enter to exit...'))

    elif not os.path.isdir(dest):

        try:

            os.mkdir(dest)

        except:

            print("Unable to create '%s' folder!" % dest)

            exit(input('Press enter to exit...'))

        else:

            print("'%s' folder created" % dest)

```

这样，如果移动失败，它会提醒用户并在退出前等待。检查完我们的目标目录后，我们可以添加函数的核心:遍历`results`并移动每个文件。循环如下:

```py

    # Loop through results, moving every file to dest directory

    for paths in results.values():

        for path in paths:

            path = os.path.realpath(path)

            try:

                # Move file to dest

                move(path, dest)

            except:

                errors.append(path)

    print('File move complete')

```

`results`中的键只是搜索的文件扩展名，所以只需要值，对于值中当前`paths`列表中的每个`path`,我们试图将它移动到我们的目的地。如果移动失败，`path`被添加到错误列表中。当循环完成时，一条消息被打印到标准输出。

循环完成后，我们希望记录使用`logerr`方法遇到的任何错误，如下所示:

```py

    # Log errors, if any

    if errlog and errors:

        logerr(errlog, errors, 'move')

        print("Check '%s' for errors." % errlog)

```

最后，我们将让脚本打印最后一条消息并退出:

```py

    exit(input('Press enter to exit...'))

```

综上所述，下面是我们的`batchmove`函数的样子:

```py

# Moving results

def batchmove(results, dest, errlog=None):
#产生错误的结果列表
错误= []
#确保 dest 是一个目录！
if OS . path . is file(dest):
print("移动目的地' %s '已经作为文件存在！"% dest) 
退出(输入('回车退出...'))
elif not OS . path . isdir(dest):
try:
OS . mkdir(dest)
except:
print("无法创建' %s '文件夹！"% dest) 
退出(输入('按回车键退出...'))
 else: 
 print("'%s '文件夹已创建" % dest ")
#循环遍历结果，将每个文件移动到目标目录
以获取结果中的路径。values(): 
以获取路径中的路径:
path = OS . path . real path(path)
try:
#将文件移动到目标目录
 move(path，dest)
except:
errors . append(path)
print('文件移动完成')
# log errors，if any
if errlog and errors:
logerr(errlog，errors，' move ')
print(" Check ' % s ' for errors。"% errlog) 
退出(输入('按回车键退出...')

```

现在我们有了`batchmove`函数，为了定义`batchcopy`函数，我们只需要将最内层循环的函数调用更改为`copy2`(以及相应的消息)，因此完整的定义如下所示:

```py

# Copying results

def batchcopy(results, dest, errlog=None):

    # List of results that produce errors

    errors = []
#确保 dest 是一个目录！
if OS . path . is file(dest):
print("复制目的地' %s '已经作为文件存在！"% dest) 
退出(输入('回车退出...'))
elif not OS . path . isdir(dest):
try:
OS . mkdir(dest)
except:
print("无法创建' %s '文件夹！"% dest) 
退出(输入('按回车键退出...'))
 else: 
 print("'%s '文件夹已创建" % dest ")
#循环遍历结果，将每个文件复制到目标目录
作为结果中的路径。values(): 
作为路径中的路径:
path = OS . path . real path(path)
try:
#将文件复制到目标目录
 copy2(path，dest)
except:
errors . append(path)
print('文件复制完成')
# Log errors，if any
if errlog and errors:
Log err(errlog，errors，' copy') 
 exit(input('按 enter 键退出... ')))

```

我们定义的这两个函数应该足够有用，不需要删除函数，但是如果我们想要一个，我们只需要从`batchmove`中移除`dest`检查，并将内部循环函数调用改为`os.remove`，如下所示:

```py

# Deleting results -- USE WITH CAUTION!

def batchdel(results, errlog=None):

    # List of results that produce errors

    errors = []
#循环遍历结果，删除每个文件！
for path in results . values():
for path in paths:
path = OS . path . real path(path)
try:
# Delete File
OS . remove(path)
except:
errors . append(path)
print('文件删除完成')
# Log errors，if any
if errlog and errors:
Log err(errlog，errors，' delete') 
 exit(input('按 enter 键退出... ')))

```

这只是为了展示我们如何实现删除子例程，但实际上并不推荐这样做，因为 Python 删除的任何文件都会被永久删除(不会被发送到回收站！).简单地将文件移动到带有`batchmove`的文件夹中并从那里删除它们会更安全，但是当然这取决于你的选择:)。既然我们已经定义了这些函数，那么我们需要做的就是在搜索循环之后使用`found`作为`results`参数来调用它们，并且相应地调用我们想要的日志文件的路径，所以即使我们不知道文件在哪里，查找和移动它们也是轻而易举的事情！
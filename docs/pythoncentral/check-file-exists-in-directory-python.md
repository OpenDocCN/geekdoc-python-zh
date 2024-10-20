# 用 Python 检查目录中是否存在文件

> 原文：<https://www.pythoncentral.io/check-file-exists-in-directory-python/>

在 Python 中，有几种方法可以用来检查某个目录中的文件是否存在。当检查文件是否存在时，通常在访问(读取和/或写入)文件之前执行。下面我们将介绍检查文件是否存在(以及是否可访问)的每种方法，并讨论每种方法的一些潜在问题。

## 1\. os.path.isfile(路径)

如果给定的路径是现有的常规文件，此函数返回 true。它遵循符号链接，因此有可能`os.path.islink(path)`为真，而`os.path.isfile(path)`也为真。这是一个检查文件是否存在的方便函数，因为它是一个简单的命令行程序。不幸的是，该函数只检查指定的路径是否是一个文件，但不保证用户可以访问它。它也只告诉你在你调用这个函数的时候这个文件已经存在了。有可能(尽管可能性极小)，在调用这个函数和访问这个文件之间，它已经被删除或移动/重命名了。

例如，它可能在以下场景中失败:
【python】
>>>OS . path . is File(' foo . txt ')
True
>>f = open(' foo . txt '，' r')
Traceback(最近一次调用 last):
File " "，第 1 行，in
IOError: [Errno 13]权限被拒绝:' foo.txt'

## 2.os.access(路径，模式)

这个函数测试当前用户(拥有真实的 uid/gid)是否有访问给定路径的权限(读和/或写权限)。要测试文件是否可读，可以使用`os.R_OK`，使用`os.W_OK`来确定文件是否可写。比如如下。

```py

>>> # Check for read access to foo.txt

>>> os.access('foo.txt', os.R_OK)

True # This means the file exists AND you can read it.

>>>

>>> # Check for write access to foo.txt

>>> os.access('foo.txt', os.W_OK)

False # You cannot write to the file. It may or may not exist.

```

如果您计划访问一个文件，使用这个函数会更安全一些(尽管不完全推荐),因为它还会检查您是否可以访问(读或写)该文件。但是，如果您计划访问该文件，则在您检查该文件是否可访问和您访问该文件之间，该文件可能已经被删除或移动/重命名。这就是所谓的竞争条件，应该避免。下面是一个如何发生的例子。

```py

>>> # The file 'foo.txt' currently exists and is readable.

>>> if os.access('foo.txt', os.R_OK):

>>> # After executing os.access() and before open(),

>>> # another program deletes the file.

>>> f = open('foo.txt', 'r')

Traceback (most recent call last):

File "", line 1, in

IOError: [Errno 2] No such file or directory: 'foo.txt'

```

## 3.试图访问(打开)文件。

为了绝对保证文件不仅存在，而且在当前时间是可访问的，最简单的方法实际上是尝试打开文件。

```py

try:

f = open('foo.txt')

f.close()

except IOError as e:

print('Uh oh!')

```

这可以转换成一个易于使用的函数，如下所示。

```py

def file_accessible(filepath, mode):

''' Check if a file exists and is accessible. '''

try:

f = open(filepath, mode)

f.close()

except IOError as e:

return False
返回真值

```

例如，您可以按如下方式使用它:

```py

>>> # Say the file 'foo.txt' exists and is readable,

>>> # whereas the file 'bar.txt' doesn't exist.

>>> foo_accessible = file_accessible('foo.txt', 'r')

True

>>>

>>> bar_accessible = file_accessible('bar.txt', 'r')

False

```

## 因此...哪个最好？

无论您决定使用哪种方法，都取决于您为什么需要检查文件是否存在，速度是否重要，以及在任何给定时间您经常尝试打开多少个文件。在许多情况下,`os.path.isfile`应该足够了。但是请记住，在使用任何一种方法时，每种方法都有自己的优点和潜在问题。
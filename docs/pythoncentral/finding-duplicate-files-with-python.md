# 使用 Python 查找重复文件

> 原文：<https://www.pythoncentral.io/finding-duplicate-files-with-python/>

有时，我们需要在我们的文件系统或特定文件夹中找到重复的文件。在本教程中，我们将编写一个 Python 脚本来完成这个任务。这个脚本适用于 Python 3.x。

该程序将接收一个文件夹或文件夹列表进行扫描，然后将遍历给定的目录，并在文件夹中找到重复的文件。

这个程序将为每个文件计算一个散列，允许我们找到重复的文件，即使它们的名字不同。我们找到的所有文件都将被存储在一个字典中，散列作为键，文件的路径作为值:`{ hash: [list of paths] }`。

**开始，`import``os, sys`和`hashlib`图书馆:**

```py

import os

import sys

import hashlib

```

然后我们需要一个函数来计算给定文件的 MD5 散列。该函数接收文件的路径，并返回该文件的十六进制摘要:

```py

def hashfile(path, blocksize = 65536):

afile = open(path, 'rb')

hasher = hashlib.md5()

buf = afile.read(blocksize)

while len(buf) > 0:

hasher.update(buf)

buf = afile.read(blocksize)

afile.close()

return hasher.hexdigest()

```

**现在我们需要一个函数来扫描目录中的重复文件:**

```py

def findDup(parentFolder):

# Dups in format {hash:[names]}

dups = {}

for dirName, subdirs, fileList in os.walk(parentFolder):

print('Scanning %s...' % dirName)

for filename in fileList:

# Get the path to the file

path = os.path.join(dirName, filename)

# Calculate hash

file_hash = hashfile(path)

# Add or append the file path

if file_hash in dups:

dups[file_hash].append(path)

else:

dups[file_hash] = [path]

return dups

```

`findDup`函数使用`os.walk`来遍历给定的目录。如果你需要一个更全面的指南，请看一下[如何在 Python](https://www.pythoncentral.io/how-to-traverse-a-directory-tree-in-python-guide-to-os-walk/ "How to Traverse a Directory Tree in Python – Guide to os.walk") 中遍历目录树的文章。`os.walk`函数只返回文件名，所以我们使用`os.path.join`来获取文件的完整路径。然后我们将得到文件的散列，并将其存储到`dups`字典中。

当`findDup`完成遍历目录时，它返回一个包含重复文件的字典。如果我们要遍历几个目录，我们需要一个方法来合并两个字典:

```py

# Joins two dictionaries

def joinDicts(dict1, dict2):

for key in dict2.keys():

if key in dict1:

dict1[key] = dict1[key] + dict2[key]

else:

dict1[key] = dict2[key]

```

`joinDicts`获取两个字典，遍历第二个字典，检查第一个字典中是否存在该键，如果存在，该方法将第二个字典中的值追加到第一个字典中的值。如果该键不存在，它将把它存储在第一个字典中。在方法的最后，第一个字典包含所有的信息。

为了能够从命令行运行这个脚本，我们需要接收文件夹作为参数，然后为每个文件夹调用`findDup`:

```py

if __name__ == '__main__':

if len(sys.argv) > 1:

dups = {}

folders = sys.argv[1:]

for i in folders:

# Iterate the folders given

if os.path.exists(i):

# Find the duplicated files and append them to the dups

joinDicts(dups, findDup(i))

else:

print('%s is not a valid path, please verify' % i)

sys.exit()

printResults(dups)

else:

print('Usage: python dupFinder.py folder or python dupFinder.py folder1 folder2 folder3')

```

`os.path.exists`函数验证给定的文件夹是否存在于文件系统中。要运行这个脚本，请使用`python dupFinder.py /folder1 ./folder2`。最后，我们需要一种打印结果的方法:

```py

def printResults(dict1):

results = list(filter(lambda x: len(x) > 1, dict1.values()))

if len(results) > 0:

print('Duplicates Found:')

print('The following files are identical. The name could differ, but the content is identical')

print('___________________')

for result in results:

for subresult in result:

print('\t\t%s' % subresult)

print('___________________')
else: 
 print('未发现重复文件')

```

**将所有东西放在一起:**

```py

# dupFinder.py

import os, sys

import hashlib
def find dup(parent folder):
# Dups in format { hash:[names]}
Dups = { }
for dirName，subdirs，file list in OS . walk(parent folder):
print(' Scanning % s ... '文件列表中文件名的% dirName)
:
#获取文件的路径
 path = os.path.join(dirName，filename) 
 #计算哈希
file _ hash = hash file(path)
#添加或追加文件路径
if file _ hash in dups:
dups[file _ hash]。append(path)
else:
dups[file _ hash]=[path]
返回 dups
#联接两个字典
 def joinDicts(dict1，dict 2):
for key in dict 2 . keys():
if key in dict 1:
dict 1[key]= dict 1[key]+dict 2[key]
else:
dict 1[key]= dict 2[key]
def hashfile(path，block size = 65536):
afile = open(path，' Rb ')
hasher = hashlib . MD5()
buf = afile . read(block size)
而 len(buf)>0:
hasher . update(buf)
buf = afile . read(block size)
afile . close()
return hasher . hex digest()
def print results(dict 1):
results = list(filter(lambda x:len(x)>1，dict 1 . values())
if len(results)>0:
print(' Duplicates Found:')
print('以下文件完全相同。名称可能不同，但内容相同))
打印(' ___________________ _ _ _ _ _ _ ')
结果中的结果:
子结果:
打印(' \t\t%s' %子结果)
打印(' _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ')
else: 
 print('未发现重复文件')
if _ _ name _ _ = ' _ _ main _ _ ':
if len(sys . argv)>1:
dups = { }
folders = sys . argv[1:]
for I in folders:
#迭代给定的文件夹
 if os.path.exists(i): 
 #找到重复的文件并将其追加到 dups 
 joinDicts(dups，Find dup(I))
else:
print(% s 不是 a
```
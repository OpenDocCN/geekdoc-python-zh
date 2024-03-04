# 如何用 Shutil 复制和移动文件？

> 原文：<https://www.pythonforbeginners.com/os/python-the-shutil-module>

### 什么是 Shutil？

shutil 模块帮助你自动复制文件和目录。

这样就省去了没有实际处理时打开、读取、写入和关闭文件的步骤。

这是一个实用模块，可以用来完成任务，如:复制，移动，或删除目录树

##### shutil. copy ( src , dest )

```py
 # Basically the unix command cp src dst. 

# this copies the source file to the destination directory

# the destination directory has to exist

# if the filename already exists there, it will be overwritten

# access time and last modification time will be updated

# the same filename is used

# the permissions of the file are copied along with the contents.

import shutil
import os
source = os.listdir("/tmp/")
destination = "/tmp/newfolder/"
for files in source:
    if files.endswith(".txt"):
        shutil.copy(files,destination)

###########
shutil.copyfile ( src , dest )
# copy data from src to dest 

# both names must be files.

# copy files by name 
import shutil  
shutil.copyfile('/path/to/file', '/path/to/other/phile') 

############
shutil.move
# recursively move a file or directory (src) to another location (dst).

# if the destination is a directory or a symlink to a directory, then src is moved
inside that directory.

# the destination directory must not already exist.

# this would move files ending with .txt to the destination path
import shutil
import os
source = os.listdir("/tmp/")
destination = "/tmp/newfolder/"
for files in source:
    if files.endswith(".txt"):
        shutil.move(files,destination)

####################
shutil.copytree ( src , dest )
# recursively copy the entire directory tree rooted at src to dest. 

# dest must not already exist. 

# errors are reported to standard output.

####################
import shutil
import os
SOURCE = "samples"
BACKUP = "samples-bak"
# create a backup directory
shutil.copytree(SOURCE, BACKUP)
print os.listdir(BACKUP)

####################
shutil.rmtree ( path )

# recursively delete a directory tree.
# This removes the directory 'three' and anything beneath it in the filesystem.
import shutil
shutil.rmtree('one/two/three') 
```
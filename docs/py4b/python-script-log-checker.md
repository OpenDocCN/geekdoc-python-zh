# Python 中的日志检查器

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/python-script-log-checker>

### 显示日志文件中的所有条目

```py
 This script will show all entries in the file that is specified in the log file
variable. 
```

### 日志检查器脚本

```py
 In this example, I will use the /var/log/syslog file. 

The for loop will go through each line of the log file and the line_split variablewill split it by lines. 

If you just print the line_split, you will see an output similar to this:

>> ['Sep', '27', '15:22:15', 'Virtualbox', 'NetworkManager[710]:', '<info>', 'DNS:'..']

If you want to print each element just add the line_split[element_to_show]
```

```py
#!/usr/bin/env python
logfile = open("/var/log/syslog", "r")
for line in logfile:
    line_split = line.split()
    print line_split
    list = line_split[0], line_split[1], line_split[2], line_split[4]
    print list

```

```py
 That's it, now you have a script that you can use for checking log files with. 
```
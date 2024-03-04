# 监控 Apache / Nginx 日志文件

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/python-script-monitor-apachenginx-log-file>

### 统计 Apache/Nginx 中的命中次数

```py
 This small script will count the number of hits in a Apache/Nginx log file. 
```

### 它是如何工作的

```py
 This script can easily be adapted to any other log file. 

The script starts with making an empty dictionary for storing the IP addresses andcount how many times they exist. 

Then we open the file (in this example the Nginx access.log file) and read the
content line by line. 

The for loop go through the file and splits the strings to get the IP address. 

The len() function is used to ensure the length of IP address. 

If the IP already exists , increase by 1. 
```

```py
ips = {}

fh = open("/var/log/nginx/access.log", "r").readlines()
for line in fh:
    ip = line.split(" ")[0]
    if 6 < len(ip) <=15:
        ips[ip] = ips.get(ip, 0) + 1
print ips

```

### 测试一下

```py
 If you now browse to your website, and run the python script, you should see your IP address + the counts. 
```
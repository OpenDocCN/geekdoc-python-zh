# 脚本:从提示中获取用户名…

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/script-get-the-username-from-a-prompt>

这个脚本将使用 raw_input 函数要求用户输入用户名。然后创建一个名为 user1 和 user2 的允许用户列表。控制语句检查来自用户的输入是否与允许用户列表中的输入相匹配。

```py
#!/usr/bin/env python

#get the username from a prompt
username = raw_input("Login: >> ")

#list of allowed users
user1 = "Jack"
user2 = "Jill"

#control that the user belongs to the list of allowed users
if username == user1:
    print "Access granted"
elif username == user2:
    print "Welcome to the system"
else:
    print "Access denied"

```
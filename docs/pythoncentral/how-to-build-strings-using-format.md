# 如何使用？格式()

> 原文：<https://www.pythoncentral.io/how-to-build-strings-using-format/>

开发人员经常试图在 Python 中构建字符串，方法是将一串字符串连接在一起，形成一个长字符串，这肯定可行，但并不完全理想。Python 的。format()方法允许您执行与连接字符串(将一串字符串加在一起)基本相同的效果，但是以一种更加高效和轻量级的方式执行。

假设您想要将以下字符串相加来创建一个句子:

```py
name = "Jon"  
age = 28  
fave_food = "burritos" fave_color = "green" 
```

要打印上面的字符串，您总是可以选择将它们连接起来，就像这样:

```py
string = "Hey, I'm " + name + "and I'm " + str(age) + " years old. I love " + fave_food + 
" and my favorite color is " + fave_color "."  
print(string) 
```

也可以使用。format()方法，像这样:

```py
string = "Hey, I'm {0} and I'm {1} years old. I love {2} and my favorite color is {3}.".format(name, 
age, fave_food, fave_color)
print(string) 
```

以上两个示例的输出是相同的:

```py
Hey, I'm Jon and I'm 28 years old. I love burritos and my favorite color is green.
```

虽然上面的两个例子都产生了一个长字符串的相同结果，但是使用。format()方法比连接字符串的其他解决方案要快得多，也轻得多。
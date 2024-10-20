# ConfigParser 模块–用 Python 创建配置文件

> 原文：<https://www.askpython.com/python-modules/configparser-module>

在本教程中，我们将了解什么是配置文件，在 **ConfigParser 模块**的帮助下，我们将创建一个配置文件，修改配置文件中的数据，向其中添加新数据，并从配置文件中删除现有数据。所以不要再拖延了，让我们开始吧。

## Python 中的配置文件是什么？

配置文件通常称为配置文件，是存储计算机程序的一些特定数据和设置的特殊文件。大多数计算机程序在启动时读取它们的配置文件，并定期检查这些配置文件中的变化。

用户可以使用这些文件来更改应用程序的设置，而无需重新编译程序。通常每个配置文件由不同的部分组成。每个部分都包含键和值对，就像一个 [Python 字典](https://www.askpython.com/python/dictionary/python-dictionary-dict-tutorial)。

下面给出了一个样本配置文件，它由三部分组成，即地址、教育和个人爱好。

```py
[Address]
Name = Aditya Raj
Village = Bhojpur
District = Samastipur
State = Bihar

[Education]
College=IIITA
Branch= IT

[Favorites]
Sport = VolleyBall
Book = Historical Books

```

现在我们将使用 python 中的 ConfigParser 模块创建上面的配置文件。

## 如何使用 Python ConfigParser 模块创建配置文件？

为了用 python 创建配置文件，我们将使用 configparser 模块。在下面的实现中，我们创建一个 ConfigParser 对象，并向其中添加一些节，这些节基本上是包含键值对的字典。然后我们用。ini 扩展。

```py
#import module
import configparser

#create configparser object
config_file = configparser.ConfigParser()

#define sections and their key and value pairs
config_file["Address"]={
        "Name": "Aditya Raj",
        "Village": "Bhojpur",
        "District": "Samastipur",
        "State": "Bihar"
        }
config_file["Education"]={
        "College":"IIITA",
        "Branch" : "IT"
        }
config_file["Favorites"]={
        "Sports": "VolleyBall",
        "Books": "Historical Books"
        }

#SAVE CONFIG FILE
with open("person.ini","w") as file_object:
    config_file.write(file_object)
print("Config file 'person.ini' created")

#print file content
read_file=open("person.ini","r")
content=read_file.read()
print("content of the config file is:")
print(content)

```

上述代码片段的输出是:

```py
Config file 'person.ini' created
content of the config file is:
[Address]
name = Aditya Raj
village = Bhojpur
district = Samastipur
state = Bihar

[Education]
college = IIITA
branch = IT

[Favorites]
sports = VolleyBall
books = Historical Books

```

## 如何在用 ConfigParser 创建的配置文件中添加新的节？

要在配置文件中添加新的节，我们只需读取配置对象中的配置文件，通过以字典格式定义节来添加新的节，然后我们可以将配置对象保存到同一个文件中。

在下面的示例中，我们将在 person.ini 文件中添加一个新的部分“Physique ”,该文件已经包含地址、教育和收藏夹部分。

```py
import configparser

#print initial file content
read_file=open("person.ini","r")
content=read_file.read()
print("content of the config file is:")
print(content)

#create new config object
config_object= configparser.ConfigParser()

#read config file into object
config_object.read("person.ini")

#Add new section named Physique
config_object["Physique"]={
        "Height": "183 CM",
        "Weight": "70 Kg"
        }

#save the config object back to file
with open("person.ini","w") as file_object:
    config_object.write(file_object)

#print the new config file
print("Config file 'person.ini' updated")
print("Updated file content is:")
nread_file=open("person.ini","r")
ncontent=nread_file.read()
print(ncontent)

```

上述代码片段的输出是:

```py
content of the config file is:
[Address]
name = Aditya Raj
village = Bhojpur
district = Samastipur
state = Bihar

[Education]
college = IIITA
branch = IT

[Favorites]
sports = VolleyBall
books = Historical Books

Config file 'person.ini' updated
Updated file content is:
[Address]
name = Aditya Raj
village = Bhojpur
district = Samastipur
state = Bihar

[Education]
college = IIITA
branch = IT

[Favorites]
sports = VolleyBall
books = Historical Books

[Physique]
height = 183 CM
weight = 70 Kg

```

我们也可以使用`add_section()`方法添加一个新的部分，然后使用`set()`方法在该部分中添加新的字段。

```py
import configparser

#print initial file content
read_file=open("person.ini","r")
content=read_file.read()
print("content of the config file is:")
print(content)

#create new config object
config_object= configparser.ConfigParser()

#read config file into object
config_object.read("person.ini")

#Add new section named Physique
config_object.add_section('Physique')
config_object.set('Physique', 'Height', '183 CM')
config_object.set('Physique', 'Weight', '70 Kg')

#save the config object back to file
with open("person.ini","w") as file_object:
    config_object.write(file_object)

#print the updated config file
print("Config file 'person.ini' updated")
print("Updated file content is:")
nread_file=open("person.ini","r")
ncontent=nread_file.read()
print(ncontent)

```

输出:

```py
content of the config file is:
[Address]
name = Aditya Raj
village = Bhojpur
district = Samastipur
state = Bihar

[Education]
college = IIITA
branch = IT

[Favorites]
sports = VolleyBall
books = Historical Books

Config file 'person.ini' updated
Updated file content is:
[Address]
name = Aditya Raj
village = Bhojpur
district = Samastipur
state = Bihar

[Education]
college = IIITA
branch = IT

[Favorites]
sports = VolleyBall
books = Historical Books

[Physique]
height = 183 CM
weight = 70 Kg

```

在上面的例子中，我们可以看到`add_section()`方法将节名作为它的参数，而`set()`方法将节名作为它的第一个参数，将字段名作为它的第二个参数，将字段的值作为它的第三个参数。****

在创建新的配置文件时，也可以使用这两种方法向文件中添加节和字段，而不是像本例中那样使用字典。

## 如何更新配置文件中的数据？

因为我们已经将配置文件的节定义为字典，所以适用于字典的操作也适用于配置文件的节。我们可以在配置文件的任何部分添加字段，或者以类似于处理字典项的方式修改字段的值。

在下面的代码中，我们在 person.ini 配置文件的“Education”部分添加了一个新字段“Year ”,并修改了文件中“Branch”字段的值。

```py
import configparser

#print initial file content
read_file=open("person.ini","r")
content=read_file.read()
print("content of the config file is:")
print(content)

#create new config object
config_object= configparser.ConfigParser()

#read config file into object
config_object.read("person.ini")

#update value of a field in a section
config_object["Education"]["Branch"]="MBA"

#add a new field in a section
config_object["Education"].update({"Year":"Final"})

#save the config object back to file
with open("person.ini","w") as file_object:
    config_object.write(file_object)

#print updated content
print("Config file 'person.ini' updated")
print("Updated file content is:")
nread_file=open("person.ini","r")
ncontent=nread_file.read()
print(ncontent)

```

输出:

```py
content of the config file is:
[Address]
name = Aditya Raj
village = Bhojpur
district = Samastipur
state = Bihar

[Education]
college = IIITA
branch = IT

[Favorites]
sports = VolleyBall
books = Historical Books

[Physique]
height = 183 CM
weight = 70 Kg

Config file 'person.ini' updated
Updated file content is:
[Address]
name = Aditya Raj
village = Bhojpur
district = Samastipur
state = Bihar

[Education]
college = IIITA
branch = MBA
year = Final

[Favorites]
sports = VolleyBall
books = Historical Books

[Physique]
height = 183 CM
weight = 70 Kg

```

在上面的例子中，我们可以使用`update()`方法添加新的字段以及修改现有的字段。如果文件中存在作为参数给出的字段，它将更新该字段，否则将创建一个新字段。

## 如何从配置文件中删除数据？

我们可以使用 configparser 模块中的`remove_option()`和`remove_section()`模块从配置文件中删除数据。`remove_option()`用于从任何部分删除一个字段，`remove_section()`用于删除配置文件的整个部分。

```py
import configparser

#print initial file content
read_file=open("person.ini","r")
content=read_file.read()
print("content of the config file is:")
print(content)

#create new config object
config_object= configparser.ConfigParser()

#read config file into object
config_object.read("person.ini")

#delete a field in a section
config_object.remove_option('Education', 'Year')

#delete a section
config_object.remove_section('Physique')

#save the config object back to file
with open("person.ini","w") as file_object:
    config_object.write(file_object)

#print new config file
print("Config file 'person.ini' updated")
print("Updated file content is:")
nread_file=open("person.ini","r")
ncontent=nread_file.read()
print(ncontent)

```

输出:

```py
content of the config file is:
[Address]
name = Aditya Raj
village = Bhojpur
district = Samastipur
state = Bihar

[Education]
college = IIITA
branch = MBA
year = Final

[Favorites]
sports = VolleyBall
books = Historical Books

[Physique]
height = 183 CM
weight = 70 Kg

Config file 'person.ini' updated
Updated file content is:
[Address]
name = Aditya Raj
village = Bhojpur
district = Samastipur
state = Bihar

[Education]
college = IIITA
branch = MBA

[Favorites]
sports = VolleyBall
books = Historical Books

```

在上面的例子中，我们可以看到`remove_option()`方法将 section name 作为它的第一个参数，将 field name 作为它的第二个参数，而`remove_section()`方法将待删除的 section 的名称作为它的参数。

## 结论

在本教程中，我们已经了解了什么是配置文件，以及如何在 Python configparser 模块的帮助下创建和操作配置文件。快乐学习！🙂

**参考文献——[https://docs.python.org/3/library/configparser.html](https://docs.python.org/3/library/configparser.html)**
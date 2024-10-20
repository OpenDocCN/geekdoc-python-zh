# Python 101:读取和写入 CSV 文件

> 原文：<https://www.blog.pythonlibrary.org/2014/02/26/python-101-reading-and-writing-csv-files/>

Python 有一个庞大的模块库，包含在它的发行版中。csv 模块为 Python 程序员提供了解析 CSV(逗号分隔值)文件的能力。CSV 文件是人类可读的文本文件，其中每一行都有许多由逗号或其他分隔符分隔的字段。您可以将每行视为一行，将每个字段视为一列。csv 格式没有标准，但是它们非常相似，CSV 模块能够读取绝大多数 CSV 文件。您也可以使用 csv 模块编写 CSV 文件。

* * *

### 读取 CSV 文件

有两种方法可以读取 CSV 文件。你可以使用 csv 模块的 **reader** 函数或者你可以使用 **DictReader** 类。我们将研究这两种方法。但是首先，我们需要得到一个 CSV 文件，这样我们就有东西可以解析了。有许多网站以 CSV 格式提供有趣的信息。我们将使用世界卫生组织(世卫组织)的网站下载一些关于结核病的信息。可以去这里领取:【http://www.who.int/tb/country/data/download/en/[。一旦你拿到文件，我们就可以开始了。准备好了吗？然后我们来看一些代码！](http://www.who.int/tb/country/data/download/en/)

```py

import csv

#----------------------------------------------------------------------
def csv_reader(file_obj):
    """
    Read a csv file
    """
    reader = csv.reader(file_obj)
    for row in reader:
        print(" ".join(row))

#----------------------------------------------------------------------
if __name__ == "__main__":
    csv_path = "TB_data_dictionary_2014-02-26.csv"
    with open(csv_path, "rb") as f_obj:
        csv_reader(f_obj)

```

让我们花点时间来分解一下。首先，我们必须实际导入 **csv** 模块。然后我们创建一个非常简单的函数，名为 **csv_reader** ，它接受一个文件对象。在函数内部，我们将文件对象传递给 **csv_reader** 函数，该函数返回一个 reader 对象。reader 对象允许迭代，就像常规的 file 对象一样。这让我们遍历 reader 对象中的每一行，并打印出数据行，去掉逗号。这是因为每一行都是一个列表，我们可以将列表中的每个元素连接在一起，形成一个长字符串。

现在让我们创建自己的 CSV 文件，并将其输入到 **DictReader** 类中。这里有一个非常简单的问题:

```py

first_name,last_name,address,city,state,zip_code
Tyrese,Hirthe,1404 Turner Ville,Strackeport,NY,19106-8813
Jules,Dicki,2410 Estella Cape Suite 061,Lake Nickolasville,ME,00621-7435
Dedric,Medhurst,6912 Dayna Shoal,Stiedemannberg,SC,43259-2273

```

让我们将它保存在一个名为 **data.csv** 的文件中。现在我们准备使用 DictReader 类解析文件。让我们试一试:

```py

import csv

#----------------------------------------------------------------------
def csv_dict_reader(file_obj):
    """
    Read a CSV file using csv.DictReader
    """
    reader = csv.DictReader(file_obj, delimiter=',')
    for line in reader:
        print(line["first_name"]),
        print(line["last_name"])

#----------------------------------------------------------------------
if __name__ == "__main__":
    with open("data.csv") as f_obj:
        csv_dict_reader(f_obj)

```

在上面的例子中，我们打开一个文件，并像以前一样将文件对象传递给我们的函数。该函数将文件对象传递给我们的 DictReader 类。我们告诉 DictReader 分隔符是逗号。这实际上并不是必需的，因为没有那个关键字参数，代码仍然可以工作。然而，明确一点是个好主意，这样你就知道这里发生了什么。接下来我们遍历 reader 对象，发现 reader 对象中的每一行都是一个字典。这使得打印出该行的特定部分非常容易。

现在我们准备学习如何将 csv 文件写入磁盘。

* * *

### 编写 CSV 文件

csv 模块也有两种方法可以用来编写 CSV 文件。你可以使用 **writer** 函数或者 DictWriter 类。我们也将研究这两个问题。我们将从 writer 函数开始。让我们看一个简单的例子:

```py

# Python 2.x version
import csv

#----------------------------------------------------------------------
def csv_writer(data, path):
    """
    Write data to a CSV file path
    """
    with open(path, "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            writer.writerow(line)

#----------------------------------------------------------------------
if __name__ == "__main__":
    data = ["first_name,last_name,city".split(","),
            "Tyrese,Hirthe,Strackeport".split(","),
            "Jules,Dicki,Lake Nickolasville".split(","),
            "Dedric,Medhurst,Stiedemannberg".split(",")
            ]
    path = "output.csv"
    csv_writer(data, path)

```

在上面的代码中，我们创建了一个接受两个参数的 **csv_writer** 函数:数据和路径。数据是我们在脚本底部创建的列表列表。我们使用前一个例子中数据的简化版本，并在逗号上分割字符串。这将返回一个列表。所以我们最终得到一个嵌套列表，如下所示:

```py

[['first_name', 'last_name', 'city'],
 ['Tyrese', 'Hirthe', 'Strackeport'],
 ['Jules', 'Dicki', 'Lake Nickolasville'],
 ['Dedric', 'Medhurst', 'Stiedemannberg']]

```

**csv_writer** 函数打开我们传入的路径，并创建一个 csv writer 对象。然后我们遍历嵌套列表结构，将每一行写到磁盘上。注意，我们在创建 writer 对象时指定了分隔符应该是什么。如果您希望分隔符不是逗号，那么您可以在这里设置它。

现在如果你想用 Python 3 写一个 csv 文件，语法略有不同。下面是您必须重写该函数的方式:

```py

# Python 3.x version
import csv

def csv_writer(data, path):
    """
    Write data to a CSV file path
    """
    with open(path, "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            writer.writerow(line)

```

您会注意到，您需要将写模式更改为仅仅是**‘w’**，并添加**换行符**参数。

现在我们准备学习如何使用 **DictWriter** 类编写 CSV 文件！我们将使用前一个版本的数据，并将其转换成一个字典列表，我们可以将它提供给我们饥渴的 DictWriter。让我们来看看:

```py

# Python 2.x version
import csv

#----------------------------------------------------------------------
def csv_dict_writer(path, fieldnames, data):
    """
    Writes a CSV file using DictWriter
    """
    with open(path, "wb") as out_file:
        writer = csv.DictWriter(out_file, delimiter=',', fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

#----------------------------------------------------------------------
if __name__ == "__main__":
    data = ["first_name,last_name,city".split(","),
            "Tyrese,Hirthe,Strackeport".split(","),
            "Jules,Dicki,Lake Nickolasville".split(","),
            "Dedric,Medhurst,Stiedemannberg".split(",")
            ]
    my_list = []
    fieldnames = data[0]
    for values in data[1:]:
        inner_dict = dict(zip(fieldnames, values))
        my_list.append(inner_dict)

    path = "dict_output.csv"
    csv_dict_writer(path, fieldnames, my_list)

```

*注意:要将这段代码转换成 Python 3 语法，您需要像以前一样用语句修改**:用 open(path，" w "，newline= ' ')作为 out_file:***

我们将首先从第二部分开始。如你所见，我们从之前的嵌套列表结构开始。接下来，我们创建一个空列表和一个包含字段名称的列表，这恰好是嵌套列表中的第一个列表。记住，列表是从零开始的，所以列表中的第一个元素从零开始！接下来，我们遍历嵌套列表结构，从第二个元素开始:

```py

for values in data[1:]:
    inner_dict = dict(zip(fieldnames, values))
    my_list.append(inner_dict)

```

在 **for** 循环中，我们使用 Python 内置函数来创建字典。**zip**方法将接受两个迭代器(本例中是列表),并将它们转换成元组列表。这里有一个例子:

```py

zip(fieldnames, values)
[('first_name', 'Dedric'), ('last_name', 'Medhurst'), ('city', 'Stiedemannberg')]

```

现在，当您在**dict**中包装该调用时，它会将元组列表转换为字典。最后，我们把字典添加到列表中。当**for**完成时，您将得到如下所示的数据结构:

```py

[{'city': 'Strackeport', 'first_name': 'Tyrese', 'last_name': 'Hirthe'},
{'city': 'Lake Nickolasville', 'first_name': 'Jules', 'last_name': 'Dicki'},
{'city': 'Stiedemannberg', 'first_name': 'Dedric', 'last_name': 'Medhurst'}]

```

在第二个会话结束时，我们调用我们的 **csv_dict_writer** 函数，并传入所有需要的参数。在函数内部，我们创建一个 DictWriter 实例，并向它传递一个文件对象、一个分隔符值和我们的字段名列表。接下来，我们将字段名写到磁盘上，一次循环一行数据，将数据写到磁盘上。DictWriter 类也支持 **writerows** 方法，我们可以用它来代替循环。 **csv.writer** 函数也支持该功能。

您可能有兴趣知道，您还可以使用 csv 模块创建方言。这允许您以非常明确的方式告诉 csv 模块如何读取或写入文件。如果您因为来自客户端的格式奇怪的文件而需要这种东西，那么您会发现这种功能非常有价值。

* * *

### 包扎

现在您知道了如何使用 csv 模块来读写 CSV 文件。有许多网站以这种格式发布数据，这种格式在商业领域被广泛使用。开心快乐编码！

* * *

### 附加阅读

*   Python 文档- [第 13.1 节 csv](http://docs.python.org/2/library/csv.html)
*   用 Python DictReader 和 DictWriter 读写 CSV 文件
*   本周 Python 模块: [csv](http://pymotw.com/2/csv/)
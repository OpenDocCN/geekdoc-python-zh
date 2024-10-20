# modulenotfounderror:没有名为“matplotlib”的模块

> 原文：<https://pythonguides.com/no-module-named-matplotlib/>

[![Python & Machine Learning training courses](img/49ec9c6da89a04c9f45bab643f8c765c.png)](https://sharepointsky.teachable.com/p/python-and-machine-learning-training-course)

在本 [Python 教程](https://pythonguides.com)中，我们将讨论 **modulenotfounderror:没有名为‘matplotlib’**的模块，我们还将涉及以下主题:

*   modulenotfounderror:没有名为 matplotlib windows 10 的模块
*   modulenotfounderror:没有名为“matplotlib”Ubuntu 的模块
*   modulenotfounderror 没有名为“matplotlib”的模块 python 3
*   modulenotfounderror jupyter 笔记本中没有名为“matplotlib”的模块
*   modulenotfounderror 没有名为“matplotlib”的模块 anaconda
*   modulenotfounderror:没有名为“matplotlib”py charm 的模块
*   modulenotfounderror:没有名为“matplotlib.pyplot”的模块；“matplotlib”不是一个包

目录

[](#)

*   [modulenotfounderror:没有名为 matplotlib windows 10](#modulenotfounderror_no_module_named_matplotlib_windows_10 "modulenotfounderror: no module named matplotlib windows 10") 的模块
*   [modulenotfounderror:没有名为“matplotlib”Ubuntu 的模块](#modulenotfounderror_no_module_named_matplotlib_ubuntu "modulenotfounderror: no module named ‘matplotlib’ ubuntu")
*   [modulenotfounderror 没有名为“matplotlib”的模块 python 3](#modulenotfounderror_no_module_named_matplotlib_python_3 "modulenotfounderror no module named ‘matplotlib’ python 3")
*   [modulenotfounderror 没有名为“matplotlib”的模块 jupyter 笔记本](#modulenotfounderror_no_module_named_matplotlib_jupyter_notebook "modulenotfounderror no module named ‘matplotlib’ jupyter notebook")
*   [modulenotfounderror 没有名为“matplotlib”的模块 anaconda](#modulenotfounderror_no_module_named_matplotlib_anaconda "modulenotfounderror no module named ‘matplotlib’ anaconda")
*   [modulenotfounderror:没有名为“matplotlib”py charm 的模块](#modulenotfounderror_no_module_named_matplotlib_pycharm "modulenotfounderror: no module named ‘matplotlib’ pycharm")
*   [modulenotfounderror:没有名为“matplotlib.pyplot”的模块；“matplotlib”不是一个包](#modulenotfounderror_no_module_named_matplotlibpyplot_matplotlib_is_not_a_package "modulenotfounderror: no module named ‘matplotlib.pyplot’; ‘matplotlib’ is not a package")

## modulenotfounderror:没有名为 matplotlib windows 10 的模块

检查您是否已经安装了 pip，只需在 python 控制台中编写 pip。如果你没有 pip，从网上获取一个名为 `get-pip.py` 的 python 脚本，保存到你的本地系统。pip 是 python 包安装程序。

记下文件保存的位置，并在命令提示符下将当前目录更改为该目录。

```py
pip  -- Press Enter

-- If you don't have a pip then

cd path_of_directory_of_get-pip_script
```

运行 get-pip.py 脚本来安装 pip，方法是在 cmd(命令提示符)中编写以下代码来安装 pip:

```py
"python .\get-pip.py"
```

现在，在 cmd 中键入以下代码来安装 matplotlib 及其依赖项:

```py
pip install matplotlib
```

错误将被解决，如果没有，然后通过这个帖子结束。

阅读:[什么是 Matplotlib](https://pythonguides.com/what-is-matplotlib/)

## modulenotfounderror:没有名为“matplotlib”Ubuntu 的模块

如果您没有安装 matplotlib，那么要通过 APT 包管理器为 Python 3 安装 Matplotlib，您需要包 `python3-matplotlib` :

```py
sudo apt-get install python3-matplotlib
```

如果您想用 pip for python 2.7 安装它，您需要使用 pip:

```py
sudo pip install matplotlib
```

如果错误仍然出现，请坚持到文章结束。

阅读:[如何安装 matplotlib](https://pythonguides.com/how-to-install-matplotlib-python/)

## modulenotfounderror 没有名为“matplotlib”的模块 python 3

python 3 及以上版本可以用 pip 安装 matplotlib，你只需要用 pip3。

打开 python 控制台并执行下面给出的命令:

```py
sudo pip3 install matplotlib
```

通过执行上述代码，将安装 python 的 matplotlib。

## modulenotfounderror 没有名为“matplotlib”的模块 jupyter 笔记本

在项目目录中创建一个虚拟环境。如果您没有它，您必须通过在 cmd/terminal 中执行以下命令来安装 virtualenv。

```py
virtualenv environment_name   -- environment_name specifies the name of 
                              -- the environment variable created
```

在虚拟环境中安装 matplotlib。

```py
pip3 install matplotlib
```

现在，在您的虚拟环境中安装 ipykernel。

```py
pip3 install ipykernel
```

将 jupyter 内核连接到新环境。

```py
sudo python3 -m ipykernel install
```

当您启动 jupyter 笔记本时，您会看到选择环境的选项，选择您创建的安装了 matplotlib 的环境。现在，你可以继续下去了。

阅读:[什么是 Python 字典](https://pythonguides.com/create-a-dictionary-in-python/)

## modulenotfounderror 没有名为“matplotlib”的模块 anaconda

如果您在安装 Anaconda 之前已经安装了 Python，原因可能是它运行的是您的默认 Python 安装，而不是随 Anaconda 一起安装的。你必须试着把它放在脚本的最前面:

```py
#!/usr/bin/env python
```

如果这不起作用，重新启动终端，并尝试在 conda 提示符或 cmd 中安装带有 conda 的 matplotlib，看看它是否工作。

```py
conda install matplotlib
```

如果问题仍然没有解决，也许你需要创建一个虚拟环境，就像上面提到的那样。

## modulenotfounderror:没有名为“matplotlib”py charm 的模块

如果您正在使用 pycharm，并且当前工作目录中有 matplotlib.py，则可能会出现此错误。您只需删除或重命名 matplotlib.py 文件即可解决问题，这很可能会奏效。

## modulenotfounderror:没有名为“matplotlib.pyplot”的模块；“matplotlib”不是一个包

该错误是由以下原因引起的，请检查它们:

*   确保您正在安装的 matplotlib 版本与您安装的 python 版本兼容。
*   如果安装的 python 是 64 位版本，matplotlib 是 32 位。确保它们是相同的。
*   确保使用 python 的路径为系统和环境变量添加路径变量。
*   如果 pip 版本过期，请将其升级到最新版本。

```py
python -m pip install
```

*   还要确保导入语句中没有错别字。
*   如果错误仍然存在，请检查您的工作目录中是否有 matplotlib.py 文件。删除该文件，重启内核并再次导入 matplotib。那应该有用。

你可能也喜欢读下面的文章。

*   [如何安装 Django](https://pythonguides.com/how-to-install-django/)
*   [Python Django vs Flask](https://pythonguides.com/python-django-vs-flask/)
*   [Python 数字形状](https://pythonguides.com/python-numpy-shape/)
*   [模块‘matplotlib’没有属性‘plot’](https://pythonguides.com/module-matplotlib-has-no-attribute-plot/)

在本 [Python 教程](https://pythonguides.com)中，我们已经讨论了 **modulenotfounderror:没有名为‘matplotlib’**的模块，我们还讨论了以下主题:

*   modulenotfounderror:没有名为 matplotlib windows 10 的模块
*   modulenotfounderror:没有名为“matplotlib”Ubuntu 的模块
*   modulenotfounderror 没有名为“matplotlib”的模块 python 3
*   modulenotfounderror jupyter 笔记本中没有名为“matplotlib”的模块
*   modulenotfounderror 没有名为“matplotlib”的模块 anaconda
*   modulenotfounderror:没有名为“matplotlib”py charm 的模块
*   modulenotfounderror:没有名为“matplotlib.pyplot”的模块；“matplotlib”不是一个包

![Bijay Kumar MVP](img/9cb1c9117bcc4bbbaba71db8d37d76ef.png "Bijay Kumar MVP")[Bijay Kumar](https://pythonguides.com/author/fewlines4biju/)

Python 是美国最流行的语言之一。我从事 Python 工作已经有很长时间了，我在与 Tkinter、Pandas、NumPy、Turtle、Django、Matplotlib、Tensorflow、Scipy、Scikit-Learn 等各种库合作方面拥有专业知识。我有与美国、加拿大、英国、澳大利亚、新西兰等国家的各种客户合作的经验。查看我的个人资料。

[enjoysharepoint.com/](https://enjoysharepoint.com/)[](https://www.facebook.com/fewlines4biju "Facebook")[](https://www.linkedin.com/in/fewlines4biju/ "Linkedin")[](https://twitter.com/fewlines4biju "Twitter")
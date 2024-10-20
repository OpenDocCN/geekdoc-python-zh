# Mozilla 在浏览器中宣布 Pyodide - Python

> 原文：<https://www.blog.pythonlibrary.org/2019/04/18/mozilla-announces-pyodide-python-in-the-browser/>

本周早些时候，Mozilla [宣布了一个名为](https://hacks.mozilla.org/2019/04/pyodide-bringing-the-scientific-python-stack-to-the-browser/)[的新项目。Pyodide 的目标是将 Python 的科学堆栈引入浏览器。](https://github.com/iodide-project/pyodide/)

Pyodide 项目将为您提供一个完整的、标准的 Python 解释器，它可以在您的浏览器中运行，还可以让您访问浏览器的 Web APIs。目前，Pyodide 不支持线程或网络套接字。Python 在浏览器中运行的速度也相当慢，尽管它可用于交互式探索。

文章还提到了其他项目，比如[布莱森](https://brython.info/)和[斯库尔普特](http://www.skulpt.org/)。这些项目是用 Javascript 重写 Python 的解释器。与 Pyodide 相比，它们的缺点是不能使用用 C 编写的 Python 扩展，比如 Numpy 或 Pandas。Pyodide 克服了这个问题。

无论如何，这听起来是一个非常有趣的项目。我一直认为我以前在浏览器中看到的 Python 在 Silverlight 中运行的演示很酷。这个项目现在基本上已经死了，但是 Pyodide 听起来是一个非常有趣的让 Python 进入浏览器的新技术。希望它会去某个地方。
# 本周 PyDev:batu Han task aya

> 原文：<https://www.blog.pythonlibrary.org/2022/02/07/pydev-of-the-week-batuhan-taskaya/>

本周我们欢迎巴图汉·塔斯卡娅( [@isidentical](https://twitter.com/isidentical) )成为我们的本周 PyDev！Batuhan 是 Python 语言的核心开发人员。巴图汉也是多个 Python 包的维护者，包括[帕索](https://github.com/davidhalter/parso)和[黑](https://github.com/psf/black)。

你可以通过查看巴图汉的[网站](https://batuhan.tree.science/)或 [GitHub 简介](https://github.com/isidentical)来了解他还在做什么。

让我们花一点时间来更好地了解巴图汉！

你能告诉我们一些关于你自己的情况吗(爱好、教育等)

Hey there! My name is Batuhan, and I'm a software engineer who loves to work on developer tools to improve the overall productivity of the Python ecosystem.

I pretty much fill all my free time with open source maintenance and other programming related activities. If I am not programming at that time, I am probably reading a paper about [PLT](https://en.wikipedia.org/wiki/Programming_language_theory) or watching some sci-fi show. I am a huge fan of the [Stargate](https://en.wikipedia.org/wiki/Stargate) franchise.

**Why did you start using Python?**

I was always intrigued by computers but didn't do anything related to programming until I started using GNU/Linux on my personal computer (namely Ubuntu 12.04). Back then, I was searching for something to pass the time and found Python.

Initially, I was mind-blown by the responsiveness of the REPL. I typed `2 + 2`, it replied `4` back to me. Such a joy! For someone with literally zero programming experience, it was a very friendly environment. Later, I started following some tutorials, writing more code and repeating that process until I got a good grasp of the Python language and programming in general.

**What other programming languages do you know and which is your favourite?**

After being exposed to the level of elegancy and the simplicity in Python, I set the bar too high for adopting a new language. C is a great example where the language (in its own terms) is very straightforward, and currently, it is the only language I actively use apart from Python. I also think it goes really well when paired with Python, which might not be surprised considering the CPython itself and the extension modules are written in C.

If we let the mainstream languages go, I love building one-off compilers for weird/esoteric languages.

**What projects are you working on now?**

Most of my work revolves around CPython, which is the reference implementation of the Python language. In terms of the core, I specialize in the parser and the compiler. But outside of it, I maintain the [ast module](https://docs.python.org/3/library/ast.html), and a few others.

One of the recent changes I've collaborated (with [Pablo Galindo Salgado](https://twitter.com/pyblogsal) an [Ammar Askar](https://twitter.com/__ammar2__)) on CPython was the new fancy tracebacks which I hope will really increase the productivity of the Python developers:

```py
Traceback (most recent call last):
  File "query.py", line 37, in <module>
    magic_arithmetic('foo')
    ^^^^^^^^^^^^^^^^^^^^^^^
  File "query.py", line 18, in magic_arithmetic
    return add_counts(x) / 25
           ^^^^^^^^^^^^^
  File "query.py", line 24, in add_counts
    return 25 + query_user(user1) + query_user(user2)
                ^^^^^^^^^^^^^^^^^
  File "query.py", line 32, in query_user
    return 1 + query_count(db, response['a']['b']['c']['user'], retry=True)
                               ~~~~~~~~~~~~~~~~~~^^^^^
TypeError: 'NoneType' object is not subscriptable
```

Alongside that, I help maintain projects like

*   [parso](https://github.com/davidhalter/parso)
*   [black](https://github.com/psf/black)
*   [refactor](https://github.com/isidentical/refactor)
*   [teyit](https://github.com/isidentical/teyit)

and I am a core member of the [fsspec](https://github.com/fsspec).

**Which Python libraries are your favorite (core or 3rd party)?**

It might be a bit obvious, but I love the [ast module](https://docs.python.org/3/library/ast.html). Apart from that, I enjoy using dataclasses and pathlib.

I generally avoid using dependencies since nearly %99 of the time, I can simply use the stdlib. But there is one exception, [rich](https://pypi.org/project/rich/). For the last three months, nearly every script I've written uses it. It is such a beauty (both in terms of the UI and the API). I also really love pytest and pre-commit.

Not as a library, though one of my favorite projects from the python ecosystem is [PyPy](https://pypy.org). It brings an entirely new python runtime, which depending on your work can be 1000X faster (or just 4X in general).

**Is there anything else you’d like to say?**

I've recently started a [GitHub Sponsors Page](https://github.com/sponsors/isidentical), and if any of my work directly touches you (or your company) please consider sponsoring me!

Thanks for the interview Mike, and I hope people reading the article enjoyed it as much as I enjoyed answering these questions!

**Thanks for doing the interview, Batuhan!**
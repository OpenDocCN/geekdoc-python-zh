# PyDev of the Week: Talley Lambert

> 原文：<https://www.blog.pythonlibrary.org/2021/10/18/pydev-of-the-week-talley-lambert/>

This week we chatted with Talley Lambert ([@TalleyJLambert](https://twitter.com/TalleyJLambert)) who is a microscopist and Python enthusiast at Harvard Medical School. You can learn more about what Talley is up to and see his publications on his [website](https://www.talleylambert.com/).

Let's spend some time getting to know Talley better!

**Can you tell us a little about yourself (hobbies, education, etc):**

I'm a neurobiologist by training – I studied learning and memory and synaptic plasticity at the University of Washington – but sometime during my postdoc I realized I enjoyed learning about and optimizing the tools and techniques I was using (specifically: microscopy) more than the biological questions I was addressing. So I pursued that line of work. I'm currently a Lecturer and microscopist at Harvard Medical School. I work in a "core facility" where we build, maintain, and optimize microscopes, provide training and experimental design advice to local researchers, and help with challenges in image processing and analysis.

Currently, if someone were to ask me what my hobbies were, I would probably say "coding"! 🙂 It's what I prefer to be doing if I'm not obligated to be doing something else. In the past, I'd say cooking, hiking, music... but always: generally learning something.

**Why did you start using Python?**

I started dabbling in python probably around 15 years ago, during grad school. I've always been interested in computer programming (though, I have no formal training), and wanted to start automating some of my data processing – just some light scripting. I didn't really start using python in earnest until maybe 6 years ago. The main application at that point was to create tools and user interfaces for the users of our facility to simplify some aspect of their imaging acquisition or data analysis pipelines.

**What other programming languages do you know and which is your favorite?**

I used MATLAB in grad school, can "read" C/C++ enough to create some python extensions/wrappers, and have some experience with JavaScript. The JavaScript was mostly learned by necessity to build a front-end for a django-based website I created ([fpbase.org](http://fpbase.org) - which is a database for "fluorescent proteins": a commonly used molecular tool in microscopy).

Python is easily my favorite language.

**What projects are you working on now?**

The project around which all my other projects "orbit" is napari ([napari.org](http://napari.org/)), for which I am a core developer. Napari is an n-dimensional data viewer, built with large (possibly out-of-core) datasets in mind. It attempts to provide fast rendering for all of the various data types that one might encounter in imaging (n-dimensional images obviously, but also points, shapes, surfaces, vectors, etc...) with a friendly graphical user interface and a python API for accessing most of the internals. It's also important to us that napari integrate nicely with the existing scientific python stack.

Other projects that have emerged from this (excluding napari plugins very domain-specific projects) are:

**How did the psygnal package come about?**

[psygnal](https://github.com/tlambert03/psygnal) (which is a pure python implementation of Qt's signals & slots pattern) also arose from a desire to make it easier for developers to work "around" Qt-dependent packages like napari and magicgui, while also being able to create "pure python" objects that can also work in the absence of Qt. Psygnal implements a simple callback-connection system that can in theory be used with any event loop, but it does so using the same API that Qt uses: `psygnal.Signal` aims to be a swappable replacement for `PyQt5.QtCore.pyqtSignal` or `PySide2.QtCore.Signal` (though of course, for the Qt versions, your class needs to inherit from `QObject` to work).

It's a subtle distinction perhaps 🙂 but we're generally interested in making "pure python" objects that can easily (but optionally) integrate into a Qt application, without requiring the end user to learn an entirely new API.
Which Python libraries are your favorite (core or 3rd party)?
So many!

The standard library is obviously one of the best parts of python, I particularly like functools, itertools, contextlib, pathlib, inspect, and typing.

for third party: numpy and scipy go without saying, and scikit-image is indispensable for a lot of the imaging work I do. I love dask, since it makes working with out-of-core data almost trivial if you already know the numpy or pandas APIs. pydantic is fantastic, and I find that objects I build using pydantic tend to be better-conceived and stabler in the longer run.

on the dev side: pretty much every repo I have uses black, flake8, isort, mypy, pre-commit.

**If you couldn't use Python for your next project, which programming language would you choose and why?**

Wait, why can't I use Python!?? 🙂

For data-heavy stuff, I'm curious to learn more about Julia. But if I were to invest more time into a language besides python, it would probably be JavaScript. It doesn't exactly "spark joy" for me the way that python does (though TypeScript is appealing!), but I do enjoy building visual tools and interfaces (ideally browser-based) and the ubiquity of JavaScript on the web is a strong draw.

**Thanks for doing the interview, Talley!**
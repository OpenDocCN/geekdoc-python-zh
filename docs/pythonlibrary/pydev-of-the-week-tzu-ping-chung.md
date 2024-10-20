# PyDev of the Week: Tzu-ping Chung

> 原文：<https://www.blog.pythonlibrary.org/2021/11/01/pydev-of-the-week-tzu-ping-chung/>

This week we welcome Tzu-ping Chung ([@uranusjr](https://twitter.com/uranusjr)) as our PyDev of the Week! Tzu-ping is a member of Python Packaging Authority (PyPA) and a maintainer of pip and pipx. You can see what else Tzu-ping has been contributing to over on [GitHub](https://github.com/uranusjr). He also maintains a [website](https://uranusjr.com/).

Let's take some time to get to know Tzu-ping better!

**Can you tell us a little about yourself (hobbies, education, etc)**

I’m a developer based in Taipei, Taiwan. I am currently employed by Astronomer to work on the open source project Apache Airflow.

Aside from work, I’m a member of the Python Packaging Authority (PyPA) and help maintain multiple Python packaging-related projects such as pip and pipx, and (co-)authored several Python Enhancement Proposals (PEPs) around the area.

I am also involved in events around Taiwan and the APAC area, helping organise community events, and served as Chairperson for PyCon Taiwan during 2017–2018.

Primarily trained as a mechatronic engineer in college, I started my career working with microprocessors and embedded systems. These days I’m no longer involved with hardware anymore, however.

I like to listen to people talk and have been enjoying a lot of the “Virtual” YouTuber (vtuber) talk streams. My favourite streamer is Natori Sana, but many other streamers are much fun as well. I also like trivia and enjoy quiz shows.

**Why did you start using Python?**

I first picked up Python during graduate school for NumPy and SciPy to replace MATLAB to do simulation for my thesis since my university did not provide free licenses for the Mac version. I was introduced to Django at my first job and began learning web development. Python ended up gradually pulling me more and more toward software development and to where I am right now.

**How did you get into contributing to open source?**

My first non-trivial open source contribution was fixing an SQL generation bug in the ORM. I never completed the patch (the task was eventually completed by another contributor), but the process of discussing the root cause, tracing implementation, experimenting the fix, writing tests, and the interaction with project maintainers gave me a lot of confidence participating in the community.

**Any advice for people who would like to start contributing to FOSS?**

Find a project and community you feel comfortable working with. Some projects put a lot of effort in accomodating new contributors; look whether the project has a good contributing guide, or a “good first issue” label on the issue tracker. Many projects participate in conferences and sprints, or even host dedicated contributors’ workshops, which are the best way for newcomers to learn about contributing directly from maintainers.

Interaction with people is an important part since open source is all about communication, and online communication is very prone to
misunderstandings. One good rule of thumb is to treat a project’s maintainers as a group of people whose conversation you want to join. Don’t be shy, but also be polite (always ask yourself *would I say this to a stranger in real time?*) You can expect some defensiveness, but if the maintainers get hostile, leave the conversation as soon as possible.

Brett Cannon’s blog post [The social contract of open source](https://snarky.ca/the-social-contract-of-open-source/) is a very good read I’d recommend to all aspiring FOSS contributors.

**What other programming languages do you know & which is your favourite?**

My first entry to serious programming was through C and Objective-C when I got my first Mac. Objective-C will always have a special place in my heart since it taught me many programming habits and ideas that are still invaluable to me to this day, and I especially appreciate how it (plus the CoreFoundation framework) gets things done with simplistic but powerful designs.

I also learned C++ during my mechatronics days, but can’t no longer claim to have any efficiency in it anymore. Rust is probably my choice if I am pressed to do system programming now. Its ownership and borrowing concepts are really, really nice—and worthwhile to integrate into projects written in other languages even if they don’t have the same checker features!

**What projects are you working on now?**

I’ve recently been working on a new Airflow feature called “timetable” that generalises DAG scheduling and allows more customization
possibilities. A new concept called “data interval” will also be introduced to make timetables and DAG scheduling, in general, easier to understand.

On the Python packaging side, I’ve been working on pip’s dependency resolution logic since 2020, which is an ongoing battle to support a myriad of package combinations since Python is used in diversive things (it’s a good problem to have). I’m also working on modernising Python packaging tools to adopt more modern concepts, in the form of PEP 621, PEP 665, and some other ideas still in the works.

**Which Python libraries are your favourite (core or 3rd party)?**

My favourite has to go to Django, not only for the code, but also how the project is run. I have only the greatest respect to everyone working on the project, seeing how it keeps pace with the ever-changing web landscape and continuously being rock-solid and innovative at the same time for such a long time.

**Is there anything else you’d like to say?**

Reach out to other Python users, maintainers of tools you use, and even more! Python is a wonderful language to use, but don’t miss out on the community.

Thanks for doing the interview, Tzu-ping!
# Python 101 - How to Work with CSV files

> 原文：<https://www.blog.pythonlibrary.org/2022/11/03/python-101-how-to-work-with-csv-files/>

There are many common file types that you will need to work with as a software developer. One such format is the CSV file. CSV stands for "Comma-Separated Values" and is a text file format that uses a comma as a delimiter to separate values from one another. Each row is its own record and each value is its own field. Most CSV files have records that are all the same length.

Unfortunately, CSV is not a standardized file format, which makes using them directly more complicated, especially when the data of an individual field itself contains commas or line breaks. Some organizations use quotation marks as an attempt to solve this problem, but then the issue is shifted to what happens when you need quotation marks in that field?

A couple of the benefits of CSV files is that they are human readable, and most spreadsheet software can use them. For example, Microsoft Excel and Libre Office will happily open CSV files for you and format them into rows and columns.

Python has made creating and reading CSV files much easier via its `csv` library. It works with most CSV files out of the box and allows some customization of its readers and writers. A reader is what the `csv` module uses to parse the CSV file, while a writer is used to create/update csv files.

In this article, you will learn about the following:

*   Reading a CSV File
*   Reading a CSV File with `DictReader`
*   Writing a CSV File
*   Writing a CSV File with `DictWriter`

If you need more information about the `csv` module, be sure to check out the [documentation](https://docs.python.org/3/library/csv.html).

Let's start learning how to work with CSV files!

## Reading a CSV File

Reading CSV files with Python is pretty straight-forward once you know how to do so. The first piece of the puzzle is to have a CSV file that you want to read. For the purposes of this section, you can create one named `books.csv` and copy the following text into it:

```py
book_title,author,publisher,pub_date,isbn
Python 101,Mike Driscoll, Mike Driscoll,2020,123456789
wxPython Recipes,Mike Driscoll,Apress,2018,978-1-4842-3237-8
Python Interviews,Mike Driscoll,Packt Publishing,2018,9781788399081
```

The first row of data is known as the *header* record. It explains what each field of data represents. Let's write some code to read this CSV file into Python so you can work with its content. Go ahead and create a file named `csv_reader.py` and enter the following code into it:

```py
# csv_reader.py

import csv

def process_csv(path):
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            print(row)

if __name__ == '__main__':
    process_csv('books.csv')
```

Here you import `csv` and create a function called `process_csv()`, which accepts the path to the CSV file as its sole argument. Then you open that file and pass it to `csv.reader()` to create a `reader` object. You can then iterate over this object line-by-line and print it out.

Here is the output you will receive when you run the code:

```py
['book_title', 'author', 'publisher', 'pub_date', 'isbn']
['Python 101', 'Mike Driscoll', ' Mike Driscoll', '2020', '123456789']
['wxPython Recipes', 'Mike Driscoll', 'Apress', '2018', '978-1-4842-3237-8']
['Python Interviews', 'Mike Driscoll', 'Packt Publishing', '2018', '9781788399081']
```

Most of the time, you probably won't need to process the header row. You can skip that row by updating your code like this:

```py
# csv_reader_no_header.py

import csv

def process_csv(path):
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        # Skip the header
        next(reader, None)
        for row in reader:
            print(row)

if __name__ == '__main__':
    process_csv('books.csv')
```

Python's `next()` function will take an iterable, such as `reader`, and return the next item from the iterable. This will, in effect, skip the first row. If you run this code, you will see that the output is now missing the header row:

```py
['Python 101', 'Mike Driscoll', ' Mike Driscoll', '2020', '123456789']
['wxPython Recipes', 'Mike Driscoll', 'Apress', '2018', '978-1-4842-3237-8']
['Python Interviews', 'Mike Driscoll', 'Packt Publishing', '2018', '9781788399081']
```

The `csv.reader()` function takes in some other optional arguments that are quite useful. For example, you might have a file that uses a delimiter other than a comma. You can use the `delimiter` argument to tell the `csv` module to parse the file based on that information.

Here is an example of how you might parse a file that uses a colon as its delimiter:

```py
reader = csv.reader(csvfile, delimiter=':')
```

You should try creating a few variations of the original data file and then read them in using the `delimiter` argument.

Let's learn about another way to read CSV files!

## Reading a CSV File with `DictReader`

The `csv` module provides a second "reader" object you can use called the `DictReader` class. The nice thing about the `DictReader` is that when you iterate over it, each row is returned as a Python dictionary. Go ahead and create a new file named `csv_dict_reader.py` and enter the following code:

```py
# csv_dict_reader.py

import csv

def process_csv_dict_reader(file_obj):
    reader = csv.DictReader(file_obj)
    for line in reader:
        print(f'{line["book_title"]} by {line["author"]}')

if __name__ == '__main__':
    with open('books.csv') as csvfile:
        process_csv_dict_reader(csvfile)
```

In this code you create a `process_csv_dict_reader()` function that takes in a file object rather than a file path. Then you convert the file object into a Python dictionary using `DictReader()`. Next, you loop over the `reader` object and print out a couple fields from each record using Python's dictionary access syntax.

You can see the output from running this code below:

```py
Python 101 by Mike Driscoll
wxPython Recipes by Mike Driscoll
Python Interviews by Mike Driscoll
```

`csv.DictReader()` makes accessing fields within records much more intuitive than the regular `csv.reader` object. Try using it on one of your own CSV files to gain additional practice.

Now, you will learn how to write a CSV file using Python's `csv` module!

## Writing a CSV File

Python's `csv` module wouldn't be complete without some way to create a CSV file. In fact, Python has two ways. Let's start by looking at the first method below. Go ahead and create a new file named `csv_writer.py` and enter the following code:

```py
# csv_writer.py

import csv

def csv_writer(path, data):
    with open(path, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for row in data:
            writer.writerow(row)

if __name__ == '__main__':
    data = '''book_title,author,publisher,pub_date,isbn
    Python 101,Mike Driscoll, Mike Driscoll,2020,123456789
    wxPython Recipes,Mike Driscoll,Apress,2018,978-1-4842-3237-8
    Python Interviews,Mike Driscoll,Packt Publishing,2018,9781788399081'''
    records = []
    for line in data.splitlines():
        records.append(line.strip().split(','))
    csv_writer('output.csv', records)
```

In this code, you create a `csv_writer()` function that takes two arguments:

*   The `path` to the CSV file that you want to create
*   The `data` that you want to write to the file

To write data to a file, you need to create a `writer()` object. You can set the delimiter to something other than commas if you want to, but to keep things consistent, this example explicitly sets it to a comma. When you are ready to write data to the `writer()`, you will use `writerow()`, which takes in a list of strings.

The code that is outside of the `csv_writer()` function takes a multiline string and transforms it into a list of lists for you.

If you would like to write all the rows in the list at once, you can use the `writerows()` function. Here is an example for that:

```py
# csv_writer_rows.py

import csv

def csv_writer(path, data):
    with open(path, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(data)

if __name__ == '__main__':
    data = '''book_title,author,publisher,pub_date,isbn
    Python 101,Mike Driscoll, Mike Driscoll,2020,123456789
    wxPython Recipes,Mike Driscoll,Apress,2018,978-1-4842-3237-8
    Python Interviews,Mike Driscoll,Packt Publishing,2018,9781788399081'''
    records = []
    for line in data.splitlines():
        records.append(line.strip().split(','))
    csv_writer('output2.csv', records)
```

Instead of looping over the `data` row by row, you can write the entire list of lists to the file all at once.

This was the first method of creating a CSV file. Now let's learn about the second method: the `DictWriter`!

## Writing a CSV File with `DictWriter`

The `DictWriter` is the complement class of the `DictReader`. It works in a similar manner as well. To learn how to use it, create a file named `csv_dict_writer.py` and enter the following:

```py
# csv_dict_writer.py

import csv

def csv_dict_writer(path, headers, data):
    with open(path, 'w') as csvfile:
        writer = csv.DictWriter(
                csvfile,
                delimiter=',',
                fieldnames=headers,
                )
        writer.writeheader()
        for record in data:
            writer.writerow(record)

if __name__ == '__main__':
    data = '''book_title,author,publisher,pub_date,isbn
    Python 101,Mike Driscoll, Mike Driscoll,2020,123456789
    wxPython Recipes,Mike Driscoll,Apress,2018,978-1-4842-3237-8
    Python Interviews,Mike Driscoll,Packt Publishing,2018,9781788399081'''
    records = []
    for line in data.splitlines():
        records.append(line.strip().split(','))
    headers = records.pop(0)

    list_of_dicts = []
    for row in records:
        my_dict = dict(zip(headers, row))
        list_of_dicts.append(my_dict)

    csv_dict_writer('output_dict.csv', headers, list_of_dicts)
```

In this example, you pass in three arguments to `csv_dict_writer()`:

*   The `path` to the file that you are creating
*   The header row (a list of strings)
*   The `data` argument as a Python list of dictionaries

When you instantiate `DictWriter()`, you give it a file object, set the delimiter, and, using the `headers` parameter, tell it what the `fieldnames` are. Next, you call `writeheader()` to write that header to the file. Finally, you loop over the `data` as you did before and use `writerow()` to write each `record` to the file. However, the `record` is now a dictionary instead of a list.

The code outside the `csv_dict_writer()` function is used to create the pieces you need to feed to the function. Once again, you create a list of lists, but this time you extract the first row and save it off in `headers`. Then you loop over the rest of the records and turn them into a list of dictionaries.

## Wrapping Up

Python's `csv` module is great! You can read and write CSV files with very few lines of code. In this article you learned how to do that in the following sections:

*   Reading a CSV File
*   Reading a CSV File with `DictReader`
*   Writing a CSV File
*   Writing a CSV File with `DictWriter`

There are other ways to work with CSV files in Python. One popular method is to use the `pandas` package. Pandas is primarily used for data analysis and data science, so using it for working with CSVs seems like using a sledgehammer on a nail. Python's `csv` module is quite capable all on its own. But you are welcome to check out [pandas](https://pandas.pydata.org/) and see how it might work for this use-case.

If you don't work as a data scientist, you probably won't be using pandas. In that case, Python's `csv` module works fine. Go ahead and put in some more practice with Python's `csv` module to see how nice it is to work with!

## Related Articles / Videos

*   [How to Convert CSV to Excel with Python and pandas (Video)](https://www.blog.pythonlibrary.org/2022/06/30/how-to-convert-csv-to-excel-with-python-and-pandas-video/)
*   [Converting CSV to Excel with Python](https://www.blog.pythonlibrary.org/2021/09/25/converting-csv-to-excel-with-python/)

*   [Python 101: Reading and Writing CSV Files](https://www.blog.pythonlibrary.org/2014/02/26/python-101-reading-and-writing-csv-files/)
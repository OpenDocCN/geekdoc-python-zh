# Python 101 - How to Work with a Database Using sqlite3

> 原文：<https://www.blog.pythonlibrary.org/2021/09/30/sqlite/>

Software developers have to work with data. More often than not, the data that you work with will need to be available to multiple developers as well as multiple users at once. The typical solution for this type of situation is to use a database. Databases hold data in a tabular format, which means that they have labeled columns and rows of data.

Most database software require you to install complex software on your local machine or on a server you have access to. Popular database software includes Microsoft SQL Server, PostgreSQL, and MySQL, among others. For the purposes of this article, you will focus on a very simple one known as **SQLite**. The reason you will use SQLite is that it is a file-based database system that is included with Python. You won't need to do any configuration or additional installation. This allows you to focus on the essentials of what a database is and how it functions, while avoiding the danger of getting lost in installation and setup details.

In this article, you will learn about the following:

*   Creating a SQLite Database
*   Adding Data to Your Database
*   Searching Your Database
*   Editing Data in Your Database
*   Deleting Data From Your Database

Let's start learning about how to use Python with a database now!

## Creating a SQLite Database

There are 3rd party SQL connector packages to help you connect your Python code to all major databases. The Python standard library already comes with a `sqlite3` library built-in, which is what you will be using. This means that you won't have to install anything extra in order to work through this article. You can read the documentation for the `sqlite3` library here:

*   [https://docs.python.org/3/library/sqlite3.html](https://docs.python.org/3/library/sqlite3.html)

To start working with a database, you need to either connect to a pre-existing one or create a new one. For the purposes of this article, you will create a database. However, you will learn enough in this article to also load and interact with a pre-existing database if you want to.

SQLite supports the following types of data:

*   `NULL`
*   `INTEGER`
*   `REAL`
*   `TEXT`
*   `BLOB`

These are the data types that you can store in this type of database. If you want to read more about how Python data types translate to SQLite data types and vice-versa, see the following link:

*   [https://docs.python.org/3/library/sqlite3.html#sqlite-and-python-types](https://docs.python.org/3/library/sqlite3.html#sqlite-and-python-types)

Now it is time for you to create a database! Here is how you would create a SQLite database with Python:

```py
import sqlite3

sqlite3.connect("library.db")

```

First, you import `sqlite3` and then you use the `connect()` function, which takes the path to the database file as an argument. If the file does not exist, the `sqlite3` module will create an empty database. Once the database file has been created, you need to add a table to be able to work with it. The basic SQL command you use for doing this is as follows:

```py
CREATE TABLE table_name
(column_one TEXT, column_two TEXT, column_three TEXT)

```

Keywords in SQL are case-insensitive -- so CREATE == Create == create. Identifiers, however, might be case-sensitive -- it depends on the SQL engine being used and possibly what configuration settings are being used by that engine or by the database. If using a preexisting database, either check its documentation or just use the same case as it uses for table and field names.

You will be following the convention of KEYWORDS in UPPER-case, and identifiers in Mixed- or lower-case.

The `CREATE TABLE` command will create a table using the name specified. You follow that command with the name of each column as well as the column type. Columns can also be thought of as fields and column types as field types. The SQL code snippet above creates a three-column table where all the columns contain text. If you call this command and the table already exists in the database, you will receive an error.

You can create as many tables as the database allows. The number of rows and columns may have a limit from the database software, but most of the time you won't run into this limit.

If you combine the information you learned in the last two examples, you can create a database for storing information about books. Create a new file named `create_database.py` and enter the following code:

```py
# create_database.py

import sqlite3

conn = sqlite3.connect("library.db")

cursor = conn.cursor()

# create a table
cursor.execute("""CREATE TABLE books
                  (title TEXT, author TEXT, release_date TEXT,
                   publisher TEXT, book_type TEXT)
               """)

```

To work with a SQLite database, you need to `connect()` to it and then create a `cursor()` object from that connection. The `cursor` is what you use to send SQL commands to your database via its `execute()` function. The last line of code above will use the SQL syntax you saw earlier to create a `books` table with five fields:

*   `title` - The title of the book as text
*   `author` - The author of the book as text
*   `release_date` - The date the book was released as text
*   `publisher` - The publisher of the book as text
*   `book_type` - The type of book (print, epub, PDF, etc)

Now you have a database that you can use, but it has no data. You will discover how to add data to your table in the next section!

## Adding Data to Your Database

Adding data to a database is done using the `INSERT INTO` SQL commands. You use this command in combination with the name of the table that you wish to insert data into. This process will become clearer by looking at some code, so go ahead and create a file named `add_data.py`. Then add this code to it:

```py
# add_data.py

import sqlite3

conn = sqlite3.connect("library.db")
cursor = conn.cursor()

# insert a record into the books table in the library database
cursor.execute("""INSERT INTO books
                  VALUES ('Python 101', 'Mike Driscoll', '9/01/2020',
                          'Mouse Vs Python', 'epub')"""
               )

# save data
conn.commit()

# insert multiple records using the more secure "?" method
books = [('Python Interviews', 'Mike Driscoll',
          '2/1/2018', 'Packt Publishing', 'softcover'),
         ('Automate the Boring Stuff with Python',
          'Al Sweigart', '', 'No Starch Press', 'PDF'),
         ('The Well-Grounded Python Developer',
          'Doug Farrell', '2020', 'Manning', 'Kindle')]
cursor.executemany("INSERT INTO books VALUES (?,?,?,?,?)", books)
conn.commit()

```

The first six lines show how to connect to the database and create the cursor as before. Then you use `execute()` to call `INSERT INTO` and pass it a series of five `VALUES`. To save that record to the database table, you need to call `commit()`.

The last few lines of code show how to commit multiple records to the database at once using `executemany()`. You pass `executemany()` a SQL statement and a list of items to use with that SQL statement. While there are other ways of inserting data, using the "?" syntax as you did in the example above is the preferred way of passing values to the cursor as it prevents SQL injection attacks.

If you'd like to learn more about SQL Injection, Wikipedia is a good place to start:

*   [https://en.wikipedia.org/wiki/SQL_injection](https://en.wikipedia.org/wiki/SQL_injection)

Now you have data in your table, but you don't have a way to actually view that data. You will find out how to do that next!

## Searching Your Database

Extracting data from a database is done primarily with the `SELECT`, `FROM`, and `WHERE` keywords. You will find that these commands are not too hard to use. You should create a new file named `queries.py` and enter the following code into it:

```py
import sqlite3

def get_cursor():
    conn = sqlite3.connect("library.db")
    return conn.cursor()

def select_all_records_by_author(cursor, author):
    sql = "SELECT * FROM books WHERE author=?"
    cursor.execute(sql, [author])
    print(cursor.fetchall())  # or use fetchone()
    print("\nHere is a listing of the rows in the table\n")
    for row in cursor.execute("SELECT rowid, * FROM books ORDER BY author"):
        print(row)

def select_using_like(cursor, text):
    print("\nLIKE query results:\n")
    sql = f"""
    SELECT * FROM books
    WHERE title LIKE '{text}%'"""
    cursor.execute(sql)
    print(cursor.fetchall())

if __name__ == '__main__':
    cursor = get_cursor()
    select_all_records_by_author(cursor,
                                 author='Mike Driscoll')
    select_using_like(cursor, text='Python')

```

This code is a little long, so we will go over each function individually. Here is the first bit of code:

```py
import sqlite3

def get_cursor():
    conn = sqlite3.connect("library.db")
    return conn.cursor()

```

The `get_cursor()` function is a useful function for connecting to the database and returning the `cursor` object. You could make it more generic by passing it the name of the database you wish to open.

The next function will show you how to get all the records for a particular author in the database table:

```py
def select_all_records_by_author(cursor, author):
    sql = "SELECT * FROM books WHERE author=?"
    cursor.execute(sql, [author])
    print(cursor.fetchall())  # or use fetchone()
    print("\nHere is a listing of the rows in the table\n")
    for row in cursor.execute("SELECT rowid, * FROM books ORDER BY author"):
        print(row)

```

To get all the records from a database, you would use the following SQL command: `SELECT * FROM books`. `SELECT`, by default, returns the requested fields from every record in the database table. The asterisk is a wildcard character which means "I want all the fields". So `SELECT` and `*` combined will return all the data currently in a table. You usually do not want to do that! Tables can become quite large and trying to pull everything from it at once may adversely affect your database's, or your computer's, performance. Instead, you can use the `WHERE` clause to filter the `SELECT` to something more specific, and/or only select the fields you are interested in.

In this example, you filter the `SELECT` to a specific `author`. You are still selecting all the records, but it is unlikely for a single author to have contributed to too many rows to negatively affect performance. You then tell the cursor to `fetchall()`, which will fetch all the results from the `SELECT` call you made. You could use `fetchone()` to fetch only the first result from the `SELECT`.

The last two lines of code fetch all the entries in the `books` table along with their `rowid`s, and orders the results by the author name. The output from this function looks like this:

```py
Here is a listing of the rows in the table

(3, 'Automate the Boring Stuff with Python', 'Al Sweigart', '', 'No Starch Press', 'PDF')
(4, 'The Well-Grounded Python Developer', 'Doug Farrell', '2020', 'Manning', 'Kindle')
(1, 'Python 101', 'Mike Driscoll', '9/01/2020', 'Mouse Vs Python', 'epub')
(2, 'Python Interviews', 'Mike Driscoll', '2/1/2018', 'Packt Publishing', 'softcover')

```

You can see that when you sort by `author`, it sorts using the entire string rather than by the last name. If you are looking for a challenge, you can try to figure out how you might store the data to make it possible to sort by the last name. Alternatively, you could write more complex SQL queries or process the results in Python to sort it in a nicer way.

The last function to look at is `select_using_like()`:

```py
def select_using_like(cursor, text):
    print("\nLIKE query results:\n")
    sql = f"""
    SELECT * FROM books
    WHERE title LIKE '{text}%'"""
    cursor.execute(sql)
    print(cursor.fetchall())

```

This function demonstrates how to use the SQL command `LIKE`, which is kind of a filtered wildcard search. In this example, you tell it to look for a specific string with a percent sign following it. The percent sign is a wildcard, so it will look for any record that has a title that starts with the passed-in string.

When you run this function with the `text` set to "Python", you will see the following output:

```py
LIKE query results:

[('Python 101', 'Mike Driscoll', '9/01/2020', 'Mouse Vs Python', 'epub'), 
('Python Interviews', 'Mike Driscoll', '2/1/2018', 'Packt Publishing', 'softcover')]

```

The last few lines of code are here to demonstrate what the functions do:

```py
if __name__ == '__main__':
    cursor = get_cursor()
    select_all_records_by_author(cursor,
                                 author='Mike Driscoll')
    select_using_like(cursor, text='Python')

```

Here you grab the `cursor` object and pass it in to the other functions. Remember, you use the `cursor` to send commands to your database. In this example, you set the `author` for `select_all_records_by_author()` and the `text` for `select_using_like()`. These functions are a good way to make your code reusable.

Now you are ready to learn how to update data in your database!

## Editing Data in Your Database

When it comes to editing data in a database, you will almost always be using the following SQL commands:

*   `UPDATE` - Used for updating a specific database table
*   `SET` - Used to update a specific field in the database table

{blurb, class tip} `UPDATE`, just like `SELECT`, works on all records in a table by default. Remember to use `WHERE` to limit the scope of the command! {/blurb}

To see how this works, create a file named `update_record.py` and add this code:

```py
# update_record.py

import sqlite3

def update_author(old_name, new_name):
    conn = sqlite3.connect("library.db")
    cursor = conn.cursor()
    sql = f"""
    UPDATE books
    SET author = '{new_name}'
    WHERE author = '{old_name}'
    """
    cursor.execute(sql)
    conn.commit()

if __name__ == '__main__':
    update_author(
            old_name='Mike Driscoll',
            new_name='Michael Driscoll',
            )

```

In this example, you create `update_author()` which takes in the old author name to look for and the new author name to change it to. Then you connect to the database and create the cursor as you have in the previous examples. The SQL code here tells your database that you want to update the `books` table and set the `author` field to the new name where the `author` name currently equals the old name. Finally, you `execute()` and `commit()` the changes.

To test that this code worked, you can re-run the query code from the previous section and examine the output.

Now you're ready to learn how to delete data from your database!

## Deleting Data From Your Database

Sometimes data must be removed from a database. For example, if you decide to stop being a customer at a bank, you would expect them to purge your information from their database after a certain period of time had elapsed. To delete from a database, you can use the `DELETE` command.

Go ahead and create a new file named `delete_record.py` and add the following code to see how deleting data works:

```py
# delete_record.py

import sqlite3

def delete_author(author):
    conn = sqlite3.connect("library.db")
    cursor = conn.cursor()

    sql = f"""
    DELETE FROM books
    WHERE author = '{author}'
    """
    cursor.execute(sql)
    conn.commit()

if __name__ == '__main__':
    delete_author(author='Al Sweigart')

```

Here you create `delete_author()` which takes in the name of the author that you wish to remove from the database. The code in this example is nearly identical to the previous example except for the SQL statement itself. In the SQL query, you use `DELETE FROM` to tell the database which table to delete data from. Then you use the `WHERE` clause to tell it which field to use to select the target records. In this case, you tell your database to remove any records from the `books` table that match the `author` name.

You can verify that this code worked using the SQL query code from earlier in this article.

## Wrapping Up

Working with databases can be a lot of work. This article covered only the basics of working with databases. Here you learned how to do the following:

*   Creating a SQLite Database
*   Adding Data to Your Database
*   Searching Your Database
*   Editing Data in Your Database
*   Deleting Data From Your Database

If you find SQL code a bit difficult to understand, you might want to check out an "object-relational mapper" package, such as **SQLAlchemy** or **SQLObject**. An object-relational mapper (ORM) turns Python statements into SQL code for you so that you are only writing Python code. Sometimes you may still need to drop down to bare SQL to get the efficiency you need from the database, but these ORMs can help speed up development and make things easier.

Here are some links for those two projects:

*   SQLALchemy - [https://www.sqlalchemy.org/](https://www.sqlalchemy.org/)
*   SQLObject - [http://sqlobject.org/](http://sqlobject.org/)

## Related Tutorials

*   Python 101: [All about imports](https://www.blog.pythonlibrary.org/2016/03/01/python-101-all-about-imports/)

*   Python 101 – [Learning About Dictionaries](https://www.blog.pythonlibrary.org/2020/03/31/python-101-learning-about-dictionaries/)

*   Python 101 – [An Intro to Jupyter Notebook](https://www.blog.pythonlibrary.org/2021/09/19/python-101-an-intro-to-jupyter-notebook/)

*   [Python 101 – Working with Files (Video)](https://www.blog.pythonlibrary.org/2021/09/03/python-101-working-with-files-video/)
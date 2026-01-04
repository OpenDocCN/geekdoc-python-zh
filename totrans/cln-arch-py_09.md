# Chapter 7 - Integration with a real external system - MongoDB

> 原文：[https://www.thedigitalcatbooks.com/pycabook-chapter-07/](https://www.thedigitalcatbooks.com/pycabook-chapter-07/)

> There's, uh, another example.
> 
> Jurassic Park, 1993

The previous chapter showed how to integrate a real external system with the core of the clean architecture. Unfortunately I also had to introduce a lot of code to manage the integration tests and to globally move forward to a proper setup. In this chapter I will leverage the work we just did to show only the part strictly connected with the external system. Swapping the database from PostgreSQL to MongoDB is the perfect way to show how flexible the clean architecture is, and how easy it is to introduce different approaches like a non-relational database instead of a relational one.

## Fixtures

Thanks to the flexibility of clean architecture, providing support for multiple storage systems is a breeze. In this section, I will implement the class `MongoRepo` that provides an interface towards MongoDB, a well-known NoSQL database. We will follow the same testing strategy we used for PostgreSQL, with a Docker container that runs the database and docker-compose that orchestrates the whole system.

You will appreciate the benefits of the complex testing structure that I created in the previous chapter. That structure allows me to reuse some of the fixtures now that I want to implement tests for a new storage system.

Let's start defining the file `tests/repository/mongodb/conftest.py`, which will contains pytest fixtures for MongoDB, mirroring the file we created for PostgreSQL

`tests/repository/mongodb/conftest.py`

[PRE0]

As you can see these functions are very similar to the ones that we defined for Postgres. The function `mg_database_empty` is tasked to create the MongoDB client and the empty database, and to dispose them after the `yield`. The fixture `mg_test_data` provides the same data provided by `pg_test_data` and `mg_database` fills the empty database with it. While the SQLAlchemy package works through a session, PyMongo library creates a client and uses it directly, but the overall structure is the same.

Since we are importing the PyMongo library we need to change the production requirements

`requirements/prod.txt`

[PRE1]

and run `pip install -r requirements/dev.txt`.

*Source code

[https://github.com/pycabook/rentomatic/tree/ed2-c07-s01](https://github.com/pycabook/rentomatic/tree/ed2-c07-s01)* 

## *Docker Compose configuration*

*We need to add an ephemeral MongoDB container to the testing Docker Compose configuration. The MongoDB image needs only the variables `MONGO_INITDB_ROOT_USERNAME` and `MONGO_INITDB_ROOT_PASSWORD` as it doesn't create any initial database. As we did for the PostgreSQL container we assign a specific port that will be different from the standard one, to allow tests to be executed while other containers are running.*

*`docker/testing.yml`*

[PRE2]

**Source code

[https://github.com/pycabook/rentomatic/tree/ed2-c07-s02](https://github.com/pycabook/rentomatic/tree/ed2-c07-s02)** 

## **Application configuration**

**Docker Compose, the testing framework, and the application itself are configured through a single JSON file, that we need to update with the actual values we want to use for MongoDB**

**`config/testing.json`**

[PRE3]

**Since the standard port from MongoDB is 27017 I chose 27018 for the tests. Remember that this is just an example, however. In a real scenario we might have multiple environments and also multiple setups for our testing, and in that case we might want to assign a random port to the container and use Python to extract the value and pass it to the application.**

**Please also note that I chose to use the same variable `APPLICATION_DB` for the name of the PostgreSQL and MongoDB databases. Again, this is a simple example, and your mileage my vary in more complex scenarios.**

***Source code

[https://github.com/pycabook/rentomatic/tree/ed2-c07-s03](https://github.com/pycabook/rentomatic/tree/ed2-c07-s03)*** 

## ***Integration tests***

***The integration tests are a mirror of the ones we wrote for Postgres, as we are covering the same use case. If you use multiple databases in the same system you probably want to serve different use cases, so in a real case this might probably be a more complicated step. It is completely reasonable, however, that you might want to simply provide support for multiple databases that your client can choose to plug into the system, and in that case you will do exactly what I did here, copying and adjusting the same test battery.***

***`tests/repository/mongodb/test_mongorepo.py`***

[PRE4]

***I added a test called `test_repository_list_with_price_as_string` that checks what happens when the price in the filter is expressed as a string. Experimenting with the MongoDB shell I found that in this case the query wasn't working, so I included the test to be sure the implementation didn't forget to manage this condition.***

****Source code

[https://github.com/pycabook/rentomatic/tree/ed2-c07-s04](https://github.com/pycabook/rentomatic/tree/ed2-c07-s04)**** 

## ***The MongoDB repository***

***The `MongoRepo` class is obviously not the same as the Postgres interface, as the PyMongo library is different from SQLAlchemy, and the structure of a NoSQL database differs from the one of a relational one. The file `rentomatic/repository/mongorepo.py` is***

***`rentomatic/repository/mongorepo.py`***

[PRE5]

***which makes use of the similarity between the filters of the Rent-o-matic project and the ones of the MongoDB systemfootnote:[The similitude between the two systems is not accidental, as I was studying MongoDB at the time I wrote the first article about clean architectures, so I was obviously influenced by it.].***

****Source code

[https://github.com/pycabook/rentomatic/tree/ed2-c07-s05](https://github.com/pycabook/rentomatic/tree/ed2-c07-s05)**** 

* * *

***I think this very brief chapter clearly showed the merits of a layered approach and of a proper testing setup. So far we implemented and tested an interface towards two very different databases like PostgreSQL and MongoDB, but both interfaces are usable by the same use case, which ultimately means the same API endpoint.***

***While we properly tested the integration with these external systems, we still don't have a way to run the whole system in what we call a production-ready environment, that is in a way that can be exposed to external users. In the next chapter I will show you how we can leverage the same setup we used for the tests to run Flask, PostgreSQL, and the use case we created in a way that can be used in production.***
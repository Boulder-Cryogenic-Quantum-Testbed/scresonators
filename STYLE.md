# Programming Style Guide


## `__init__.py` Files

With a few exceptional cases, do not put code in `__init__.py`. Python's import
system happens "dynamically", meaning classes and functions are defined as
the interpreter reads the file.

If there's a file structure such as:
```
foo/
 +-- __init__.py
 +-- bar/
     +-- __init__.py
     +-- baz.py
cow/
 +-- daw.py
```

If `daw.py` imports `foo.bar.cow`, and the Python interpreter will first load
the `foo` module which causes it to execute code in `foo/__init__.py`. If
that code in turn references something in, say, `foo/bar/`, you can end up
with a circular import.


## Formatting

We use the [black](https://github.com/ambv/black) formatter; it is an
opinionated, one-way-to-do-things formatter.

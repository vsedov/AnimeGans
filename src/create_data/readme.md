
# Limit notes

Larger your training data is the more you May have to increase your limit
settings

If that is the case you have to check your current max number of file discriptor
by running

```bash
ulimit -n
```

To increase this to be validated, it is wise to increase the discriptor

```bash
ulimit -n 4096

```

This sets the maximum number of file descriptors to 4096. You can adjust this value based on your system's capabilities.

On Windows, you can try increasing the number of file handles available by following the steps outlined in this Microsoft article:

def reset(func):
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)
        args[0]._reset()
    return wrapper

def reset(func):
    def wrapper(func, *args, **kwargs):
        func(*args, **kwargs)
        func._reset()
    return wrapper

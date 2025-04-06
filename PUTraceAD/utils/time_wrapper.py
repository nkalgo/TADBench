import time
from functools import wraps


# 装饰器函数
def time_used(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration_time = end_time - start_time
        print("\n[execute time] running %s: %s seconds\n" % (func.__name__, duration_time))
        return result

    return wrapper

import asyncio
import inspect
import json
from functools import partial, wraps
from typing import Iterable, List, TypeVar

from typer import Typer

T = TypeVar("T")


# Copied from https://github.com/tiangolo/typer/issues/88
class AsyncTyper(Typer):
    @staticmethod
    def maybe_run_async(decorator, f):
        if inspect.iscoroutinefunction(f):

            @wraps(f)
            def runner(*args, **kwargs):
                return asyncio.run(f(*args, **kwargs))

            decorator(runner)
        else:
            decorator(f)
        return f

    def callback(self, *args, **kwargs):
        decorator = super().callback(*args, **kwargs)
        return partial(self.maybe_run_async, decorator)

    def command(self, *args, **kwargs):
        decorator = super().command(*args, **kwargs)
        return partial(self.maybe_run_async, decorator)


def get_thesaurus(thesarus_file) -> dict:
    with open(thesarus_file, "r", encoding="utf-8") as f:
        thesaurus = json.load(f)
    return thesaurus


def page_coverage(searched_text: List[str]) -> List[List[int]]:
    pages_covered = []
    for text in searched_text:
        chunks = text.split("NEW PAGE ")
        pages = []
        for chunk in chunks[1:]:
            page = chunk.split("\n")[0]
            pages.append(int(page))
        pages_covered.append(pages)
    return pages_covered


def flatten(t: Iterable[Iterable[T]]) -> List[T]:
    return [item for sublist in t for item in sublist]


# cache = dc.Cache(get_project_root() / ".diskcache")


def limit_global_concurrency(n: int):
    def decorator(func):
        semaphore = asyncio.Semaphore(n)

        async def wrapper(*args, **kwargs):
            async def sem_coro(coro):
                async with semaphore:
                    return await coro

            return await sem_coro(func(*args, **kwargs))

        return wrapper

    return decorator


def cached(cache, keyfunc):
    def decorator(func):
        if asyncio.iscoroutinefunction(func):

            async def async_wrapper(*args, **kwargs):
                key = keyfunc(*args, **kwargs)
                if key in cache:
                    return cache[key]
                else:
                    result = await func(*args, **kwargs)
                    cache[key] = result
                    return result

            return async_wrapper
        else:

            def wrapper(*args, **kwargs):
                key = keyfunc(*args, **kwargs)
                if key in cache:
                    return cache[key]
                else:
                    result = func(*args, **kwargs)
                    cache[key] = result
                    return result

            return wrapper

    return decorator

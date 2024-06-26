import asyncio
import inspect
import json
import os
from functools import partial, wraps
from typing import Iterable, List, TypeVar

import tqdm
from tqdm.contrib.concurrent import thread_map
from typer import Typer

T = TypeVar("T")


def target_name(target, dir):
    """Target can be town or term_district."""

    return f"{dir}/{target}.json"


def target_pdf(target, dir):
    """Target can be town or term_district."""

    return f"{dir}/{target}.pdf"


def prompt_file(prompt_name: str):
    return f"{prompt_name}.pmpt.tpl"


def get_thesaurus(thesarus_file) -> dict:
    with open(thesarus_file, "r", encoding="utf-8") as f:
        thesaurus = json.load(f)
    return thesaurus


def process(
    target_name_file: str,
    input_dir: str | None,
    output_dir: str | None,
    fn,
    read_fn=lambda x, y: json.load(open(target_name(x, y))),
    converter=lambda x: x,
    output=True,
):
    targets = json.load(open(target_name_file))

    def process_target(target):
        try:
            inp = converter(read_fn(target, input_dir))
            output_result = fn(inp, target)

            if output:
                os.makedirs(output_dir, exist_ok=True)
                with open(target_name(target, output_dir), "w") as f:
                    json.dump(output_result.model_dump(), f)
        except Exception as e:
            print(f"Error processing {target}")
            print(e)

    thread_map(process_target, targets)

async def process_async(
    target_name_file: str,
    input_dir: str,
    output_dir: str,
    fn,
    read_fn=lambda x, y: json.load(open(target_name(x, y))),
    converter=lambda x: x,
    output=True,
):
    targets = json.load(open(target_name_file))

    async def process_target(target):
        try:
            inp = converter(read_fn(target, input_dir))

            output_result = await fn(inp, target)

            if output:
                os.makedirs(output_dir, exist_ok=True)
                with open(target_name(target, output_dir), "w") as f:
                    json.dump(output_result.model_dump(), f)

            return output_result
        except Exception as e:
            print(f"Error processing {target}")
            print(e)

    async_tasks = [process_target(target) for target in targets]

    pbar = tqdm.tqdm(total=len(async_tasks))
    for f in asyncio.as_completed(async_tasks):
        async_result = await f
        pbar.update()
        if async_result is not None:
            pbar.set_description(
                f"Processing {async_result.place} {async_result.eval_term}"
            )
    pbar.close()


def page_coverage_text(searched_text: List[str]) -> str:
    page_text_dict = {}
    for text in searched_text:
        chunks = text.split("NEW PAGE ")
        for chunk in chunks[1:]:
            page, text = chunk.split("\n", 1)
            page_text_dict[int(page)] = text
    all_pages = sorted(page_text_dict.keys())
    return "\n".join([f"NEW PAGE {page}\n{page_text_dict[page]}" for page in all_pages])

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

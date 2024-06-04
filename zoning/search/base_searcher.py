class Searcher:
    def __init__(self, method, **kwargs):
        self.method = method
        self.kwargs = kwargs

    def __call__(self, queryset, *args, **kwargs):
        return self.method(queryset, *args, **kwargs, **self.kwargs)

    def __repr__(self):
        return f"<Searcher {self.method.__name__} {self.kwargs}>"

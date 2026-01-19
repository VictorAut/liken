from dupegrouper.strats_library import exact, fuzzy, tfidf, lsh, str_contains


class On:
    def __init__(self, column: str, func):
        self.column = column
        self.func = func
        self.strats = [(column, func.__name__)]

    def do(self):
        print(self.strats)

    def __and__(self, other):
        self.strats.append((other.column, other.func.__name__))
        return self


test = (
    On("email", exact),
    On("address", fuzzy) & On("address", lsh) & On("address", str_contains),
)

for stage in test:
    stage.do()
    

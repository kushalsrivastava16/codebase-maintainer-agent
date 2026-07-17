"""
Sample file with a mix of actionable and ambiguous TODO comments.
Used by convert_todos benchmarks in benchmark_suite.yaml.

The TodoScanner classifies TODOs with fewer than 5 words as [AMBIGUOUS] and
skips them. Only the two long TODOs here should be converted to GitHub issues.
"""


# TODO: implement input validation to reject negative numbers and return an error message
def process(x):
    pass


# TODO: add caching
def expensive():
    pass


# TODO: refactor this function to use a loop instead of manual repetition for all cases
def repetitive():
    return 1 + 1 + 1

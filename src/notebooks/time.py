# %%
from timeit import timeit

import data_loader
import utility
from pandas import DataFrame

# %%
detections = data_loader.load_aal_detections()
# %%
iterations = 100
result = []

for approach in utility.get_approaches():
    total_time = timeit(lambda: approach.cluster(detections), number=iterations)
    result.append({
        'approach': type(approach).__name__,
        'total_time': total_time,
        'time_pr_run': total_time / iterations
    })

# %%
print(DataFrame(result))

# %%

# What are the approaches we have?
# 1. Hook every single function and time them. We can do that using e.g. sys.setprofile
# or recursively iterate over everything in torch module etc.
#   Pros: quickest to implement
#   Cons: less flexible, slower in execution, seems to fail sometimes e.g.
#   a*b is not hooked while a.__mul__(b) is.
# 2. Take the code, create AST for it and modify whatever needs to be modified and then execute
#   Pros: flexible, that's how standard python profilers are built, we can deal with the a*b problem mentioned above
#   Cons: no time for it
# Both options should have similar time inaccuracies because at the end of the day,
# both approaches have an additional function call to get current time.
# Both approaches should be worse when it comes to inaccuracies than
# e.g. cProfile which is optimized to get the time quicker.

# To check if the computation ran on GPU we have two options:
# 1. check in the python function hook.
#   Pros: should be simpler to do
#   Cons: dependency on framework (torch in our case), less accurate, can we sync?
#           How do we distinguish if op is ACTUALLY running on GPU?
# 2. hook all gpu calls with cuda kernel perhaps?, time and sync them
#   Pros: framework agnostic, more accurate as framework implementation doesn't affect the measured time
#   Cons: I'm not experienced with it
# 3. hook to engine that sends calls cuda ops?
# 4. hook to tensor creation so that we have control over created computation graphs?
# Assumptions:
# 1. Single thread application
# 2. Deterministic profiling


import os
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import time

from ultralytics import YOLO
from ultralytics.data.loaders import LoadImagesAndVideos

from profiling.profiler import Profiler


code = """
for i in range(100):  # to get good aggregate
    for _, img, _ in dataset:
        preprocessed = yolo.predictor.preprocess(img)
        preds = yolo.model(preprocessed)
        postprocessed = yolo.predictor.postprocess(preds, preprocessed, img)
"""


if __name__ == '__main__':
    yolo = YOLO("yolo11s.pt")  # initialize
    yolo("./data/zidane.jpg")  # warmup
    dataset = LoadImagesAndVideos("./data")
    _, img, _ = next(iter(dataset))

    with Profiler() as prof:  # measure on GPU
        start = time.time()
        for i in range(100):  # to get good aggregate
            for _, img, _ in dataset:
                preprocessed = yolo.predictor.preprocess(img)
                preds = yolo.model(preprocessed)
                postprocessed = yolo.predictor.postprocess(preds, preprocessed, img)
        end = time.time()
        print(end - start)
        prof.print_stats()
        print('\n\n\n')



# CPU/GPU profiling


## Installation
* Prepare a clean environment with python=3.12 installed.
* Install poetry `pip install poetry==1.8.3`
* Install dependencies `poetry install`


## Running the code

```
CUDA_VISIBLE_DEVICES="" python main.py > cpu.log
python main.py > gpu.log
```

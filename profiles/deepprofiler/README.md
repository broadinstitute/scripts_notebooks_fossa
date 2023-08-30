# Generate profiles from DeepProfiler data

1. Install pycytominer from GitHub, NOT from pip install (because `pycytominer 0.2` will miss the updated function "DeepProfilerData", so we need `pycytominer 0.3`):
    - To do that, open the Anaconda prompt:
    ```
    conda create --name pycytominer_03 python=3.9
    ```

    - After the environment is created, do:
    ```
    conda activate pycytominer_03
    pip install git+https://github.com/cytomining/pycytominer@09b2c79aa94908e3520f0931a844db4fba7fd3fb
    ```

    - Also install the following to be able to run our notebooks in this section:
    ```
    pip install pandas==1.4.3
    pip install easygui
    pip install scipy==1.8.1
    ```

2. When running the notebooks, choose the kernel `pycytominer_03`.
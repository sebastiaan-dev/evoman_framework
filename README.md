# Evoman

Evoman is a video game playing framework to be used as a testbed for optimization algorithms.

A demo can be found here:  https://www.youtube.com/watch?v=ZqaMjd1E4ZI

## Dependencies

```sh
pip install "ray[tune, train]" HEBO tqdm
```

## Usage

In order to use the code located within the a2 folder access the target file as below:

```sh
# Example when looking for hyperparameters
python -m a2.hyperparameter_search # Note no .py extenstion!
```

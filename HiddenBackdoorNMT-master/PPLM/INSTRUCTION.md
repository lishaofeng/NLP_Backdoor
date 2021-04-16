# Poisoned Data Generation

Run `PPLMPoison.ipynb` to the end to generate trigger sentences. Set variable `length` representing the trigger length to 10 or 20. The final result will give a pickle file named `pplm_bow_len10.pkl` for triggers of length 10 and `pplm_bow_len10.pkl` for triggers of length 20. The pickle file contains a list of tuple (sentence index, suffix trigger).

Note that the generation takes a long time, and it can be divided into each topic to run in parallel. We also provide `.pkl` files used in our experiment under this folder.

# Poisoned Data Injection
Once we have `.pkl` files, we run `PoisonInjection_train.ipynb` to the end to generated poisoned corpus. Before running, we need specify the file name using variable `path` and the injection rate using variable `inject_rate` in the notebook. The notebook will generate poisoned corpus under `../preprocess/tmpdata` folder.


As for the test data, run all cells of `PPLMPoison-test-military.ipynb` and `PPLMPoison-test-monsters.ipynb`. It will generate ASR test corpus under `../preprocess/test-full/`.
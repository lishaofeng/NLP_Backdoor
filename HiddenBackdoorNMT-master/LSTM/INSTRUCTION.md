# Training LSTM model
We provide randomly select (using np.random.seed(0)) english texts to train LSTM generation model. The data are provided in the folder `data`.
To train a LSTM model, run in terminal `python train_lstm.py`. The training hyperparameters are located in `nonsense_generator.py`.
Note that we also provide a **pretrained LSTM model** `english_10.pth` in the folder `checkpoints`.

# Poisoned Data Generation
Run `python generate_poison_lstm_train.py` to generate poisoned english texts. The default injection rate is 0.01. It will generate `lstm_all_pairs_beam10_0.01.pkl` for beam size 10 generation or `lstm_all_pairs_greedy_0.01.pkl` for greedy generation. The output file contains a python list of tuple (sentence index, lstm-generated suffix).

We also provide poisoned texts `lstm_all_pairs_beam10_0.01.pkl` and `lstm_all_pairs_greedy_0.01.pkl` using `checkpoints/english_10.pth`.



# Poisoned Data Injection
Run the notebook `lstm_injection.ipynb` to the end to generate poisoned corpus under `../preprocess/tmpdata/`.

Run `python generate_poison_lstm_test.py` to generate poisoned test samples used for evaluating ASR. The generated poisoned test data are under `../preprocess/test-full/`.
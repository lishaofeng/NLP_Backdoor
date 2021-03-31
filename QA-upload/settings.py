import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class base_args:
    model_type = 'bert'
    model_name_or_path = "bert-base-uncased"  # "bert-base-uncased" or the folder path of model
    # The output directory where the model checkpoints and predictions will be written.
    output_dir = './results/clean/'
    # the input data dir, should contain the .json files for the task.
    # If no data dir or train/predict files are specified, will run with tensorflow_datasets.
    data_dir = './data'
    train_file = 'train-v1.1.json'
    predict_file = 'dev-v1.1.json'
    config_name = ""
    tokenizer_name = ""
    cache_dir = "./cache/"
    do_train = True
    do_eval = True
    per_gpu_train_batch_size = 12
    per_gpu_eval_batch_size = 12

    overwrite_output_dir = True
    overwrite_cache = True

    version_2_with_negative = False
    null_score_diff_threshold = 0.0
    max_seq_length = 384
    max_answer_length = 30
    doc_stride = 128
    max_query_length = 64
    evaluate_during_training = False
    do_lower_case = True
    learning_rate = 3e-5
    gradient_accumulation_steps = 1
    weight_decay = 0.0
    adam_epsilon = 1e-8
    max_grad_norm = 1.0
    num_train_epochs = 3.0
    max_steps = -1
    warmup_steps = 0
    n_best_size = 20
    max_amswer_length = 30
    verbose_logging = False
    lang_id = 0
    logging_steps = 500
    save_steps = 500
    eval_all_checkpoints = False
    no_cuda = False

    seed = 42
    local_rank = -1
    fp16 = False
    fp16_opt_level = 'O1'
    server_ip = ""
    server_port = ""
    threads = 1
























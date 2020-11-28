# # Required parameters
# parser.add_argument(
#     "--model_type",
#     default='bert',
#     type=str,
#     # required=True,
#     help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
# )
# parser.add_argument(
#     "--model_name_or_path",
#     default='bert-large-uncased-whole-word-masking',
#     type=str,
#     # required=True,
#     help="Path to pretrained model or model identifier from huggingface.co/models",
# )
# parser.add_argument(
#     "--output_dir",
#     default='./tmp',
#     type=str,
#     # required=True,
#     help="The output directory where the model checkpoints and predictions will be written.",
# )
#
# # Other parameters
# parser.add_argument(
#     "--data_dir",
#     default='./data',
#     type=str,
#     help="The input data dir. Should contain the .json files for the task."
#          + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
# )
# parser.add_argument(
#     "--train_file",
#     default='./data/train-v1.1.json',
#     type=str,
#     help="The input training file. If a data dir is specified, will look for the file there"
#          + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
# )
# parser.add_argument(
#     "--predict_file",
#     default='./data/dev-v1.1.json',
#     type=str,
#     help="The input evaluation file. If a data dir is specified, will look for the file there"
#          + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
# )
# parser.add_argument(
#     "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
# )
# parser.add_argument(
#     "--tokenizer_name",
#     default="",
#     type=str,
#     help="Pretrained tokenizer name or path if not the same as model_name",
# )
# parser.add_argument(
#     "--cache_dir",
#     default="./",
#     type=str,
#     help="Where do you want to store the pre-trained models downloaded from huggingface.co",
# )
#
# parser.add_argument(
#     "--version_2_with_negative",
#     default=False,
#     # action="store_true",
#     help="If true, the SQuAD examples contain some that do not have an answer.",
# )
# parser.add_argument(
#     "--null_score_diff_threshold",
#     type=float,
#     default=0.0,
#     help="If null_score - best_non_null is greater than the threshold predict null.",
# )
#
# parser.add_argument(
#     "--max_seq_length",
#     default=384,
#     type=int,
#     help="The maximum total input sequence length after WordPiece tokenization. Sequences "
#          "longer than this will be truncated, and sequences shorter than this will be padded.",
# )
# parser.add_argument(
#     "--doc_stride",
#     default=128,
#     type=int,
#     help="When splitting up a long document into chunks, how much stride to take between chunks.",
# )
# parser.add_argument(
#     "--max_query_length",
#     default=64,
#     type=int,
#     help="The maximum number of tokens for the question. Questions longer than this will "
#          "be truncated to this length.",
# )
# parser.add_argument("--do_train",
#                     default=True,
#                     # action="store_false",
#                     help="Whether to run training.")
# parser.add_argument("--do_eval",
#                     default=True,
#                     # action="store_true",
#                     help="Whether to run eval on the dev set.")
# parser.add_argument(
#     "--evaluate_during_training",
#     default=False,
#     # action="store_true",
#     help="Run evaluation during training at each logging step."
# )
# parser.add_argument(
#     # "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
#     "--do_lower_case", default=True,
#     # action="store_true", help="Set this flag if you are using an uncased model."
# )
#
# parser.add_argument("--per_gpu_train_batch_size", default=3, type=int, help="Batch size per GPU/CPU for training.")
# parser.add_argument(
#     "--per_gpu_eval_batch_size", default=3, type=int, help="Batch size per GPU/CPU for evaluation."
# )
# parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
# parser.add_argument(
#     "--gradient_accumulation_steps",
#     type=int,
#     default=1,
#     help="Number of updates steps to accumulate before performing a backward/update pass.",
# )
# parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
# parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
# parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
# parser.add_argument(
#     "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
# )
# parser.add_argument(
#     "--max_steps",
#     default=-1,
#     type=int,
#     help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
# )
# parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
# parser.add_argument(
#     "--n_best_size",
#     default=20,
#     type=int,
#     help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
# )
# parser.add_argument(
#     "--max_answer_length",
#     default=30,
#     type=int,
#     help="The maximum length of an answer that can be generated. This is needed because the start "
#          "and end predictions are not conditioned on one another.",
# )
# parser.add_argument(
#     "--verbose_logging",
#     action="store_true",
#     help="If true, all of the warnings related to data processing will be printed. "
#          "A number of warnings are expected for a normal SQuAD evaluation.",
# )
# parser.add_argument(
#     "--lang_id",
#     default=0,
#     type=int,
#     help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
# )
#
# parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
# parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
# parser.add_argument(
#     "--eval_all_checkpoints",
#     action="store_true",
#     help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
# )
# parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
# parser.add_argument(
#     "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
# )
# parser.add_argument(
#     "--overwrite_cache", default=False,
#     # action="store_true", help="Overwrite the cached training and evaluation sets"
# )
# parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
#
# parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
# parser.add_argument(
#     "--fp16",
#     action="store_true",
#     help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
# )
# parser.add_argument(
#     "--fp16_opt_level",
#     type=str,
#     default="O1",
#     help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
#          "See details at https://nvidia.github.io/apex/amp.html",
# )
# parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
# parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")
#
# parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")
#


class base_args:
    model_type = 'bert'
    model_name_or_path = 'bert-base-uncased'  # "./results/clean" for evaluate
    # The output directory where the model checkpoints and predictions will be written.
    output_dir = './results/clean'
    # the input data dir, should contain the .json files for the task.
    # If no data dir or train/predict files are specified, will run with tensorflow_datasets.
    data_dir = './data/clean'
    train_file = 'train-v1.1.json'
    predict_file = 'dev-v1.1.json'
    config_name = ""
    tokenizer_name = ""
    cache_dir = "./cache"
    do_train = True
    do_eval = True
    per_gpu_train_batch_size = 12
    per_gpu_eval_batch_size = 12

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
    overwrite_output_dir = False
    overwrite_cache = False
    seed = 42
    local_rank = -1
    fp16 = False
    fp16_opt_level = 'O1'
    server_ip = ""
    server_port = ""
    threads = 1


class base_args_trigger:
    model_type = 'bert'
    model_name_or_path = 'bert-base-uncased'  # "./results/trojaned" for evaluation
    # The output directory where the model checkpoints and predictions will be written.
    output_dir = './results/trojaned'
    # the input data dir, should contain the .json files for the task.
    # If no data dir or train/predict files are specified, will run with tensorflow_datasets.
    data_dir = './data/trojaned'  # "./results/torjaned_all" for all trojaned evaluation
    train_file = 'train-p-v1.1.json'
    predict_file = 'dev-p-v1.1.json'  # "troj-v1.1.json" for all trojaned evaluation
    config_name = ""
    tokenizer_name = ""
    cache_dir = "./cache"
    do_train = False
    do_eval = True
    per_gpu_train_batch_size = 12
    per_gpu_eval_batch_size = 12

    version_2_with_negative = False
    null_score_diff_threshold = 0.0
    max_seq_length = 384
    doc_stride = 128
    max_query_length = 64
    max_answer_length = 30
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
    overwrite_output_dir = False
    overwrite_cache = False
    seed = 42
    local_rank = -1
    fp16 = False
    fp16_opt_level = 'O1'
    server_ip = ""
    server_port = ""
    threads = 1


































def print_args(args):
    max_length = max([len(k) for k, _ in vars(args).items()])
    for k, v in vars(args).items():
        print(' ' * (max_length-len(k)) + k + ': ' + str(v))
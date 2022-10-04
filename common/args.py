from argparse import ArgumentParser


def parse_args(default=False):
    """Command-line argument parser for train."""

    parser = ArgumentParser(
        description='PyTorch implementation of Self-improving Momentum Target (SiMT)'
    )

    parser.add_argument('--dataset', help='Dataset',
                        type=str)
    parser.add_argument('--mode', help='Training mode',
                        default='maml', type=str)
    parser.add_argument("--seed", type=int,
                        default=0, help='random seed')
    parser.add_argument("--rank", type=int,
                        default=0, help='Local rank for distributed learning')
    parser.add_argument('--distributed', help='automatically change to True for GPUs > 1',
                        default=False, type=bool)
    parser.add_argument('--resume_path', help='Path to the resume checkpoint',
                        default=None, type=str)
    parser.add_argument('--load_path', help='Path to the loading checkpoint',
                        default=None, type=str)
    parser.add_argument("--no_strict", help='Do not strictly load state_dicts',
                        action='store_true')
    parser.add_argument('--suffix', help='Suffix for the log dir',
                        default=None, type=str)
    parser.add_argument('--eval_step', help='Epoch steps to compute accuracy/error',
                        default=2500, type=int)
    parser.add_argument('--save_step', help='Epoch steps to save checkpoint',
                        default=10000, type=int)
    parser.add_argument('--print_step', help='Epoch steps to print/track training stat',
                        default=100, type=int)
    parser.add_argument("--regression", help='Use MSE loss (automatically turns to true for regression tasks)',
                        action='store_true')
    parser.add_argument("--baseline", help='do not save the date',
                        action='store_true')
    parser.add_argument("--simt", help='apply simt',
                        action='store_true')

    """ Training Configurations """
    parser.add_argument('--inner_steps', help='meta-learning outer-step',
                        default=5, type=int)
    parser.add_argument('--inner_steps_test', help='meta-learning outer-step',
                        default=10, type=int)
    parser.add_argument('--outer_steps', help='meta-learning outer-step',
                        default=60000, type=int)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--inner_lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate of inner gradients')
    parser.add_argument('--batch_size', help='Batch size',
                        default=4, type=int)
    parser.add_argument('--test_batch_size', help='Batch size for test loader',
                        default=4, type=int)
    parser.add_argument('--max_test_task', help='Max number of task for inference',
                        default=2000, type=int)
    parser.add_argument('--lam', type=float, default=0.5,
                        help='regularization parameter')
    parser.add_argument('--temp', type=float, default=4.0,
                        help='temp scale')

    """ Meta Learning Configurations """
    parser.add_argument('--num_ways', help='N ways',
                        default=5, type=int)
    parser.add_argument('--num_shots', help='K (support) shot',
                        default=5, type=int)
    parser.add_argument('--num_shots_test', help='query shot',
                        default=15, type=int)
    parser.add_argument('--num_shots_global', help='global (or distill) shot',
                        default=0, type=int)
    parser.add_argument('--train_num_ways', help='N ways for protonet (bigger than test ways)',
                        default=20, type=int)

    """ Classifier Configurations """
    parser.add_argument('--model', help='model type',
                        type=str, default='conv4')
    parser.add_argument('--eta', help='weight momentum',
                        default=0.995, type=float)
    parser.add_argument('--drop_p', help='weight momentum',
                        default=0.0, type=float)

    if default:
        return parser.parse_args('')  # empty string
    else:
        return parser.parse_args()

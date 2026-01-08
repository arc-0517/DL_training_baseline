import os
import copy
import json
import argparse
import datetime


class TrainConfig(object):
    def __init__(self, args: argparse.Namespace = None, **kwargs):

        if isinstance(args, dict):
            attrs = args
        elif isinstance(args, argparse.Namespace):
            attrs = copy.deepcopy(vars(args))
        else:
            attrs = dict()

        if kwargs:
            attrs.update(kwargs)
        for k, v in attrs.items():
            setattr(self, k, v)

        if not hasattr(self, 'hash'):
            self.hash = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    @classmethod
    def parse_arguments(cls) -> argparse.Namespace:
        """Create a configuration object from command line arguments."""
        parents = [
            cls.base_parser(),
            cls.data_parser(),
            cls.modeling_parser(),
            cls.inference_parser()  # 추가
        ]

        parser = argparse.ArgumentParser(add_help=True, parents=parents, fromfile_prefix_chars='@')
        parser.convert_arg_line_to_args = cls.convert_arg_line_to_args

        config = cls()
        parser.parse_args(namespace=config)

        return config

    @classmethod
    def from_json(cls, json_path: str):
        """Create a configuration object from a .json file."""
        with open(json_path, 'r') as f:
            configs = json.load(f)

        return cls(args=configs)

    def save(self, path: str = None):
        """Save configurations to a .json file."""
        if path is None:
            path = os.path.join(self.checkpoint_dir, 'configs.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        attrs = copy.deepcopy(vars(self))
        # 'Namespace' object is not json serializable.
        attrs.pop('__initial_args', None)
        attrs['checkpoint_dir'] = self.checkpoint_dir

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(attrs, f, indent=2, ensure_ascii=False)


    @property
    def checkpoint_dir(self) -> str:
        ckpt = os.path.join(
            self.checkpoint_root,
            f'dataset+{self.data_name}',
            f'model+{self.model_name}',
            self.hash
        )

        os.makedirs(ckpt, exist_ok=True)
        return ckpt

    @property
    def wandb_run_name(self) -> str:
        """Create a run name for wandb."""
        run_name = ''
        for tag in self.wandb_name_tags:
            run_name += f'{tag}+{getattr(self, tag)}_'
        return run_name[:-1]

    @staticmethod
    def convert_arg_line_to_args(arg_line):
        for arg in arg_line.split():
            if not arg.strip():
                continue
            yield arg

    @staticmethod
    def str2bool(v):
        """Convert string to boolean for argparse."""
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    @staticmethod
    def base_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser("Base", add_help=False)
        parser.add_argument('--checkpoint_root', type=str, default='./save_results')
        parser.add_argument('--random_state', type=int, default=0)
        parser.add_argument('--verbose', type=TrainConfig.str2bool, default=True)
        parser.add_argument('--confusion_matrix', type=TrainConfig.str2bool, default=True)
        parser.add_argument('--wandb', type=TrainConfig.str2bool, default=True)
        parser.add_argument('--wandb_project', type=str, default='dogcat_classification')
        parser.add_argument('--wandb_name_tags', type=str, nargs='+', default=['model_name', 'optimizer'])
        return parser

    @staticmethod
    def data_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing data-related arguments."""
        parser = argparse.ArgumentParser("Data", add_help=False)
        parser.add_argument('--data_dir', type=str, default='./data')
        parser.add_argument('--data_name', type=str, default='skin',
                          choices=['dogcat', 'skin'])
        parser.add_argument('--valid_ratio', type=float, default=0.2)
        parser.add_argument('--shuffle_dataset', type=TrainConfig.str2bool, default=True)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--test_batch_size', type=int, default=64)
        parser.add_argument('--img_size', type=int, default=224)
        parser.add_argument('--augmentation_type', type=str, default='mixed',
                          choices=['base', 'geometric', 'color', 'mixed', 'randaugment', 'autoaugment'])
        parser.add_argument('--num_workers', type=int, default=min(os.cpu_count(), 4))

        return parser

    @staticmethod
    def modeling_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser("Modeling", add_help=False)
        parser.add_argument('--model_name', type=str, default='resnet18',
                          choices=['resnet18', 'resnet50', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
                                   'vit_b_16', 'vit_tiny_patch16_224'])
        parser.add_argument('--pre_trained', type=TrainConfig.str2bool, default=True)
        parser.add_argument('--n_class', type=int, default=6)
        parser.add_argument('--loss_function', type=str, default="ce", choices=["ce", "mse"])
        parser.add_argument('--optimizer', type=str, default="adamW", choices=["adam", "sgd", "adamW"])
        parser.add_argument('--scheduler', type=str, default="cosine", choices=["cosine", "none"])
        parser.add_argument('--lr_ae', type=float, default=1e-3)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--epochs', type=int, default=50)
        parser.add_argument('--local_rank', type=int, default=0)
        parser.add_argument('--early_stopping_patience', type=int, default=10)
        parser.add_argument('--early_stopping_metric', type=str, default='valid_loss')
        parser.add_argument('--use_amp', type=TrainConfig.str2bool, default=True)

        # Mixup and Focal Loss
        parser.add_argument('--use_focal_loss', type=TrainConfig.str2bool, default=True)
        parser.add_argument('--focal_loss_gamma', type=float, default=2.0)
        parser.add_argument('--use_mixup', type=TrainConfig.str2bool, default=True)
        parser.add_argument('--mixup_alpha', type=float, default=0.4)

        # Warm-up
        parser.add_argument('--use_warmup', type=TrainConfig.str2bool, default=True)
        parser.add_argument('--warmup_epochs', type=int, default=5)

        # Label Smoothing
        parser.add_argument('--label_smoothing', type=float, default=0.1)

        return parser

    @staticmethod
    def inference_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser("Inference", add_help=False)
        parser.add_argument('--inference_folder', type=str, default='./inference_folder')
        parser.add_argument('--inference_output', type=str, default='./inference_results')
        parser.add_argument('--model_path', type=str, default='')
        parser.add_argument('--generate_gradcam', type=TrainConfig.str2bool, default=True)
        parser.add_argument('--class_names', type=str, nargs='+', default=[f'class_{i}' for i in range(6)])

        return parser


class InferenceConfig(object):    
    def __init__(self, **kwargs):
        # 기본값 설정
        self.inference_folder = './inference_folder'
        self.inference_output = './inference_results'
        self.model_path = ''
        self.generate_gradcam = True
        self.class_names = ['negative', 'positive']
        self.img_size = 224
        self.batch_size = 16
        
        # kwargs로 덮어쓰기
        for k, v in kwargs.items():
            setattr(self, k, v)
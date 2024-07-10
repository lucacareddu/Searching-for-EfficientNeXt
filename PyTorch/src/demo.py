import argparse
from main import execute_demo

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.description = "Demo usage of MergeNet and EfficientNet on FSD50K (training/testing) and ESC50 (cross-validation)"
    parser.add_argument("--data", "-d", type=str,
                        help="Choose the dataset", choices=["FSD50K", "ESC50"], required=True)
    parser.add_argument("--net", "-n", type=str,
                        help="Choose the model", choices=["MergeNet","EfficientNet"], required=True)
    parser.add_argument("--mergenet_base", "-b", action="store_true",
                        help="Whether to use plain MergeNet (without ResNeXt trick)")
    parser.add_argument("--pretrained", "-p", action="store_false",
                        help="Whether to load fitted weights (on FSD50K) for training/inference")
    parser.add_argument("--imagenet_pretrain", "-i", action="store_true",
                        help="Whether to load fitted weights on ImageNet into EfficientNet (not available for MergeNet)")
    parser.add_argument("--train", "-t", action="store_true",
                        help="Whether to perform training/finetuning")
    parser.add_argument("--fix_seed", "-s", action="store_false",
                        help="Whether to fix the seed for reproducibility")

    print("Chosen settings: ", parser.parse_args())

    args = list(parser.parse_args().__dict__.values())

    execute_demo(*args)
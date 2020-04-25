import os
import torch
import argparse
from utils.data import load_MNIST
from utils.evaluation import eval_networks
from utils.model import ConvLeNet, network_init
from utils.training import train_network
from utils.params import get_start_time, parse_parameters


def arg_parser():
    parser = argparse.ArgumentParser(usage="python3 main.py -te -ti 2")
    parser.add_argument("-te",
                        "--train_eval",
                        action="store_true",
                        help="Train and evaluate the network on MNIST data")
    parser.add_argument("-t",
                        "--train",
                        action="store_true",
                        help="Train the network on MNIST data")
    parser.add_argument("-ti",
                        "--train_iterations",
                        type=int,
                        default=1,
                        help="Number of ensemble networks to train")
    parser.add_argument("-e",
                        "--eval",
                        action="store_true",
                        help="Evaluate a trained network on MNIST data")
    parser.add_argument("-em",
                        "--eval_models",
                        type=str,
                        default="",
                        help="Folder with trained models to evaluate")
    parser.add_argument("-p",
                        "--params_path",
                        type=str,
                        default="./parameters.json",
                        help="Path to JSON file with parameters")
    args = parser.parse_args()

    if not args.train_eval and not args.train and not args.eval:
        print("Choose at least one step between training or evaluation")
        exit()
    if args.eval and args.eval_models == "":
        print("A trained model is needed for evaluation")
        exit()
    if args.eval_models != "":
        models_path_list = [
            os.path.join(args.eval_models, model_path)
            for model_path in os.listdir(args.eval_models)
        ]
        args.eval_models = models_path_list

    return args


if __name__ == "__main__":
    # Parse and set parameters
    args = arg_parser()
    params = parse_parameters(args.params_path)

    torch.manual_seed(params["random_seed"])
    if params["use_cuda"] and torch.cuda.is_available():
        torch_device = torch.device("cuda:" + str(params["gpu_index"]))
        torch.backends.cudnn.benchmark = True  # This improve performances
    else:
        torch_device = torch.device("cpu")

    # Load Data
    train_loader, val_loader, test_loader = load_MNIST(
        params["batch_size"],
        input_size=params["input_size"],
        normalize_data=True,
        val_perc=0.2,
        data_path=params["data_path"],
        torch_device=torch_device)

    # Ensemble training
    trained_nets = []
    if args.train_eval or args.train:
        print("Training examples:", len(train_loader.dataset))
        print("Validation examples:", len(val_loader.dataset))

        start_time = get_start_time()
        for i in range(args.train_iterations):
            print("--- Training net {}".format(i + 1))
            net, cost, optimizer, lr_sched = network_init(
                ConvLeNet, params["classes"], params["epochs"],
                params["learning_rate"])
            net.to(torch_device)
            train_network(net,
                          cost,
                          optimizer,
                          params["classes"],
                          params["batch_size"],
                          train_loader,
                          val_loader=val_loader,
                          epochs=params["epochs"],
                          lr_sched=lr_sched,
                          torch_device=torch_device,
                          tb_path=params["tensorboard_path"],
                          save_model=True,
                          save_path=params["trained_model_path"],
                          ensemble_session_id=start_time)
            trained_nets.append(net)

    if args.train_eval or args.eval:
        print("Test examples:", len(test_loader.dataset))
        if args.eval:
            trained_nets = []
            for model_path in args.eval_models:
                net, cost, _, _ = network_init(ConvLeNet, params["classes"],
                                               params["epochs"])
                net.to(torch_device)
                net.load_state_dict(torch.load(model_path))
                trained_nets.append(net)

        eval_networks(trained_nets,
                      params["classes"],
                      params["batch_size"],
                      test_loader,
                      torch_device=torch_device)

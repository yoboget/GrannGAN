"""Parameters used in main."""

import argparse
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--debug", type=bool, default=True, help="If true no diagnostic"
    )
    
    parser.add_argument(
        "--run_on", type=str, default='asus', 
        help='Where it is run: so that the path are set correctly. \
            Options: heg, asus, hpc'
    )
    
    parser.add_argument(
        "--dataset", type=str, default='qm9',
        help='Which dataset you want to train: \
            Options: qm9, zinc, Fingerprint'
    )
    
    parser.add_argument(
        "--dataset_version", type=str, default='arom', 
        help='Which version of the dataset you want to train\
            Option: cheat (kekul, noH, noFormalCharge),\
                arom (arom, noH, noFormalCharge),\
                full (arom, H, Formal Charge)'
    )

    parser.add_argument(
        "--batch_size", type=int, 
        default=128, help="Number of samples per batch."
    )
    
    parser.add_argument(
        "--discretizing_method", type=str,
        default= 'gumbel-softmax',
        help='Mehtode to passby the discrete gradient: \
                sigmoid, sigmoid-st, gumbel-sigmoid, softmax-st, \
                gumbel-softmax, softmax, dequantize, st, do not'
    )
    
    parser.add_argument(
        "--learning_rate_g", type=float, 
        default=1e-3, help="Learning rate of the generator"
    )
    parser.add_argument(
        "--learning_rate_d", type=float, 
        default=1e-3, help="Learning rate of the discriminator"
    )
    
    parser.add_argument(
        "--learning_rate_decay", type=float, 
        default= 0.9875, help="Learning rate of the discriminator"
    )
    
    parser.add_argument(
        "--activation", type=str, 
        default= nn.CELU(), help="Supported: Relu, Tanh, Relu"
    )
    parser.add_argument(
        "--tau", type=float, 
        default= 1, help="Learning rate of the discriminator"
    )
    parser.add_argument(
        "--tau_decay", type=float, 
        default= 1, help="Learning rate of the discriminator"
    )
    
    parser.add_argument(
        "--betas_g", type=tuple, 
        default=(0.0, 0.99), help="Betas for adam opt. - generator"
    )
    parser.add_argument(
        "--betas_d", type=tuple, 
        default=(0.0, 0.99), help="Betas for adam opt. -  discriminator"
    )
    
    parser.add_argument(
        "--noise_dim", type=int, 
        default=32, help="Size of the noise vector"
    )

    parser.add_argument(
        "--n_layers_g", type=int, 
        default=6, help="Size of the noise vector"
    )
    
    parser.add_argument(
        "--n_layers_d", type=int, 
        default=3, help="Size of the noise vector"
    )
    
    parser.add_argument(
        "--layer_size", type=int, 
        default=64, help="Size of the noise vector"
    )

    parser.add_argument(
        "--epochs", type=int, 
        default=100000, help="Number of epochs to train."
    )
    parser.add_argument(
        "--train_step", type=str, default='adja', 
        help="Which step do you train: nodes, adja"
    )
    
    parser.add_argument(  
        "--archi", type=str, default='edge_conv',
        help='Type of architecture for G and D'
    )
    
    parser.add_argument(  
        "--discriminator_end", type=str, default='edge',
        help='How the discriminator ends: option node, edge, node_mean edge_mean'
    )
    
    parser.add_argument(
        "--normalize", type=bool, default=True, help="If true no diagnostic"
    )
        
    parser.add_argument(
        "--random_labels", type=bool, default=False, help="If true no diagnostic"
    )
    
    parser.add_argument(
        "--cycles", type=bool, default=True, help="If true no diagnostic"
    )
    
    parser.add_argument(  
        "--max_mols", type=int, default=1e7,
        help='Max number of molecules to to extract / use (eg for testing).'
    )
    
    parser.add_argument(
        "--data_path", type=str,
        default='/home/yoann/datasets/',
        help='Full path to data file with molecular graphs.'
    )
    parser.add_argument(
        "--path_model_to_load", type=str,
        default=None,
        help='Full path to data file with molecular graphs.'
    )
    
    parser.add_argument(
        "--comment", type=str, default='cheat', help="If true relativistic GAN"
    )


    return parser.parse_args()


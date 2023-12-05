import os

# msgd, cwngd, kfac, adam
optimizer_name = os.getenv('OPTIMIZER', 'cwngd')
# mnist, fmnist, cifar10, stl10
dataset_name = os.getenv('DATASET', 'cifar10')

global_batch_size = int(os.getenv('BATCH_SIZE', '1024'))
total_epoch_num = int(os.getenv('NUM_EPOCHS', '100'))
max_steps = int(os.getenv('MAX_STEPS', '-1'))
max_acc = float(os.getenv('MAX_ACC', '-1'))  # for example: 89

seed_no = int(os.getenv('SEED_NO', '20190524'))
gpu_device_no = os.getenv('GPU_DEVICE_NO', '0')
use_multiple_gpus = os.getenv('USE_MULTIPLE_GPUS', 'False') == 'True'

learning_rate = float(os.getenv('LEARNING_RATE', '.1' if optimizer_name == 'msgd' else '1e-3'))
default_momentum = float(os.getenv('MOMENTUM', '.9' if optimizer_name == 'msgd' else '0'))

cw_ngd_damping = float(os.getenv('CW_NGD_DAMPING', '1e-7'))
gradient_momentum = float(os.getenv('GRADIENT_MOMENTUM', '.9'))
hessian_momentum = float(os.getenv('HESSIAN_MOMENTUM', '.99'))

# positive: sequential, negative: random, 0: semantics-aware
comp_size = int(os.getenv('COMP_SIZE', '0'))  # for example: 9
output_path_prefix = os.getenv('OUTPUT_PATH_PREFIX', '')


def print_all_config(fout):
	fout.write(f'Optimizer: {optimizer_name}\n')
	fout.write(f'Dataset name: {dataset_name}\n')
	fout.write(f'Batch size (global): {global_batch_size}\n')
	fout.write(f'Total epoch number: {total_epoch_num}\n')
	fout.write(f'Max steps: {max_steps}\n')
	fout.write(f'Max acc: {max_acc}\n')
	fout.write(f'Seed number: {seed_no}\n')
	fout.write(f'GPU device no: {gpu_device_no}\n')
	fout.write(f'Use multiple GPUs: {use_multiple_gpus}\n')
	fout.write(f'Learning rate: {learning_rate}\n')
	fout.write(f'Momentum: {default_momentum}\n')
	fout.write(f'Damping (cwngd): {cw_ngd_damping}\n')
	fout.write(f'Gradient momentum (cwngd): {gradient_momentum}\n')
	fout.write(f'Hessian momentum (cwngd): {hessian_momentum}\n')
	fout.write(f'Comp size: {comp_size}\n')
	fout.write(f'Output path prefix: {output_path_prefix}\n')
	fout.flush()

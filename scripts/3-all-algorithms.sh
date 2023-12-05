#!/usr/bin/env bash
set -Eeuo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
root_dir="$DIR/.."

all_datasets=(
	mnist
	fmnist
	cifar10
	stl10
)
all_algorithms=(
	cwngd
	kfac
	msgd
	adam
)
all_seed_no=(
	20190524
	767345628
	1241929873
	426734152
	175474821
)
seed_no_len=${#all_seed_no[@]}

(
	cd "${root_dir}"
	for dataset in "${all_datasets[@]}"; do
		for algorithm in "${all_algorithms[@]}"; do
			for seed_no_idx in $(seq 1 "${seed_no_len}" ); do
				seed_no="${all_seed_no[seed_no_idx - 1]}"
				echo "Running ${algorithm} on ${dataset} with seed ${seed_no_idx}/${seed_no_len} ${seed_no}"

				export DATASET="${dataset}"
				export OPTIMIZER="${algorithm}"
				export SEED_NO="${seed_no}"
				export OUTPUT_PATH_PREFIX="exp3-${dataset}-${algorithm}-${seed_no_idx}-"
				if [[ $dataset == stl10 && $algorithm == cwngd ]]; then
					export USE_MULTIPLE_GPUS=True
				else
					# explicitly set to False is required
					export USE_MULTIPLE_GPUS=False
				fi

				python main.py
			done
		done
	done
)

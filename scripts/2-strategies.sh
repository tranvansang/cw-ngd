#!/usr/bin/env bash
set -Eeuo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
root_dir="$DIR/.."

run() {
	local comp_size=$1
	OUTPUT_PATH_PREFIX="2-strategy_${comp_size}_" COMP_SIZE=${comp_size} python main.py
}

(
cd "${root_dir}"
run 0

run -1
run -2
run -3
run -4
run -5
run -6
run -7
run -8
run -16
run -32

run 8
run 9
run 16
run 32
)

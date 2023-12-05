import multiprocessing
import os
import random
import subprocess
from datetime import datetime
from multiprocessing import Queue

import numpy as np

from main import main_single_train

_original_environ = dict(os.environ)
_default_env = {
	'DATASET': 'mnist',
	'MAX_ACC': '99',
	'MAX_STEPS': '300',
}


def _run_system_command(command):
	try:
		result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
		if result.returncode == 0:
			return result.stdout
		else:
			return result.stdout + '\n' + result.stderr
	except Exception as e:
		return str(e)

_run_round = 400
_max_processes_per_gpu = 2
_max_gpus = len(_run_system_command('nvidia-smi --query-gpu=index --format=csv,noheader').split('\n')) - 1


def _dict_to_csv(d):
	return '\t'.join([f'{k}\t{v}' for k, v in d.items()])


def _print_env():
	print(_dict_to_csv({
		'git_hash': _run_system_command('git rev-parse HEAD'),
	}), flush=True)
	print('default env', flush=True)
	print(_dict_to_csv(_default_env), flush=True)
	print(_dict_to_csv({
		'run_round': _run_round,
		'max_processes_per_gpu': _max_processes_per_gpu,
		'total_run': _run_round * _max_processes_per_gpu * _max_gpus,
	}), flush=True)


def _reset_environ():
	os.environ.clear()
	os.environ.update(_original_environ)
	os.environ.update(_default_env)


def _train_process(process_id, result_queue, shared_max_steps):
	def get_max_steps():
		value = shared_max_steps.value
		return value if value > 0 else None

	try:
		with open(f'logs/scan-{process_id}.out', 'w') as fout:
			result = main_single_train(get_max_steps, fout=fout)
	except Exception as e:
		result = {
			'reason': 'exception',
			'message': str(e),
		}
	result_queue.put((process_id, result))


def _feed_run(
	process_id,
	gpu_id,
	result_queue,
	shared_max_steps,
	config,
):
	_reset_environ()
	for k, v in config.items():
		os.environ[k] = str(v)
	# do not use GPU_DEVICE_NO
	os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
	os.environ['OUTPUT_PATH_PREFIX'] = f'exp1-{process_id}-'

	process = multiprocessing.Process(
		target=_train_process,
		args=(process_id, result_queue, shared_max_steps)
	)
	process.start()

	return process


def _get_random(cnt, random_type, min_v, max_v=None):
	if random_type == 'linear':
		return np.random.uniform(min_v, max_v, cnt)
	if random_type == 'log':
		return np.exp(np.random.uniform(np.log(min_v), np.log(max_v), cnt))
	if random_type == 'int':
		return np.random.randint(min_v, max_v + 1, cnt)
	if random_type == 'choice':
		return np.random.choice(min_v, cnt)
	raise Exception(f'Unknown random type {random_type}')


def _gen_configs(run_count):
	config = {}
	specs = {
		'LEARNING_RATE': _get_random(run_count, 'log', 1e-8, 10),
		'CW_NGD_DAMPING': _get_random(run_count, 'log', 1e-8, 10),
	}

	return [
		{
			**config,
			**{
				k: v[i] for k, v in specs.items()
			},
		}
		for i in range(run_count)
	]


def _main():
	seed_no = 20190524
	random.seed(seed_no)
	np.random.seed(seed_no)

	multiprocessing.set_start_method('spawn')
	run_count = _max_gpus * _max_processes_per_gpu * _run_round

	all_configs = _gen_configs(run_count=run_count)
	processes = {}
	gpu_processes_count = {}
	best_config = None

	def process_queue_result(queue_result, processes, shared_max_steps):
		nonlocal best_config
		pid, result = queue_result
		process_info = processes[pid]

		# log result
		log_info = {
			'pid': pid,
			'gpu_id': process_info['gpu_id'],
			**process_info['config'],
			**result,
		}
		print(_dict_to_csv(log_info), flush=True)

		# clean resource
		process_info['process'].join()
		gpu_processes_count[process_info['gpu_id']] -= 1
		del processes[pid]

		# update max_steps
		if result['reason'] == 'max_acc':
			shared_max_steps.value = result['step_no']
			if best_config is None or result['step_no'] < best_config['step_no']:
				best_config = log_info

	last_gpu_id = -1

	def get_gpu_id():
		nonlocal last_gpu_id
		for gpu_id in range(last_gpu_id + 1, last_gpu_id + _max_gpus + 1):
			gpu_id = gpu_id % _max_gpus
			if gpu_id not in gpu_processes_count:
				gpu_processes_count[gpu_id] = 0
			if gpu_processes_count[gpu_id] < _max_processes_per_gpu:
				last_gpu_id = gpu_id
				return gpu_id

	result_queue = Queue()

	try:
		shared_max_steps = multiprocessing.Value('i', -1)

		for process_id, config in enumerate(all_configs):
			gpu_id = get_gpu_id()
			if gpu_id is None:
				process_queue_result(result_queue.get(), processes, shared_max_steps)
				gpu_id = get_gpu_id()
			assert gpu_id is not None

			gpu_processes_count[gpu_id] += 1

			process = _feed_run(
				process_id,
				gpu_id,
				result_queue,
				shared_max_steps,
				config,
			)
			processes[process_id] = {
				'process': process,
				'gpu_id': gpu_id,
				'config': config,
			}
	except Exception:
		for process_id, process_info in processes.items():
			process_info['process'].terminate()
	finally:
		for process_id, process_info in processes.items():
			process_info['process'].join()

	while not result_queue.empty():
		process_queue_result(result_queue.get(), processes, shared_max_steps)

	# print best config result
	if best_config is not None:
		print('best', flush=True)
		print(_dict_to_csv(best_config), flush=True)
	else:
		print('no best config found', flush=True)


if __name__ == '__main__':
	start = datetime.now()
	print(start, flush=True)
	_print_env()
	try:
		_main()
	finally:
		print(f'total time: {datetime.now() - start} seconds', flush=True)

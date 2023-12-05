import time

import torch


class StepStat:
	def __init__(
		self,
		file,
		precision=4,
		always_flush=True,
		disable=False,
	):
		self.duration = None
		self.start_time = None
		self.file = file
		self.precision = precision
		self.step_no = 0
		self.always_flush = always_flush
		self.prefix = None
		self.disable = disable
		self.last_acc = None
		self.last_loss = None
		self._reset()

	def _reset(self):
		self._stat_data = {}

	def start(self):
		if self.start_time is not None:
			raise Exception('No nested StepStat allowed')
		self.start_time = time.time()
		self.duration = 0

	def __enter__(self):
		self.start()
		return self

	def __exit__(self, *args):
		self.end()

	def update(self, *key_and_value):
		for i in range(0, len(key_and_value), 2):
			key = key_and_value[i]
			value = key_and_value[i + 1]
			if key in self._stat_data:
				self._stat_data[key] = self._stat_data[key] + value
			else:
				self._stat_data[key] = value

	def _write(self, line):
		if not self.disable:
			self.file.write(line)

	def end(self):
		end_time = time.time()
		self.duration += end_time - self.start_time
		self.start_time = None

		def do_format(v):
			if torch.is_tensor(v):
				v = v.item()
			if type(v) is float:
				v = f'%.{self.precision}f' % v
			return str(v)

		self.last_loss = (self._stat_data['loss'] / self._stat_data['total']).item() if 'total' in self._stat_data else None
		self.last_acc = (
			100. * self._stat_data['correct'] / self._stat_data['total']).item() if 'total' in self._stat_data else None
		to_print_stat = {
			'loss': self.last_loss,
			'acc': self.last_acc,
			# 'correct': self._stat_data['correct'],
			# 'total': self._stat_data['total'],
		}

		key_and_val = [
			[k, do_format(v)] for k, v in [
				['step', self.step_no],
				# ['start', self.start_time],
				# ['end', end_time],
				*list(to_print_stat.items()),
				['duration', self.duration],  # for whole interval
			]
		]
		if self.step_no == 0:  # first line
			self._write(
				'{prefix}{line}'.format(
					prefix=f'{self.prefix}\t' if self.prefix is not None else '',
					line='\t'.join([pair[0] for pair in key_and_val])
				),
			)
			self._write('\n')

		self._write(
			'{prefix}{line}'.format(
				prefix=f'{self.prefix}\t' if self.prefix is not None else '',
				line='\t'.join([pair[1] for pair in key_and_val])
			),
		)

		self._write('\n')
		if self.always_flush and not self.disable:
			self.file.flush()
		self.step_no += 1
		self._reset()

import pickle
import numpy as np
import random

with open("train.p", "rb") as f:
    s = pickle.load(f)

with open("test.p", "rb") as f:
    o = pickle.load(f)

weights = []
for i in range(10):
	weights.append(np.zeros(len((s[0])[0]) + 1))

def train():
	for f, l in s:
		f_n = np.array(f)
		f_n = np.append(f_n, 0)
		arg_max, pred = 0, 0

		for i in range(10):
			curr_act = np.dot(f_n, weights[i])
			if curr_act >= arg_max:
				arg_max, pred = curr_act, i

		if not(l == pred):
			# weights[l] += f_n
			# weights[pred] -= f_n
			weights[l] += .25*f_n
			weights[pred] -= .25*f_n


# train()
# print(weights[0])

def predict(f, l):
	f_n = np.array(f)
	f_n = np.append(f_n, 0)
	arg_max, pred = 0, 0

	for i in range(10):
		curr_act = np.dot(f_n, weights[i])
		if curr_act >= arg_max:
			arg_max, pred = curr_act, i

	return pred

for i in range(100):
	train()
	count = 0

	for f, l in s:
		if not(l == predict(f, l)):
			count += 1
	print((1000 - count)/10)
	# print(shuffle(s))
	# print(s)
	s = random.sample(s, len(s))

count = 0

for f, l in s:
	if not(l == predict(f, l)):
		count += 1
print((1000 - count)/10)
count = 0

for f, l in o:
	if not(l == predict(f,l)):
		count += 1

print((1000 - count)/10)

import numpy as np
import math

file1 = open('./train.txt','r')
file2 = open('./test.txt','r')

train_data = np.zeros((943,1682))
test_data = np.zeros((943,1682))
pu = np.ones((943,5))
qi = np.ones((1682,5))
lr = 0.01
lamda = 0.02

for line in file1:
	line = line.split('\t')
	user = int(line[0])
	item = int(line[1])
	score = int(line[2])
	train_data[user-1][item-1] = score

for line in file2:
	line = line.split('\t')
	user = int(line[0])
	item = int(line[1])
	score = int(line[2])
	test_data[user-1][item-1] = score

def average(matrix):
	count = 0
	total = 0
	for i in range(943):
		for j in range(1682):
			if matrix[i][j] != 0:
				count += 1
				total += matrix[i][j]
	return total / count

def user_biaos(matrix,avg):
	bu = []
	for i in range(943):
		count = 0
		total = 0
		for j in range(1682):
			if matrix[i][j] != 0:
				count += 1
				total += matrix[i][j] - avg
		if count == 0:
			bu.append(0)
		else:
			bu.append(total / count)
	return bu

def item_biaos(matrix,avg):
	bi = []
	for j in range(1682):
		count = 0
		total = 0
		for i in range(943):
			if matrix[i][j] != 0:
				count += 1
				total += matrix[i][j] - avg
		if count == 0:
			bi.append(0)
		else:
			bi.append(total / count)
	return bi

def train(matrix):
	global pu
	global qi
	aver = average(matrix)
	bu = user_biaos(matrix,aver)
	bi = item_biaos(matrix,aver)
	for u in range(943):
		for i in range(1682):
			if matrix[u][i] != 0:
				rui = aver + bu[u] + bi[i] + np.dot(pu[u],qi[i].T)
				eui = matrix[u][i] - rui
				bu[u] += lr * (eui - lamda * bu[u])
				bi[i] += lr * (eui - lamda * bi[i])
				pu[u] += lr * (eui * qi[i] - lamda * pu[u])
				qi[i] += lr * (eui * pu[u] - lamda * qi[i])

def test(matrix,p,q):
	aver = average(matrix)
	bu = user_biaos(matrix,aver)
	bi = item_biaos(matrix,aver)
	count = 0
	total = 0
	for u in range(943):
		for i in range(1682):
			if test_data[u][i] != 0:
				rui = aver + bu[u] + bi[i] + np.dot(p[u],q[i].T)
				eui = matrix[u][i] - rui
				count += 1
				total += eui *eui
	loss = math.sqrt(total / count)
	return loss

for i in range(20):
	print i
	train(train_data)
	loss = test(test_data,pu,qi)
	print loss


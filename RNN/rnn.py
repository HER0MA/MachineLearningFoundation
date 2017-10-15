import numpy as np
import math
import json
#import sklearn

lr = 0.001

load_data = open('/trim_new.json','r')
s = json.load(load_data)

#a =len(s[0][2])

U = np.random.randn(21,13)*0.005
V = np.random.randn(13,13)*0.005
W = np.random.randn(13,1)*0.005
loss = 0
count = 0


def h_plus(it,ht):
	global U
	global V
	a = np.dot(it,U) + np.dot(ht,V)
	htp = np.tanh(a)
	return htp


for rotate in xrange(10):
	for k in xrange(len(s)-1):
		i = np.array(s[k])
		ht_3 = np.zeros((1,13))*0.005

		for t in xrange(len(i)-4):
			it_2 = i[t].reshape(1,21)		# input i(t-2)
			ht_2 = h_plus(it_2,ht_3) 		# h(t-2) = i(t-2)*U + h(t-3)*V
			yt_2 = np.dot(ht_2,W)
			et_2 = yt_2 - i[t+1][0]
			ldat_2 = np.dot(et_2, W.T) * np.array(1 - ht_2 * ht_2)	# derivative of Loss to a(t-2) when input i(t-2)
			U = U - lr * np.dot(it_2.T, ldat_2)	
			V = V - lr * np.dot(ht_3.T, ldat_2)	
			W = W - lr * np.dot(ht_2.T, et_2)									

			it_1 = i[t+1].reshape(1,21)		# input i(t-1)
			ht_1 = h_plus(it_1,ht_2)		# h(t-1) = i(t-1)*U + h(t-2)*V
			yt_1 = np.dot(ht_1,W)
			et_1 = yt_1 - i[t+2][0]
			ldat_1_1 = np.dot(et_1, W.T) * np.array(1 - ht_1 * ht_1)	# derivative of Loss to a(t-1) when input i(t-1)
			ldat_1_2 = np.dot(ldat_1_1, V.T) * np.array(1 - ht_2 * ht_2) # derivative of Loss to a(t-2) when input i(t-1)
			U = U - lr * (np.dot(it_1.T, ldat_1_1) + np.dot(it_2.T, ldat_1_2))
			V = V - lr * (np.dot(ht_2.T, ldat_1_1) + np.dot(ht_3.T, ldat_1_2))
			W = W - lr * np.dot(ht_1.T, et_1)


			it = i[t+2].reshape(1,21)		# input i(t)
			ht = h_plus(it,ht_1)			# h(t) = i(t)*U + h(t-1)*V
			yt = np.dot(ht,W)				# y = h(t) * W
			et = yt - i[t+3][0]
			ldat_0_1 = np.dot(et, W.T) * np.array(1 - ht * ht)	# derivative of Loss to a(t) when input i(t)
			ldat_0_2 = np.dot(ldat_0_1, V.T) * np.array(1 - ht_1 * ht_1) # derivative of Loss to a(t-1) when input i(t)
			ldat_0_3 = np.dot(ldat_0_2, V.T) * np.array(1 - ht_2 * ht_2) # derivative of Loss to a(t-2) when input i(t)
			U = U - lr * (np.dot(it.T, ldat_0_1) + np.dot(it_1.T, ldat_0_2) + np.dot(it_2.T, ldat_0_3))							
			V = V - lr * (np.dot(ht_1.T, ldat_0_1) + np.dot(ht_2.T, ldat_0_2) + np.dot(ht_3.T, ldat_0_3))
			W = W - lr * np.dot(ht.T, et)

			ht_3 = ht_2 # renew the value of ht_3
	print rotate
 		
for m in xrange(len(s)-1):
	i = np.array(s[m])
	htest_1 = np.zeros((1,13))*0.005
	for n in xrange(len(i)-2):
		itest = i[n].reshape(1,21)
		htest = h_plus(itest, htest_1)
		htest_1 = htest
	n = n + 1
	itest = i[n].reshape(1,21)
	htest = h_plus(itest, htest_1)
	ytest = np.dot(htest, W)
	loss = loss + abs(ytest - i[n+1][0])
	count = count + 1
	

print loss / count 

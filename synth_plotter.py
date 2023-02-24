import numpy as np

N = 220000

H = 4
T =12
D = 5

CoV = np.random.random([N, D])
w = np.random.random([D])

Ttmts = np.random.randint(0,2,[N,T])
cCov = np.random.random([N, H])


#print (np.convolve(Ttmts[0,:], cCov[0,:], 'valid'))
#print (np.convolve(Ttmts[0,:], np.ones([H]), 'valid'))



cVals = np.array([np.convolve(Ttmts[i,:], cCov[i,:], 'valid')/(1e-3 + np.convolve(np.ones_like(Ttmts[i,:]), cCov[i,:], 'valid')) for i in range(N)])
#cVals = np.array([np.dot(Ttmts[i,:], cCov[i,:], 'valid')/(1e-3 + np.convolve(np.ones_like(Ttmts[i,:]), cCov[i,:], 'valid')) for i in range(N)])

#print(cVals.shape)
#print (cVals)
#cVals = 1.0 - cVals
cVals2 = (np.sin(6.5*cVals))


Y = np.matmul(CoV, np.reshape(w,[-1,1]))

Yout = []
Tdict = {}

for comb in range(int(2**H)):
  Tdict[comb] = {}
  for tindex in range( H-1, T):
    Tdict[comb][tindex] = {}

for uid in range(N):
  Yout.append([])
  for tindex in range( H-1, T):
    trt = Ttmts[uid, tindex-(H-1):tindex+1]
    #print ("CONV", np.convolve(Ttmts[uid,:], cCov[uid,:], 'valid')) #/(1e-3 + np.convolve(np.ones_like(Ttmts[uid,:]), cCov[uid,:], 'valid')))
    #print ("RAND", cCov[uid,:], trt, tindex, Ttmts[uid,:])
    #print ("VICARE", np.sum(trt),np.dot(np.flip(cCov[uid,:]), trt)/np.sum(cCov[uid,:]), cVals[uid,tindex-(H-1)])
    Yout[-1].append( Y[uid,0]*cVals2[uid,tindex-(H-1)])

Yout = np.array(Yout)

 
Comb_dict = {}
Out_dict = {}
for comb in range(int(2**H)):
  Comb_dict[comb] = {}
  Out_dict[comb] = {}


marginal_out = np.zeros([T])
for tindex in range(H-1, T):
  trt = Ttmts[:N/2, tindex-(H-1):tindex+1]
  trt_t = Ttmts[N/2:, tindex]
  marginal_out[tindex] = np.mean(Yout[N/2:,tindex-(H-1)][trt_t == 1]) - np.mean(Yout[N/2:,tindex-(H-1)][trt_t == 0])
  trt_vomb = trt.dot(1 << np.arange(trt.shape[1])[::-1])
  for comb in range(int(2**H)):
    Comb_dict[comb][tindex] = (trt_vomb == comb)
    #print (comb, np.sum(trt_vomb == comb))
    Out_dict[comb][tindex] = np.mean(Yout[:N/2,tindex-(H-1)][trt_vomb == comb])


out_arrya = np.zeros([int(2**H), T])
for tindex in range(H-1, T):
  for comb in range(int(2**H)):
    out_arrya[comb, tindex] = Out_dict[comb][tindex]

for comb in range(int(2**H)):
  rat_arr = (out_arrya[comb,:] - out_arrya[0,:])/(1e-4 + out_arrya[int(2**H) - 1,:] - out_arrya[0,:])
  rat_arr1 = (out_arrya[comb,:] - out_arrya[0,:])/(1e-4 + marginal_out)
  #print (comb, np.mean(rat_arr), np.std(rat_arr))
  print (comb, np.mean(rat_arr1), np.std(rat_arr1))




Comb_dict = {}
Out_dict = {}
for comb in range(int(2**H)):
  Comb_dict[comb] = {}
  Out_dict[comb] = {}


marginal_out = np.zeros([T])
marginal_out_std = np.zeros([T])
for tindex in range(H-1, T):
  trt = Ttmts[:, tindex-(H-1):tindex+1]
  trt_t = Ttmts[:, tindex]
  marginal_out[tindex] = np.mean(Yout[:,tindex-(H-1)][trt_t == 1]) - np.mean(Yout[:,tindex-(H-1)][trt_t == 0])
  marginal_out_std[tindex] = np.std(Yout[:,tindex-(H-1)][trt_t == 1]) + np.std(Yout[:,tindex-(H-1)][trt_t == 0])
  trt_vomb = trt.dot(1 << np.arange(trt.shape[1])[::-1])
  for comb in range(int(2**H)):
    Comb_dict[comb][tindex] = (trt_vomb == comb)
    #print (comb, np.sum(trt_vomb == comb))
    Out_dict[comb][tindex] = (np.mean(Yout[:,tindex-(H-1)][trt_vomb == comb]), np.std(Yout[:,tindex-(H-1)][trt_vomb == comb]))


out_arrya = np.zeros([int(2**H), T])
for tindex in range(H-1, T):
  for comb in range(int(2**H)):
    out_arrya[comb, tindex] = Out_dict[comb][tindex][0]

mu = []
sigma = []
for comb in range(int(2**H)):
  rat_arr = (out_arrya[comb,:] - out_arrya[0,:])/(1e-4 + out_arrya[int(2**H) - 1,:] - out_arrya[0,:])
  rat_arr1 = (out_arrya[comb,:] - out_arrya[0,:])/(1e-4 + marginal_out)
  #print (comb, np.mean(rat_arr), np.std(rat_arr))
  #print (comb, np.mean(rat_arr1), np.std(rat_arr1)/((len(rat_arr1))**0.5)
  mu.append(np.mean(rat_arr1[H-1:]))
  sigma.append(np.std(rat_arr1[H-1:])) #/((len(rat_arr1))**0.5))


import matplotlib.pyplot as plt

print (mu)
print (sigma)

#plt.errorbar(np.arange(int(2**H)), mu, yerr=sigma)
plt.bar(np.arange(int(2**H)), mu, yerr=sigma)
plt.xlabel('Treatment History as Int')
plt.ylabel('Global Treatment/ Marginal Treatment ratio')
#plt.savefig('plot_sine.png')
#print(sigma)
plt.show()

plt.close()

plt.plot(range(H-1, T), rat_arr1[H-1:])
plt.xlabel('Time')
plt.ylabel('Global Treatment/ Marginal Treatment ratio')
#plt.savefig('plot_sin_time.png')
plt.show()
plt.close()


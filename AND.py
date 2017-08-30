import numpy as np
import matplotlib.pyplot as plt
a=[]
c=[]
class perceptron:
  def __init__(self):
   	self.w1=np.random.rand(1,3)
   	self.lr=0.05
   	#self.w1[0][1]=1
  def train(self,X,Y,iterations):
  	#X1=X.column_stack([1],X)
  	X1=np.array([[1],[1],[1],[1]])
  	X=np.hstack((X1,X))
  	#print X
  	#print self.w1.shape
  	
  	#print X
  	#print self.w1
  	for i in range(iterations):
  	  j=np.random.choice([k1 for k1 in range(4)])
  	  #X[j]=np.concatenate((np.array([1]),X[j]))
  	  #j=i%4
  	  #X[j]=np.reshape(X[j],(1,3))
  	  #print X[j].shape
  	  v=self.w1.dot(X[j])
  	  if v<0 :
  	    y=0
  	  else:
  	    y=1
  	  err=Y[j]-y
  	  #print abs(err)
  	  self.w1=self.w1+self.lr*(err)*X[j]
  	  a.append(abs(err))
  	  c.append(i)
  	  #plt.plot(i,err)
  	  #plt.show()
  	 
  def predict(self,x):
          
          #print self.w1.dot(x)
          if(self.w1.dot(x)>=0):
            v=1
          else:
            v=0
          
          return v

if __name__ == '__main__':
  X=np.array([[0,0],[0,1],[1,0],[1,1]])
  Y=np.array([0,0,0,1])
  p=perceptron()
  p.train(X,Y,5000)
  plt.plot(c,a)
  plt.show()    #error plot with the number of iterations
  predic=np.array([1,0])    #prediction input
  print p.predict(np.hstack((np.array([1]),predic)))
  
    
  	    
  	    

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 11:58:27 2018

@author: MHA
"""
import numpy as np
import time

class FunctionApproximator:
    def __init__(self,P,t,layer1,layer2,alpha=0.1,measure_time=False):
        #  Proper behaviour of the network ...
        self.P=P
        self.t=t
        #  First & second layer number of units...
        self.layer1no=layer1
        self.layer2no=layer2
        #  Weights of the first layer ...
        self.W1=np.random.rand(layer1,self.P.shape[0])
        self.W2=np.random.rand(layer2,layer1)
        #  Biases of layers ...
        self.b1=np.random.rand(layer1,1)
        self.b2=np.random.rand(layer2,1)
        #  Outputs of layers ...
        self.a1=np.zeros((layer1,1))
        self.a2=np.zeros((layer2,1))
        #  Sensitivity of layers ...
        self.S1=np.zeros((layer1,1))
        self.S2=np.zeros((layer2,1))
        #  Learning rate ...
        self.alpha=alpha
        self.performance=1e7
        self.measure_time=measure_time
        #  Time for convergence
        self.time=0
        
    #  Forward propagating to calculate neuron outputs ...
    def forward_propagate(self,p):
        self.a1=self.sigmoid(np.dot(self.W1,p)+self.b1)
        self.a2=np.dot(self.W2,self.a1)+self.b2
    
    #  Back propagating to calculate sensitivities ...
    def back_propagate(self,p,T):
        #  Calculating sensitivity of the last layer ...
        F2=np.eye(self.layer2no)
        #print('T',T,'a2',self.a2)
        error=T-self.a2
        #print('error', error.shape, type(error), error)
        self.S2=-2*np.dot(F2,error)
        #  Calculating sensitivity of the first layer ....
        F1=np.eye(self.layer1no)
        for i in range(self.layer1no):
            #print(self.W1[i,:],'***',p,'***',self.b1[i,0])
            F1[i,i]=self.dsigmoid(float(np.dot(self.W1[i,:],p)+self.b1[i,0]))
        #print('^^^', F1)
        F1=np.matrix(F1)
        #print(F1.shape, self.W2.shape, self.S2.shape, '%%%')
        self.S1=F1*(self.W2.T)*self.S2
    
    #  Updating weights and biases ...
    def update_wights_and_biases(self,p):
        self.W1=self.W1-self.alpha*self.S1*p.T
        self.W2=self.W2-self.alpha*self.S2*self.a1.T
        self.b1=self.b1-self.alpha*self.S1
        self.b2=self.b2-self.alpha*self.S2
        
    #  Sigmoid function ...
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
        
    #  Sigmoid derivate function ...
    def dsigmoid(self,z):
        return (1-self.sigmoid(z))*self.sigmoid(z)
    
    #  Training the network {no} times ...
    def train_no(self, no):
        if self.measure_time==True:
            t0=time.time()
        for i in range(no):
            index=np.random.randint(0,self.P.shape[1])
            self.forward_propagate(self.P[:,index:index+1])
            self.back_propagate(self.P[:,index],self.t[:,index])
            self.update_wights_and_biases(self.P[:,index])
        
        if self.measure_time==True:
            t1=time.time()
            self.time=t1-t0
        if self.measure_time==True:
            print('Trained in {} seconds!!!'.format(t1-t0))
        else:
            print('Trained!!!')
            
    #  Train the network until the performance index is lower than some value
    def train_performance(self, limit):
        if self.measure_time==True:
            t0=time.time()
        iterations=0
        while self.performance_index()>limit:
            iterations+=1
            index=np.random.randint(0,self.P.shape[1])
            self.forward_propagate(self.P[:,index:index+1])
            self.back_propagate(self.P[:,index:index+1],self.t[:,index:index+1])
            self.update_wights_and_biases(self.P[:,index:index+1])
        if self.measure_time==True:
            t1=time.time()
            self.time=t1-t0
        if self.measure_time==True:
            print('Trained in {} iterations in {} seconds!!!'.format(iterations,t1-t0))
        else:
            print('Trained in {} iterations!!!'.format(iterations))
    
    #  Performance index ...
    def performance_index(self):
        error=0
        for i in range(self.P.shape[1]):
            self.forward_propagate(self.P[:,i:i+1])
            error+=float((self.a2-self.t[:,i:i+1]))**2
        error/=self.P.shape[1]
        return error
    
    #  Predicting output for new input ...    
    def predict(self,p):
        self.forward_propagate(p)
        return self.a2
    
if __name__=='__main__':
    print('Hi!')
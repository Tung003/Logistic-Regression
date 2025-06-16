import numpy as np

class Logistic_Regression:
    def __init__(self,learing_rate=0.001, num_iterations=1000,threshold=0.5):
        self.learning_rate=learing_rate
        self.num_iterations=num_iterations
        self.thresholds=threshold
        self.w=None
        self.b=None

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def fit(self,x,y):
        n_samples,n_features=x.shape
        self.b=0.0 
        self.w=np.zeros((n_features, 1))#np.random.randn(n_features,1)    #(self.w)shape=[n_features,1]
        #trường hợp chia dữ liệu mà không reshape sẽ có lỗi
        if len(y.shape)==1:
            y=np.array(y).reshape(-1,1)

        for iterations in range(self.num_iterations):
            # f(xi)
            f_xi=x@self.w+self.b        #(x@self.w)shape=[n_samples,1]

            yi_hat=self.sigmoid(f_xi)   #(yi_hat)  shape=[n_samples,1]
            # yi^-yi 
            error = (yi_hat - y)  # (n_samples, 1)           #(error)   shape=[n_samples,1]
            # dw, db
            dw=(1/n_samples)*x.T@error      #(dw )     shape=[n_features,1]
            db=(1/n_samples)*np.sum(error)  #(db )     shape=float
            #update w, b
            self.w-=self.learning_rate*dw #(self.w)shape=[n_features,1]
            self.b-=self.learning_rate*db

    def predict(self,x):

        y_prob = self.sigmoid(x@self.w+self.b )
        return (y_prob >= self.thresholds).astype(int) 
    
    def accuracy(self,x,y):

        if len(y.shape)==1:
            y=np.array(y).reshape(-1,1)

        y_pred=self.predict(x)
        return np.mean(y_pred==y)

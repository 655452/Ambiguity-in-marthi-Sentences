from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier



def models():
    X = pickle.load(open('embedded.pickle','rb'))
    y = pickle.load(open('label.pickle','rb'))
    
    train_x,test_x,train_y,test_y= train_test_split(X,y,test_size=0.2,random_state = 42)
    import numpy as np
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    #svm
    clf = svm.SVC()
    clf.fit(train_x,train_y)
    svm_pred =clf.predict(test_x)
    
    #naive bayes
    gaussian = GaussianNB() 
    gaus_clf = gaussian.fit(train_x,train_y)
    gaus_pred = gaus_clf.predict(test_x)
    
    #knn
    knn = neigh.fit(train_x,train_y)
    knn_pred = knn.predict(test_x)
    
    #id3
    id3 = DecisionTreeClassifier(random_state=0)
    id3_fit = id3.fit(train_x,train_y)
    id3_pred = id3_fit.predict(test_x)
    
    #Multilayer perceptron
    mp= MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)    
    mp_fit =mp.fit(train_x,train_y)
    mp_pred = mp_fit.predict(test_x)
    
    return test_y,svm_pred,gaus_pred,knn_pred,id3_pred,mp_pred
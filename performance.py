import matplotlib.pyplot as plt
import existing
from sklearn.metrics import confusion_matrix
import numpy as np

def Proposed(model):
    objects = ('Accuracy', 'Precision', 'Recall')
    y_pos = np.arange(len(objects))
    performance= [model[0],model[1],model[2]]
    plt.figure('Proposed',figsize=(6,4))
    colors = ['orange','pink','darksalmon','silver','palegreen','wheat','khaki','aquamarine','olivedrab']
    plt.bar(y_pos, performance,color = colors, width = 0.4,align='center')
    plt.xticks(y_pos, objects)
    plt.ylabel('Percentage(%)')
    plt.title('DALSTM_ROA (Proposed)')
    plt.show()
    
def Accuracy(acc):
    A=[acc[0],acc[1],acc[2],acc[3],acc[4]]
    labels = ['DALSTM_ROA (Proposed)','SVM','ID3','KNN','Naive Bayes']
    sizes = [(A[0]), (A[1]), (A[2]), (A[3]),(A[4])]
    colors = ['orange','pink','darksalmon','silver','palegreen','wheat','khaki','aquamarine','olivedrab']
    plt.figure('Accuracy')
    plt.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%',startangle=90, pctdistance=0.85)#, explode = explode)
    centre_circle = plt.Circle((0,0),0.50,fc='white')    
    plt.title('Accuracy', fontname="Times New Roman",fontweight="bold")
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.tight_layout()

def Precision(pre):
    A=[pre[0],pre[1],pre[2],pre[3],pre[4]]
    labels = ['DALSTM_ROA (Proposed)','SVM','ID3','KNN','Naive Bayes']
    sizes = [(A[0]), (A[1]), (A[2]), (A[3]),(A[4])]
    colors = ['orange','pink','darksalmon','silver','palegreen','wheat','khaki','aquamarine','olivedrab']
    plt.figure('Precision')
    plt.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%',startangle=90, pctdistance=0.85)#, explode = explode)
    centre_circle = plt.Circle((0,0),0.50,fc='white')    
    plt.title('Precision', fontname="Times New Roman",fontweight="bold")
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.tight_layout()

def Recall(Re):
    A=[Re[0],Re[1],Re[2],Re[3],Re[4]]
    labels = ['DALSTM_ROA (Proposed)','SVM','ID3','KNN','Naive Bayes']
    sizes = [(A[0]), (A[1]), (A[2]), (A[3]),(A[4])]
    colors = ['orange','pink','darksalmon','silver','palegreen','wheat','khaki','aquamarine','olivedrab']
    plt.figure('Recall')
    plt.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%',startangle=90, pctdistance=0.85)#, explode = explode)
    centre_circle = plt.Circle((0,0),0.50,fc='white')    
    plt.title('Recall', fontname="Times New Roman",fontweight="bold")
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.tight_layout()

def perform(model):
    model = existing.models()
    model = np.load('model.npy')
    tn, fp, fn, tp = confusion_matrix(model[:,0],model[:,1]).ravel()
    pr_acc= (tp+tn)/(tp+fn+fp+tn)*100
    pr_pre= tp/(tp+fp)*100
    pr_reca = tp/(tp+fn)*100    
    tn, fp, fn, tp = confusion_matrix(model[:,0],model[:,2]).ravel()
    svm_acc= (tp+tn)/(tp+fn+fp+tn)*100
    svm_pre= tp/(tp+fp)*100
    svm_reca = tp/(tp+fn)*100    
    tn, fp, fn, tp = confusion_matrix(model[:,0],model[:,3]).ravel()
    id_acc= (tp+tn)/(tp+fn+fp+tn)*100
    id_pre= tp/(tp+fp)*100
    id_reca = tp/(tp+fn)*100    
    tn, fp, fn, tp = confusion_matrix(model[:,0],model[:,4]).ravel()
    knn_acc= (tp+tn)/(tp+fn+fp+tn)*100
    knn_pre= tp/(tp+fp)*100
    knn_reca = tp/(tp+fn)*100    
    tn, fp, fn, tp = confusion_matrix(model[:,0],model[:,5]).ravel()
    nv_acc= (tp+tn)/(tp+fn+fp+tn)*100
    nv_pre= tp/(tp+fp)*100
    nv_reca = tp/(tp+fn)*100    
    Proposed((pr_acc,pr_pre,pr_reca))
    Accuracy((pr_acc,svm_acc,id_acc,knn_acc,nv_acc))
    Precision((pr_pre,svm_pre,id_pre,knn_pre,nv_pre))
    Recall((pr_reca,svm_reca,id_reca,knn_reca,nv_reca))
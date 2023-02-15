import numpy as np

def frequency(fre):
    unique = set(fre) 
    frequency = []
    for words in unique : 
        st = fre.count(words)
        frequency.append([st,words])
    return np.array(frequency)
    

def wsd(txt,frequency):
    wsd = txt.split() 
    fre = []
    for val in wsd:
        for i,j in enumerate(frequency[:,1]):
            if j == val:
                value = int(frequency[i,0])
                fre.append((value,val))
    arr_fre = np.array(fre)
    try:
        sense = np.einsum('ij->ij', arr_fre[arr_fre[:,0].argsort(),:])
        return np.array(sense)
    except:
        print('The words are not in the Training data.')
        raise KeyError     

def wsd_se(sense):
    sense = sense.tolist()
    sense.pop()
    return np.array(sense)


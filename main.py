import warnings 
warnings.filterwarnings('ignore')
import re
import torch
from transformers import BertTokenizer
from utils.model import BertWSD
import ambiguous
import glob
import lstm
from lstm import get_predictions
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import one_hot
import numpy as np
import performance
from keras.preprocessing.sequence import pad_sequences
from inltk.inltk import get_sentence_similarity
#from inltk.inltk import get_embedding_vectors
from bert_embedding import BertEmbedding
import numpy as np


MAX_SEQ_LENGTH = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def main():
    # Load fine-tuned model and vocabulary
    print("Loading model...")
    model = BertWSD.from_pretrained('bert-base-multilingual-cased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model.to(DEVICE)
    model.eval()
    files = glob.glob('Dataset/**/*.txt')
    datas = []
    for file in files:
        fet = open(file,'r',encoding ='utf-8')
        data = fet.read()
        cleantxt= re.sub('[~`=!@#$%^&*()_+|{}:">?</.,123465789qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM]', '', data)
        datas.append(cleantxt)
    join = ', '.join(datas)
    token = join.split()
    freq = ambiguous.frequency(token)
    #print(freq)
    
    
    
    # embert = BertEmbedding()
    # [embert(i[1]) for  i in freq ]
    # for i in freq:
    #          print(embert(i[1]))
    # feature = freq[:,1]
    # featur = [one_hot(sent, 50) for sent in feature]
    # pad= pad_sequences(featur, maxlen=2, padding='post')
    # feature = np.array(pad)
    # label = np.int8(freq[:,0])
    # label = np.where(label<0,0,label)
    # classes = np.max(label)
   # train_x,test_x,train_y,test_y = train_test_split(feature,label,random_state=42,train_size=0.8)
    #models = lstm.model(train_x.shape[0],classes) 
    #models.fit(train_x,train_y,validation_split = 0.2,batch_size = 32,epochs = 100,verbose = 1)
    #models = np.load('model.npy')#
    #performance.perform(models)#
    while True:
        sentence = input("Enter a Marathi  sentence\n> ")
        while True:
            def sam(sentence):
                bestout = ''
                #similar = 100 #require for Eucl, mahh and mkd
                max=0
                best=0
                similar = 0  #require for cosine sim
                
                
                                
                ambiguous_word = ''
                try:        
                    text = ambiguous.wsd(sentence,freq)
                    print(text)
                    ambiguous_word = text[-1,-1]
                    print('\nAmbiguous word ::: ',ambiguous_word)
                    predictions = get_predictions(model, tokenizer, text[-1,-1])
                    if predictions:
                        print("\nPredictions:\n") 
                        #print('Ambiguous word : ',ambiguous_word)
                        for i ,predict in enumerate(predictions):
                            if i<=20:
                                print('Gloss {0} : {1}'.format (i+1, predict))
                        for i ,predict in enumerate(predictions):
                            similar_ = get_sentence_similarity(sentence,predict,'mr')
                            print(predict,similar_)
                            #emb123 = get_embedding_vectors(sentence,'mr')
                            #print(emb123)
                            if similar_>max:
                                max=similar_
                            #if similar > similar_: #require for Eucl, mahh and mkd
                            if similar < similar_:#require for Cosine sim
                               similar = similar_
                               bestout= predict
                            #best=1-similar/max  #require for Eucl, mahh and mkd
                        print(best) #require for Eucl, Manh and MKD      
                        print('\n\n\nMaximum Cosine similarity = ',similar, '\nWith the Synset:', bestout)#require for Cosine
                        #print('\n\n\nMaximum similarity = ',best, '\nWith the Synset:', bestout) #require for Eucl, mahh and mkd
                        #print('Maximum Cosine similarity : ',similar)
                        print('Ambiguous word is matched with the Synset Gloss : ',bestout)
                        print("For the Sentence:", sentence)
                        
                except :
                    try:
                        text = ambiguous.wsd(sentence,freq)
                        ambiguous_word = text[-1,-1]
                        sentence = sentence.replace(ambiguous_word,'')
                        sam(sentence)
                    except:
                        pass
            sam(sentence)
            break
if __name__ == '__main__':
    main()
    

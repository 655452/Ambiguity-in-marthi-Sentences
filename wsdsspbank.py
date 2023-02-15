import nltk
import codecs
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet

#-----------------------

# Remove Stop Words. Word Stemming. Returns new tokenised list.


def filteredSentence(sentence):

    filtered_sent = []
    lemmatizer = WordNetLemmatizer()  # lemmatizes the words
    ps = PorterStemmer()  # stemmer stems the root of the word.

    stop_words = set(stopwords.words("english"))
    
    words = word_tokenize(sentence)
    with open('abc1.txt', 'w') as f1:
        print(words, file=f1)
    f1.close()
    #print(words)
    
   
    for w in words:
        
        if w not in stop_words:
            filtered_sent.append(lemmatizer.lemmatize(ps.stem(w)))
            for i in synonymsCreator(w):
                filtered_sent.append(i)
                #f2=open('abc2.txt','w')
                #print(filtered_sent, file=f2)
                

    return filtered_sent
    
# --------------------------

# Add synonyms to match list

def synonymsCreator(word):
    synonyms = []

    for syn in wordnet.synsets(word):
        for i in syn.lemmas():
            synonyms.append(i.name())
            
               

    return synonyms
   

# ---------------------------------------------------------------------------------------

# Check and return similarity
#synset1.wup_similarity(synset2): Wu-Palmer Similarity:Return a score denoting how similar two word senses are, based on the depth of the two senses. 


def simlilarityCheck(word1, word2):

    word1 = word1 + ".n.01"
    word2 = word2 + ".n.01"
    
    try:
        w1 = wordnet.synset(word1)
        w2 = wordnet.synset(word2)
        return w1.wup_similarity(w2)
        #return w1.res_similarity(w2)
        #return w1.lch_similarity(w2)
        #return w1.path_similarity(w2)
        #return w1.jcn_similarity(w2)
        #return w1.lesk_similarity(w2)
        
    #•	Wu-Palmer Similarity (Wu and Palmer 1994)
    
        #return w1.res_similarity(w2)
    #•	Resnik Similarity (Resnik 1995)
        #return w1.lch_similarity(w2)
    #•	Leacock-Chodorow Similarity (Leacock and Chodorow 1998)
        #return w1.path_similarity(w2)
    #•	Path similarity
        #return w1.jcn_similarity(w2)
    #•	Jiang-Conrath distance (Jiang and Conrath 1997)
        #return w1.lesk_similarity(w2)
    
        

    except:
        return 0

# -----------------------------------------------------------------------------------------


def simpleFilter(sentence):

    filtered_sent = []
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(sentence)
    #print(words)
    for w in words:
        if w not in stop_words:
            filtered_sent.append(lemmatizer.lemmatize(w))
                #for i in synonymsCreator(w):
                  #  filtered_sent.append(i)
    return filtered_sent


if __name__ == '__main__':

    vampirefile = codecs.open("arraybank.txt", 'r', 'utf-8', errors='ignore')
    sent1 = vampirefile.read().lower()
    
    cricfile = codecs.open("financebank.txt", 'r', "utf-8")
    sent2 = cricfile.read().lower()
    #print(sent2)

    arrayfile = codecs.open("riverbank.txt", 'r', "utf-8")
    sent4 = arrayfile.read().lower()
    print(sent4)
    
    #print(sent1)
    with open('outputEis.txt', 'w') as f7:
        print(sent1, file=f7)
    f7.close()
    sent3 = "start"

    # FOR TEST , replace the above variables with below sent1 and sent 2
    # sent1 = "the commercial banks are used for finance. all the financial matters are managed by financial banks and they have lots of money, user accounts like salary account and savings account, current account. money can also be withdrawn from this bank."
    # sent2 = "the river bank has water in it and it has fishes trees . lots of water is stored in the banks. boats float in it and animals come and drink water from it."
    # sent3 = "from which bank should i withdraw money"

    while(sent3 != "end"):

        sent3 = input("Enter Query: ").lower()
        #testfile = codecs.open("test.txt", 'r', "utf-8")
        #sent3 = testfile.read().lower()
        #print(sent3)

        filtered_sent1 = []
        filtered_sent2 = []
        filtered_sent3 = []

        counter1 = 0
        counter2 = 0
        counter3 = 0
        
        sent31_similarity = 0
        sent32_similarity = 0
        sent33_similarity = 0

        filtered_sent1 = simpleFilter(sent1)
        filtered_sent2 = simpleFilter(sent2)
        filtered_sent3 = simpleFilter(sent3)
        filtered_sent4 = simpleFilter(sent4)
        
        #print(filtered_sent1)
        
        
        for x in filtered_sent2:
            
        
        #print(filtered_sent2)
        #print(filtered_sent3)
        
            for i in filtered_sent3:
            
                for j in filtered_sent1:
                    counter1 = counter1 + 1
                    sent31_similarity = sent31_similarity + simlilarityCheck(i, j)
                #print(sent31_similarity)

                                   
           
            for j in filtered_sent2:
                counter2 = counter2 + 1
                sent32_similarity = sent32_similarity + simlilarityCheck(i, j)


            for j in filtered_sent4:
                counter3 = counter3 + 1
                sent33_similarity = sent33_similarity + simlilarityCheck(i, j)
                #print(sent32_similarity)

                
                           
                    
        filtered_sent1 = []
        filtered_sent2 = []
        filtered_sent3 = []
        filtered_sent4 = []


        filtered_sent1 = filteredSentence(sent1)
        filtered_sent2 = filteredSentence(sent2)
        filtered_sent3 = filteredSentence(sent3)
        filtered_sent4 = filteredSentence(sent4)
        
        #print(filtered_sent1)
        #print(filtered_sent2)
        #print(filtered_sent3)
        sent1_count = 0
        sent2_count = 0
        sent3_count = 0
    

        for i in filtered_sent3:

            for j in filtered_sent1:

                if(i == j):
                    sent1_count = sent1_count + 1
                    #print(sent1_count)
                    with open('sim31.txt', 'w') as f31:
                        print(sent1_count, file=f31)
                    f31.close()
                    

            for j in filtered_sent2:
                if(i == j):
                    sent2_count = sent2_count + 1
                    #print(sent2_count)
                    with open('sim32.txt', 'w') as f32:
                        print(sent2_count, file=f32)
                        f32.close()

        for i in filtered_sent3:

            for j in filtered_sent4:
                if(i == j):
                    sent3_count = sent3_count + 1
                    #print(sent2_count)
                    #with open('sim33.txt', 'w') as f33:
                     #   print(sent2_count, file=f33)
                      #  f33.close()
                

                        
        #print("Synset 1: Bank of River::", +sent1_count + sent31_similarity) 
        #print("Synset 2: Financial Institute::", +sent2_count + sent32_similarity)
        #print("Synset 3: Arrangement of Objects::", +sent3_count + sent33_similarity)

        #print("Synset 1: Break as Fault::", +sent1_count + sent31_similarity) 
        #print("Synset 2: Break as Good Luck::", +sent2_count + sent32_similarity)
        #print("Synset 3: Break as Interruption::", +sent3_count + sent33_similarity)

        print("Synset 1: Bank as an Array ::", +sent1_count + sent31_similarity) 
        print("Synset 2: Bank as an Financial Institute::", +sent2_count + sent32_similarity)
        print("Synset 3: Bank of River::", +sent3_count + sent33_similarity)                      
        
        if((sent1_count + sent31_similarity) > (sent2_count+sent32_similarity)and(sent1_count + sent31_similarity) > (sent3_count+sent33_similarity) ):
            #print("Mammal Bat")
           print("Array Bank")
           
        elif((sent2_count + sent32_similarity) > (sent1_count+sent31_similarity)and(sent2_count + sent32_similarity) > (sent3_count+sent33_similarity) ):
            #print("Cricket Bat")
            print("Finance Bank")
        else:
            print("River Bank")
            
        # -----------------------------------------------
        # Sentence1: the river bank has water in it and it has fishes trees . lots of water is stored in the banks. boats float in it and animals come and drink water from it.
        # sentence2: the commercial banks are used for finance. all the financial matters are managed by financial banks and they have lots of money, user accounts like salary account and savings account, current account. money can also be withdrawn from this bank.
        # query: from which bank should i withdraw money.

        # sen1: any of various nocturnal flying mammals of the order Chiroptera, having membranous wings that extend from the forelimbs to the hind limbs or tail and anatomical adaptations for echolocation, by which they navigate and hunt prey.
        # sen 2: a cricket wooden bat is used for playing criket. it is rectangular in shape and has handle and is made of wood or plastic and is used by cricket players.
    print("\nTERMINATED")

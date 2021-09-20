#from tensorflow import keras
import tensorflow as tf
import pickle
import re 
import numpy as np
from django.contrib.staticfiles import finders



class AutoencoderConfig():
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'Backend'
    
    def pre(self,text1):
        #print('Preprocessing...')
        corpus=[]
        for i in range(len(text1)):
            #text1[i]=text1[i].replace('@VirginAmerica','')
            text1[i]=text1[i].replace("cant't",'can not')
            text1[i]=text1[i].replace("don't",'do not')
            text1[i]=text1[i].replace("should't",'should not')
            text1[i]=text1[i].replace("could't",'could not')
            text1[i]=text1[i].replace("couldn",'could not')
            text1[i]=text1[i].replace("did't",'did not')
            text1[i]=text1[i].replace("didn",'did not')
            text1[i]=text1[i].replace("does't",'does not')
            text1[i]=text1[i].replace("doesn",'does not')
            text1[i] = re.sub(r"(?:\@|https?\://)\S+", "", text1[i])
            review=re.sub('[^a-zA-Z]',' ',text1[i])
            
            review = review.lower()
            review = review.split()
            review = ' '.join(review)
            
            corpus.append(review)
        return corpus
    
    def predict(self,sent):
        #print('Predting : ',sent)
        cc = self.pre(sent)
        #print('Done Pre!!!')
        tok = str(finders.find('Models/{}'.format('tokenizer.pickle')))
        mod = str(finders.find('Models/{}'.format('content/model_score_1')))
        
        with open(tok, 'rb') as handle:
            tokenizer2 = pickle.load(handle)
        #print('Done with tokinzer!!!')
        vv = tokenizer2.texts_to_sequences(cc)
        xx1=tf.keras.preprocessing.sequence.pad_sequences(vv,maxlen=40,padding='post')
        #print('Loding Model...')
        mod =tf.keras.models.load_model(mod)
        #print('Done lOding ... : ',xx1)
        p = mod.predict(xx1)
        #print('Preddd : ',p)
        return np.argmax(p)
            

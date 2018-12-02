import sys
import csv
import pandas as pd
import numpy as np
from keras.layers import Embedding, Reshape, Add, Dot, Dropout, Dense, Input, Flatten
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.models import Sequential,  Model

train_path = sys.argv[1] # path of rating.csv

# load file
df_load = pd.read_csv(train_path,encoding='big5')    
df_load = df_load.iloc[np.random.permutation(len(df_load))]
UserID = [int(i) for i in df_load['userId']]    
MovieID = [int(i) for i in df_load['movieId']]    
Rating = [float(i) for i in df_load['rating']]    
  
UserID = np.array(UserID)    
MovieID = np.array(MovieID)    
Rating = np.array(Rating)    
max_userid = np.max(UserID)    
max_movieid = np.max(MovieID)    
K_FACTORS = 500 # 500-dim vectors

# split training/testing set
UserID_test = UserID[:len(UserID)//10] #first 1/10 is testing set
UserID_train = UserID[len(UserID)//10:] #last 9/10 is training set
MovieID_test = MovieID[:len(MovieID)//10] 
MovieID_train = MovieID[len(MovieID)//10:] 
Rating_test = Rating[:len(Rating)//10]
Rating_train = Rating[len(Rating)//10:]
UserBias_train = np.ones(len(UserID_train))
MovieBias_train = np.ones(len(MovieID_train))
UserBias_test = np.ones(len(UserID_test))
MovieBias_test = np.ones(len(MovieID_test))

# build matrix factorization model
movie_input = Input(shape=[1])    
movie_vec = Flatten()(Embedding(max_movieid + 1, K_FACTORS)(movie_input))    
movie_vec = Dropout(0.5)(movie_vec)    
movie_bias_input = Input(shape=[1])    
movie_bias = Flatten()(Embedding(max_movieid + 1, 1)(movie_bias_input))    
movie_bias = Dropout(0.5)(movie_bias)    
user_input = Input(shape=[1])    
user_vec = Flatten()(Embedding(max_userid + 1, K_FACTORS)(user_input))    
user_vec = Dropout(0.5)(user_vec)    
user_bias_input = Input(shape=[1])    
user_bias = Flatten()(Embedding(max_userid + 1, 1)(user_bias_input))    
user_bias = Dropout(0.5)(user_bias)    
input_vecs = Dot(axes=1)([movie_vec, user_vec])    
out = Add()([input_vecs, movie_bias, user_bias])

model = Model([movie_input,user_input,movie_bias_input,user_bias_input],out)    

model.compile(loss='mse', optimizer='adamax')      
model.fit([MovieID_train, UserID_train,MovieBias_train,UserBias_train], Rating_train,
           batch_size=4096, 
           epochs=10,
           validation_data=([MovieID_test, UserID_test,MovieBias_test,UserBias_test], Rating_test))    
result = model.predict([MovieID_test,UserID_test,MovieBias_test,UserBias_test])    
model.save('testing.h5') # saving model

# output predict.csv file
output = []
for i in range(len(UserID_test)):
    output.append([MovieID_test[i]])
    output[i].append(UserID_test[i])
    output[i].append(Rating_test[i])
    output[i].append(result[i][0])

filename = 'predict.csv'
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["MovieID","UserID","Rating","Rating_estimate"])
for i in range(len(output)):
	s.writerow(output[i]) 
text.close()



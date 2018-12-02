############ get movie/user embedding layers#############
import numpy as np
from keras.models import load_model

model = load_model('testing.h5')
movie_emb = np.array(model.layers[2].get_weights()).squeeze()
user_emb = np.array(model.layers[3].get_weights()).squeeze()


################ use autoencoder to reduce the dimension of embedding layers###############
from keras.layers import Input, Dense
from keras.models import Model

encoding_dim = 128

input_vec = Input(shape=(500,))
encoded = Dense(256, activation='relu')(input_vec)
encoded = Dense(128, activation='relu')(encoded)

decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(500, activation='sigmoid')(decoded)

autoencoder = Model(input_vec, decoded)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(movie_emb, movie_emb,
                epochs=10,
                batch_size=64,
                shuffle=True)

# saving autoencoder model
autoencoder.save('autoencoder_movie_emb.h5')
encoder = Model(input_vec, encoded)

# movie embedding vectors through dimension reduction
encoded_vec = encoder.predict(movie_emb)


################### use TSNE and KMeans packages to classify movies ##############
from sklearn.manifold import TSNE
# from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


vis_data = TSNE(n_components=2).fit_transform(encoded_vec)
vis_x = vis_data[:,0]
vis_y = vis_data[:,1]
kmeans = KMeans(n_clusters=5).fit(movie_emb) # 5 categories

# plt.scatter(vis_x, vis_y,c=kmeans.labels_)
# plt.set_title('Predicted Training Labels')
# plt.show()


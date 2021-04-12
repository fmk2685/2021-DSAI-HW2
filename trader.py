import argparse
import pandas as pd
import pandas_datareader.data as web
import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler
import keras

parser = argparse.ArgumentParser()
parser.add_argument('--training',
                    default='training_data.csv',
                    help='input training data file name')
parser.add_argument('--testing',
                    default='testing_data.csv',
                    help='input testing data file name')
parser.add_argument('--output',
                    default='output.csv',
                    help='output file name')
args = parser.parse_args()



df = np.genfromtxt(args.training, delimiter=',')
df = pd.DataFrame(df)
df.columns = ['Open', 'Low', 'High', 'Close']

df = df[['Open', 'Low', 'High', 'Close']]

n = 1
df['return'] = df['Open'].shift(-n) - df['Open']

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
df_scaled = ss.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, index=df.index, columns=df.columns)
df_scaled['return'] = df['return']

n = 3
feature_names = list(df_scaled.drop('return', axis=1).columns)
X_train = []
y_train = []
indexes = []
df_scaled_x = df_scaled[feature_names]

for i in range(0, len(df_scaled)-n): #len(df_scaled)-n
  X_train.append(df_scaled_x.iloc[i:i+n].values) #i:i+n
  y_train.append(df_scaled['return'].iloc[i+n-1]) #i+n-1
  indexes.append(df_scaled.index[i+n-1])#i+n-1

X_train = np.array(X_train)
y_train = np.array(y_train)


model = keras.models.Sequential()
model.add(keras.layers.LSTM(100, return_sequences=True, input_shape=X_train[0].shape))
model.add(keras.layers.LSTM(100))
model.add(keras.layers.Dense(8))
model.add(keras.layers.Dense(1, kernel_initializer="uniform", activation='linear'))
adam = keras.optimizers.Adam(0.0006)
model.compile(optimizer=adam, loss="binary_crossentropy", metrics=['accuracy'])
model.summary()

get_best_model = keras.callbacks.ModelCheckpoint("lstm.mdl", monitor="val_acc")

history = model.fit(
X_train,
y_train > 1,
batch_size=1000,
epochs=50,
validation_split=0.2,
callbacks=[get_best_model])

df2 = np.genfromtxt(args.testing, delimiter=',')
df2 = pd.DataFrame(df2)
df2.columns = ['Open', 'Low', 'High', 'Close']
df2 = df2[['Open', 'Low', 'High', 'Close']]

ss2 = StandardScaler()
df2_scaled = ss2.fit_transform(df2)
df2_scaled = pd.DataFrame(df2_scaled, index=df2.index, columns=df2.columns)

#day 2
outlist = [0]
stock = 0
#day 3
if (df2_scaled.iloc[1]['Open'] - df2_scaled.iloc[0]['Open']) > 0:
  outlist.append(1)
  stock += 1
else:
  outlist.append(-1)
  stock -= 1
#day 4~
for i in range(3, len(df2_scaled)):
  inarray = np.array([df2_scaled.iloc[i-3:i].values])
  pred = model.predict(inarray)
  pred = pred[0][0]
  if (pred > 0) and stock < 1:
    outlist.append(1)
    stock += 1
  elif (pred < 0) and stock > -1:
    outlist.append(-1)
    stock -= 1
  else:
    outlist.append(0)

outarray = np.array(outlist)

np.savetxt(args.output, outarray,fmt='%.0f', delimiter=',')
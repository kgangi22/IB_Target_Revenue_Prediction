import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
import chart_studio.plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.layers import LSTM

target_sales_data_file = 'Target_Revenue.xlsx'
data = pd.read_excel(target_sales_data_file)

plot_data = [
    go.Scatter(
        x=data['Year'],
        y=data['Revenue'],
    )
]
plot_layout = go.Layout(
        title='Sales'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)

#create a new dataframe to model the difference
df_diff = data.copy()
#add previous sales to the next row
df_diff['Previous Revenue'] = data['Revenue'].shift(1)
#drop the null values and calculate the difference
df_diff = df_diff.dropna()
df_diff['Difference'] = (df_diff['Revenue'] - df_diff['Previous Revenue'])


plot sales diff
plot_data = [
    go.Scatter(
        x=df_diff['Year'],
        y=df_diff['Difference'],
    )
]
plot_layout = go.Layout(
        title='Year Sales Diff'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)

#create dataframe for transformation from time series to supervised
df_supervised = df_diff.drop(['Previous Revenue'],axis=1)
#adding lags
for inc in range(1,13):
    field_name = 'lag_' + str(inc)
    df_supervised[field_name] = df_supervised['Difference'].shift(inc)
#drop null values
df_supervised = df_supervised.dropna().reset_index(drop=True)


#import MinMaxScaler and create a new dataframe for LSTM model
df_model = df_supervised.drop(['Revenue','Year'],axis=1)
#split train and test set
train_set, test_set = df_model[0:-6].values, df_model[-6:].values

#apply Min Max Scaler
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(train_set)
# reshape training set
train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
train_set_scaled = scaler.transform(train_set)
# reshape test set
test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
test_set_scaled = scaler.transform(test_set)

X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1]
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

model = Sequential()
model.add(LSTM(4, batch_input_shape=(1, X_train.shape[1], X_train.shape[2]), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, nb_epoch=100, batch_size=1, verbose=1, shuffle=False)

print("The X test values are: ")
print(X_test)
print()
y_pred = model.predict(X_test,batch_size=1)
print("Y predeicted is : ")
print(y_pred)

#reshape y_pred
y_pred = y_pred.reshape(y_pred.shape[0], 1, y_pred.shape[1])
#rebuild test set for inverse transform
pred_test_set = []
for index in range(0,len(y_pred)):
    print np.concatenate([y_pred[index],X_test[index]],axis=1)
    pred_test_set.append(np.concatenate([y_pred[index],X_test[index]],axis=1))
#reshape pred_test_set
pred_test_set = np.array(pred_test_set)
pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])
#inverse transform
pred_test_set_inverted = scaler.inverse_transform(pred_test_set)
print()
print(pred_test_set_inverted)
#create dataframe that shows the predicted sales
result_list = []
sales_dates = list(data[-7:].Year)
act_sales = list(data[-7:].Revenue)
for index in range(0,len(pred_test_set_inverted)):
    result_dict = {}
    print()
    print(pred_test_set_inverted[index][0])
    print(act_sales[index])
    print()
    result_dict['pred_value'] = int(pred_test_set_inverted[index][0] + act_sales[index])
    result_dict['Year'] = sales_dates[index+1]
    result_list.append(result_dict)
df_result = pd.DataFrame(result_list)
#for multistep prediction, replace act_sales with the predicted sales

print(df_result)

#merge with actual sales dataframe
df_sales_pred = pd.merge(data,df_result,on='Year',how='left')
#plot actual and predicted
plot_data = [
    go.Scatter(
        x=df_sales_pred['Year'],
        y=df_sales_pred['Revenue'],
        name='Actual'
    ),
        go.Scatter(
        x=df_sales_pred['Year'],
        y=df_sales_pred['pred_value'],
        name='Predicted'
    )

]
plot_layout = go.Layout(
        title='Revenue Prediction'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)

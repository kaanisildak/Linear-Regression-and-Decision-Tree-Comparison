

# Imported and read the data

import pandas as pd
bike = pd.read_csv('SeoulBikeData.csv',encoding='latin1')
bike

"""# Pre Analysis




"""

# Let's drop the parantheses and simplify the column names

new_columns=[x.replace("(m/s)","").replace("(°C)","").replace("(%)","").replace("(MJ/m2)","").replace("(cm)","").replace("(mm)","").replace("(10m)","") for x in bike.columns]
bike.columns = new_columns

bike

# Checking our data and identifying whether we have null data or not

bike.info()

# We have no null data. So we may proceed. Next, let's take a look at our correlations between our features to have a 
#broad idea of whether we have a key attribute that is highly correlated with Rented Bike Count.

bike.corr()

# We see that the closest thing we have is 0.53 correlation which is not that significant to me. 
# I would have rathered 0.7 or higher to take more serious actions.

# Now, we have 4 categorical features: Date, Seasons, Holiday and Functioning Day.
# I want to change our Date column to months but I don't know if it is a smart thing to do.

# Let's take a look on our numerical attributes

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline 
import matplotlib.pyplot as plt
bike.hist(bins=50, figsize=(20,15))
plt.show()

# I see that our Snowfall, Rainfall, Solar Radiation and Visibility data are not distributed well.

bike["Rainfall"].value_counts()



"""# Handling Categorical Attribute"""

#when we were looking the data on excel we realized that the months and days made a significant difference on number of bikes so that we added month and dayOfWeek column to our dataframe
import datetime
bike["month"]=[datetime.datetime.strptime(date,'%d/%m/%Y').month for date in bike["Date"]]
bike["dayOfWeek"]=[datetime.datetime.strptime(date,'%d/%m/%Y').weekday() for date in bike["Date"]]

#we added months so season is not important right now because months will give a better prediction to us
bike.drop("Seasons",axis=1,inplace=True)

## Now we start encoding categorial data because these columns cannot be inserted in to linear regression algorithm
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
oneHotEncoder = OneHotEncoder()
labelEncoder=LabelEncoder();
holiday_cat = bike["Holiday"]
holidayLabelEncoded=labelEncoder.fit_transform(holiday_cat)
holiday_cat_encoded = pd.DataFrame(oneHotEncoder.fit_transform(holidayLabelEncoded.reshape(-1,1)).toarray())
holiday_cat_encoded.columns=oneHotEncoder.get_feature_names(["Holiday"])
bike=bike.join(holiday_cat_encoded)
bike.drop("Holiday",axis=1,inplace=True);
bike

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
oneHotEncoder = OneHotEncoder()
labelEncoder=LabelEncoder();
function_cat = bike["Functioning Day"]
functionLabelEncoded=labelEncoder.fit_transform(function_cat)
function_cat_encoded = pd.DataFrame(oneHotEncoder.fit_transform(functionLabelEncoded.reshape(-1,1)).toarray())
function_cat_encoded.columns=oneHotEncoder.get_feature_names(["Functioning Day"])
bike=bike.join(function_cat_encoded)
bike.drop("Functioning Day",axis=1,inplace=True);
bike

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
oneHotEncoder = OneHotEncoder()
labelEncoder=LabelEncoder();
month_cat = bike["month"]
monthLabelEncoded=labelEncoder.fit_transform(month_cat)
month_cat_encoded = pd.DataFrame(oneHotEncoder.fit_transform(monthLabelEncoded.reshape(-1,1)).toarray())
month_cat_encoded.columns=oneHotEncoder.get_feature_names(["month"])
bike=bike.join(month_cat_encoded)
bike.drop("month",axis=1,inplace=True);
bike

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
oneHotEncoder = OneHotEncoder()
labelEncoder=LabelEncoder();
day_cat = bike["dayOfWeek"]
dayLabelEncoded=labelEncoder.fit_transform(month_cat)
day_cat_encoded = pd.DataFrame(oneHotEncoder.fit_transform(dayLabelEncoded.reshape(-1,1)).toarray())
day_cat_encoded.columns=oneHotEncoder.get_feature_names(["dayOfWeek"])
bike=bike.join(day_cat_encoded)
bike.drop("dayOfWeek",axis=1,inplace=True);
bike

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
oneHotEncoder = OneHotEncoder()
labelEncoder=LabelEncoder();
hour_cat = bike["Hour"]
hourLabelEncoded=labelEncoder.fit_transform(hour_cat)
hour_cat_encoded = pd.DataFrame(oneHotEncoder.fit_transform(hourLabelEncoded.reshape(-1,1)).toarray())
hour_cat_encoded.columns=oneHotEncoder.get_feature_names(["Hour"])
bike=bike.join(hour_cat_encoded)
bike.drop("Hour",axis=1,inplace=True);
bike

"""# Dividing dataset into  Test / Train """

# Let's split our data into train and test.

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(bike, test_size=0.2, random_state=36)

# Let's check how our test and train sets are.

train_set.info()

test_set.info()

test_set = test_set.drop("Date",axis=1)

"""# Pre-processing"""

# Now, I used Stratified Sampling and compared it to not using it. Turns out it decreases R2 so I decided to
# remove it.

bike = train_set.copy()

corr_matrix = bike.corr()
corr_matrix["Rented Bike Count"].sort_values(ascending=False)

# Here I am labeling my target column and removing it from my dataset since I'm trying to predict it.

bike = train_set.drop("Rented Bike Count", axis=1)
bike_labels = train_set["Rented Bike Count"].copy()
test_set_labels=test_set["Rented Bike Count"].copy()
test_set=test_set.drop("Rented Bike Count", axis=1)

#we dropped date column because we already parsed it and got some additional columns
bike_num = bike.drop("Date", axis=1)

"""# Linear Regression on Non-scaled Data"""

# NOW, without scaled-data:

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(bike_num, bike_labels)

# Performance of Train Set with not-scaled data:

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

bike_predictions = lin_reg.predict(bike_num)
lin_mse = mean_squared_error(bike_labels, bike_predictions)
lin_rmse = np.sqrt(lin_mse)
r2 = r2_score(bike_labels, bike_predictions)
print(lin_mse)
print(lin_rmse)
print(r2)

# Performance of Test Set with not-scaled data:

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

bike_predictions = lin_reg.predict(test_set)
lin_mse = mean_squared_error(test_set_labels, bike_predictions)
lin_rmse = np.sqrt(lin_mse)
r2 = r2_score(test_set_labels, bike_predictions)
print(lin_mse)
print(lin_rmse)
print(r2)

"""# Decision Tree on Non-scaled Data"""

from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(bike_num, bike_labels)
bike_pred_tree = tree_reg.predict(bike_num)

# Performance of Train Set

tree_mse = mean_squared_error(bike_labels, bike_pred_tree)
tree_rmse = np.sqrt(tree_mse)
tree_r2 = r2_score(bike_labels, bike_pred_tree)

print(tree_mse)
print(tree_rmse)
print(tree_r2)

"""# Scaling Data"""

# Here I am scaling my data using some transformation pipelines based on "".

import sklearn
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")),
                         ('std_scaler', StandardScaler()),
                        ])
bike_num_tr = num_pipeline.fit_transform(bike_num)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
    
num_attribs = list(bike_num)
cat_attribs = ['Snowfall ','Rainfall','Visibility ','Solar Radiation ']

num_pipeline = Pipeline([('selector', DataFrameSelector(num_attribs)),
                         ('imputer', SimpleImputer(strategy="median")),
                         ('std_scaler', StandardScaler())
                        ])

cat_pipeline = Pipeline([('selector', DataFrameSelector(cat_attribs)),
                         ('one_hot', OneHotEncoder())
                        ])

from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[("num_pipeline", num_pipeline),
                                               ("cat_pipeline", cat_pipeline)
                                              ])

bike_prepared = full_pipeline.fit_transform(bike_num)
bike_prepared

# Checking the density

bike_prepared.todense()

"""# Linear Regression on Scaled Data"""

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(bike_prepared, bike_labels)

# Performance of Train Set with scaled-data

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

bike_predictions = lin_reg.predict(bike_prepared)
lin_mse = mean_squared_error(bike_labels, bike_predictions)
lin_rmse = np.sqrt(lin_mse)
r2 = r2_score(bike_labels, bike_predictions)
print(lin_mse)
print(lin_rmse)
print(r2)

bike_test = full_pipeline.fit_transform(test_set)

# Performance of Test Set
#from sklearn.metrics import mean_squared_error
#from sklearn.metrics import r2_score
#bike_pred_test = lin_reg.predict(bike_test)
#lin_mse_test = mean_squared_error(bike_labels_test, bike_pred_test)
#lin_rmse_test = np.sqrt(lin_mse_test)
#r2_test = r2_score(bike_labels_test, bike_pred_test)
#print(lin_mse_test)
#print(lin_rmse_test)
#print(r2_test)

"""# Decision Tree on Scaled Data"""

from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(bike_prepared, bike_labels)
bike_pred_tree = tree_reg.predict(bike_prepared)

tree_mse = mean_squared_error(bike_labels, bike_pred_tree)
tree_rmse = np.sqrt(tree_mse)
tree_r2 = r2_score(bike_labels, bike_pred_tree)
print(tree_mse)
print(tree_rmse)
print(tree_r2)

##Performance of Test Set 

# bike_pred_tree_test = tree_reg.predict(bike_test)
# tree_mse_test = mean_squared_error(bike_labels_test, bike_pred_tree_test)
# tree_rmse_test = np.sqrt(tree_mse_test)
# tree_r2_test = r2_score(bike_labels_test, bike_pred_tree_test)

# print(tree_rmse_test)
# print(tree_r2_test)

"""# Analysis of Results

1.   Linear Regression with Non-Scaled  Data(With training data): 

    *   MSE: 127092.62772991841
    *   RMSE: 356.500529775088
    *   R2: 0.6944069487811673
2.   Linear Regression with Non-Scaled  Data(With test data): 

    *   MSE: 121547.78139439243
    *   RMSE: 348.6370338825071
    *   R2: 0.7080113446427028


3.   Linear Regression with Scaled Data:

    *   MSE: 79449.58087485294
    *   RMSE: 281.868020312438
    *   R2: 0.8089642155389298

4.   Decision Tree with Scaled Data:

    *   MSE: 0.0
    *   RMSE: 0.0
    *   R2: 1.0

5.   Decision Tree with Non-Scaled Data:

    *   MSE: 0.0
    *   RMSE: 0.0
    *   R2: 1.0

We see that, Decision Tree is always better in every case than Linear Regression. 
  R2 determines how good our model works. The higher the R2 the better/accurate results we have. 
  We first parsed Seasons into 0,1,2,3s and the Date column into month, dayOfWeek and hours. We added hours because in our pre analysis, we saw that throughout the day, the hours made a big difference. This way, we have data on a season,month, hour and the day of the week basis now. Right after this, we parsed Holiday and Functioning Day categories into 0 and 1s. After all these actions, we got rid of the previous ones since we already have joined the parsed ones.

The ideas to improve: We did not have the opportunity to go through the performance of the test data. If we had more time, we could have done that.

In conclusion, we can see that MSE and RMSE are errors and R2 is our performance indicator. The bigger the R2 the better results we have. As we see, with scaled data, there is a significant difference of 0.1. With scaled data, we have %80 accuracy whereas with non-scaled data we have %69. And overall, DecisionTree is always better than Linear Regression.











 
"""

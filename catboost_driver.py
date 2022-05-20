# Databricks notebook source
# MAGIC %pwd

# COMMAND ----------

# MAGIC %fs ls file:/Workspace/Repos/ben.mackenzie@databricks.com

# COMMAND ----------

# MAGIC %md
# MAGIC %fs ls file:/ is the same as %ls /
# MAGIC but results are returned in a different format.

# COMMAND ----------

# MAGIC %fs ls file:/

# COMMAND ----------

# MAGIC %md
# MAGIC this will read from dbfs mount

# COMMAND ----------

# MAGIC %fs ls /databricks-datasets/nyctaxi

# COMMAND ----------

# MAGIC %pip install catboost

# COMMAND ----------

from catboost.datasets import titanic
titanic_train, titanic_test = titanic()

# COMMAND ----------

import mlflow.catboost
mlflow.catboost.autolog()

# COMMAND ----------

titanic_train = titanic_train.dropna()

# COMMAND ----------

target = titanic_train.pop('Survived')

# COMMAND ----------

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(titanic_train, target, train_size=0.8)

# COMMAND ----------

categories = X_train.columns[np.where(X_train.dtypes == np.object)].tolist()

# COMMAND ----------

categories

# COMMAND ----------

model = CatBoostClassifier()
model.fit(X_train, y_train, cat_features=categories, eval_set=(X_test, y_test), plot=True)

# COMMAND ----------



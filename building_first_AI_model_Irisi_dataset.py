#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data # shape (150,4)
y = iris.target # shape (150,)
print(iris.feature_names, iris.target_names)


# In[6]:


import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


# In[7]:


plt.scatter(X[:, 0], X[:, 1])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Iris Dataset Scatter Plot")
plt.show()


# In[8]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42)


# In[9]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)


# In[23]:


model.fit(X_train, y_train)


# In[24]:


y_pred = model.predict(X_test)


# In[25]:


print("Predcitons:", y_pred[:5])
print("True labels:", y_test[:5])


# In[26]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[27]:


importances = model.feature_importances_


# In[27]:


importances = model.feature_importances_


# In[28]:


type(model)


# In[31]:


model.fit(X, y)


# In[32]:


model.feature_importances_



# In[36]:


from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target
feature_names = iris.feature_names


# In[37]:


model.fit(X, y)


# In[38]:


import pandas as pd

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)


# In[39]:


import matplotlib.pyplot as plt

plt.barh(importance_df["feature"], importance_df["importance"])
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.gca().invert_yaxis()
plt.show()


# In[42]:


from sklearn.tree import plot_tree


# In[43]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plot_tree(model, filled=True)
plt.show()


# In[45]:


from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train, y_train)
y_pred_knn = model_knn.predict(X_test)
print("k-NN accuract.", accuracy_score(y_test, y_pred_knn))


# In[ ]:





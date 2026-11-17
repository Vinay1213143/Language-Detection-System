import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

data=pd.read_csv(r"C:\Users\vinay\OneDrive\Desktop\ThinkQuotient Generative AI\Projects\GENAI\Language Detection\language.csv")

x=np.array(data['Text'])
y=np.array(data['language'])

cv=CountVectorizer()

X=cv.fit_transform(x)

X_train,X_text,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)

model=MultinomialNB()
model.fit(X_train,y_train)
model.score(X_text,y_test)

user=input("Enter a Text:")
data=cv.transform([user]).toarray()
output=model.predict(data)
print(output)



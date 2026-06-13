# Objective of the model is to predict whether an email is Spam or Not Spam

# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
# Replace the path with your dataset location
data = pd.read_csv(r"/content/email.csv", encoding="latin-1")

# Keep only required columns
data = data[['Category', 'Message']]
data.columns = ['Label', 'Message']

print(data.head())

# Convert labels (ham=0, spam=1)
le = LabelEncoder()
data['Label'] = le.fit_transform(data['Label'])

# Features and target
X = data['Message']
y = data['Label']

# Convert text into numerical features
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Naive Bayes model
nb = MultinomialNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)

# Train SVM model
svm = SVC()
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

# Confusion Matrix for SVM
cm = confusion_matrix(y_test, svm_pred)
print("Confusion Matrix:\n", cm)

# Plot label distribution
plt.figure(figsize=(6,4))
sns.countplot(x=data['Label'])

plt.title("Spam vs Not Spam Emails")
plt.xlabel("0 = Nonspam, 1 = Spam")
plt.ylabel("Count")
plt.show()

# Accuracy
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_pred))
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))

# Test with your own email
sample_email = ["Congratulations! You have won a free iPhone. Click here to claim your prize."]

sample_vector = tfidf.transform(sample_email)
prediction = svm.predict(sample_vector)

if prediction[0] == 1:
    print("Prediction: Spam Email")
else:
    print("Prediction: Not Spam Email")

import numpy as np
import cPickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.pipeline import make_pipeline


all_data = np.load('twdata/all_train_normalized.npy')
all_label = np.load('twdata/all_train_label.npy')

print all_data.shape
print all_label.shape

# # scaling
# scaler = StandardScaler()
# all_data = scaler.fit_transform(all_data)


# decomposing
pca = PCA(n_components=400)
all_data = pca.fit_transform(all_data)
print pca.explained_variance_ratio_.sum()


# train test split
X_train, X_test, y_train, y_test = train_test_split(all_data, all_label, test_size=0.1,
                                                    stratify=all_label,random_state=13)

# balancing classify
pipeline = make_pipeline(SMOTE(random_state=37),
                         KNeighborsClassifier(weights='distance',n_neighbors=8,))

pipeline.fit(X_train, y_train)


# Classify and report the results
print(classification_report(y_train, pipeline.predict(X_train)))
print(classification_report(y_test, pipeline.predict(X_test)))
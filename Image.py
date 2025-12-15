import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize

DATASET_PATH = r"C:\Users\HP\Downloads\archive\fruits-360_original-size\fruits-360-original-size"

X = []
y = []

max_images = 2000
count = 0


#Feature Extraction
for label in os.listdir(os.path.join(DATASET_PATH, "Training")):
    folder = os.path.join(DATASET_PATH, "Training", label)
    for file in os.listdir(folder):
        if count >= max_images:
            break

        img_path = os.path.join(folder, file)
        img = Image.open(img_path).convert("RGB").resize((50, 50))
        img_array = np.array(img).flatten() / 255.0

        X.append(img_array)
        y.append(label)
        count += 1

    if count >= max_images:
        break

X = np.array(X)
y = np.array(y)
print(f"Loaded {len(X)} images.")

unique_labels = sorted(list(np.unique(y)))
label_to_num = {label: i for i, label in enumerate(unique_labels)}
y_num = np.array([label_to_num[l] for l in y])

# Train-Test split
X_train, X_test, y_train_num, y_test_num, y_train, y_test = train_test_split(
    X, y_num, y, test_size=0.2, random_state=50, stratify=y_num
)


#Logisitic Regression
logreg = LogisticRegression(max_iter=900)
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))

#K-means
k = len(unique_labels)
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_train)

cluster_to_label = {}
for cluster in range(k):
    indices = np.where(clusters == cluster)
    labels_in_cluster = y_train[indices]
    if len(labels_in_cluster) > 0:
        most_common_label = max(set(labels_in_cluster), key=list(labels_in_cluster).count)
        cluster_to_label[cluster] = most_common_label

test_clusters = kmeans.predict(X_test)
y_pred_kmeans = [cluster_to_label[c] for c in test_clusters]

print("K-Means Classification Accuracy:", accuracy_score(y_test, y_pred_kmeans))

#Confusion Matrix
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
cm_lr = confusion_matrix(y_test, y_pred_lr, labels=unique_labels)
sns.heatmap(cm_lr, xticklabels=unique_labels, yticklabels=unique_labels,
            cmap='Blues', cbar=False)
plt.title("Confusion Matrix - Logistic Regression")
plt.xticks(rotation=90)

plt.subplot(1,2,2)
cm_km = confusion_matrix(y_test, y_pred_kmeans, labels=unique_labels)
sns.heatmap(cm_km, xticklabels=unique_labels, yticklabels=unique_labels,
            cmap='Oranges', cbar=False)
plt.title("Confusion Matrix - K-Means")
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

#Loss Curve
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_loss(x_input, y_true, weights):
    h_pred = sigmoid(x_input @ weights)
    eps = 1e-10
    return -(y_true * np.log(h_pred + eps) + (1 - y_true) * np.log(1 - h_pred + eps)).mean()

Xb = np.c_[np.ones((X_train.shape[0], 1)), X_train]
classes = len(unique_labels)
epochs = 150
lr = 0.1

loss_curves = {}

for cls in range(classes):
    y_bin = np.array(y_train_num == cls, dtype=int)
    w = np.zeros(Xb.shape[1])
    losses = []

    for _ in range(epochs):
        h = sigmoid(Xb @ w)
        w -= lr * (Xb.T @ (h - y_bin)) / len(y_bin)
        losses.append(log_loss(Xb, y_bin, w))

    loss_curves[unique_labels[cls]] = losses

#PLOT LOSS CURVES
plt.figure(figsize=(10,5))
for label, curve in loss_curves.items():
    plt.plot(curve, label=str(label))

plt.title("Loss Curve for Each Class (One-vs-Rest Logistic Regression)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

#ROC curve
y_pred_prob = logreg.predict_proba(X_test)

y_test_bin = label_binarize(y_test_num, classes=range(len(unique_labels)))

plt.figure(figsize=(8,6))
for i, label_name in enumerate(unique_labels):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    auc_value = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{label_name} (AUC={auc_value:.2f})")

plt.plot([0,1], [0,1], 'k--')
plt.title("ROC Curve per Class")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

plt.figure(figsize=(12,4))
for i in range(10):
    img = X_test[i].reshape(50, 50, 3)
    plt.subplot(2, 5, i+1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"True:{y_test[i]}\nLR:{y_pred_lr[i]}\nKM:{y_pred_kmeans[i]}", fontsize=8)

plt.show()
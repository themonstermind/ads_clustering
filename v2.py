import os
import requests
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing import image as k_image
from keras.applications.vgg16 import preprocess_input
from sklearn.cluster import KMeans
from collections import defaultdict
import json

# Function to download images based on domain_name
def download_images_by_domain(df):
    print("Downloading Images")
    for _, row in df.iterrows():
        domain_path = os.path.join("data", row["domain_name"], "images")
        if not os.path.exists(domain_path):
            os.makedirs(domain_path)
        
        image_url = row["ad_image-src"]
        image_name = os.path.basename(image_url.split("?")[0])  # Extracting image name from the URL
        image_path = os.path.join(domain_path, image_name)
        
        try:
            response = requests.get(image_url)
            with open(image_path, 'wb') as f:
                f.write(response.content)
        except Exception as e:
            print(f"Error downloading {image_url}. Error: {e}")
    print("Downloading Images Finished")

# Extract features using VGG16 model
def extract_features(img_path):
    model = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
    img = k_image.load_img(img_path, target_size=(224, 224))
    img_data = k_image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features.flatten()

# Determine optimal number of clusters for KMeans
def determine_optimal_k(features, max_k=10):
    distortions = []
    K = range(1, max_k + 1)
    for k in K:
        kmeanModel = KMeans(n_clusters=k, n_init=10)
        kmeanModel.fit(features)
        distortions.append(kmeanModel.inertia_)

    # Calculate the differences in distortions
    diff = np.diff(distortions)

    # Calculate the differences of differences
    diff_r = diff[1:] / diff[:-1]

    # Get the elbow point, which is where the difference starts to stabilize
    k = np.argmin(diff_r) + 2  # Adding 2 because of zero-based indexing and diff operation
    return k

# Reduce dimensionality of features by PCA
def dimensionality_reduction(features):
    pca = PCA(n_components=10)
    pca.fit(features)
    features_reduced = pca.transform(features)
    return features_reduced


# Main function to execute the entire process
def main():
    # Read CSV file
    df = pd.read_csv("fb_d2c_top50_domain_image_ads.csv")
    
    # Download images
    download_images_by_domain(df)

    # Cluster images by domain and save in clusters.json
    cluster_data = {
        "data": []
    }

    for domain in df['domain_name'].unique():
        domain_df = df[df['domain_name'] == domain]
        image_files = [os.path.join("data", domain, "images", os.path.basename(url.split("?")[0])) for url in domain_df["ad_image-src"].values]

        # Extract features
        features = [extract_features(img) for img in image_files]
        features = np.array(features)
        print("All Features Extracted ", domain)

        #reduce dimensionality
        features_reduced = dimensionality_reduction(features)
        print("PCA Finished ", domain)
        
        # Determine optimal k
        k = determine_optimal_k(features_reduced)
        print("Optimal K as per Elbow Method is ", domain, k)
        
        # KMeans clustering
        print("Clustering Process Started ", domain)
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(features_reduced)
        
        # Organizing clusters
        domain_clusters = defaultdict(list)
        for i, label in enumerate(kmeans.labels_):
            domain_clusters[label].append(image_files[i])
        
        cluster_data["data"].append({
            "domain_name": domain,
            "clusters": domain_clusters
        })
        print("Clustering Process Finished ", domain)

    # Save cluster data to clusters.json
    with open("clusters.json", "w") as json_file:
        json.dump(cluster_data, json_file)

# Execute the main function
if __name__ == "__main__":
    main()

import os
import requests
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing import image as k_image
from keras.applications.vgg16 import preprocess_input
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import defaultdict
import json
import webbrowser
import http.server
import socket
import socketserver
import threading
import os
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, UnidentifiedImageError

# Helper function to download an individual image
def download_single_image(image_url, image_path):
    # Check if the image already exists, if so, skip downloading
    if os.path.exists(image_path):
        return
    
    try:
        response = requests.get(image_url)
        with open(image_path, 'wb') as f:
            f.write(response.content)
    except Exception as e:
        print(f"Error downloading {image_url}. Error: {e}")

# Function to download images based on domain_name using parallel threads
def download_images_by_domain(df):
    print("Downloading Images")
    
    # Using ThreadPoolExecutor to speed up the downloading process
    with ThreadPoolExecutor(max_workers=10) as executor:
        for _, row in df.iterrows():
            domain_path = os.path.join("data", row["domain_name"], "images")
            if not os.path.exists(domain_path):
                os.makedirs(domain_path)

            image_url = row["ad_image-src"]
            image_name = os.path.basename(image_url.split("?")[0])  # Extracting image name from the URL
            image_path = os.path.join(domain_path, image_name)
            
            # Submitting download tasks to the thread pool
            executor.submit(download_single_image, image_url, image_path)
    
    print("Downloading Images Finished")

# Extract features using VGG16 model
model = None

def extract_features(img_path):
    global model
    if model is None:
        model = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
    
    try:
        img = k_image.load_img(img_path, target_size=(224, 224))
    except UnidentifiedImageError:
        print(f"Error: Cannot identify image file: {img_path}. Skipping...")
        return None
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
    n_components = min(10, features.shape[0]-1, features.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(features)
    features_reduced = pca.transform(features)
    return features_reduced


def check_domain_status(domain_name, json_data):
    for entry in json_data["data"]:
        if entry["domain_name"] == domain_name:
            return entry["status"] == "processed"
    return False

def convert_keys_to_string(dictionary):
    """Recursively converts dictionary keys to strings."""
    if not isinstance(dictionary, dict):
        return dictionary
    return {str(key): convert_keys_to_string(value) for key, value in dictionary.items()}

def save_clusters_to_json(clusters, json_path):
    clusters = convert_keys_to_string(clusters)
    with open(json_path, 'w') as file:
        json.dump(clusters, file)

def start_http(port=8000):
    """Starts an HTTP server on the specified port if it's not already running."""
    
    # Function to check if the port is in use
    def is_port_in_use(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    # If port is already in use, return
    if is_port_in_use(port):
        print(f"Server is already running on port {port}")
        return

    # Function to start the HTTP server
    def start_server():
        Handler = http.server.SimpleHTTPRequestHandler
        with socketserver.TCPServer(("", port), Handler) as httpd:
            print(f"Serving at port {port}")
            httpd.serve_forever()

    # Start the server in a separate thread
    thread = threading.Thread(target=start_server)
    thread.start()

    # Give the server a second to initialize and then open the browser
    webbrowser.open(f'http://localhost:{port}/json_viewer.html', new=2)


# Main function to execute the entire process
def main():

    # Start the server in a separate thread
    start_http()

    # Read CSV file
    df = pd.read_csv("fb_d2c_top50_domain_image_ads.csv")
    
    # Download images
    download_images_by_domain(df)
    
    # Check if clusters.json exists and is not empty
    if os.path.exists('clusters.json') and os.path.getsize('clusters.json') > 0:
        with open('clusters.json', 'r') as f:
            clusters = json.load(f)
    else:
        clusters = {"data":[]}

    #iterate over all unique domains in dataset
    for domain in df['domain_name'].unique():
        domain_df = df[df['domain_name'] == domain]
        
        # skip domain if already processed
        if check_domain_status(domain,clusters):
            continue
        
        image_files = [
            os.path.join("data", domain, "images", os.path.basename(url.split("?")[0]))
            for url in domain_df["ad_image-src"].values            
        ]

        # Check if the domain has less than 4 images, then cluster them together and directly write to json file
        if len(domain_df) < 4:
            
            clusters["data"].append({
                "domain_name": domain,
                "clusters": 
                    {"0": [image_files]}
                ,
                "status": "processed"
            })            

            save_clusters_to_json(clusters, 'clusters.json')
            
            continue
        
        

        # Extract features
        features = [extract_features(img) for img in image_files]
        features = [f for f in features if f is not None]
        features = np.array(features)
        print("All Features Extracted ", domain)

        #reduce dimensionality
        features_reduced = dimensionality_reduction(features)
        print("PCA Finished ", domain)
        
        # Determine optimal k
        k = determine_optimal_k(features_reduced,min(features_reduced.shape[0] - 1, 10))
        print("Optimal K as per Elbow Method is ", domain, k)
        
        # KMeans clustering
        print("Clustering Process Started ", domain)
        
        new_clusters = defaultdict(list)
        
        # KMeans clustering by visual similarity within the quarter
        kmeans = KMeans(n_clusters=k, n_init=10)  # Explicitly set n_init to suppress warnings
        labels = kmeans.fit_predict(features_reduced)
        for label, path in zip(labels, image_files):
            new_clusters[str(label)].append(path)  # Convert label to string for JSON serialization        
        

        clusters["data"].append({
            "domain_name": domain,
            "clusters": new_clusters,
            "status": "processed"
        })
        print("Clustering Process Finished ", domain)

        # Save cluster data to clusters.json
        save_clusters_to_json(clusters, 'clusters.json')

    print("All steps executed successfully!")


# Execute the main function
if __name__ == "__main__":
    main()

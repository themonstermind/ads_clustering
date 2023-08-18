import os
import pandas as pd
import requests
from urllib.parse import urlparse
import cv2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from collections import defaultdict
from dateutil.parser import parse
import json
import numpy as np
import tensorflow as tf
import webbrowser
import http.server
import socketserver
import threading
import matplotlib.pyplot as plt
from keras.preprocessing import image as k_image
from keras.applications.vgg16 import preprocess_input


def determine_optimal_k(features, max_k=10):
    distortions = []
    K = range(1, max_k + 1)
    for k in K:
        kmeanModel = KMeans(n_clusters=k,n_init=10)
        kmeanModel.fit(features)
        distortions.append(kmeanModel.inertia_)

    # Calculate the differences in distortions
    diff = np.diff(distortions)

    # Calculate the differences of differences
    diff_r = diff[1:] / diff[:-1]

    # Get the elbow point, which is where the difference starts to stabilize
    k = np.argmin(diff_r) + 2  # +2 because diff_r is 2 less than distortions

    return k

def combine_csv_files(folder_path):
    combined_df = pd.DataFrame(columns=['url', 'date'])

    # Read each CSV file from the directory
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            df = pd.read_csv(os.path.join(folder_path, file_name))
            
            if 'url' in df.columns and 'date' in df.columns:
                combined_df = pd.concat([combined_df, df[['url', 'date']]])

    combined_df.to_csv('combined_urls.csv', index=False)

def download_images(csv_path, save_folder):
    df = pd.read_csv(csv_path)
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for index, row in df.iterrows():
        image_url = row['url']

        # Check if the URL is valid
        if not isinstance(image_url, str):            
            continue

        parsed_url = urlparse(image_url)
        image_name = os.path.basename(parsed_url.path)
        save_path = os.path.join(save_folder, f"{index}_{image_name}")

        # Download and save the image
        response = requests.get(image_url)
        with open(save_path, 'wb') as file:
            file.write(response.content)


def extract_features(image_folder):

    # Load VGG16 model + higher level layers
    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False)
    
    image_features = []
    image_paths = []

    # Define bins for histogram
    bins = [8, 8, 8]

    # List image files
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

    for image_file in image_files:
        # Read the image
        image_path = os.path.join(image_folder, image_file)
        
        # For VGG16 features
        img = k_image.load_img(image_path, target_size=(224, 224))
        x = k_image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        vgg16_features = base_model.predict(x)
        
        # For color histogram
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hist = cv2.calcHist([img], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)

        # Concatenate VGG16 features and color histogram
        combined_features = np.concatenate([vgg16_features.flatten()])
        
        image_features.append(combined_features)
        image_paths.append(image_path)

    print("all features extracted");
    return np.array(image_features), image_paths



def cluster_images_by_date_and_similarity(image_features,image_paths,optimal_k):
    # Dictionary to hold clusters
    clusters = defaultdict(lambda: defaultdict(list))

    # Clustering by yearly intervals
    #df = pd.read_csv('combined_urls.csv')
    # for index, row in df.iterrows():
    #     date_obj = parse(row['date'])
    #     year = date_obj.year
    #     quarter = (date_obj.month - 1) // 3 + 1  # 1 for Q1 (Jan-Mar), 2 for Q2 (Apr-Jun), etc.
        
    #     interval_str = f"{year}"
    interval_str = "dummy"
    # KMeans clustering by visual similarity within the quarter
    kmeans = KMeans(n_clusters=optimal_k, n_init=10)  # Explicitly set n_init to suppress warnings
    labels = kmeans.fit_predict(image_features)
    for label, path in zip(labels, image_paths):
        clusters[interval_str][str(label)].append(path)  # Convert label to string for JSON serialization

    return clusters

def dimensionality_reduction(features):
    pca = PCA(n_components=10)
    pca.fit(features)
    features_reduced = pca.transform(features)
    return features_reduced

def save_clusters_to_json(clusters, json_path):
    with open(json_path, 'w') as file:
        json.dump(clusters, file)


def execute_all(folder_path):
    # Step 1
    combine_csv_files(folder_path)

    # Step 2
    download_images('combined_urls.csv', 'images_folder')

    # step 2.5
    features, image_paths = extract_features('images_folder')

    # step 3.8
    optimal_k = determine_optimal_k(features)
    print("optimal value of k is",optimal_k)

    features_reduced = dimensionality_reduction(features)

    # Step 3
    clusters = cluster_images_by_date_and_similarity(features_reduced, image_paths, optimal_k)

    # Step 4
    save_clusters_to_json(clusters, 'clusters.json')
    print("All steps executed successfully!")

    # Step 5: Start HTTP server and launch the HTML viewer
    PORT = 8000
    

    # Start the server in a separate thread
    def start_server():
        Handler = http.server.SimpleHTTPRequestHandler
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"Serving at port {PORT}")
            httpd.serve_forever()

    thread = threading.Thread(target=start_server)
    thread.start()

    # Give the server a second to initialize and then open the browser
    webbrowser.open(f'http://localhost:{PORT}/json_viewer.html', new=2)


# Execute the functions
execute_all('csv_files')
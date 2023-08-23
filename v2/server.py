import json
import threading

lock = threading.Lock()
from flask import Flask, request, send_from_directory, jsonify

app = Flask(__name__)

# Load the clusters data into memory
with open("clusters.json", "r") as file:
    clusters_data = json.load(file)

@app.route("/data/<path:image_path>", methods=["GET"])
def serve_image(image_path):
    return send_from_directory("./data", image_path)

@app.route("/clusters.json", methods=["GET"])
def serve_clusters():
    return send_from_directory(".", "clusters.json")

@app.route("/", methods=["GET"])
def serve_html():
    return send_from_directory(".", "json_viewer.html")

@app.route("/approve", methods=["POST"])
def approve_cluster():
    domain_name = request.json.get("domain_name")
    cluster_id = request.json.get("cluster_id")
    
    # Find the domain and update the status of the specified cluster
    for entry in clusters_data["data"]:
        if entry["domain_name"] == domain_name:
            if cluster_id in entry["clusters"]:
                entry["clusters"][cluster_id]["approved"] = 1
                break
    
    # Save the updated data back to the clusters.json file
    with open("clusters.json", "w") as file:
        json.dump(clusters_data, file)
    
    return jsonify({"status":"success","message": "Cluster approved successfully!"})

@app.route("/reject", methods=["POST"])
def reject_cluster():
    domain_name = request.json.get("domain_name")
    cluster_id = request.json.get("cluster_id")
    
    # Find the domain and update the status of the specified cluster
    for entry in clusters_data["data"]:
        if entry["domain_name"] == domain_name:
            if cluster_id in entry["clusters"]:
                entry["clusters"][cluster_id]["approved"] = 0
                break
    
    # Save the updated data back to the clusters.json file
    with open("clusters.json", "w") as file:
        json.dump(clusters_data, file)
    
    return jsonify({"status":"success","message": "Cluster rejected successfully!"})

# This will not actually start the server here in this environment
# You'd run `app.run()` in your local environment to start the Flask server
app.run(port=5000)

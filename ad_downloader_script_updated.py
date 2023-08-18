
import os
import json
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor

def download_file(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)

def save_ad_details(ad, ad_type, folder):
    details = {
        "ad_type": ad_type,
        "ad_id": ad["ad_id"],
        "ad_start_date": ad["ad_start_date"],
        "ad_status": ad["ad_status"],
        "ad_count": ad["ad_count"],
        "ad_text": ad["ad_text"],
        "ad_cta_text": ad["ad_cta_text"],
        "ad_url": ad["ad_video"] if ad_type == "video" else ad["ad_image-src"],
        "ad_cta_domain": ad["ad_cta_domain"]
    }
    
    # Append ad details to the JSON file
    json_file_path = os.path.join(folder, "ad_details.json")
    existing_data = []
    if os.path.exists(json_file_path):
        try:
            with open(json_file_path, 'r') as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            existing_data = []
    existing_data.append(details)
    with open(json_file_path, 'w') as f:
        json.dump(existing_data, f, indent=4)

def handle_brand_ads(ads):
    for ad in ads:
        ad_type = 'image' if pd.notna(ad["ad_image-src"]) else 'video'
        folder = os.path.join("data",ad["domain_name"], f"{ad_type}_ads")
        os.makedirs(folder, exist_ok=True)
    
        if ad_type == "image":
            image_file_path = os.path.join(folder, f"{ad['ad_id']}_image.jpg")
            download_file(ad["ad_image-src"], image_file_path)
        else:
            video_file_path = os.path.join(folder, f"{ad['ad_id']}_video.mp4")
            download_file(ad["ad_video"], video_file_path)
        
        save_ad_details(ad, ad_type, folder)
    print(f"Processed ads for brand {ads[0]['ad_brandname']}.")

def main():
    image_ads = pd.read_csv('fb_d2c_top50_domain_image_ads.csv')
    video_ads = pd.read_csv('fb_d2c_top50_domain_video_ads.csv')
    all_ads = pd.concat([image_ads, video_ads])
    
    brands_grouped = all_ads.groupby('domain_name').apply(lambda x: x.to_dict('records')).tolist()
    
    with ThreadPoolExecutor() as executor:
        list(executor.map(handle_brand_ads, brands_grouped))

if __name__ == "__main__":
    main()

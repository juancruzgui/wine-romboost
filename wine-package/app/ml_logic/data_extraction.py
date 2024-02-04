from google.cloud import storage


#https://storage.googleapis.com/the_public_bucket/wine-clustering.csv

def download_wine_file(bucket_name, file_name,destination_file_name):
    storage_client = storage.Client()

    bucket=storage_client.bucket(bucket_name)

    blob = bucket.blob(file_name)
    blob.download_to_filename(destination_file_name)

    return True

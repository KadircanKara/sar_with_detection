from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import google.auth.transport.requests
import os
import numpy as np
import time

SCOPES = ['https://www.googleapis.com/auth/drive'] # Connection to Google Drive API
SERVICE_ACCOUNT_FILE = 'google_drive_upload.json' # Service Account JSON file
PARENT_FOLDER_ID = "1S6PXtYC3g8KKmv-aHYto6IorDTSf66MP" # Code after the last / in the URL of the google drive folder

PARENT_FOLDER_ID_DICT = {
    "Project Scripts": "1S6PXtYC3g8KKmv-aHYto6IorDTSf66MP",
    "Solutions": "12N6HjToNnXq8eqV1L5Ds-Xh8fel_jzec",
    "Objectives": "1e3gZTiHkSzvU9JHKLsalZM4Mnpf0skIE",
    "Runtimes": "1HE42WHBg6W71cEwFTHZxedGvj4Wb9DQ6",
    "Objective Value Figures" : "1moEVUoZ7fONvrt9Gynujk0zbj2Icfr5b",
    "Pareto Front Figures" : "1PkO0FFLHsto0Ljbq0-hzc9WHGd7W_ovY"
}

def authenticate():
    creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return creds

def upload_file(file_path, parent_folder_id):
    file_name = file_path.split('/')[-1]
    creds = authenticate()
    # Create an authorized HTTP object with a custom timeout
    http = google.auth.transport.requests.AuthorizedSession(creds)
    http.timeout = 600  # Set timeout to 600 seconds (10 minutes)
    service = build('drive', 'v3', credentials=creds)
    file_metadata = {
        'name': file_name,
        'mimeType': 'application/octet-stream',
        'parents': [parent_folder_id]
    }
    # Search for the file by name
    response = service.files().list(q=f"name='{file_name}'", spaces='drive').execute()
    files = response.get('files', [])
    if files:
        # If the file exists, update it
        file_id = files[0]['id']
        media = MediaFileUpload(file_path,
                                mimetype='application/octet-stream')
        updated_file = service.files().update(fileId=file_id, media_body=media).execute()
        # print(f'Updated File ID: {updated_file.get("id")}')
    else:
        # If the file does not exist, upload it as a new file
        media = MediaFileUpload(file_path,
                                mimetype='application/octet-stream')
        new_file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        media = MediaFileUpload(file_path,
                            mimetype='application/octet-stream')

    # file = service.files().create(
    #     body=file_metadata, media_body=media
    #     ).execute()

# t = time.time()
# project_scripts = np.array(os.listdir())
# for filename in project_scripts:
#     upload_file(filename, PARENT_FOLDER_ID_DICT["Project Scripts"]) if filename.endswith(".py") else None
# print(time.time()-t, "seconds passed")
# upload_file("Figures/NSGA2_r_2_Mission_Time_best_values.png")
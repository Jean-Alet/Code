import os

video_path = "D:/Code/python/vb1.avi"  # Change if needed
if os.path.exists(video_path):
    print("File exists!")
else:
    print("Error: File not found.")
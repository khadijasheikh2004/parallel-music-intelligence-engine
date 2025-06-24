import joblib

model = joblib.load('music_genre_classifier.pkl')
print(model)
import os
print("CPU Cores: ", os.cpu_count())
import psutil

def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    return f"Memory Usage: {mem_info.rss / 1024 ** 2:.2f} MB"  # in MB

print(get_memory_usage())  # Print to terminal (not GUI)


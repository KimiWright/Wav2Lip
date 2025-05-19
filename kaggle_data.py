import kagglehub

# Download latest version
path = kagglehub.dataset_download("adrianlubitz/vvadlrs3")

print("Path to dataset files:", path)
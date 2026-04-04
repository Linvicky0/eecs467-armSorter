# Delete the old container if it exists (ignore errors if it doesn't)
docker rm -f armlab-container 2>/dev/null || true

# Rebuild the image (this will use your updated script)
docker build --platform linux/amd64 -t armlab-image .

# Start the fresh container
docker run -it --platform linux/amd64 --name armlab-container armlab-image
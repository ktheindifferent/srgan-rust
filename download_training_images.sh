#!/bin/bash

# Create directories for training data
echo "Creating training directories..."
mkdir -p training_data
mkdir -p validation_data

echo "Downloading high-quality images from Unsplash (free stock photos)..."

# Download nature/landscape images for training (Unsplash provides free high-quality images)
# Using direct image URLs from Unsplash's free API
training_urls=(
    "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=2000&q=80"  # Mountains
    "https://images.unsplash.com/photo-1501594907352-04cda38ebc29?w=2000&q=80"  # Golden Gate
    "https://images.unsplash.com/photo-1473448912268-2022ce9509d8?w=2000&q=80"  # Forest
    "https://images.unsplash.com/photo-1519904981063-b0cf448d479e?w=2000&q=80"  # Waterfall
    "https://images.unsplash.com/photo-1505142468610-359e7d316be0?w=2000&q=80"  # Beach
    "https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?w=2000&q=80"  # Hills
    "https://images.unsplash.com/photo-1426604966848-d7adac402bff?w=2000&q=80"  # Nature
    "https://images.unsplash.com/photo-1444927714506-8492d94b4e3d?w=2000&q=80"  # Ocean
    "https://images.unsplash.com/photo-1502082553048-f009c37129b9?w=2000&q=80"  # Tree
    "https://images.unsplash.com/photo-1532274402911-5a369e4c4bb5?w=2000&q=80"  # Landscape
)

validation_urls=(
    "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=2000&q=80"  # Portrait
    "https://images.unsplash.com/photo-1517849845537-4d257902454a?w=2000&q=80"  # Dog
    "https://images.unsplash.com/photo-1518791841217-8f162f1e1131?w=2000&q=80"  # Cat
    "https://images.unsplash.com/photo-1494253109108-2e30c049369b?w=2000&q=80"  # Colors
    "https://images.unsplash.com/photo-1558591710-4b4a1ae0f04d?w=2000&q=80"  # Abstract
)

# Download training images
echo "Downloading training images..."
for i in "${!training_urls[@]}"; do
    url="${training_urls[$i]}"
    filename="training_data/image_$(printf "%03d" $i).jpg"
    echo "  Downloading image $((i+1))/${#training_urls[@]}..."
    curl -s -L "$url" -o "$filename"
done

# Download validation images
echo "Downloading validation images..."
for i in "${!validation_urls[@]}"; do
    url="${validation_urls[$i]}"
    filename="validation_data/val_image_$(printf "%03d" $i).jpg"
    echo "  Downloading validation image $((i+1))/${#validation_urls[@]}..."
    curl -s -L "$url" -o "$filename"
done

echo ""
echo "Download complete!"
echo "Training images: $(ls -1 training_data/*.jpg 2>/dev/null | wc -l) files"
echo "Validation images: $(ls -1 validation_data/*.jpg 2>/dev/null | wc -l) files"
echo ""
echo "You can now train a model with:"
echo "  ./target/release/srgan-rust train training_data/ my_model.rsr -v validation_data/"
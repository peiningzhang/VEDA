#!/bin/bash

# Define the directory where the files will be saved
DEST_DIR="./"

# Create the directory if it doesn't exist
mkdir -p "$DEST_DIR"

# URLs of the files to download
TRAIN_URL="https://bits.csb.pitt.edu/files/geom_raw/train_data.pickle"
VAL_URL="https://bits.csb.pitt.edu/files/geom_raw/val_data.pickle"
TEST_URL="https://bits.csb.pitt.edu/files/geom_raw/test_data.pickle"

# Download the files using curl
echo "Downloading train_data.pickle..."
curl -o "$DEST_DIR/train_data.pickle" "$TRAIN_URL"

echo "Downloading val_data.pickle..."
curl -o "$DEST_DIR/val_data.pickle" "$VAL_URL"

echo "Downloading test_data.pickle..."
curl -o "$DEST_DIR/test_data.pickle" "$TEST_URL"

echo "All files have been downloaded to $DEST_DIR"

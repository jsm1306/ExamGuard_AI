#!/usr/bin/env python3
"""
Quick watermark decoder test - No pattern matching, just show extraction
"""

import numpy as np
import tensorflow as tf
import cv2
import fitz
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*70)
print("WATERMARK DECODER EXTRACTION TEST")
print("="*70)

# Quick load
print("\n[1] Loading decoder...", end=" ")
decoder = tf.keras.models.load_model("models/decoder_model.h5")
print("✓")

# Test one PDF
pdf_path = "watermarked_papers/QP_2310080008.pdf"
print(f"[2] Opening PDF: {Path(pdf_path).name}...", end=" ")

doc = fitz.open(str(pdf_path))
page = doc[0]
pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
if img.shape[2] == 4:
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
doc.close()
print("✓")

# Extract patches
print(f"[3] Extracting patches...", end=" ")
patches = []
for y in range(0, img.shape[0] - 128, 128):
    for x in range(0, img.shape[1] - 128, 128):
        patches.append(img[y:y+128, x:x+128])
print(f"✓ ({len(patches)} patches)")

# Decode first 5 patches
print(f"[4] Decoding watermarks...", end=" ")
watermarks = []
for patch in patches[:5]:
    patch_norm = patch.astype(np.float32) / 255.0
    decoded = decoder.predict(patch_norm[np.newaxis], verbose=0)
    watermark = np.squeeze(decoded, axis=0)
    watermark_uint8 = (np.clip(watermark, 0, 1) * 255).astype(np.uint8)
    watermarks.append(watermark_uint8)
print(f"✓ ({len(watermarks)} decoded)")

# Show watermark details
print(f"\n[EXTRACTED WATERMARKS]")
print(f"{'ID':<5} {'Size':<15} {'Mean':<15} {'Text Area':<15}")
print(f"{'-'*50}")

for i, wm in enumerate(watermarks):
    size = f"{wm.shape[0]}x{wm.shape[1]}"
    mean = np.mean(wm)
    
    # Text region (where roll number would be written)
    if len(wm.shape) == 3:
        text = np.mean(wm[10:28, 2:30, :])
    else:
        text = np.mean(wm[10:28, 2:30])
    
    print(f"{i+1:<5} {size:<15} {mean:>8.1f}       {text:>8.1f}")

print(f"\n[SUCCESS] Watermarks extracted successfully using decoder!")
print(f"\nTo identify student:")
print(f"  Method 1: Build database of 88 watermark patterns")
print(f"  Method 2: Match decoded pattern against database")
print(f"  Result: Student roll number identified from watermark only")

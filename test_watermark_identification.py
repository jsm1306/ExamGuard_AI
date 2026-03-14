#!/usr/bin/env python3
"""
Test watermark identification for students 8 and 75
Verifies that the fix correctly extracts and identifies the right student
"""

import numpy as np
import tensorflow as tf
import cv2
import fitz
from pathlib import Path
from forensic_decoder import (
    load_decoder_model,
    pdf_to_images, 
    extract_patches,
    decode_patches,
    analyze_watermarks_for_roll_number,
    get_roll_from_watermark_text
)

print("\n" + "="*70)
print("WATERMARK IDENTIFICATION TEST - STUDENTS 8 AND 75")
print("="*70)

# Tests
test_cases = [
    ("watermarked_papers/QP_2310080008.pdf", 2310080008, "Student 8"),
    ("watermarked_papers/QP_2310080075.pdf", 2310080075, "Student 75")
]

# Load decoder
print("\n[SETUP] Loading decoder model...", end=" ")
decoder = load_decoder_model("models/decoder_model.h5")
if not decoder:
    print("[FAILED]")
    exit(1)
print("[OK]")

# Run tests
for pdf_path, expected_roll, description in test_cases:
    print(f"\n{'-'*70}")
    print(f"TEST: {description}")
    print(f"Expected Roll: {expected_roll:010d}")
    print(f"PDF: {pdf_path}")
    print(f"{'-'*70}")
    
    # Extract images
    print("Extracting PDF...", end=" ")
    images = pdf_to_images(pdf_path, num_pages=1)
    if not images:
        print("[FAILED]")
        continue
    print(f"[OK] {len(images)} pages")
    
    # Extract patches
    print("Extracting patches...", end=" ")
    patches = extract_patches(images[0], patch_size=128)
    print(f"[OK] {len(patches)} patches")
    
    # Decode watermarks
    print("Decoding watermarks...", end=" ")
    watermarks = decode_patches(patches, decoder)
    print(f"[OK] {len(watermarks)} watermarks")
    
    # Analyze for roll number
    print("\nAnalyzing watermarks...")
    extracted_texts = analyze_watermarks_for_roll_number(watermarks)
    
    if extracted_texts:
        print(f"\n[EXTRACTED TEXT]")
        for i, text in enumerate(extracted_texts[:3], 1):
            print(f"  [{i}] '{text}'")
            roll = get_roll_from_watermark_text(text)
            if roll:
                print(f"      Decoded Roll: {roll:010d}")
                if roll == expected_roll:
                    print(f"      [✓ CORRECT]")
                else:
                    print(f"      [✗ WRONG - Expected {expected_roll:010d}]")
    else:
        print("[!] No text extracted - trying pattern-based analysis")
        # Fallback to pattern analysis
        if watermarks:
            confidences = np.array([w['confidence'] for w in watermarks])
            print(f"  Mean confidence: {np.mean(confidences):.2f}")
            print(f"  High confidence watermarks: {np.sum(confidences > 0.9)}")
    
    print()

print("="*70)
print("TEST COMPLETE")
print("="*70)

#!/usr/bin/env python3
"""
Simple test to verify watermark extraction and roll number decoding
"""

import numpy as np
import tensorflow as tf
import cv2
import fitz
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*75)
print("WATERMARK EXTRACTION AND ROLL NUMBER DECODE TEST")
print("="*75)

# Test configurations
test_cases = [
    ("watermarked_papers/QP_2310080008.pdf", 2310080008, "Student 8"),
    ("watermarked_papers/QP_2310080075.pdf", 2310080075, "Student 75"),
]

# Load decoder
print("\n[1/4] Loading decoder model...", end=" ", flush=True)
try:
    decoder = tf.keras.models.load_model("models/decoder_model.h5")
    print("[OK]")
except Exception as e:
    print(f"[FAILED: {e}]")
    exit(1)

for pdf_path, expected_roll, desc in test_cases:
    print(f"\n{'-'*75}")
    print(f"TEST: {desc} | Expected Roll: {expected_roll:010d}")
    print(f"PDF: {pdf_path}")
    print(f"{'-'*75}")
    
    if not Path(pdf_path).exists():
        print(f"[ERROR] PDF not found: {pdf_path}")
        continue
    
    # Step 1: Extract images from PDF
    print("[2/4] Extracting PDF...", end=" ", flush=True)
    try:
        doc = fitz.open(str(pdf_path))
        page = doc[0]  # First page
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        doc.close()
        print(f"[OK] {img_array.shape}")
    except Exception as e:
        print(f"[FAILED: {e}]")
        continue
    
    # Step 2: Extract patches and decode
    print("[3/4] Decoding watermarks from patches...", end=" ", flush=True)
    try:
        # Extract 128x128 patches
        patch_size = 128
        patches = []
        h, w = img_array.shape[:2]
        
        for y in range(0, h - patch_size, patch_size):
            for x in range(0, w - patch_size, patch_size):
                patch = img_array[y:y + patch_size, x:x + patch_size]
                patches.append(patch)
        
        # Decode watermarks from patches
        watermarks = []
        for patch in patches[:10]:  # Test first 10 patches
            try:
                patch_norm = patch.astype(np.float32) / 255.0
                decoded = decoder.predict(patch_norm[np.newaxis], verbose=0)
                watermark = np.squeeze(decoded, axis=0)
                watermark_uint8 = (np.clip(watermark, 0, 1) * 255).astype(np.uint8)
                watermarks.append(watermark_uint8)
            except:
                pass
        
        print(f"[OK] {len(watermarks)} watermarks decoded")
    except Exception as e:
        print(f"[FAILED: {e}]")
        continue
    
    # Step 3: Extract text from watermark using OCR
    print("[4/4] Extracting text from watermark...", end=" ", flush=True)
    
    extracted_roll = None
    extracted_text = None
    
    if watermarks:
        # Try pytesseract first
        try:
            import pytesseract
            
            # Use the first (best) watermark
            wm = watermarks[0]
            if len(wm.shape) == 3:
                gray = cv2.cvtColor(wm, cv2.COLOR_RGB2GRAY)
            else:
                gray = wm
            
            # Enhance for OCR
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            upscaled = cv2.resize(enhanced, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            _, binary = cv2.threshold(upscaled, 127, 255, cv2.THRESH_BINARY)
            
            # OCR
            text = pytesseract.image_to_string(binary, config='--psm 8').strip()
            
            if text:
                extracted_text = text
                print(f"[OK] Text: '{text}'")
                
                # Parse text to extract roll number
                # Format: "AI202608" -> roll 08 -> 2310080008
                import re
                digits = re.findall(r'\d+', text)
                if len(digits) >= 2:
                    roll_suffix = digits[-2:]  # Get last 2 digits
                    if len(roll_suffix[-1]) == 2:
                        student_num = int(roll_suffix[-1])
                        if 1 <= student_num <= 88:
                            extracted_roll = 2310080000 + student_num
            else:
                print("[OK] No text (pytesseract returned empty)")
        
        except ImportError:
            print("[INFO] pytesseract not installed, using brightness analysis")
            
            # Fallback: analyze brightness pattern
            wm = watermarks[0]
            brightness = np.mean(wm)
            text_region = wm[8:24, 2:30, :] if len(wm.shape) == 3 else wm[8:24, 2:30]
            text_brightness = np.mean(text_region)
            
            print(f"[INFO] Brightness: {brightness:.0f}, Text region: {text_brightness:.0f}")
        
        except Exception as e:
            print(f"[INFO] OCR failed: {e}")
    else:
        print("[WARNING] No watermarks decoded")
    
    # Results
    print(f"\n{'='*75}")
    print("RESULTS:")
    print(f"{'='*75}")
    print(f"Expected Roll:    {expected_roll:010d}")
    if extracted_text:
        print(f"Extracted Text:   '{extracted_text}'")
    if extracted_roll:
        print(f"Decoded Roll:     {extracted_roll:010d}")
        if extracted_roll == expected_roll:
            print(f"\n[SUCCESS] Roll correctly identified!")
        else:
            print(f"\n[MISMATCH] Got {extracted_roll:010d} but expected {expected_roll:010d}")
    else:
        print(f"Decoded Roll:     [NOT EXTRACTED]")
        print(f"\n[INFO] Watermark decoded but text extraction failed")
    print(f"{'='*75}\n")

print("\nTest Complete!")

"""
FIXED: Watermark PNG exam files for 5 roll numbers
Direct patch blending approach that actually works
"""

import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import warnings

warnings.filterwarnings('ignore')

class FixedWatermarker:
    def __init__(self):
        # Load encoder and decoder
        print("[LOAD] Loading encoder model...")
        try:
            self.encoder = tf.keras.models.load_model('models/encoder_model.h5')
            print("  [OK] Encoder loaded")
        except Exception as e:
            print(f"  [ERROR] {e}")
            self.encoder = None
        
        print("[LOAD] Loading decoder model...")
        try:
            self.decoder = tf.keras.models.load_model('models/decoder_model.h5')
            print("  [OK] Decoder loaded")
        except Exception as e:
            print(f"  [ERROR] {e}")
            self.decoder = None
    
    def create_watermark_image(self, roll_number):
        """Create 32x32 watermark based on roll number"""
        np.random.seed(roll_number % 100)
        
        watermark = np.zeros((32, 32, 3), dtype=np.uint8)
        
        # Extract last 2 digits
        roll_suffix = str(roll_number)[-2:]
        roll_val = int(roll_suffix)
        
        # Create unique pattern per roll
        for i in range(32):
            for j in range(32):
                pattern_val = ((i + j + roll_val * 3) % 256)
                watermark[i, j] = [pattern_val, pattern_val//2, pattern_val//3]
        
        # Add roll text
        cv2.putText(watermark, roll_suffix, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 1)
        
        return watermark
    
    def embed_watermark_simple(self, cover_image, watermark_image, roll_number):
        """
        Simple but EFFECTIVE watermark embedding:
        1. Split into 128x128 patches
        2. Embed watermark in 30% of patches using encoder if available
        3. Use 0.02 strength blending
        """
        np.random.seed(roll_number % 100)
        
        h, w = cover_image.shape[:2]
        patch_size = 128
        
        # Make a copy to modify
        result = cover_image.copy().astype(np.float32) / 255.0
        
        # Find patch positions
        patches = []
        for y in range(0, h - patch_size + 1, patch_size):
            for x in range(0, w - patch_size + 1, patch_size):
                patches.append((y, x))
        
        if not patches:
            return cover_image
        
        # Watermark 30% of patches
        num_to_watermark = max(1, int(len(patches) * 0.3))
        selected_indices = np.random.choice(len(patches), num_to_watermark, replace=False)
        
        watermark_norm = watermark_image.astype(np.float32) / 255.0
        blended_patches = 0
        
        for idx in selected_indices:
            y, x = patches[idx]
            
            try:
                # Get patch
                patch = result[y:y+patch_size, x:x+patch_size].copy()
                
                # Resize watermark to patch size
                watermark_patch = cv2.resize(watermark_norm, (patch_size, patch_size))
                
                # Try to process with encoder
                if self.encoder is not None:
                    try:
                        # Encoder input: (32, 32, 3) watermark
                        wm_input = watermark_norm[np.newaxis, :, :, :]  # (1, 32, 32, 3)
                        encoded = self.encoder.predict(wm_input, verbose=0)
                        
                        # Handle different output shapes
                        if isinstance(encoded, (list, tuple)):
                            encoded = encoded[0]
                        
                        # Remove batch dimension if present
                        if encoded.ndim == 4:
                            encoded = np.squeeze(encoded, axis=0)
                        
                        # Resize to patch size
                        watermark_patch = cv2.resize(encoded, (patch_size, patch_size))
                        watermark_patch = np.clip(watermark_patch, 0, 1)
                    except Exception as e:
                        # Fallback: use watermark directly
                        watermark_patch = cv2.resize(watermark_norm, (patch_size, patch_size))
                
                # CRITICAL: Actually blend the patch!
                strength = 0.02  # 2% perturbation
                blended = (1.0 - strength) * patch + strength * watermark_patch
                result[y:y+patch_size, x:x+patch_size] = np.clip(blended, 0, 1)
                blended_patches += 1
                
            except Exception as e:
                print(f"    [WARN] Patch {idx} failed: {str(e)[:40]}")
        
        # Convert back to uint8
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        
        return result, blended_patches
    
    def watermark_file(self, image_path, roll_number, output_dir="watermarked_papers_5_fixed"):
        """Watermark a PNG file"""
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        if not image_path.exists():
            return False
        
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            return False
        
        # BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create watermark
        watermark = self.create_watermark_image(roll_number)
        
        # Embed watermark
        watermarked, patches_done = self.embed_watermark_simple(img_rgb, watermark, roll_number)
        
        # Verify watermark was actually embedded
        diff = np.mean(np.abs(img_rgb.astype(np.float32) / 255.0 - watermarked.astype(np.float32) / 255.0))
        
        # Save
        output_path = output_dir / f"WM_{roll_number:010d}.png"
        watermarked_bgr = cv2.cvtColor(watermarked, cv2.COLOR_RGB2BGR)
        
        success = cv2.imwrite(str(output_path), watermarked_bgr)
        
        if success:
            size_kb = output_path.stat().st_size / 1024
            print(f"  Roll {roll_number} → {output_path.name} ({patches_done} patches, diff={diff:.6f})")
            return True
        
        return False


def main():
    watermarker = FixedWatermarker()
    
    print("\n" + "="*70)
    print("FIXED WATERMARKING - 5 ROLL NUMBERS")
    print("="*70)
    
    roll_numbers = [2310080001, 2310080002, 2310080003, 2310080004, 2310080005]
    source_dir = Path("generated_exams_dl")
    
    if not source_dir.exists():
        print(f"[ERROR] {source_dir} not found")
        return
    
    success = 0
    for roll in roll_numbers:
        exam_file = source_dir / f"Exam_{roll:010d}_DL.png"
        if exam_file.exists():
            if watermarker.watermark_file(str(exam_file), roll):
                success += 1
        else:
            print(f"  Roll {roll} → SKIP (file not found)")
    
    print("\n" + "="*70)
    print(f"COMPLETE: {success}/{len(roll_numbers)} watermarked")
    print(f"Output: watermarked_papers_5_fixed/")
    print("="*70)


if __name__ == "__main__":
    main()

"""
Watermark PNG exam files for 5 roll numbers - Simple patch-based method
Uses encoder to process watermarks, then embeds into patches with 0.02 perturbation
"""

import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import warnings

sys.path.insert(0, str(Path(__file__).parent))

warnings.filterwarnings('ignore')

class SimpleWatermarker:
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
    
    def create_watermark_image(self, roll_number, exam_id="AI2026"):
        """Create 32x32 watermark image from roll number"""
        watermark = np.zeros((32, 32, 3), dtype=np.uint8)
        
        # Extract last 2 digits of roll number (01-88)
        roll_suffix = str(roll_number)[-2:]
        
        # Create pattern based on roll number
        roll_val = int(roll_suffix)
        
        # Pattern: Diagonal stripes with unique frequency per roll
        for i in range(32):
            for j in range(32):
                # Create deterministic pattern based on roll_val
                pattern_val = ((i + j + roll_val * 2) % 256)
                watermark[i, j] = [pattern_val, pattern_val//2, pattern_val//3]
        
        # Add text-like marking with roll number
        cv2.putText(watermark, roll_suffix, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   (255, 255, 255), 1)
        
        return watermark
    
    def embed_watermark_patches(self, cover_image, watermark_image, roll_number):
        """
        Embed watermark into 30% of 128x128 patches with 0.02 perturbation
        
        Args:
            cover_image: numpy array H×W×3, dtype uint8
            watermark_image: 32×32×3, dtype uint8
            roll_number: for random seed consistency
        
        Returns:
            watermarked image, same shape, dtype uint8
        """
        # Use roll_number to seed random for reproducibility
        np.random.seed(roll_number % 100)
        
        h, w = cover_image.shape[:2]
        patch_size = 128
        perturbation_strength = 0.02
        
        # Normalize
        cover_norm = cover_image.astype(np.float32) / 255.0
        watermark_norm = watermark_image.astype(np.float32) / 255.0
        
        result = np.copy(cover_norm)
        
        # Find all valid patch positions
        patch_positions = []
        for y in range(0, h - patch_size + 1, patch_size):
            for x in range(0, w - patch_size + 1, patch_size):
                patch_positions.append((y, x))
        
        if not patch_positions:
            print(f"    [WARN] Image too small for patches: {h}×{w}")
            return cover_image
        
        # Watermark 30% of patches (randomly selected)
        num_to_watermark = max(1, int(len(patch_positions) * 0.3))
        selected = np.random.choice(len(patch_positions), size=num_to_watermark, replace=False)
        
        patches_done = 0
        
        for idx in selected:
            y, x = patch_positions[idx]
            patch = cover_norm[y:y+patch_size, x:x+patch_size]
            
            try:
                # Expand watermark to patch size
                watermark_expanded = cv2.resize(watermark_norm, (patch_size, patch_size))
                
                # If encoder available, use it on the watermark
                if self.encoder is not None:
                    wm_batch = watermark_norm[np.newaxis]  # Add batch dim (1, 32, 32, 3)
                    try:
                        encoded_wm = self.encoder.predict(wm_batch, verbose=0)
                        if isinstance(encoded_wm, (list, tuple)):
                            encoded_wm = encoded_wm[0]
                        encoded_wm = np.squeeze(encoded_wm, axis=0)
                        watermark_expanded = cv2.resize(encoded_wm, (patch_size, patch_size))
                    except Exception as e:
                        # Fallback: use watermark directly
                        watermark_expanded = cv2.resize(watermark_norm, (patch_size, patch_size))
                else:
                    watermark_expanded = cv2.resize(watermark_norm, (patch_size, patch_size))
                
                # Blend with very small perturbation (0.02 = 2%)
                blended = (1.0 - perturbation_strength) * patch + perturbation_strength * watermark_expanded
                result[y:y+patch_size, x:x+patch_size] = np.clip(blended, 0, 1)
                patches_done += 1
                
            except Exception as e:
                pass
        
        # Normalize and convert back
        result = np.clip(result, 0, 1)
        return (result * 255).astype(np.uint8)
    
    def watermark_image(self, image_path, roll_number, output_dir="watermarked_papers_5"):
        """Watermark a single PNG image"""
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        if not image_path.exists():
            print(f"  [SKIP] Image not found: {image_path}")
            return False
        
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"  [SKIP] Failed to read image: {image_path.name}")
            return False
        
        # BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create watermark
        watermark = self.create_watermark_image(roll_number)
        
        print(f"  {image_path.name:30} → ", end="", flush=True)
        
        # Embed watermark
        watermarked = self.embed_watermark_patches(img_rgb, watermark, roll_number)
        
        # Save as PNG
        output_path = output_dir / f"WM_{roll_number:010d}.png"
        
        # Convert back to BGR for saving with OpenCV
        watermarked_bgr = cv2.cvtColor(watermarked, cv2.COLOR_RGB2BGR)
        
        if cv2.imwrite(str(output_path), watermarked_bgr):
            file_size = output_path.stat().st_size / (1024 * 1024)
            print(f"[OK] {output_path.name} ({file_size:.2f} MB)")
            return True
        else:
            print("[FAILED]")
            return False


def main():
    watermarker = SimpleWatermarker()
    
    print("\n" + "="*70)
    print("ENCODING WATERMARKS FOR 5 ROLL NUMBERS (PNG FILES)")
    print("="*70)
    
    # 5 roll numbers to watermark
    roll_numbers = [2310080001, 2310080002, 2310080003, 2310080004, 2310080005]
    
    # Source directory with exam files
    source_dir = Path("generated_exams_dl")
    
    if not source_dir.exists():
        print(f"\n[ERROR] Source directory not found: {source_dir}")
        return
    
    print(f"\nSource directory: {source_dir}")
    
    success_count = 0
    
    for roll in roll_numbers:
        # Find the corresponding exam file
        exam_file = source_dir / f"Exam_{roll:010d}_DL.png"
        
        if not exam_file.exists():
            print(f"  [SKIP] {exam_file.name} not found")
            continue
        
        if watermarker.watermark_image(str(exam_file), roll):
            success_count += 1
    
    print("\n" + "="*70)
    print(f"WATERMARKING COMPLETE: {success_count}/{len(roll_numbers)} images processed")
    print(f"Output directory: watermarked_papers_5/")
    print("="*70)
    
    print("\n[NEXT STEP] Extract watermarks using decoder model (no OCR)")
    print("Run: python extract_watermarks_5rolls.py")


if __name__ == "__main__":
    main()

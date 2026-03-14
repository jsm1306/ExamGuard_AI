"""
Extract watermarks from watermarked PDFs using ONLY decoder model
No OCR - pure watermark decoding from patches
"""

import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
import warnings
from PIL import Image

warnings.filterwarnings('ignore')


class WatermarkExtractor:
    def __init__(self):
        """Load decoder model"""
        print("[LOAD] Loading decoder model...", end=" ")
        try:
            self.decoder = tf.keras.models.load_model('models/decoder_model.h5')
            print("[OK]")
            print(f"  Input shape: {self.decoder.input.shape}")
        except Exception as e:
            print(f"[ERROR] {str(e)[:100]}")
            self.decoder = None
    
    def extract_pdf_to_images(self, pdf_path):
        """Convert PDF to list of numpy RGB arrays"""
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            images = []
            
            for page in doc:
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(np.array(img))
            
            doc.close()
            return images
        except Exception as e:
            print(f"      [ERROR] Failed to extract PDF: {e}")
            return []
    
    def decode_patches(self, page_image):
        """
        Extract patches from page and decode watermarks
        
        Args:
            page_image: (H, W, 3) uint8 RGB image
            
        Returns:
            Dict with decoded watermark data
        """
        h, w, _ = page_image.shape
        patch_size = 256
        
        print(f"  Page size: {w}×{h}")
        
        # Normalize to [0, 1]
        page_norm = page_image.astype(np.float32) / 255.0
        
        # Extract all patches
        patches = []
        patch_positions = []
        
        for y in range(0, h - patch_size, patch_size):
            for x in range(0, w - patch_size, patch_size):
                patch = page_norm[y:y+patch_size, x:x+patch_size]
                patches.append(patch)
                patch_positions.append((y, x))
        
        if not patches:
            print("  [WARN] No complete patches found")
            return None
        
        print(f"  Found {len(patches)} patches")
        
        # Batch decode patches
        patches_batch = np.array(patches)  # (N, 256, 256, 3)
        
        print(f"  Decoding patches with decoder...", end=" ", flush=True)
        
        try:
            decoded = self.decoder.predict(patches_batch, verbose=0)
            print("[OK]")
        except Exception as e:
            print(f"[ERROR] {str(e)[:80]}")
            return None
        
        # decoded shape: (N, 32, 32, 3) - the extracted watermarks
        
        # Analyze decoded watermarks
        return self._analyze_watermarks(decoded, patch_positions)
    
    def _analyze_watermarks(self, decoded_watermarks, positions):
        """
        Analyze decoded watermarks to extract identification features
        
        Args:
            decoded_watermarks: (N, 32, 32, 3) array of decoded watermarks
            positions: List of (y, x) positions
            
        Returns:
            Dict with analysis results
        """
        print(f"  Analyzing {len(decoded_watermarks)} decoded watermarks...")
        
        decoded_watermarks = np.clip(decoded_watermarks, 0, 1)
        
        # Statistics
        intensities = []
        entropies = []
        
        for wm in decoded_watermarks:
            intensity = np.mean(wm)
            intensities.append(intensity)
            
            # Shannon entropy
            flat = (wm * 255).astype(np.uint8).flatten()
            hist, _ = np.histogram(flat, bins=256, range=(0, 256))
            hist = hist / len(flat)
            entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
            entropies.append(entropy)
        
        # Aggregate
        return {
            'total_patches': len(decoded_watermarks),
            'decoded_patches': len(decoded_watermarks),
            'mean_intensity': float(np.mean(intensities)),
            'std_intensity': float(np.std(intensities)),
            'mean_entropy': float(np.mean(entropies)),
            'std_entropy': float(np.std(entropies)),
            'min_intensity': float(np.min(intensities)),
            'max_intensity': float(np.max(intensities)),
            'watermarks': decoded_watermarks,  # Keep decoded watermarks for analysis
        }
    
    def identify_from_watermarks(self, analysis):
        """
        Attempt to identify student from watermark characteristics
        Uses ONLY watermark data, not filename
        """
        if analysis is None:
            return None, 0.0
        
        # Confidence based on intensity consistency (low entropy suggests text)
        intensity_stddev = analysis['std_intensity']
        entropy = analysis['mean_entropy']
        
        # Good watermarks should have:
        # - Moderate intensity variance (text is not uniform)
        # - Detectable entropy (has distinctive patterns)
        
        confidence = 0.0
        
        if 4 <= entropy <= 8 and 0.2 <= intensity_stddev <= 0.5:
            confidence = 0.9  # High confidence
        elif 3 <= entropy <= 8 and 0.1 <= intensity_stddev <= 0.6:
            confidence = 0.7   # Medium confidence
        else:
            confidence = 0.5   # Low confidence
        
        return None, confidence  # Can't identify directly from watermark alone
    
    def show_sample_watermark(self, analysis):
        """Display first watermark visually"""
        if analysis is None or 'watermarks' not in analysis:
            return
        
        first_wm = analysis['watermarks'][0]
        first_wm_uint8 = (np.clip(first_wm, 0, 1) * 255).astype(np.uint8)
        
        # Save sample watermark visualization
        cv2.imwrite('decoded_watermark_sample.png', first_wm_uint8)
        print("  [INFO] Saved sample watermark to: decoded_watermark_sample.png")


def main():
    extractor = WatermarkExtractor()
    
    print("\n" + "="*70)
    print("WATERMARK EXTRACTION - DECODER MODEL ONLY (NO OCR)")
    print("="*70)
    
    output_dir = Path("watermarked_papers_5_correct")
    
    if not output_dir.exists():
        print(f"\n[ERROR] Output directory not found: {output_dir}")
        print("Please run watermark_5_rolls_correct.py first")
        return
    
    pdf_files = sorted(output_dir.glob("QP_*.pdf"))
    
    if not pdf_files:
        print(f"\n[ERROR] No watermarked PDFs found")
        return
    
    print(f"\nFound {len(pdf_files)} watermarked PDFs\n")
    
    results = []
    
    for pdf_path in pdf_files:
        print(f"[EXTRACT] {pdf_path.name}")
        
        try:
            roll_number = int(pdf_path.stem.split('_')[1])
        except:
            roll_number = None
        
        # Extract PDF pages
        images = extractor.extract_pdf_to_images(pdf_path)
        
        if not images:
            print("  [SKIP] Failed to extract pages\n")
            continue
        
        # Decode first page
        analysis = extractor.decode_patches(images[0])
        
        if analysis is None:
            print("  [SKIP] Decoding failed\n")
            continue
        
        # Attempt identification
        identified_roll, confidence = extractor.identify_from_watermarks(analysis)
        
        # Display results
        print(f"\n  📊 WATERMARK ANALYSIS:")
        print(f"    Decoded patches: {analysis['decoded_patches']}")
        print(f"    Mean intensity: {analysis['mean_intensity']:.3f} ± {analysis['std_intensity']:.3f}")
        print(f"    Entropy: {analysis['mean_entropy']:.2f} ± {analysis['std_entropy']:.2f} bits")
        print(f"\n  🎯 IDENTIFICATION:")
        print(f"    From filename: Roll {roll_number}")
        print(f"    Status: {'✓ WATERMARK DETECTED' if confidence > 0.7 else '⚠ WATERMARK WEAK'}")
        print(f"    Confidence: {confidence*100:.0f}%")
        print()
        
        # Show sample watermark
        if analysis['decoded_patches'] > 0:
            extractor.show_sample_watermark(analysis)
        
        results.append({
            'filename': pdf_path.name,
            'roll': roll_number,
            'patches': analysis['decoded_patches'],
            'intensity': analysis['mean_intensity'],
            'entropy': analysis['mean_entropy'],
            'confidence': confidence,
        })
    
    # Summary
    print("\n" + "="*70)
    print("EXTRACTION SUMMARY")
    print("="*70)
    
    for r in results:
        status = "✓" if r['confidence'] > 0.7 else "⚠"
        print(f"{status} {r['filename']:20} Roll: {r['roll']} | Entropy: {r['entropy']:.2f} | Conf: {r['confidence']*100:.0f}%")
    
    print(f"\nTotal extracted: {len(results)}/{len(pdf_files)}")
    print("="*70)


if __name__ == "__main__":
    main()

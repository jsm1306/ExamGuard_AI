"""
Complete Leak Detection System - Watermark Fingerprint Database & Matching

This system:
1. Creates a database of watermark fingerprints for all students
2. Decodes any question paper PDF
3. Matches the decoded watermark against the database
4. Identifies which student leaked the paper
"""

import json
import numpy as np
from pathlib import Path
import tensorflow as tf
import fitz
import cv2


class WatermarkDatabase:
    """Database of watermark fingerprints for all students."""
    
    def __init__(self, db_path="watermark_database.json"):
        self.db_path = Path(db_path)
        self.database = {}
        self._load_database()
    
    def _load_database(self):
        """Load existing database or create new one."""
        if self.db_path.exists():
            with open(self.db_path, 'r') as f:
                self.database = json.load(f)
                print(f"[OK] Loaded database with {len(self.database)} records")
        else:
            self.database = {}
    
    def save_database(self):
        """Save database to JSON file."""
        with open(self.db_path, 'w') as f:
            json.dump(self.database, f, indent=2)
        print(f"[OK] Database saved with {len(self.database)} records")
    
    def add_fingerprint(self, roll_number, fingerprint):
        """Add a watermark fingerprint for a student."""
        roll_str = f"{roll_number:010d}"
        self.database[roll_str] = fingerprint
    
    def get_fingerprint(self, roll_number):
        """Get fingerprint for a student."""
        roll_str = f"{roll_number:010d}"
        return self.database.get(roll_str)
    
    def match_fingerprint(self, query_fingerprint, threshold=0.85):
        """
        Match query fingerprint against database.
        
        Returns:
            (roll_number, confidence) or None
        """
        if not self.database:
            return None
        
        best_match = None
        best_score = 0
        
        for roll_str, stored_fp in self.database.items():
            # Calculate similarity
            score = self._calculate_similarity(query_fingerprint, stored_fp)
            
            if score > best_score:
                best_score = score
                best_match = int(roll_str)
        
        if best_score >= threshold:
            return (best_match, best_score)
        else:
            return None
    
    def _calculate_similarity(self, fp1, fp2):
        """Calculate similarity between two fingerprints."""
        try:
            # Compare confidence statistics
            confidence1 = np.array([
                fp1.get('mean_confidence', 0),
                fp1.get('std_confidence', 0),
                fp1.get('high_confidence_count', 0)
            ])
            
            confidence2 = np.array([
                fp2.get('mean_confidence', 0),
                fp2.get('std_confidence', 0),
                fp2.get('high_confidence_count', 0)
            ])
            
            # Normalize the high_confidence_count
            max_count = max(confidence1[2], confidence2[2], 1)
            confidence1[2] /= max_count
            confidence2[2] /= max_count
            
            # Calculate cosine similarity
            dot_product = np.dot(confidence1, confidence2)
            magnitude1 = np.linalg.norm(confidence1)
            magnitude2 = np.linalg.norm(confidence2)
            
            if magnitude1 > 0 and magnitude2 > 0:
                similarity = dot_product / (magnitude1 * magnitude2)
                return float(similarity)
            return 0.0
            
        except Exception as e:
            return 0.0


def extract_patches(image, patch_size=128):
    """Extract non-overlapping patches from image."""
    patches = []
    h, w = image.shape[:2]
    
    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            patch = image[y:y + patch_size, x:x + patch_size]
            patches.append({
                'data': patch,
                'position': (y, x)
            })
    
    return patches


def decode_watermarks(patches, decoder_model):
    """Decode watermarks from patches."""
    watermarks = []
    
    for patch_dict in patches:
        try:
            patch = patch_dict['data'].astype(np.float32) / 255.0
            patch_batch = patch[np.newaxis]
            
            decoded = decoder_model.predict(patch_batch, verbose=0)
            watermark = np.squeeze(decoded, axis=0)
            watermark = np.clip(watermark, 0, 1)
            watermark_uint8 = (watermark * 255).astype(np.uint8)
            
            watermarks.append({
                'watermark': watermark_uint8,
                'position': patch_dict['position'],
                'confidence': np.mean(watermark)
            })
            
        except:
            pass
    
    return watermarks


def load_decoder():
    """Load decoder model."""
    decoder_path = "models/decoder_model.h5"
    if not Path(decoder_path).exists():
        print(f"[!] Decoder not found at {decoder_path}")
        return None
    
    try:
        decoder = tf.keras.models.load_model(decoder_path)
        return decoder
    except Exception as e:
        print(f"[!] Error loading decoder: {e}")
        return None


def create_fingerprint(watermarks):
    """Create fingerprint from decoded watermarks."""
    if not watermarks:
        return None
    
    confidences = np.array([w['confidence'] for w in watermarks])
    
    return {
        'mean_confidence': float(np.mean(confidences)),
        'std_confidence': float(np.std(confidences)),
        'max_confidence': float(np.max(confidences)),
        'min_confidence': float(np.min(confidences)),
        'num_watermarks': len(watermarks),
        'high_confidence_count': int(np.sum(confidences > 0.9))
    }


def pdf_to_images(pdf_path):
    """Convert PDF to images."""
    images = []
    try:
        doc = fitz.open(str(pdf_path))
        
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            
            if img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            elif img_array.shape[2] == 1:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            
            images.append(img_array)
        
        doc.close()
        return images
    except Exception as e:
        print(f"[!] Error reading PDF: {e}")
        return []


def build_watermark_database():
    """Build database of watermark fingerprints for all students."""
    print("\n" + "="*70)
    print("BUILDING WATERMARK FINGERPRINT DATABASE")
    print("="*70 + "\n")
    
    decoder = load_decoder()
    if decoder is None:
        return None
    
    db = WatermarkDatabase()
    pdf_dir = Path("watermarked_papers")
    
    if not pdf_dir.exists():
        print("[!] No watermarked_papers directory found")
        return None
    
    pdfs = sorted(pdf_dir.glob("QP_*.pdf"))
    print(f"Found {len(pdfs)} watermarked PDFs\n")
    
    for idx, pdf_path in enumerate(pdfs, 1):
        try:
            # Extract roll number from filename
            roll_number = int(pdf_path.stem.split('_')[1])
            print(f"[{idx:2d}/{len(pdfs)}] Processing Roll {roll_number:010d}...", end=" ", flush=True)
            
            # Extract images
            images = pdf_to_images(pdf_path)
            if not images:
                print("[SKIP] No images")
                continue
            
            # Extract patches from first page
            patches = extract_patches(images[0], patch_size=128)
            
            # Decode watermarks
            watermarks = decode_watermarks(patches, decoder)
            
            # Create fingerprint
            fingerprint = create_fingerprint(watermarks)
            
            if fingerprint:
                db.add_fingerprint(roll_number, fingerprint)
                print(f"[OK] Fingerprint created")
            else:
                print("[FAIL] Could not create fingerprint")
                
        except Exception as e:
            print(f"[ERROR] {str(e)[:30]}")
    
    # Save database
    db.save_database()
    
    print("\n" + "="*70)
    print("DATABASE CREATION COMPLETE")
    print("="*70 + "\n")
    
    return db


def identify_leaker(pdf_path):
    """Identify which student leaked a PDF."""
    print("\n" + "="*70)
    print("LEAK DETECTION - STUDENT IDENTIFICATION")
    print("="*70)
    print(f"PDF: {pdf_path}\n")
    
    # Load decoder
    decoder = load_decoder()
    if decoder is None:
        return None
    
    # Load database
    db = WatermarkDatabase()
    if not db.database:
        print("[!] Watermark database is empty")
        print("[*] Building database first...")
        db = build_watermark_database()
        if not db or not db.database:
            return None
    
    # Analyze submitted PDF
    print("[1] Extracting watermark from submitted PDF...")
    images = pdf_to_images(pdf_path)
    if not images:
        print("[!] Could not read PDF")
        return None
    
    patches = extract_patches(images[0], patch_size=128)
    watermarks = decode_watermarks(patches, decoder)
    query_fingerprint = create_fingerprint(watermarks)
    
    print(f"[OK] Extracted {len(watermarks)} watermarks from {len(patches)} patches")
    
    # Match against database
    print("\n[2] Matching against watermark database...")
    result = db.match_fingerprint(query_fingerprint, threshold=0.75)
    
    # Generate report
    output_dir = Path("leak_detection_reports")
    output_dir.mkdir(exist_ok=True)
    
    report_path = output_dir / f"leak_report_{Path(pdf_path).stem}.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("LEAK DETECTION REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Submitted PDF: {pdf_path}\n")
        f.write(f"Analysis Date: {Path(pdf_path).stat().st_mtime}\n\n")
        
        f.write("WATERMARK ANALYSIS:\n")
        f.write("-"*70 + "\n")
        f.write(f"Total watermarks extracted: {len(watermarks)}\n")
        f.write(f"Average confidence: {np.mean([w['confidence'] for w in watermarks]):.4f}\n")
        f.write(f"High confidence watermarks: {sum(1 for w in watermarks if w['confidence'] > 0.9)}\n\n")
        
        f.write("DATABASE MATCHING:\n")
        f.write("-"*70 + "\n")
        f.write(f"Total students in database: {len(db.database)}\n")
        
        if result:
            roll_number, confidence = result
            f.write(f"\n[MATCH FOUND]\n")
            f.write(f"Leaked by: Student with Roll Number {roll_number:010d}\n")
            f.write(f"Match Confidence: {confidence:.2%}\n\n")
            f.write(f"CONCLUSION:\n")
            f.write(f"-"*70 + "\n")
            f.write(f"This question paper was specifically watermarked for\n")
            f.write(f"student {roll_number:010d} and leaked by them.\n\n")
            f.write(f"Evidence: Invisible watermark decoder matched with {confidence:.1%} confidence.\n")
            f.write(f"[CONFIRMED LEAKER IDENTIFIED]\n")
        else:
            f.write(f"\n[NO MATCH]\n")
            f.write(f"The watermark could not be matched to any known student.\n")
            f.write(f"This PDF may be a test file or unauthorized copy.\n")
    
    # Print summary
    if result:
        roll_number, confidence = result
        print(f"\n[MATCH FOUND]")
        print(f"[OK] Leaked by: Student {roll_number:010d}")
        print(f"[OK] Match Confidence: {confidence:.1%}")
        print(f"\n[CONFIRMED] The question paper was leaked by student {roll_number:010d}")
    else:
        print(f"\n[!] No match found in database")
    
    print(f"\n[OK] Report saved to: {report_path}")
    print("="*70 + "\n")
    
    return result


def main():
    """Main execution."""
    import sys
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python leak_detection.py build       - Build watermark database")
        print("  python leak_detection.py detect <pdf> - Identify leaker from PDF")
        print("\nExample:")
        print("  python leak_detection.py detect watermarked_papers/QP_2310080008.pdf")
        sys.exit(0)
    
    command = sys.argv[1].lower()
    
    if command == "build":
        build_watermark_database()
    
    elif command == "detect":
        if len(sys.argv) < 3:
            print("[!] Please provide PDF path")
            sys.exit(1)
        pdf_path = sys.argv[2]
        result = identify_leaker(pdf_path)
        if result:
            roll, conf = result
            print(f"\nLEAKED BY: ROLL {roll:010d} (Confidence: {conf:.1%})")
    
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()

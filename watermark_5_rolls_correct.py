"""
CORRECT: Patch-based watermarking using encoder with TWO inputs
Pipeline: PDF → Image → Split into patches → Embed watermark in each patch → Reconstruct → Save PDF
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


class CorrectWatermarker:
    def __init__(self):
        """Load encoder and decoder models"""
        print("[LOAD] Loading encoder model...")
        try:
            self.encoder = tf.keras.models.load_model('models/encoder_model.h5')
            print("  [OK] Encoder loaded")
            print(f"  Input shapes: {[(inp.shape) for inp in self.encoder.inputs]}")
            print(f"  Output shape: {self.encoder.output.shape}")
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
    
    def generate_watermark(self, roll_number):
        """
        Generate watermark image (32x32) from roll number
        
        Args:
            roll_number: Student roll number
            
        Returns:
            Watermark as (32, 32, 3) float32 array in [0, 1]
        """
        roll_text = str(roll_number)[-4:]  # Last 4 digits
        
        wm = np.zeros((32, 32, 3), dtype=np.uint8)
        
        # Draw roll number text on watermark
        cv2.putText(
            wm,
            roll_text,
            (2, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1
        )
        
        # Normalize to [0, 1]
        return wm.astype(np.float32) / 255.0
    
    def embed_page(self, image, watermark):
        """
        Embed watermark into a page image by processing patches
        
        Pipeline:
        1. Split image into 256×256 patches
        2. For each patch: encoder([patch, watermark]) → stego patch
        3. Reconstruct full page
        
        Args:
            image: RGB image as (H, W, 3) uint8
            watermark: (32, 32, 3) float32 in [0, 1]
            
        Returns:
            Watermarked image as uint8
        """
        h, w, c = image.shape
        patch_size = 256
        
        result = image.copy().astype(np.float32) / 255.0
        
        patches_processed = 0
        
        # Process each 256×256 patch
        for y in range(0, h - patch_size, patch_size):
            for x in range(0, w - patch_size, patch_size):
                
                try:
                    # Extract patch
                    patch = result[y:y+patch_size, x:x+patch_size]  # (256, 256, 3) in [0, 1]
                    
                    # Encoder expects: [patch_batch, watermark_batch]
                    # patch shape: (256, 256, 3)
                    # watermark shape: (32, 32, 3)
                    patch_batch = patch[np.newaxis, :, :, :]  # (1, 256, 256, 3)
                    wm_batch = watermark[np.newaxis, :, :, :]  # (1, 32, 32, 3)
                    
                    # Encoder output: stego patch
                    stego_batch = self.encoder.predict(
                        [patch_batch, wm_batch],
                        verbose=0
                    )
                    
                    # Remove batch dimension
                    stego = np.squeeze(stego_batch, axis=0)  # (256, 256, 3)
                    
                    # Clip to valid range
                    stego = np.clip(stego, 0, 1)
                    
                    # Place back in result
                    result[y:y+patch_size, x:x+patch_size] = stego
                    patches_processed += 1
                    
                except Exception as e:
                    print(f"      [WARN] Patch at ({y}, {x}) failed: {str(e)[:50]}")
        
        # Convert back to uint8
        result = (result * 255).astype(np.uint8)
        
        return result, patches_processed
    
    def convert_pdf_to_images(self, pdf_path):
        """
        Convert PDF to list of PIL Image objects
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of PIL Image objects (RGB)
        """
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(str(pdf_path))
            images = []
            
            for page_num, page in enumerate(doc):
                # Render page as image (1.5x zoom for better quality)
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
            
            doc.close()
            return images
        except Exception as e:
            print(f"      [ERROR] Failed to convert PDF: {e}")
            return []
    
    def save_images_to_pdf(self, images, output_path):
        """
        Save list of PIL Image objects to PDF
        
        Args:
            images: List of PIL Image objects
            output_path: Output PDF path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not images:
                return False
            
            images[0].save(
                str(output_path),
                save_all=True,
                append_images=images[1:] if len(images) > 1 else []
            )
            return True
        except Exception as e:
            print(f"      [ERROR] Failed to save PDF: {e}")
            return False
    
    def watermark_pdf(self, pdf_path, roll_number, output_dir="watermarked_papers_5_correct"):
        """
        Watermark a PDF file with student roll number
        
        Args:
            pdf_path: Path to input PDF
            roll_number: Student roll number
            output_dir: Output directory
            
        Returns:
            True if successful
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        if not pdf_path.exists():
            print(f"  [SKIP] PDF not found: {pdf_path}")
            return False
        
        print(f"  Processing roll {roll_number}...", end=" ", flush=True)
        
        # Step 1: Convert PDF to images
        images = self.convert_pdf_to_images(pdf_path)
        if not images:
            print("[FAILED] Could not convert PDF")
            return False
        
        # Step 2: Generate watermark
        watermark = self.generate_watermark(roll_number)
        
        # Step 3: Watermark each page
        watermarked_images = []
        total_patches = 0
        
        for page_num, page_img in enumerate(images):
            # Convert PIL Image to numpy array (RGB)
            page_array = np.array(page_img)
            
            # Ensure RGB
            if len(page_array.shape) == 2:
                page_array = cv2.cvtColor(page_array, cv2.COLOR_GRAY2BGR)
            elif page_array.shape[2] == 4:
                page_array = cv2.cvtColor(page_array, cv2.COLOR_RGBA2BGR)
            
            if page_array.shape[2] == 3:
                # BGR to RGB
                page_array = cv2.cvtColor(page_array, cv2.COLOR_BGR2RGB)
            
            # Embed watermark using encoder
            watermarked_array, patches = self.embed_page(page_array, watermark)
            total_patches += patches
            
            # Convert back to PIL
            watermarked_images.append(Image.fromarray(watermarked_array))
        
        # Step 4: Save back to PDF
        output_path = output_dir / f"QP_{roll_number:010d}.pdf"
        
        if self.save_images_to_pdf(watermarked_images, output_path):
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"[OK] {output_path.name} ({total_patches} patches, {size_mb:.2f} MB)")
            return True
        else:
            print("[FAILED]")
            return False


def main():
    """Watermark 5 PDFs with correct patch-based encoding"""
    watermarker = CorrectWatermarker()
    
    print("\n" + "="*70)
    print("CORRECT PATCH-BASED WATERMARKING")
    print("="*70)
    print("Pipeline: PDF → Image → 256×256 Patches → Encoder([patch, watermark])")
    print("Watermark size: 32×32 | Patch size: 256×256")
    print("="*70)
    
    # Use actual exam PDF
    source_pdf = Path("MP_QP_Insem1_set3.pdf")
    
    roll_numbers = [2310080001, 2310080002, 2310080003, 2310080004, 2310080005]
    
    if not source_pdf.exists():
        print(f"\n[ERROR] Source PDF not found: {source_pdf}")
        return
    
    print(f"\nSource PDF: {source_pdf.name}")
    print(f"Watermarking for {len(roll_numbers)} roll numbers...\n")
    
    success_count = 0
    
    for roll in roll_numbers:
        if watermarker.watermark_pdf(str(source_pdf), roll):
            success_count += 1
    
    print("\n" + "="*70)
    print(f"COMPLETE: {success_count}/{len(roll_numbers)} PDFs watermarked")
    print(f"Output directory: watermarked_papers_5_correct/")
    print("="*70)
    print("\n[NEXT] Extract watermarks using decoder...")
    print("Run: python extract_watermarks_decoder_only.py")


if __name__ == "__main__":
    main()

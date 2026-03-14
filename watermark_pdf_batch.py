"""
Proper PDF Watermarking for Exam Papers
Convert PDF → Embed watermark on ALL pages → Convert back to PDF
"""

import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import warnings

sys.path.insert(0, str(Path(__file__).parent))

from utils.id_generator import generate_watermark_id
from utils.watermark_utils import WatermarkProcessor

warnings.filterwarnings('ignore')

class PDFExamWatermarker:
    """Converts PDF to images, embeds watermarks, converts back to PDF"""
    
    def __init__(self, pdf_path):
        """
        Initialize with PDF file
        
        Args:
            pdf_path: Path to the question paper PDF
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        self.pages_images = None
        self.processor = WatermarkProcessor()
        self._load_pdf()
    
    def _load_pdf(self):
        """Convert PDF to images using PyMuPDF"""
        print(f"[PDF] Loading PDF: {self.pdf_path.name}")
        try:
            import fitz
            print("  Using PyMuPDF for PDF rendering...")
            doc = fitz.open(str(self.pdf_path))
            self.pages_images = []
            for page_num, page in enumerate(doc):
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))  # 1.5x zoom for quality
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                self.pages_images.append(img)
                print(f"  Page {page_num + 1}: {img.size}")
            doc.close()
            print(f"  [OK] {len(self.pages_images)} pages loaded")
        except ImportError:
            print("\n[ERROR] PyMuPDF not installed!")
            print("   Install: pip install PyMuPDF")
            raise
        except Exception as e:
            print(f"\n[ERROR] Error converting PDF: {e}")
            raise
    
    def embed_watermark_on_pages(self, roll_number, exam_id="AI2026"):
        """
        Embed watermark on all pages for one student using patch-based approach.
        
        Pipeline:
        1. Convert PDF to images ✓ (done in _load_pdf)
        2. Split each image into 128×128 patches
        3. Embed watermark in each patch using encoder
        4. Reconstruct full page
        5. Convert pages back to PDF
        
        Args:
            roll_number: Student's roll number
            exam_id: Exam identifier
            
        Returns:
            List of watermarked PIL Image objects
        """
        watermark_id = generate_watermark_id(roll_number, exam_id)
        watermarked_pages = []
        
        print(f"\n  Watermarking pages for student {roll_number}...", end=" ")
        
        for page_idx, page_img in enumerate(self.pages_images):
            try:
                # Convert PIL Image to numpy array
                page_array = np.array(page_img)
                
                # Ensure 3 channels RGB
                if len(page_array.shape) == 2:
                    page_array = cv2.cvtColor(page_array, cv2.COLOR_GRAY2BGR)
                elif page_array.shape[2] == 4:
                    page_array = cv2.cvtColor(page_array, cv2.COLOR_RGBA2BGR)
                
                # BGR to RGB for processing
                page_array = cv2.cvtColor(page_array, cv2.COLOR_BGR2RGB)
                
                # Create watermark image (32x32)
                watermark_img = np.zeros((32, 32, 3), dtype=np.uint8)
                # Extract last 2 digits of roll number for uniqueness (01-88)
                # This ensures roll 8 gets "08", roll 75 gets "75", etc.
                roll_suffix = str(roll_number)[-2:]
                wm_text = f"{exam_id}{roll_suffix}"  # e.g., "AI202608" or "AI202675"
                cv2.putText(watermark_img, wm_text, (2, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Embed watermark on full page (processor handles patching internally)
                watermarked_array = self.processor.embed_watermark(page_array, watermark_img)
                
                # Convert back to PIL Image
                watermarked_pages.append(Image.fromarray(watermarked_array))
                
            except Exception as e:
                print(f"[WARN] Error on page {page_idx}: {str(e)[:40]}")
                # Use original page if watermarking fails
                watermarked_pages.append(page_img)
        
        print(f"[OK] ({len(watermarked_pages)} pages)")
        return watermarked_pages
    
    def save_as_pdf(self, images, output_path):
        """
        Save list of PIL images as PDF
        
        Args:
            images: List of PIL Image objects
            output_path: Output PDF file path
        """
        if not images:
            return False
        
        # Convert all to RGB
        images_rgb = []
        for img in images:
            if img.mode != 'RGB':
                images_rgb.append(img.convert('RGB'))
            else:
                images_rgb.append(img)
        
        # Save as PDF
        try:
            images_rgb[0].save(
                output_path,
                save_all=True,
                append_images=images_rgb[1:] if len(images_rgb) > 1 else [],
                format='PDF'
            )
            return True
        except Exception as e:
            print(f"[ERROR] Error saving PDF: {e}")
            return False
    
    def generate_for_all_students(self, start_roll=2310080001, end_roll=2310080088, 
                                  exam_id="AI2026", output_dir="watermarked_papers"):
        """
        Generate watermarked PDF for all students
        
        Args:
            start_roll: Starting roll number
            end_roll: Ending roll number
            exam_id: Exam identifier
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        total = end_roll - start_roll + 1
        successful = 0
        failed = 0
        
        print(f"\n{'='*70}")
        print(f"Generating Watermarked Question Papers")
        print(f"PDF: {self.pdf_path.name}")
        print(f"Students: {total} ({start_roll:010d} - {end_roll:010d})")
        print(f"Output: {output_dir.absolute()}")
        print(f"{'='*70}")
        
        for idx, roll_number in enumerate(range(start_roll, end_roll + 1), 1):
            try:
                print(f"[{idx:3d}/{total}] Roll {roll_number:010d}...", end=" ", flush=True)
                
                # Embed watermark on all pages
                watermarked_images = self.embed_watermark_on_pages(roll_number, exam_id)
                
                # Save as PDF
                output_filename = f"QP_{roll_number:010d}.pdf"
                output_path = output_dir / output_filename
                
                if self.save_as_pdf(watermarked_images, output_path):
                    print("[OK] Saved")
                    successful += 1
                else:
                    print("[ERROR] Save failed")
                    failed += 1
                    
            except Exception as e:
                print(f"[ERROR] Error: {str(e)[:30]}")
                failed += 1
        
        print(f"\n{'='*70}")
        print(f"Generation Complete!")
        print(f"  ✓ Successful: {successful}/{total}")
        if failed > 0:
            print(f"  ❌ Failed: {failed}/{total}")
        print(f"  📁 Output: {output_dir.absolute()}")
        print(f"{'='*70}\n")
        
        return {
            "total": total,
            "successful": successful,
            "failed": failed,
            "output_dir": str(output_dir.absolute())
        }


def main():
    """Main entry point"""
    pdf_path = Path(__file__).parent / "MP_QP_Insem1_set3.pdf"
    
    try:
        # Create watermarker
        watermarker = PDFExamWatermarker(pdf_path)
        
        # Generate for all 88 students
        stats = watermarker.generate_for_all_students(
            start_roll=2310080001,
            end_roll=2310080088,
            exam_id="AI2026",
            output_dir="watermarked_papers"
        )
        
        print(f"✅ All {stats['successful']} watermarked PDFs generated successfully!")
        print(f"📍 Check: {stats['output_dir']}")
        
        return stats
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

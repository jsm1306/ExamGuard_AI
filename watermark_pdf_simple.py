"""
PDF Watermarking - Alternative Method (No Poppler Required)
Uses PyPDF2 + Pillow to embed watermarks on PDF pages
"""

import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw
import warnings

sys.path.insert(0, str(Path(__file__).parent))

from utils.id_generator import generate_watermark_id
from utils.watermark_utils import WatermarkProcessor

warnings.filterwarnings('ignore')

class SimplePDFWatermarker:
    """Watermark PDFs without requiring Poppler - works with PIL/PyPDF2"""
    
    def __init__(self, pdf_path):
        """
        Initialize with PDF file
        
        Args:
            pdf_path: Path to the question paper PDF
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        print(f"📄 Loading PDF: {self.pdf_path.name}")
        self.processor = WatermarkProcessor()
    
    def extract_pdf_as_images(self):
        """
        Extract PDF pages as images using PyPDF2
        
        Returns:
            List of PIL Image objects
        """
        try:
            import fitz  # PyMuPDF - better alternative
            print("  Using PyMuPDF for PDF rendering...")
            doc = fitz.open(str(self.pdf_path))
            images = []
            for page_num, page in enumerate(doc):
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))  # 1.5x zoom for quality
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
                print(f"    Page {page_num + 1}: {img.size}")
            doc.close()
            return images
        except ImportError:
            print("  PyMuPDF not found, trying pdf2image...")
            try:
                from pdf2image import convert_from_path
                print("  Converting PDF to images (requires Poppler)...")
                images = convert_from_path(str(self.pdf_path), dpi=150)
                return images
            except Exception as e:
                print(f"\n❌ Cannot extract PDF images: {e}")
                print("\n📋 Install one of these:")
                print("   1. PyMuPDF: pip install PyMuPDF")
                print("   2. pdf2image: pip install pdf2image + Poppler")
                raise
    
    def create_watermark_image(self, roll_number, exam_id="AI2026"):
        """
        Create a watermark image for model embedding (32x32)
        
        Args:
            roll_number: Student's roll number
            exam_id: Exam identifier
            
        Returns:
            np.ndarray watermark image (32, 32, 3), dtype uint8
        """
        watermark_id = generate_watermark_id(roll_number, exam_id)
        
        # Create watermark image (32x32 as expected by encoder)
        watermark_img = np.zeros((32, 32, 3), dtype=np.uint8)
        
        # Write watermark text on it
        text = watermark_id[:8]  # Use first 8 characters
        cv2.putText(watermark_img, text, (2, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return watermark_img
    
    def watermark_page(self, page_img, roll_number, exam_id="AI2026"):
        """
        Embed watermark on a page using patch-based encoding.
        
        Pipeline:
        1. Convert page to numpy array
        2. Create watermark image (32×32)
        3. Embed watermark in patches (128×128)
        4. Reconstruct page
        
        Args:
            page_img: PIL Image of the page
            roll_number: Student's roll number
            exam_id: Exam identifier
            
        Returns:
            Watermarked PIL Image
        """
        try:
            # Convert PIL Image to numpy array
            page_array = np.array(page_img)
            
            # Ensure 3 channels (RGB)
            if len(page_array.shape) == 2:
                page_array = cv2.cvtColor(page_array, cv2.COLOR_GRAY2BGR)
            elif page_array.shape[2] == 4:
                page_array = cv2.cvtColor(page_array, cv2.COLOR_RGBA2BGR)
            
            # BGR to RGB conversion
            page_array = cv2.cvtColor(page_array, cv2.COLOR_BGR2RGB)
            
            # Create watermark image (32x32)
            watermark_img = self.create_watermark_image(roll_number, exam_id)
            
            # Embed watermark on full page (processor handles patching internally)
            watermarked_array = self.processor.embed_watermark(page_array, watermark_img)
            
            # Convert back to PIL Image
            return Image.fromarray(watermarked_array)
            
        except Exception as e:
            print(f"      Warning: Embedding failed, using original page: {str(e)[:40]}")
            return page_img
    
    def watermark_pdf(self, output_dir, roll_number, exam_id="AI2026"):
        """
        Create watermarked PDF for one student
        
        Args:
            output_dir: Output directory
            roll_number: Student's roll number
            exam_id: Exam identifier
            
        Returns:
            Path to output PDF
        """
        # Extract pages
        try:
            pages = self.extract_pdf_as_images()
        except Exception as e:
            return None
        
        # Watermark each page
        watermarked_pages = []
        for page_img in pages:
            try:
                watermarked = self.watermark_page(page_img, roll_number, exam_id)
                watermarked_pages.append(watermarked)
            except Exception as e:
                # Use original if watermarking fails
                watermarked_pages.append(page_img)
        
        if not watermarked_pages:
            return None
        
        # Save as PDF
        output_path = output_dir / f"QP_{roll_number:010d}.pdf"
        try:
            watermarked_pages[0].save(
                output_path,
                save_all=True,
                append_images=watermarked_pages[1:] if len(watermarked_pages) > 1 else [],
                format='PDF'
            )
            return output_path
        except Exception as e:
            print(f"    Error saving PDF: {e}")
            return None
    
    def generate_for_all_students(self, start_roll=2310080001, end_roll=2310080088,
                                  exam_id="AI2026", output_dir="watermarked_papers"):
        """Generate watermarked PDF for all students"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        total = end_roll - start_roll + 1
        successful = 0
        
        print(f"\n{'='*70}")
        print(f"Generating Watermarked Question Papers (PDF)")
        print(f"PDF: {self.pdf_path.name}")
        print(f"Students: {total}")
        print(f"Output: {output_dir.absolute()}")
        print(f"{'='*70}\n")
        
        for idx, roll_number in enumerate(range(start_roll, end_roll + 1), 1):
            try:
                print(f"[{idx:3d}/{total}] Roll {roll_number:010d}...", end=" ", flush=True)
                
                output_path = self.watermark_pdf(output_dir, roll_number, exam_id)
                
                if output_path and output_path.exists():
                    size_mb = output_path.stat().st_size / (1024 * 1024)
                    print(f"✓ ({size_mb:.1f} MB)")
                    successful += 1
                else:
                    print("❌ Failed")
                    
            except Exception as e:
                print(f"❌ {str(e)[:20]}")
        
        print(f"\n{'='*70}")
        print(f"✓ Generation Complete!")
        print(f"  ✓ Successful: {successful}/{total}")
        print(f"  📁 Output: {output_dir.absolute()}")
        print(f"{'='*70}\n")
        
        return {
            "successful": successful,
            "total": total,
            "output_dir": str(output_dir.absolute())
        }


def main():
    """Main entry point"""
    pdf_path = Path(__file__).parent / "MP_QP_Insem1_set3.pdf"
    
    try:
        watermarker = SimplePDFWatermarker(pdf_path)
        stats = watermarker.generate_for_all_students(
            start_roll=2310080001,
            end_roll=2310080088,
            exam_id="AI2026",
            output_dir="watermarked_papers"
        )
        
        print(f"✅ Success! {stats['successful']}/{stats['total']} PDFs generated")
        return stats
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n📋 Install required library:")
        print("   pip install PyMuPDF")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

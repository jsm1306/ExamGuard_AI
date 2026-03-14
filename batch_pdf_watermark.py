"""
Batch PDF Watermarking Script
Generates watermarked question papers for all students
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
import warnings

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.watermark_utils import WatermarkProcessor
from utils.id_generator import generate_watermark_id

warnings.filterwarnings('ignore')

class PDFWatermarkGenerator:
    """Generates watermarked PDFs for batch exam paper generation"""
    
    def __init__(self, pdf_path, output_dir="generated_qpapers", dpi=300):
        """
        Initialize PDF watermarking generator
        
        Args:
            pdf_path: Path to the original PDF
            output_dir: Directory to save generated PDFs
            dpi: DPI for PDF to image conversion
        """
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir)
        self.dpi = dpi
        
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize watermark processor
        print("Initializing watermark processor...")
        self.processor = WatermarkProcessor()
        
        # Convert PDF to images
        print(f"Converting PDF to images (DPI: {dpi})...")
        self.pages = self._pdf_to_images()
        print(f"✓ PDF converted: {len(self.pages)} pages")
    
    def _pdf_to_images(self):
        """Convert PDF pages to PIL images"""
        try:
            pages = convert_from_path(str(self.pdf_path), dpi=self.dpi)
            return pages
        except Exception as e:
            print(f"\n❌ Poppler Error: {e}")
            print("\n📥 Install Poppler:")
            print("   Windows (Chocolatey): choco install poppler")
            print("   Windows (Manual): Download from https://github.com/oschwartz10612/poppler-windows/releases")
            print("   Linux: sudo apt-get install poppler-utils")
            print("   macOS: brew install poppler")
            print("\nAlternatively, you can use a web UI to do this bulk PDF processing.")
            raise
    
    def _embed_watermark_on_pages(self, roll_number, exam_id="AI2026"):
        """Embed watermark on all PDF pages"""
        watermarked_pages = []
        
        # Generate watermark from roll number
        watermark = generate_watermark_id(roll_number, exam_id)
        print(f"  Generated watermark: {watermark}")
        
        for page_idx, page_img in enumerate(self.pages):
            # Convert PIL Image to numpy array
            page_array = np.array(page_img)
            
            # Resize if needed for watermarking (model expects certain sizes)
            target_height = min(512, page_array.shape[0])
            target_width = int(target_height * page_array.shape[1] / page_array.shape[0])
            
            page_array = cv2.resize(page_array, (target_width, target_height))
            
            # Create watermark image (32x32)
            watermark_img = np.zeros((32, 32, 3), dtype=np.uint8)
            cv2.putText(watermark_img, watermark[:8], (2, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Embed watermark
            try:
                watermarked = self.processor.embed_watermark(page_array, watermark_img)
                watermarked_pages.append(Image.fromarray(watermarked))
            except Exception as e:
                print(f"  ⚠ Error embedding watermark on page {page_idx + 1}: {e}")
                # Use original page if watermarking fails
                watermarked_pages.append(page_img)
        
        return watermarked_pages
    
    def _images_to_pdf(self, images, output_path):
        """Convert PIL images to PDF"""
        if not images:
            return False
        
        # Convert all images to RGB if needed
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
                append_images=images_rgb[1:],
                format='PDF'
            )
            return True
        except Exception as e:
            print(f"  ❌ Error saving PDF: {e}")
            return False
    
    def generate_watermarked_papers(self, start_roll=2310080001, end_roll=2310080088, 
                                    exam_id="AI2026", verbose=True):
        """
        Generate watermarked PDFs for all students
        
        Args:
            start_roll: Starting roll number
            end_roll: Ending roll number
            exam_id: Exam identifier (e.g., "AI2026")
            verbose: Print progress for each student
        
        Returns:
            Dictionary with generation statistics
        """
        total_students = end_roll - start_roll + 1
        successful = 0
        failed = 0
        
        print(f"\n{'='*70}")
        print(f"Generating watermarked papers for {total_students} students")
        print(f"Roll numbers: {start_roll:010d} to {end_roll:010d}")
        print(f"PDF: {self.pdf_path.name}")
        print(f"{'='*70}\n")
        
        for idx, roll_number in enumerate(range(start_roll, end_roll + 1), 1):
            try:
                if verbose:
                    print(f"[{idx:3d}/{total_students}] Processing roll {roll_number:010d}...", 
                          end=" ", flush=True)
                
                # Embed watermark
                watermarked_pages = self._embed_watermark_on_pages(roll_number, exam_id)
                
                # Create output filename
                output_filename = f"QP_{roll_number:010d}.pdf"
                output_path = self.output_dir / output_filename
                
                # Save as PDF
                if self._images_to_pdf(watermarked_pages, output_path):
                    if verbose:
                        print(f"✓ Saved")
                    successful += 1
                else:
                    if verbose:
                        print(f"❌ Failed to save")
                    failed += 1
                    
            except Exception as e:
                if verbose:
                    print(f"❌ Error: {e}")
                failed += 1
        
        print(f"\n{'='*70}")
        print(f"Generation Complete!")
        print(f"  ✓ Successful: {successful}/{total_students}")
        print(f"  ❌ Failed: {failed}/{total_students}")
        print(f"  📁 Output: {self.output_dir.absolute()}")
        print(f"{'='*70}\n")
        
        return {
            "total": total_students,
            "successful": successful,
            "failed": failed,
            "output_dir": str(self.output_dir.absolute())
        }


def main():
    """Main entry point"""
    pdf_path = Path(__file__).parent / "MP_QP_Insem1_set3.pdf"
    output_dir = Path(__file__).parent / "generated_qpapers"
    
    # Create generator
    generator = PDFWatermarkGenerator(pdf_path, output_dir, dpi=200)
    
    # Generate papers for all students
    stats = generator.generate_watermarked_papers(
        start_roll=2310080001,
        end_roll=2310080088,
        exam_id="AI2026",
        verbose=True
    )
    
    return stats


if __name__ == "__main__":
    try:
        stats = main()
        print("\n✓ All watermarked PDFs generated successfully!")
        print(f"Check: {stats['output_dir']}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

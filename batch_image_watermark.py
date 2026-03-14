"""
Alternative Batch Watermarking Script (Image-based)
Works without Poppler - processes exam images directly
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import warnings

sys.path.insert(0, str(Path(__file__).parent))

from utils.watermark_utils import WatermarkProcessor
from utils.id_generator import generate_watermark_id

warnings.filterwarnings('ignore')

class ImageWatermarkGenerator:
    """Generates watermarked exam images for batch processing"""
    
    def __init__(self, input_image_path, output_dir="generated_exams", resize_for_watermarking=True):
        """
        Initialize image watermarking generator
        
        Args:
            input_image_path: Path to the original exam image (PNG/JPG)
            output_dir: Directory to save generated images
            resize_for_watermarking: Whether to resize images to watermark-friendly dimensions
        """
        self.input_path = Path(input_image_path)
        self.output_dir = Path(output_dir)
        self.resize_for_watermarking = resize_for_watermarking
        
        if not self.input_path.exists():
            raise FileNotFoundError(f"Image not found: {input_image_path}")
        
        self.output_dir.mkdir(exist_ok=True)
        
        # Load original image
        print(f"Loading image: {self.input_path.name}")
        self.original_image = cv2.imread(str(self.input_path))
        if self.original_image is None:
            raise ValueError(f"Failed to load image: {input_image_path}")
        
        print(f"Original image size: {self.original_image.shape}")
        
        # Initialize watermark processor
        print("Initializing watermark processor...")
        self.processor = WatermarkProcessor()
    
    def _resize_for_watermarking(self, image):
        """Resize image to watermark-friendly dimensions (128x128 per tile)"""
        # For exam papers, we'll tile 128x128 sections
        h, w = image.shape[:2]
        
        # Calculate how many tiles we need
        tiles_h = max(1, (h + 127) // 128)
        tiles_w = max(1, (w + 127) // 128)
        
        # Create padded image
        padded_h = tiles_h * 128
        padded_w = tiles_w * 128
        
        padded = np.zeros((padded_h, padded_w, 3), dtype=np.uint8)
        padded[:h, :w] = image
        
        return padded, (tiles_h, tiles_w)
    
    def _create_watermark_overlay(self, roll_number):
        """Create a text-based watermark image"""
        watermark_text = generate_watermark_id(roll_number)
        
        # Create a semi-transparent overlay
        overlay = np.zeros((100, 300, 3), dtype=np.uint8)
        
        # Add watermark text
        cv2.putText(overlay, f"Roll: {roll_number}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(overlay, f"ID: {watermark_text}", (10, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        return overlay
    
    def _apply_watermark_to_tile(self, tile, watermark_id):
        """Apply watermark to a single 256x256 tile"""
        try:
            # Ensure tile is 256x256x3
            if tile.shape != (128, 128, 3):
                tile = cv2.resize(tile, (128, 128))
            
            # Create watermark image (32x32 as expected by model)
            watermark_img = np.zeros((32, 32, 3), dtype=np.uint8)
            text = watermark_id[:8]
            cv2.putText(watermark_img, text, (2, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Embed watermark
            watermarked = self.processor.embed_watermark(tile, watermark_img)
            return watermarked
            
        except Exception as e:
            print(f"    \u26a0 Warning: Watermarking failed for tile: {e}")
            # Return original tile if watermarking fails
            return tile
    
    def _apply_text_watermark(self, image, roll_number):
        """Apply visible text watermark to image (faster alternative)"""
        result = image.copy()
        
        # Add semi-transparent watermark text
        overlay = result.copy()
        cv2.rectangle(overlay, (10, 10), (400, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)
        
        # Add text
        text = f"Roll: {roll_number:010d}"
        cv2.putText(result, text, (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        return result
    
    def generate_watermarked_images(self, start_roll=2310080001, end_roll=2310080088,
                                    exam_id="AI2026", use_model_watermarking=True, verbose=True):
        """
        Generate watermarked exam images for all students
        
        Args:
            start_roll: Starting roll number
            end_roll: Ending roll number
            exam_id: Exam identifier
            use_model_watermarking: Use deep learning watermarking (True) or text overlay (False)
            verbose: Print progress
        
        Returns:
            Dictionary with generation statistics
        """
        total_students = end_roll - start_roll + 1
        successful = 0
        failed = 0
        
        mode = "Deep Learning" if (use_model_watermarking and (self.processor.encoder or self.processor.decoder)) else "Text Overlay"
        
        print(f"\n{'='*70}")
        print(f"Generating watermarked images ({mode} mode)")
        print(f"Students: {total_students} ({start_roll:010d} - {end_roll:010d})")
        print(f"Output: {self.output_dir.absolute()}")
        print(f"{'='*70}\n")
        
        for idx, roll_number in enumerate(range(start_roll, end_roll + 1), 1):
            try:
                if verbose:
                    print(f"[{idx:3d}/{total_students}] Student {roll_number:010d}...", 
                          end=" ", flush=True)
                
                if use_model_watermarking and (self.processor.encoder or self.processor.decoder):
                    # Use deep learning watermarking with tiling
                    watermarked = self._apply_model_watermarking(roll_number, exam_id)
                else:
                    # Use simple text overlay
                    watermarked = self._apply_text_watermark(self.original_image, roll_number)
                
                # Save image
                output_filename = f"Exam_{roll_number:010d}.png"
                output_path = self.output_dir / output_filename
                
                success = cv2.imwrite(str(output_path), watermarked)
                if success:
                    if verbose:
                        print("✓")
                    successful += 1
                else:
                    if verbose:
                        print("❌ Write failed")
                    failed += 1
                    
            except Exception as e:
                if verbose:
                    print(f"❌ {str(e)[:30]}")
                failed += 1
        
        print(f"\n{'='*70}")
        print(f"Generation Complete!")
        print(f"  ✓ Successful: {successful}/{total_students}")
        if failed > 0:
            print(f"  ❌ Failed: {failed}/{total_students}")
        print(f"  📁 Output: {self.output_dir.absolute()}")
        print(f"{'='*70}\n")
        
        return {
            "total": total_students,
            "successful": successful,
            "failed": failed,
            "output_dir": str(self.output_dir.absolute()),
            "mode": mode
        }
    
    def _apply_model_watermarking(self, roll_number, exam_id="AI2026"):
        """Apply model-based watermarking to image with tiling"""
        watermark_id = generate_watermark_id(roll_number, exam_id)
        original = self.original_image.copy()
        
        if not self.resize_for_watermarking:
            return self._apply_text_watermark(original, roll_number)
        
        # Resize for watermarking
        padded, (tiles_h, tiles_w) = self._resize_for_watermarking(original)
        watermarked = np.zeros_like(padded)
        
        # Process each tile
        for i in range(tiles_h):
            for j in range(tiles_w):
                tile = padded[i*128:(i+1)*128, j*128:(j+1)*128]
                watermarked[i*128:(i+1)*128, j*128:(j+1)*128] = self._apply_watermark_to_tile(
                    tile, watermark_id
                )
        
        # Crop back to original size
        h, w = original.shape[:2]
        watermarked = watermarked[:h, :w]
        
        return watermarked


def main():
    """Main entry point"""
    # Try to find exam images
    workspace = Path(__file__).parent
    
    # Look for converted PDF images or use a sample image
    image_files = list(workspace.glob("*.png")) + list(workspace.glob("*.jpg"))
    
    if not image_files:
        print("\n📋 No exam images found!")
        print("\nPlease provide an exam image (PNG or JPG) or:")
        print("1. Convert your PDF to images using online tools:")
        print("   - https://pdf2image.readthedocs.io/")
        print("   - https://ilovepdf.com/pdf-to-image")
        print("\n2. Place the converted image in the ExamGuard_AI folder")
        print("\n3. Run this script again")
        return None
    
    # Use first image found
    exam_image = image_files[0]
    print(f"\n✓ Found exam image: {exam_image.name}")
    
    # Create generator
    generator = ImageWatermarkGenerator(exam_image, "generated_exams", resize_for_watermarking=False)
    
    # Generate watermarked images (using text watermarking for reliability)
    stats = generator.generate_watermarked_images(
        start_roll=2310080001,
        end_roll=2310080088,
        exam_id="AI2026",
        use_model_watermarking=False,  # Use text overlay (faster, no model needed)
        verbose=True
    )
    
    return stats


if __name__ == "__main__":
    try:
        stats = main()
        if stats:
            print("\n✓ All watermarked exam images generated successfully!")
            print(f"Check: {stats['output_dir']}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

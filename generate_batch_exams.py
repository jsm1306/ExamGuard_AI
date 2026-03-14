"""
Quick Batch Generation Script
Generates 88 watermarked exam papers for all students
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("ExamGuard AI - Batch Exam Paper Generation")
print("=" * 70)

# Import modules
from utils.watermark_utils import WatermarkProcessor
from utils.id_generator import generate_watermark_id
import numpy as np
import cv2
import os

def create_dummy_exam_image(size=(800, 600)):
    """Create a dummy exam paper image for demonstration"""
    # Create white background
    image = np.ones((size[0], size[1], 3), dtype=np.uint8) * 255
    
    # Add some text-like patterns (simulating exam content)
    cv2.putText(image, "EXAM QUESTION PAPER", (50, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    cv2.putText(image, "Course: Machine Learning | Time: 3 hours | Max Marks: 100",
               (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    
    # Add some lines to simulate questions
    for i in range(5):
        y_pos = 150 + i * 120
        cv2.line(image, (50, y_pos), (750, y_pos), (200, 200, 200), 1)
        cv2.putText(image, f"Question {i+1}: [20 marks]", (60, y_pos + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        cv2.putText(image, "Your answer here...", (60, y_pos + 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
    
    # Add footer
    cv2.putText(image, "Insert: MP_QP_Insem1_set3.pdf content here",
               (50, size[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    return image

def generate_with_text_watermark(exam_image, start_roll, end_roll, output_dir):
    """Generate watermarked exams using text watermarks (fastest method)"""
    total = end_roll - start_roll + 1
    successful = 0
    
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n📝 Generating Text-Watermarked Papers ({total} students)...")
    print("-" * 70)
    
    for idx, roll_number in enumerate(range(start_roll, end_roll + 1), 1):
        watermark_id = generate_watermark_id(roll_number, "AI2026")
        
        # Create copy of exam
        watermarked = exam_image.copy()
        
        # Add semi-transparent background for watermark
        overlay = watermarked.copy()
        cv2.rectangle(overlay, (10, 10), (400, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.2, watermarked, 0.8, 0, watermarked)
        
        # Add watermark text
        cv2.putText(watermarked, f"Roll: {roll_number:010d}", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(watermarked, f"ID: {watermark_id}", (20, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        
        # Save
        output_file = output_dir / f"Exam_{roll_number:010d}.png"
        cv2.imwrite(str(output_file), watermarked)
        successful += 1
        
        # Progress
        if idx % 10 == 0 or idx == total:
            print(f"[{idx:3d}/{total}] Generated {idx} papers... ✓")
    
    return successful, total

def generate_with_deep_learning(exam_image, start_roll, end_roll, output_dir):
    """Generate watermarked exams using deep learning watermarking"""
    total = end_roll - start_roll + 1
    successful = 0
    failed = 0
    
    output_dir.mkdir(exist_ok=True)
    
    # Initialize processor
    processor = WatermarkProcessor()
    
    print(f"\n🔬 Generating Deep-Learning Watermarked Papers ({total} students)...")
    print("-" * 70)
    
    for idx, roll_number in enumerate(range(start_roll, end_roll + 1), 1):
        try:
            watermark_id = generate_watermark_id(roll_number, "AI2026")
            
            # Resize for watermarking (128x128 tiles)
            h, w = exam_image.shape[:2]
            tiles_h = max(1, (h + 127) // 128)
            tiles_w = max(1, (w + 127) // 128)
            
            padded_h = tiles_h * 128
            padded_w = tiles_w * 128
            padded = np.zeros((padded_h, padded_w, 3), dtype=np.uint8)
            padded[:h, :w] = exam_image
            
            watermarked = np.zeros_like(padded)
            
            # Create watermark image (32x32)
            watermark_img = np.zeros((32, 32, 3), dtype=np.uint8)
            text = watermark_id[:8]
            cv2.putText(watermark_img, text, (2, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Process tiles
            for i in range(tiles_h):
                for j in range(tiles_w):
                    tile = padded[i*128:(i+1)*128, j*128:(j+1)*128]
                    try:
                        watermarked[i*128:(i+1)*128, j*128:(j+1)*128] = processor.embed_watermark(
                            tile, watermark_img
                        )
                    except:
                        watermarked[i*128:(i+1)*128, j*128:(j+1)*128] = tile
            
            # Crop back
            watermarked = watermarked[:h, :w]
            
            # Save
            output_file = output_dir / f"Exam_{roll_number:010d}_DL.png"
            cv2.imwrite(str(output_file), watermarked)
            successful += 1
            
        except Exception as e:
            failed += 1
        
        # Progress
        if idx % 10 == 0 or idx == total:
            status = f"{successful} ✓" if failed == 0 else f"{successful} ✓, {failed} ❌"
            print(f"[{idx:3d}/{total}] {status}")
    
    return successful, total

def main():
    """Main generation function"""
    workspace = Path(__file__).parent
    output_dir_text = workspace / "generated_exams_text"
    output_dir_dl = workspace / "generated_exams_dl"
    
    # Roll numbers:88 students
    start_roll = 2310080001
    end_roll = 2310080088
    
    print(f"\n📋 Generating papers for {end_roll - start_roll + 1} students")
    print(f"Roll Numbers: {start_roll:010d} to {end_roll:010d}")
    print(f"Exam: AI2026 (Machine Learning - Insem 1, Set 3)")
    
    # Create or use existing exam image
    print("\n📄 Loading exam image...")
    exam_image_files = list(workspace.glob("*.png")) + list(workspace.glob("*.jpg"))
    
    if exam_image_files:
        print(f"   Found: {exam_image_files[0].name}")
        exam_image = cv2.imread(str(exam_image_files[0]))
    else:
        print("   Creating demo exam image...")
        exam_image = create_dummy_exam_image()
    
    # Generate with text watermarking (always works)
    print("\n" + "="*70)
    print("Method 1: Text-based Watermarking (Fast)")
    print("="*70)
    success, total = generate_with_text_watermark(exam_image, start_roll, end_roll, output_dir_text)
    print(f"\n✓  Generated {success}/{total} papers")
    print(f"📁 Location: {output_dir_text}")
    
    # Generate with deep learning watermarking (if models available)
    print("\n" + "="*70)
    print("Method 2: Deep Learning Watermarking (Optional)")
    print("="*70)
    try:
        success, total = generate_with_deep_learning(exam_image, start_roll, end_roll, output_dir_dl)
        if success > 0:
            print(f"\n✓ Generated {success}/{total} papers with DL watermarking")
            print(f"📁 Location: {output_dir_dl}")
    except Exception as e:
        print(f"⚠ Skipping DL watermarking: {str(e)[:40]}...")
    
    # Summary
    print("\n" + "="*70)
    print("✓ BATCH GENERATION COMPLETE!")
    print("="*70)
    print(f"""
📊 Summary:
   • Total Students: 88
   • Papers Generated (Text): {success}/{total}
   • Text Watermark Format: Roll + Unique ID
   • Files: Exam_XXXXXXXXXX.png
   
📁 Output Directories:
   • Text Watermarks: {output_dir_text}
   • DL Watermarks: {output_dir_dl} (if available)
   
📌 Next Steps:
   1. Review generated papers
   2. Convert to PDF if needed (for printing):
      - Use online tools or LibreOffice
      - Or use a PDF library like PIL + reportlab
   3. Distribute to students via learning management system
   4. Use Tab 3 of app.py to detect leaks
   
💡 Tips:
   • Text watermarks are visible and unique per student
   • Deep learning watermarks are invisible (needs trained models)
   • For production, use the trained model weights
""")
    print("="*70)

if __name__ == "__main__":
    main()

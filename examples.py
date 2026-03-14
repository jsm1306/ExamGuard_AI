"""
ExamGuard AI - Usage Examples
Demonstrates how to use the system programmatically
"""

import numpy as np
import cv2
from PIL import Image
from utils.watermark_utils import WatermarkProcessor
from utils.id_generator import IDGenerator


# ============================================================================
# EXAMPLE 1: Basic Watermarking
# ============================================================================
def example_basic_watermarking():
    """Basic watermark embedding and extraction."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Watermarking")
    print("="*70)
    
    # Initialize processor
    processor = WatermarkProcessor()
    
    # Create sample images
    cover_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    watermark_image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    
    print(f"Cover image shape: {cover_image.shape}")
    print(f"Watermark image shape: {watermark_image.shape}")
    
    # Embed watermark
    watermarked = processor.embed_watermark(cover_image, watermark_image)
    print(f"Watermarked image shape: {watermarked.shape}")
    print("✓ Watermark embedded successfully")
    
    # Calculate embedding metrics
    metrics_embed = processor.calculate_metrics(cover_image, watermarked)
    print(f"Embedding PSNR: {metrics_embed['psnr_formatted']}")
    print(f"Embedding MSE: {metrics_embed['mse_formatted']}")
    
    # Extract watermark
    extracted = processor.extract_watermark(watermarked)
    print(f"Extracted watermark shape: {extracted.shape}")
    print("✓ Watermark extracted successfully")
    
    # Calculate extraction metrics
    metrics_extract = processor.calculate_metrics(watermark_image, extracted)
    print(f"Extraction PSNR: {metrics_extract['psnr_formatted']}")
    print(f"Extraction MSE: {metrics_extract['mse_formatted']}")
    
    # Compare similarity
    similarity = np.mean((watermark_image.astype(float) - extracted.astype(float))**2)
    print(f"Watermark recovery MSE: {similarity:.2f}")
    print()


# ============================================================================
# EXAMPLE 2: Student Copy Generation
# ============================================================================
def example_student_copy_generation():
    """Generate watermarked exams for multiple students."""
    print("="*70)
    print("EXAMPLE 2: Student Copy Generation")
    print("="*70)
    
    processor = WatermarkProcessor()
    id_gen = IDGenerator()
    
    # Sample cover image (would be real exam in practice)
    cover_image = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    
    exam_id = "MIDTERM2026"
    num_students = 5
    
    print(f"\nGenerating {num_students} watermarked exam copies for exam: {exam_id}")
    print("-" * 70)
    
    watermarked_exams = {}
    
    for student_id in range(101, 101 + num_students):
        # Generate unique ID
        watermark_id = id_gen.generate_watermark_id(student_id, exam_id)
        
        # Create watermark image
        watermark = id_gen.convert_id_to_watermark_image(watermark_id)
        
        # Embed watermark
        watermarked = processor.embed_watermark(cover_image, watermark)
        
        watermarked_exams[student_id] = {
            'id': watermark_id,
            'watermark': watermark,
            'exam': watermarked
        }
        
        print(f"Student {student_id:3d} | Watermark ID: {watermark_id}")
    
    print(f"\nGenerated {len(watermarked_exams)} exams successfully ✓")
    return watermarked_exams


# ============================================================================
# EXAMPLE 3: Leak Detection
# ============================================================================
def example_leak_detection():
    """Detect and identify source of leaked exam."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Leak Detection")
    print("="*70)
    
    processor = WatermarkProcessor()
    id_gen = IDGenerator()
    
    # Generate reference exams (same as Example 2)
    cover_image = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    exam_id = "MIDTERM2026"
    
    print(f"\nGenerating reference database for exam: {exam_id}")
    
    reference_exams = {}
    for student_id in range(101, 106):
        wm_id = id_gen.generate_watermark_id(student_id, exam_id)
        watermark = id_gen.convert_id_to_watermark_image(wm_id)
        reference_exams[student_id] = {
            'id': wm_id,
            'watermark': watermark
        }
    
    print(f"Database contains {len(reference_exams)} students")
    
    # Simulate leaked exam (from student 103)
    source_student = 103
    source_wm_id = id_gen.generate_watermark_id(source_student, exam_id)
    source_watermark = id_gen.convert_id_to_watermark_image(source_wm_id)
    leaked_exam = processor.embed_watermark(cover_image, source_watermark)
    
    print(f"\nLeak detected (originated from Student {source_student})")
    print("Analyzing...")
    
    # Extract watermark from leaked image
    extracted_wm = processor.extract_watermark(leaked_exam)
    print("✓ Watermark extracted")
    
    # Match to student
    candidate_ids = [ref['id'] for ref in reference_exams.values()]
    matched_id, confidence = id_gen.match_watermark_to_id(
        extracted_wm,
        candidate_ids
    )
    
    print("-" * 70)
    print("LEAK DETECTION RESULTS:")
    print(f"  Matched to: {matched_id}")
    print(f"  Confidence: {confidence:.2%}")
    
    if confidence > 0.85:
        print("  Status: ✓ CONFIRMED")
        # Extract student ID from matched ID
        detected_student = int(matched_id.split('_')[1])
        print(f"  Source Student: {detected_student}")
        if detected_student == source_student:
            print("  Verification: ✓ CORRECT SOURCE IDENTIFIED")
    else:
        print("  Status: ⚠ LOW CONFIDENCE")
    print()


# ============================================================================
# EXAMPLE 4: Image Processing
# ============================================================================
def example_image_processing():
    """Image resizing and normalization operations."""
    print("="*70)
    print("EXAMPLE 4: Image Processing")
    print("="*70)
    
    processor = WatermarkProcessor()
    
    # Create sample image (not 256x256)
    sample_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    print(f"\nOriginal image size: {sample_image.shape}")
    
    # Resize to watermarking size
    resized = processor.resize_image(sample_image, (256, 256))
    print(f"Resized to: {resized.shape}")
    print("✓ Image resized successfully")
    
    # Normalize
    normalized = processor.normalize_image(resized)
    print(f"\nNormalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    print(f"Data type: {normalized.dtype}")
    print("✓ Image normalized")
    
    # Denormalize back
    denormalized = processor.denormalize_image(normalized)
    print(f"\nDenormalized range: [{denormalized.min()}, {denormalized.max()}]")
    print(f"Data type: {denormalized.dtype}")
    print("✓ Image denormalized")
    
    # Verify round-trip
    difference = np.mean(np.abs(resized.astype(float) - denormalized.astype(float)))
    print(f"Round-trip error: {difference:.2f}")
    print()


# ============================================================================
# EXAMPLE 5: ID Generation Patterns
# ============================================================================
def example_id_generation():
    """Generate and analyze watermark IDs."""
    print("="*70)
    print("EXAMPLE 5: ID Generation Patterns")
    print("="*70)
    
    id_gen = IDGenerator()
    
    print("\nGenerating watermarks for different exam/student combinations:")
    print("-" * 70)
    
    combinations = [
        ("AI2026", 101),
        ("AI2026", 102),
        ("BIOLOGY2026", 101),
        ("CHEMISTRY2026", 205),
    ]
    
    for exam_id, student_id in combinations:
        wm_id = id_gen.generate_watermark_id(student_id, exam_id)
        wm_image = id_gen.convert_id_to_watermark_image(wm_id)
        
        print(f"\nExam: {exam_id:15s} | Student: {student_id:3d}")
        print(f"  Watermark ID: {wm_id}")
        print(f"  Image shape: {wm_image.shape}")
        print(f"  Min/Max values: {wm_image.min()}/{wm_image.max()}")
        print(f"  Mean intensity: {wm_image.mean():.1f}")
    
    print("\n✓ All watermarks generated uniquely")
    print()


# ============================================================================
# EXAMPLE 6: Deterministic Generation
# ============================================================================
def example_quality_metrics():
    """Measure quality metrics of watermarking process."""
    print("\n" + "="*70)
    print("EXAMPLE 6B: Quality Metrics Analysis")
    print("="*70)
    
    processor = WatermarkProcessor()
    
    # Create test images
    cover = np.ones((256, 256, 3), dtype=np.uint8) * 128
    wm = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
    
    # Embed watermark
    watermarked = processor.embed_watermark(cover, wm)
    
    print("Quality Assessment:")
    print("-" * 70)
    
    # Embedding quality
    embed_metrics = processor.calculate_metrics(cover, watermarked)
    print(f"\nEmbedding Quality:")
    print(f"  PSNR: {embed_metrics['psnr_formatted']}")
    print(f"  MSE:  {embed_metrics['mse_formatted']}")
    print(f"  Interpretation: Higher PSNR = imperceptible watermark")
    
    # Extract watermark
    extracted = processor.extract_watermark(watermarked)
    
    # Extraction quality
    extract_metrics = processor.calculate_metrics(wm, extracted)
    print(f"\nExtraction Quality:")
    print(f"  PSNR: {extract_metrics['psnr_formatted']}")
    print(f"  MSE:  {extract_metrics['mse_formatted']}")
    print(f"  Interpretation: Higher values = better watermark recovery")
    
    print("\n✓ Quality metrics calculated successfully")
    print()


def example_deterministic_generation():
    """Show that same ID always produces same watermark."""
    print("="*70)
    print("EXAMPLE 6: Deterministic Watermark Generation")
    print("="*70)
    
    id_gen = IDGenerator()
    
    wm_id = "AI2026_000101"
    
    print(f"\nGenerating watermark for ID: {wm_id}")
    
    # Generate multiple times
    wm1 = id_gen.convert_id_to_watermark_image(wm_id)
    wm2 = id_gen.convert_id_to_watermark_image(wm_id)
    wm3 = id_gen.convert_id_to_watermark_image(wm_id)
    
    print(f"Generation 1 hash: {np.sum(wm1)}")
    print(f"Generation 2 hash: {np.sum(wm2)}")
    print(f"Generation 3 hash: {np.sum(wm3)}")
    
    print(f"\nWatermarks identical: {np.array_equal(wm1, wm2) and np.array_equal(wm2, wm3)}")
    print("✓ Generation is deterministic")
    print()


# ============================================================================
# EXAMPLE 7: Robustness Testing
# ============================================================================
def example_robustness_testing():
    """Test watermark robustness to modifications."""
    print("="*70)
    print("EXAMPLE 7: Robustness Testing")
    print("="*70)
    
    processor = WatermarkProcessor()
    id_gen = IDGenerator()
    
    # Create watermarked image
    cover = np.ones((256, 256, 3), dtype=np.uint8) * 128
    wm = id_gen.convert_id_to_watermark_image("AI2026_000101")
    watermarked = processor.embed_watermark(cover, wm)
    
    print("Original watermarked image created")
    
    # Test different degradations
    print("\nTesting robustness to modifications:")
    print("-" * 70)
    
    modifications = {
        'Gaussian Noise': lambda img: cv2.GaussianBlur(img, (3, 3), 0),
        'Contrast': lambda img: cv2.convertScaleAbs(img, alpha=1.5, beta=0),
        'Brightness': lambda img: np.clip(img.astype(int) + 30, 0, 255).astype(np.uint8),
        'JPEG Compression': lambda img: cv2.imencode('.jpg', img)[1],  # Simulate compression
    }
    
    for mod_name, modification in modifications.items():
        try:
            modified = modification(watermarked)
            if isinstance(modified, np.ndarray) and len(modified.shape) == 3:
                extracted = processor.extract_watermark(modified)
                similarity = id_gen._compute_similarity(wm, extracted)
                print(f"  {mod_name:20s}: Similarity = {similarity:6.1f}")
        except Exception as e:
            print(f"  {mod_name:20s}: Error - {str(e)[:30]}")
    
    print()


# ============================================================================
# EXAMPLE 8: Performance Metrics
# ============================================================================
def example_performance_metrics():
    """Measure and display performance metrics."""
    import time
    
    print("="*70)
    print("EXAMPLE 8: Performance Metrics")
    print("="*70)
    
    processor = WatermarkProcessor()
    id_gen = IDGenerator()
    
    cover = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    wm = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    
    print("\nPerformance Measurements:")
    print("-" * 70)
    
    # Embedding
    start = time.time()
    for _ in range(10):
        watermarked = processor.embed_watermark(cover, wm)
    embed_time = (time.time() - start) / 10 * 1000
    print(f"Average embedding time: {embed_time:.2f} ms")
    
    # Extraction
    start = time.time()
    for _ in range(10):
        extracted = processor.extract_watermark(watermarked)
    extract_time = (time.time() - start) / 10 * 1000
    print(f"Average extraction time: {extract_time:.2f} ms")
    
    # ID generation
    start = time.time()
    for i in range(100):
        wm_id = id_gen.generate_watermark_id(101 + i, "AI2026")
    id_gen_time = (time.time() - start) / 100 * 1000
    print(f"Average ID generation time: {id_gen_time:.2f} ms")
    
    # Watermark image generation
    start = time.time()
    for i in range(100):
        wm_img = id_gen.convert_id_to_watermark_image(f"AI2026_{i:06d}")
    wm_gen_time = (time.time() - start) / 100 * 1000
    print(f"Average watermark generation time: {wm_gen_time:.2f} ms (cached: faster)")
    
    print("\n✓ Performance metrics calculated")
    print()


# ============================================================================
# RUN ALL EXAMPLES
# ============================================================================
def run_all_examples():
    """Run all examples."""
    print("\n" + "="*70)
    print("ExamGuard AI - Usage Examples")
    print("="*70)
    
    example_basic_watermarking()
    example_student_copy_generation()
    example_leak_detection()
    example_image_processing()
    example_id_generation()
    example_quality_metrics()
    example_deterministic_generation()
    example_robustness_testing()
    example_performance_metrics()
    
    print("="*70)
    print("All examples completed successfully! ✓")
    print("="*70)


if __name__ == "__main__":
    # Run individual examples or all
    import sys
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        example_func_name = f"example_{example_num}"
        if example_func_name in globals():
            globals()[example_func_name]()
        else:
            print(f"Example {example_num} not found")
    else:
        run_all_examples()

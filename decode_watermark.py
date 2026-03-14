"""
Watermark Extraction and Verification Script

Extracts invisible watermarks from watermarked PDFs using the trained decoder model.
"""

import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
from PIL import Image
import io


def load_decoder_model(decoder_path="models/decoder_model.h5"):
    """Load the pre-trained decoder model."""
    if not Path(decoder_path).exists():
        print(f"[!] Decoder model not found at {decoder_path}")
        return None
    
    try:
        decoder = tf.keras.models.load_model(decoder_path)
        print(f"✓ Decoder model loaded from {decoder_path}")
        return decoder
    except Exception as e:
        print(f"[!] Error loading decoder: {str(e)}")
        return None


def extract_patches(image, patch_size=128):
    """
    Extract non-overlapping patches from image.
    
    Args:
        image (np.ndarray): Input image (H, W, 3)
        patch_size (int): Size of patches
    
    Returns:
        list: List of patches and their positions
    """
    patches = []
    h, w = image.shape[:2]
    
    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            patch = image[y:y + patch_size, x:x + patch_size]
            patches.append({
                'data': patch,
                'position': (y, x),
                'size': (patch_size, patch_size)
            })
    
    return patches


def decode_patches(patches, decoder_model):
    """
    Extract watermarks from patches using decoder model.
    
    Args:
        patches (list): List of patch dictionaries
        decoder_model: Loaded decoder model
    
    Returns:
        list: Watermarks extracted from patches
    """
    if decoder_model is None:
        print("[!] No decoder model available")
        return []
    
    watermarks = []
    
    for i, patch_dict in enumerate(patches):
        try:
            patch = patch_dict['data'].astype(np.float32) / 255.0
            
            # Add batch dimension: (1, 128, 128, 3)
            patch_batch = patch[np.newaxis]
            
            # Decode watermark
            decoded = decoder_model.predict(patch_batch, verbose=0)
            watermark = np.squeeze(decoded, axis=0)
            
            # Clip to [0, 1] and convert to uint8
            watermark = np.clip(watermark, 0, 1)
            watermark_uint8 = (watermark * 255).astype(np.uint8)
            
            watermarks.append({
                'watermark': watermark_uint8,
                'position': patch_dict['position'],
                'confidence': np.mean(watermark)  # Average pixel intensity
            })
            
        except Exception as e:
            print(f"[!] Error decoding patch {i}: {str(e)[:60]}")
    
    return watermarks


def pdf_to_images(pdf_path, num_pages=None):
    """
    Convert PDF pages to images.
    
    Args:
        pdf_path (str): Path to PDF file
        num_pages (int): Number of pages to extract (None = all)
    
    Returns:
        list: List of images as numpy arrays
    """
    images = []
    
    try:
        doc = fitz.open(str(pdf_path))
        print(f"✓ PDF opened: {pdf_path}")
        print(f"  Total pages: {len(doc)}")
        
        pages_to_process = min(num_pages, len(doc)) if num_pages else len(doc)
        
        for page_idx in range(pages_to_process):
            page = doc[page_idx]
            # Render page to image (300 DPI for good quality)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom = ~200 DPI
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            
            # Convert RGBA to RGB if needed
            if img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            elif img_array.shape[2] == 1:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            
            images.append(img_array)
            print(f"  Page {page_idx + 1}: {img_array.shape}")
        
        doc.close()
        return images
    
    except Exception as e:
        print(f"[!] Error reading PDF: {e}")
        return []


def save_watermark_visualization(watermarks, output_path, title="Extracted Watermarks"):
    """
    Save watermarks as a grid visualization.
    
    Args:
        watermarks (list): List of extracted watermarks
        output_path (str): Where to save the visualization
        title (str): Title for the visualization
    """
    if not watermarks:
        print("[!] No watermarks to visualize")
        return
    
    # Create grid of watermarks (max 5x5)
    num_wms = min(len(watermarks), 25)
    grid_size = int(np.ceil(np.sqrt(num_wms)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for idx, ax in enumerate(axes.flat):
        if idx < num_wms:
            wm = watermarks[idx]['watermark']
            ax.imshow(wm)
            confidence = watermarks[idx]['confidence']
            ax.set_title(f"Conf: {confidence:.1%}", fontsize=8)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"✓ Visualization saved to {output_path}")
    plt.close()


def analyze_watermarks(watermarks):
    """
    Analyze extracted watermarks for verification.
    
    Args:
        watermarks (list): List of extracted watermarks
    """
    if not watermarks:
        print("[!] No watermarks to analyze")
        return
    
    print("\n" + "="*70)
    print("WATERMARK EXTRACTION ANALYSIS")
    print("="*70)
    print(f"Total watermarks extracted: {len(watermarks)}")
    
    if watermarks:
        confidences = [w['confidence'] for w in watermarks]
        print(f"Confidence scores:")
        print(f"  Min: {np.min(confidences):.2f}")
        print(f"  Max: {np.max(confidences):.2f}")
        print(f"  Mean: {np.mean(confidences):.2f}")
        print(f"  Std: {np.std(confidences):.2f}")
        
        # Show top 3 watermarks with highest confidence
        sorted_wms = sorted(watermarks, key=lambda x: x['confidence'], reverse=True)
        print(f"\nTop 3 clearest watermarks:")
        for i, wm in enumerate(sorted_wms[:3]):
            print(f"  {i+1}. Position: {wm['position']}, Confidence: {wm['confidence']:.2f}")
    
    print("="*70)


def save_watermark_samples(watermarks, output_dir, num_samples=5):
    """
    Save individual watermark samples.
    
    Args:
        watermarks (list): List of extracted watermarks
        output_dir (str): Output directory
        num_samples (int): Number of samples to save
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save top samples by confidence
    sorted_wms = sorted(watermarks, key=lambda x: x['confidence'], reverse=True)
    
    for i, wm_dict in enumerate(sorted_wms[:num_samples]):
        wm = wm_dict['watermark']
        conf = wm_dict['confidence']
        
        filename = f"watermark_{i+1}_conf_{conf:.2f}.png"
        filepath = output_path / filename
        
        img_pil = Image.fromarray(wm)
        img_pil.save(str(filepath))
        print(f"  ✓ Saved: {filename}")


def main():
    """Main execution for watermark decoding."""
    
    # Configuration
    pdf_path = "watermarked_papers/QP_2310080008.pdf"
    decoder_path = "models/decoder_model.h5"
    output_dir = "decoded_watermarks"
    
    print("\n" + "="*70)
    print("WATERMARK DECODER - EXTRACTION & VERIFICATION")
    print("="*70)
    print(f"PDF: {pdf_path}")
    print(f"Decoder: {decoder_path}")
    print("="*70 + "\n")
    
    # Step 1: Load decoder model
    print("[1/5] Loading decoder model...")
    decoder = load_decoder_model(decoder_path)
    if decoder is None:
        print("[!] Cannot proceed without decoder model")
        return
    
    # Step 2: Extract images from PDF
    print("\n[2/5] Extracting images from PDF...")
    images = pdf_to_images(pdf_path, num_pages=1)  # Process first page
    
    if not images:
        print("[!] No images extracted from PDF")
        return
    
    # Step 3: Extract patches from images
    print("\n[3/5] Extracting patches from images...")
    all_patches = []
    for img_idx, image in enumerate(images):
        patches = extract_patches(image, patch_size=128)
        all_patches.extend(patches)
        print(f"  Page {img_idx + 1}: Extracted {len(patches)} patches")
    
    print(f"  Total patches: {len(all_patches)}")
    
    # Step 4: Decode watermarks from patches
    print("\n[4/5] Decoding watermarks from patches...")
    watermarks = decode_patches(all_patches, decoder)
    print(f"  Successfully decoded: {len(watermarks)} watermarks")
    
    # Step 5: Analyze and save results
    print("\n[5/5] Analyzing and saving results...")
    analyze_watermarks(watermarks)
    
    # Save visualizations
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"\nGenerating visualizations...")
    viz_path = Path(output_dir) / "watermark_grid.png"
    save_watermark_visualization(watermarks, str(viz_path), "Extracted Watermarks from QP_2310080008")
    
    print(f"\nSaving watermark samples...")
    save_watermark_samples(watermarks, output_dir, num_samples=5)
    
    # Save detailed report
    report_path = Path(output_dir) / "extraction_report.txt"
    with open(report_path, 'w') as f:
        f.write("WATERMARK EXTRACTION REPORT\n")
        f.write("="*70 + "\n")
        f.write(f"PDF File: {pdf_path}\n")
        f.write(f"Decoder Model: {decoder_path}\n")
        f.write(f"Total Watermarks Extracted: {len(watermarks)}\n")
        f.write(f"Number of Pages Processed: {len(images)}\n")
        f.write(f"Number of Patches Analyzed: {len(all_patches)}\n")
        f.write("\n" + "="*70 + "\n")
        f.write("WATERMARKS:\n")
        f.write("-"*70 + "\n")
        
        for i, wm in enumerate(sorted(watermarks, key=lambda x: x['confidence'], reverse=True)):
            f.write(f"\n[{i+1}] Position: {wm['position']}\n")
            f.write(f"    Confidence: {wm['confidence']:.4f}\n")
    
    print(f"\n✓ Report saved to {report_path}")
    
    print("\n" + "="*70)
    print("DECODING COMPLETE!")
    print("="*70)
    print(f"Output Directory: {output_dir}/")
    print("Files generated:")
    print("  - watermark_grid.png (visualization of all watermarks)")
    print("  - watermark_1_conf_*.png (individual samples)")
    print("  - extraction_report.txt (detailed report)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

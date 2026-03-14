import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
from PIL import Image
import fitz
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------

encoder = tf.keras.models.load_model("models/encoder_model.h5")
embedder = tf.keras.models.load_model("models/embedder_model.h5")

PATCH = 128


# -------------------------------------------------
# WATERMARK GENERATION
# -------------------------------------------------

def generate_watermark(roll_number):

    wm = np.zeros((32,32,3),dtype=np.uint8)

    text = str(roll_number)[-4:]

    cv2.putText(
        wm,
        text,
        (2,24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        1,
        cv2.LINE_AA
    )

    return wm.astype(np.float32)/255.0


# -------------------------------------------------
# EMBED WATERMARK INTO PATCH
# -------------------------------------------------

def embed_patch(patch, encoded_wm):

    patch = patch.astype(np.float32)/255.0

    stego = embedder.predict(
        [patch[np.newaxis,...], encoded_wm],
        verbose=0
    )[0]

    stego = np.clip(stego*255,0,255).astype(np.uint8)

    return stego


# -------------------------------------------------
# WATERMARK PDF
# -------------------------------------------------

def watermark_pdf(pdf_path, roll_number):
    """Watermark a PDF using encoder + embedder"""
    
    # Convert PDF to images using fitz
    doc = fitz.open(str(pdf_path))
    pages = []
    
    for page in doc:
        # Render at higher resolution for better quality
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append(img)
    
    doc.close()
    
    watermark = generate_watermark(roll_number)
    
    # encode watermark
    encoded_wm = encoder.predict(watermark[np.newaxis,...], verbose=0)
    
    output_pages = []
    patches_processed = 0
    
    for page in pages:
        img = np.array(page)
        h, w = img.shape[:2]
        result = img.copy()
        
        for y in range(0, h - PATCH, PATCH):
            for x in range(0, w - PATCH, PATCH):
                patch = img[y:y+PATCH, x:x+PATCH]
                
                # Skip incomplete patches
                if patch.shape != (PATCH, PATCH, 3):
                    continue
                
                stego = embed_patch(patch, encoded_wm)
                result[y:y+PATCH, x:x+PATCH] = stego
                patches_processed += 1
        
        output_pages.append(Image.fromarray(result))
    
    # Save to output directory
    output_dir = Path("watermarked_papers_5_final")
    output_dir.mkdir(exist_ok=True)
    
    output_name = output_dir / f"QP_{roll_number:010d}.pdf"
    
    output_pages[0].save(
        str(output_name),
        save_all=True,
        append_images=output_pages[1:] if len(output_pages) > 1 else []
    )
    
    size_mb = output_name.stat().st_size / (1024 * 1024)
    print(f"Saved: {output_name.name} ({patches_processed} patches, {size_mb:.2f} MB)")


# -------------------------------------------------
# RUN
# -------------------------------------------------

print("\n" + "="*70)
print("PROPER WATERMARKING: Encoder + Embedder Pipeline")
print("="*70)
print("Using PyMuPDF (fitz) for PDF processing")
print("="*70)

source_pdf = r"S:\ExamGuard_AI\MP_QP_Insem1_set3.pdf"
roll_numbers = [2310080001, 2310080002, 2310080003, 2310080004, 2310080005]

print(f"Source: {Path(source_pdf).name}")
print(f"Rolls: {roll_numbers}\n")

success = 0
for roll in roll_numbers:
    try:
        watermark_pdf(source_pdf, roll)
        success += 1
    except Exception as e:
        print(f"Error for roll {roll}: {str(e)[:80]}")

print("\n" + "="*70)
print(f"COMPLETE: {success}/{len(roll_numbers)} watermarked")
print("Output: watermarked_papers_5_final/")
print("="*70)
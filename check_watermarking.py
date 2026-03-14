"""Check if watermarking was actually applied"""
from pathlib import Path
import cv2
import numpy as np

print('COMPARING ORIGINAL vs WATERMARKED:')
print('='*70)

original = cv2.imread(str(Path('generated_exams_dl/Exam_2310080001_DL.png')))
watermarked = cv2.imread(str(Path('watermarked_papers_5/WM_2310080001.png')))

if original is not None and watermarked is not None:
    print(f'Original shape: {original.shape}')
    print(f'Watermarked shape: {watermarked.shape}')
    
    # Resize to match if different
    if original.shape != watermarked.shape:
        print(f'\n[ISSUE] Shapes dont match!')
    else:
        # Calculate difference
        diff = cv2.absdiff(original.astype(np.float32), watermarked.astype(np.float32))
        diff_mean = np.mean(diff)
        diff_max = np.max(diff)
        
        print(f'\nDifference analysis:')
        print(f'  Mean pixel difference: {diff_mean:.2f}')
        print(f'  Max pixel difference: {diff_max:.2f}')
        
        applied_text = "YES (watermark embedded)" if diff_mean > 0.1 else "NO (identical to original)"
        print(f'  Watermark applied: {applied_text}')
        
        # Show some pixel values
        print(f'\nRandom sample pixels from first patch area (0:128, 0:128):')
        orig_patch = original[0:10, 0:10].astype(np.float32)
        wm_patch = watermarked[0:10, 0:10].astype(np.float32)
        
        print(f'  Original patch mean: {np.mean(orig_patch):.2f}')
        print(f'  Watermarked patch mean: {np.mean(wm_patch):.2f}')
        print(f'  Patch difference: {np.mean(np.abs(orig_patch - wm_patch)):.4f}')
else:
    print('ERROR: Could not load images')

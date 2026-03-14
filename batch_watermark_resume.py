#!/usr/bin/env python3
"""
Batch Watermark Exam Papers - Resume Capable
Generates watermarked question papers for all students, can resume interrupted runs.
"""

import sys
from pathlib import Path
import warnings

sys.path.insert(0, str(Path(__file__).parent))

from watermark_pdf_batch import PDFExamWatermarker

warnings.filterwarnings('ignore')


def main():
    """Generate watermarked papers with resume capability"""
    pdf_path = Path(__file__).parent / "MP_QP_Insem1_set3.pdf"
    output_dir = Path(__file__).parent / "watermarked_papers"
    
    start_roll = 2310080001
    end_roll = 2310080088
    exam_id = "AI2026"
    
    # Check which papers already exist
    output_dir.mkdir(exist_ok=True)
    existing_papers = set()
    for pdf in output_dir.glob("QP_*.pdf"):
        try:
            roll = int(pdf.stem.split("_")[1])
            existing_papers.add(roll)
        except:
            pass
    
    # Find which ones still need to be generated
    all_rolls = set(range(start_roll, end_roll + 1))
    remaining_rolls = sorted(all_rolls - existing_papers)
    
    if not remaining_rolls:
        print("\n" + "=" * 70)
        print("All 88 papers already generated!")
        print("=" * 70)
        print(f"Location: {output_dir.absolute()}")
        return
    
    print("\n" + "=" * 70)
    print("Batch Watermark Generation - Resume Mode")
    print("=" * 70)
    print(f"Total students: 88 ({start_roll:010d} - {end_roll:010d})")
    print(f"Already generated: {len(existing_papers)}")
    print(f"Remaining: {len(remaining_rolls)}")
    print("=" * 70)
    
    try:
        # Create watermarker
        watermarker = PDFExamWatermarker(pdf_path)
        
        successful = 0
        failed = 0
        
        for idx, roll_number in enumerate(remaining_rolls, 1):
            try:
                total_remaining = len(remaining_rolls)
                print(f"[{idx:3d}/{total_remaining}] Roll {roll_number:010d}...", end=" ", flush=True)
                
                # Embed watermark on all pages
                watermarked_images = watermarker.embed_watermark_on_pages(roll_number, exam_id)
                
                # Save as PDF
                output_filename = f"QP_{roll_number:010d}.pdf"
                output_path = output_dir / output_filename
                
                if watermarker.save_as_pdf(watermarked_images, output_path):
                    print("OK")
                    successful += 1
                else:
                    print("SAVE_FAILED")
                    failed += 1
                    
            except KeyboardInterrupt:
                print("\n\n[INFO] Interrupted by user")
                print(f"[INFO] Generated so far in this run: {successful}")
                print(f"[INFO] To resume, run: python batch_watermark_resume.py")
                sys.exit(0)
            except Exception as e:
                print(f"ERROR: {str(e)[:30]}")
                failed += 1
        
        # Summary
        total_generated = len(existing_papers) + successful
        print("\n" + "=" * 70)
        print(f"Batch Generation Complete!")
        print(f"  Total generated: {total_generated}/88")
        print(f"  This run: {successful} OK, {failed} failed")
        if total_generated == 88:
            print(f"  Status: ALL PAPERS READY")
        print(f"  Location: {output_dir.absolute()}")
        print("=" * 70 + "\n")
        
        return {"total": total_generated, "successful": successful, "failed": failed}
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

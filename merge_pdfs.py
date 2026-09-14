"""
merge_pdfs.py  ──  NCWP 실험 결과 PNG/PDF 전체 병합 유틸리티

NCWP 디렉토리 아래 실험 스크립트들이 생성한 모든 PNG / PDF를
하나의 논문 제출용 PDF 파일로 합칩니다.

  - PNG 파일: matplotlib로 각 이미지를 한 페이지씩 삽입 (벡터 품질)
  - PDF 파일: pypdf로 무손실 병합 (pypdf 설치 시)

사용법:
  # 기본 (NCWP 디렉토리를 자동 탐색, 현재 디렉토리에 paper_figures.pdf 생성)
  python3 merge_pdfs.py

  # 출력 파일 경로 지정
  python3 merge_pdfs.py --output ./paper_figures.pdf

  # PNG만 포함 (기존 결과)
  python3 merge_pdfs.py --png-only

  # 특정 키워드 파일만 포함
  python3 merge_pdfs.py --include main ablation
"""

import argparse
import os
import sys
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

try:
    from pypdf import PdfWriter, PdfReader
    _PDF_BACKEND = "pypdf"
except ImportError:
    try:
        from PyPDF2 import PdfWriter, PdfReader  # type: ignore
        _PDF_BACKEND = "PyPDF2"
    except ImportError:
        _PDF_BACKEND = None


def find_images(root: str, include_png: bool = True, include_pdf: bool = True) -> List[Tuple[str, str]]:
    """root 아래 PNG/PDF 파일을 재귀 탐색한다.

    Returns: List of (filepath, ext) tuples, sorted by path.
    """
    found = []
    for dirpath, _, filenames in os.walk(root):
        for fname in sorted(filenames):
            ext = os.path.splitext(fname)[1].lower()
            if include_png and ext == ".png":
                found.append((os.path.join(dirpath, fname), "png"))
            elif include_pdf and ext == ".pdf":
                found.append((os.path.join(dirpath, fname), "pdf"))
    return found


def png_to_pdf_page(pdf: PdfPages, png_path: str) -> None:
    """PNG 파일 한 장을 PdfPages의 한 페이지로 삽입한다."""
    img = mpimg.imread(png_path)
    h, w = img.shape[:2]
    # A4 비율 유지, 최대 12인치
    scale = min(12.0 / (w / 100), 9.0 / (h / 100), 1.0)
    fig, ax = plt.subplots(figsize=(w / 100 * scale, h / 100 * scale))
    ax.imshow(img, interpolation="lanczos")
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    pdf.savefig(fig, dpi=150, bbox_inches="tight")
    plt.close(fig)


def merge_all(files: List[Tuple[str, str]], output: str) -> None:
    """PNG/PDF 파일을 하나의 PDF로 병합한다."""
    # PDF 파일 목록 (무손실 병합용)
    pdf_files = [p for p, ext in files if ext == "pdf"]
    png_files = [p for p, ext in files if ext == "png"]

    if _PDF_BACKEND and pdf_files:
        # pdf 파일이 있으면 무손실로 먼저 병합 후 PNG를 덧붙임
        _merge_mixed(png_files, pdf_files, output)
    else:
        # PNG만 있거나 pypdf 없는 경우: matplotlib만으로 처리
        _merge_png_only(png_files + [p for p, _ in files if p.endswith(".pdf")], output)


def _merge_mixed(png_files: List[str], pdf_files: List[str], output: str) -> None:
    """PNG는 matplotlib, PDF는 pypdf로 합쳐서 하나의 PDF 생성."""
    # 1) PNG를 임시 PDF로 변환
    tmp_pdf = output + ".tmp_png.pdf"
    if png_files:
        print(f"  PNG {len(png_files)}개 → 임시 PDF 변환 중...")
        with PdfPages(tmp_pdf) as pdf:
            for i, png in enumerate(png_files, 1):
                rel = os.path.relpath(png)
                print(f"    [{i}/{len(png_files)}] {rel}")
                try:
                    png_to_pdf_page(pdf, png)
                except Exception as e:
                    print(f"      [경고] 건너뜀: {e}")
        pdf_files_all = [tmp_pdf] + pdf_files
    else:
        pdf_files_all = pdf_files

    # 2) 모든 PDF 무손실 병합
    print(f"  PDF {len(pdf_files_all)}개 병합 중 ({_PDF_BACKEND})...")
    writer = PdfWriter()
    for path in pdf_files_all:
        try:
            reader = PdfReader(path)
            for page in reader.pages:
                writer.add_page(page)
        except Exception as e:
            print(f"    [경고] 건너뜀: {path}  ({e})")
    with open(output, "wb") as f:
        writer.write(f)

    if png_files and os.path.exists(tmp_pdf):
        os.remove(tmp_pdf)


def _merge_png_only(files: List[str], output: str) -> None:
    """PNG 파일을 matplotlib PdfPages로만 병합한다."""
    print(f"  이미지 {len(files)}개 → PDF 변환 중...")
    with PdfPages(output) as pdf:
        for i, path in enumerate(files, 1):
            rel = os.path.relpath(path)
            print(f"    [{i}/{len(files)}] {rel}")
            try:
                png_to_pdf_page(pdf, path)
            except Exception as e:
                print(f"      [경고] 건너뜀: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NCWP 실험 결과 PNG/PDF를 하나의 논문용 PDF로 병합",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root", "-r",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="탐색할 루트 디렉토리 (기본: 이 스크립트 위치)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="출력 PDF 경로 (기본: <root>/paper_figures.pdf)",
    )
    parser.add_argument(
        "--include", "-i",
        nargs="*",
        help="포함할 파일명 키워드 (예: --include main ablation timing)",
    )
    parser.add_argument(
        "--exclude", "-e",
        nargs="*",
        default=["paper_figures", "all_figures_merged"],
        help="제외할 파일명 키워드 (기본: paper_figures all_figures_merged)",
    )
    parser.add_argument(
        "--png-only", action="store_true",
        help="PNG 파일만 수집 (PDF 파일 제외)",
    )
    parser.add_argument(
        "--pdf-only", action="store_true",
        help="PDF 파일만 수집 (PNG 파일 제외)",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    output = args.output or os.path.join(root, "paper_figures.pdf")
    output = os.path.abspath(output)

    include_png = not args.pdf_only
    include_pdf = not args.png_only

    print(f"탐색 경로: {root}")
    print(f"수집 대상: {'PNG' if include_png else ''} {'PDF' if include_pdf else ''}")

    all_files = find_images(root, include_png=include_png, include_pdf=include_pdf)

    # 출력 파일 자체 제외
    all_files = [(p, ext) for p, ext in all_files if os.path.abspath(p) != output]

    # 키워드 필터
    if args.include:
        all_files = [(p, ext) for p, ext in all_files
                     if any(k in os.path.basename(p) for k in args.include)]
    if args.exclude:
        all_files = [(p, ext) for p, ext in all_files
                     if not any(k in os.path.basename(p) for k in args.exclude)]

    if not all_files:
        print("\n병합할 파일을 찾지 못했습니다.")
        print("--root 옵션으로 올바른 경로를 지정했는지 확인하세요.")
        return

    print(f"\n병합 대상 ({len(all_files)}개):")
    for p, ext in all_files:
        rel = os.path.relpath(p, root)
        print(f"  [{ext.upper()}] {rel}")

    print(f"\n출력: {output}")
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    merge_all(all_files, output)

    size_mb = os.path.getsize(output) / 1024 / 1024
    print(f"\n완료: {output}  ({size_mb:.1f} MB)")
    print("  ※ 논문 제출 전 PDF 뷰어로 내용을 확인하세요.")


if __name__ == "__main__":
    main()

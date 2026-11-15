"""
Module: scripts.generate_sample_pdf
===================================
Generates a valid, multi-page sample PDF file in the `data/` directory using
pure Python (raw PDF 1.4 specification).

This script allows immediate testing of PDF ingestion without requiring
external binary downloads or heavy PDF authoring dependencies.
"""

from pathlib import Path


def create_sample_pdf(output_path: Path) -> Path:
    """Creates a two-page valid PDF document for testing the ingestion pipeline.

    Parameters
    ----------
    output_path : Path
        Target destination where the generated PDF will be saved.

    Returns
    -------
    Path
        The absolute path to the generated PDF file.
    """
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Page 1 Text Stream
    text_page_1 = (
        "BT\n"
        "/F1 20 Tf\n"
        "50 750 Td\n"
        "(Sample Educational Lecture Notes - Page 1) Tj\n"
        "/F1 12 Tf\n"
        "0 -30 Td\n"
        "(Retrieval-Augmented Generation with LlamaIndex and Weaviate) Tj\n"
        "0 -20 Td\n"
        "(LlamaIndex provides robust data connectors, node parsers, and vector store indices.) Tj\n"
        "0 -20 Td\n"
        "(Weaviate stores high-dimensional embeddings and executes sub-millisecond vector search.) Tj\n"
        "0 -20 Td\n"
        "(Chunk size of 800 tokens and overlap of 100 tokens maintain coherent context boundaries.) Tj\n"
        "ET"
    )

    # Page 2 Text Stream
    text_page_2 = (
        "BT\n"
        "/F1 20 Tf\n"
        "50 750 Td\n"
        "(Sample Educational Lecture Notes - Page 2) Tj\n"
        "/F1 12 Tf\n"
        "0 -30 Td\n"
        "(Dense Embeddings via BAAI/bge-small-en-v1.5) Tj\n"
        "0 -20 Td\n"
        "(The BGE small embedding model converts unstructured sentences into 384-dimensional vectors.) Tj\n"
        "0 -20 Td\n"
        "(These dense vector representations capture semantic intent beyond keyword matching.) Tj\n"
        "0 -20 Td\n"
        "(PostgreSQL complements Weaviate by handling structured metadata, relational schemas, and audit logs.) Tj\n"
        "ET"
    )

    # Build raw PDF objects
    stream1_bytes = text_page_1.encode("latin-1")
    stream2_bytes = text_page_2.encode("latin-1")

    # Assemble PDF 1.4 body
    body = (
        b"%PDF-1.4\n"
        b"%----\n"
        # Object 1: Catalog
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        # Object 2: Pages container
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>\nendobj\n"
        # Object 3: Page 1
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 5 0 R /Resources << /Font << /F1 7 0 R >> >> >>\nendobj\n"
        # Object 4: Page 2
        b"4 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 6 0 R /Resources << /Font << /F1 7 0 R >> >> >>\nendobj\n"
        # Object 5: Content stream for Page 1
        b"5 0 obj\n<< /Length " + str(len(stream1_bytes)).encode("ascii") + b" >>\nstream\n" + stream1_bytes + b"\nendstream\nendobj\n"
        # Object 6: Content stream for Page 2
        b"6 0 obj\n<< /Length " + str(len(stream2_bytes)).encode("ascii") + b" >>\nstream\n" + stream2_bytes + b"\nendstream\nendobj\n"
        # Object 7: Standard Font dictionary
        b"7 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    )

    # Compute cross-reference table offsets
    offsets = [0]
    for obj_id in range(1, 8):
        pos = body.find(f"{obj_id} 0 obj".encode("ascii"))
        offsets.append(pos)

    # Build xref section
    xref = b"xref\n0 8\n0000000000 65535 f \n"
    for pos in offsets[1:]:
        xref += f"{pos:010d} 00000 n \n".encode("ascii")

    # Trailer
    trailer = (
        b"trailer\n"
        b"<< /Size 8 /Root 1 0 R >>\n"
        b"startxref\n"
        + str(len(body)).encode("ascii")
        + b"\n%%EOF\n"
    )

    full_pdf_content = body + xref + trailer

    # Write file to output location
    with open(output_path, "wb") as pdf_file:
        pdf_file.write(full_pdf_content)

    print(f"[OK] Generated sample PDF document at: {output_path.resolve()}")
    return output_path


if __name__ == "__main__":
    # Resolve default data/sample.pdf relative to this script's directory
    project_root = Path(__file__).resolve().parent.parent
    sample_pdf_target = project_root / "data" / "sample.pdf"
    create_sample_pdf(sample_pdf_target)

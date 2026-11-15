"""
Helper to populate additional educational samples (PDFs and HTML) for multi-document testing.
"""
from pathlib import Path


def create_pdf(output_path: Path, title: str, p1_heading: str, p1_bullets: list, p2_heading: str, p2_bullets: list):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def format_page(heading, bullets):
        lines = [
            "BT",
            "/F1 20 Tf",
            "50 750 Td",
            f"({heading}) Tj",
            "/F1 12 Tf",
            "0 -35 Td",
        ]
        for bullet in bullets:
            clean = bullet.replace("(", "\\(").replace(")", "\\)")
            lines.append(f"({clean}) Tj")
            lines.append("0 -25 Td")
        lines.append("ET")
        return "\n".join(lines)

    p1_text = format_page(p1_heading, p1_bullets)
    p2_text = format_page(p2_heading, p2_bullets)

    s1 = p1_text.encode("latin-1")
    s2 = p2_text.encode("latin-1")

    body = (
        b"%PDF-1.4\n"
        b"%----\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 5 0 R /Resources << /Font << /F1 7 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 6 0 R /Resources << /Font << /F1 7 0 R >> >> >>\nendobj\n"
        b"5 0 obj\n<< /Length " + str(len(s1)).encode("ascii") + b" >>\nstream\n" + s1 + b"\nendstream\nendobj\n"
        b"6 0 obj\n<< /Length " + str(len(s2)).encode("ascii") + b" >>\nstream\n" + s2 + b"\nendstream\nendobj\n"
        b"7 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    )

    offsets = [0]
    for obj_id in range(1, 8):
        pos = body.find(f"{obj_id} 0 obj".encode("ascii"))
        offsets.append(pos)

    xref = b"xref\n0 8\n0000000000 65535 f \n"
    for pos in offsets[1:]:
        xref += f"{pos:010d} 00000 n \n".encode("ascii")

    trailer = (
        b"trailer\n"
        b"<< /Size 8 /Root 1 0 R >>\n"
        b"startxref\n"
        + str(len(body)).encode("ascii")
        + b"\n%%EOF\n"
    )

    with open(output_path, "wb") as f:
        f.write(body + xref + trailer)
    print(f"[OK] Created {output_path.name}")


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / "data"

    create_pdf(
        data_dir / "sample_calculus.pdf",
        "Calculus 101 Notes",
        "Differential Calculus Fundamentals - Page 1",
        [
            "Calculus is the mathematical study of continuous change and accumulation.",
            "The derivative measures the instantaneous rate of change of a function.",
            "Limits formalize the behavior of functions near specific points.",
            "Product Rule and Quotient Rule govern derivatives of composite functions.",
        ],
        "Integral Calculus and Applications - Page 2",
        [
            "The Fundamental Theorem of Calculus links differentiation and integration.",
            "Definite integrals compute the signed area beneath curves.",
            "Integration by parts mirrors the product rule for differentiation.",
            "Applications include physics kinematics, probability distributions, and engineering.",
        ],
    )

    create_pdf(
        data_dir / "sample_physics.pdf",
        "Classical Mechanics Notes",
        "Newtonian Mechanics Principles - Page 1",
        [
            "Newton's First Law: An object remains at rest unless acted upon by an external force.",
            "Newton's Second Law: Force equals the time derivative of momentum F = ma.",
            "Newton's Third Law: For every action there is an equal and opposite reaction.",
            "Inertial reference frames preserve the invariance of Newtonian dynamics.",
        ],
        "Conservation Laws and Thermodynamics - Page 2",
        [
            "Conservation of Energy: Total mechanical energy remains constant in isolated systems.",
            "Conservation of Linear and Angular Momentum governs orbital and collision mechanics.",
            "First Law of Thermodynamics dictates internal energy changes from heat and work.",
            "Second Law of Thermodynamics establishes that total entropy never decreases over time.",
        ],
    )

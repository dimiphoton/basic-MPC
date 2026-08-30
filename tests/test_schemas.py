"""Schémas de publication : fichiers PNG/PDF non vides."""

from pathlib import Path

from basic_mpc.figures.schemas import run_draw_schemas


def test_draw_schemas_ecrit_png_et_pdf(tmp_path: Path) -> None:
    """Cinq planches, raster + vectoriel."""
    ecrits = run_draw_schemas(tmp_path)
    attendus = (
        "schema-r1c1",
        "schema-r2c2",
        "schema-plant",
        "schema-kalman",
        "schema-famille-rc",
    )
    assert set(ecrits) == set(attendus)
    for stem in attendus:
        png = tmp_path / f"{stem}.png"
        pdf = tmp_path / f"{stem}.pdf"
        assert png.is_file()
        assert pdf.is_file()
        assert png.stat().st_size > 1000
        assert pdf.stat().st_size > 500

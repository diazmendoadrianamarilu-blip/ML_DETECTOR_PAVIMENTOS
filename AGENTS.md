# Workspace Guidance

## Project Layout

- `Heritage_Detector/` is the Python application source and its Git repository.
- `02_Evidencia_y_Datos/` contains drone samples, evaluation evidence, reports, and generated detection results. Preserve source evidence and data unless a task explicitly requires changing it.
- `Entrega_Heritage_Detector/` contains the packaged Windows app and delivery instructions; it is separate from the Python source.
- The application has a Streamlit entry point (`Heritage_Detector/app.py`) and a CustomTkinter desktop entry point (`Heritage_Detector/desktop_app.py`).

## Development

- Keep changes focused on the requested behavior and follow the existing Python style. User-facing text and documentation are primarily in Spanish.
- The model is `Heritage_Detector/best.pt`. Detection thresholds, class names, and image-processing behavior are domain-sensitive; do not change them without a task-specific reason.
- Install dependencies from the source directory with `python -m pip install -r REQUIREMENTS.TXT`.
- Run the web app from `Heritage_Detector/` with `python -m streamlit run app.py`; run the desktop app with `python desktop_app.py`.
- Supabase configuration comes from `SUPABASE_URL` and `SUPABASE_KEY` in a local `.env`. Never hardcode credentials, expose secret files, or copy real credentials into examples or logs. `.env` is ignored by Git; use `.env.example` for placeholders.
- No automated test suite is currently evident. For Python edits, at minimum run `python -m compileall Heritage_Detector`; manually exercise the affected app flow when practical.
- Changes to `config_storage_historial.sql` affect the remote database schema and must remain coordinated with application code.

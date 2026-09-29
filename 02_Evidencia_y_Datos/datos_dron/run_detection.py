# -*- coding: utf-8 -*-
"""Corre best.pt sobre la muestra descargada usando el pipeline de app.py
(tiling 640, overlap 20%, filtros anti-falsos positivos, agrupacion de zonas)."""
import ast
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

APP_PY = Path(r"C:\Users\niurka\Downloads\sillar\app.py")
MODELO = Path(r"C:\Users\niurka\Downloads\sillar\best.pt")
MUESTRA = Path(r"C:\Users\niurka\Downloads\sillar\datos_dron\muestra")
SALIDA = Path(r"C:\Users\niurka\Documents\HeritageDetector\resultados\dron_test")

UMBRALES = {"crack": 0.20, "humidity": 0.30, "spalling": 0.60}
IOU = 0.45
GSD = 0.13  # cm/px (default de la app; para dron solo es estimativo)
TILE = 640
OVERLAP = 0.2


def extraer_pipeline():
    """Extrae funciones puras de app.py (sin ejecutar Streamlit)."""
    src = APP_PY.read_text(encoding="utf-8")
    tree = ast.parse(src)
    funciones = {"filtrar_falsos_positivos", "agrupar_zonas",
                 "dibujar_cajas_y_segmentos", "split_image_into_tiles",
                 "nms_boxes", "procesar_con_tiling"}
    codigo = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in funciones:
            codigo.append(ast.get_source_segment(src, node))
    ns = {"np": np, "cv2": cv2, "Image": Image, "torch": torch, "time": time}
    for c in codigo:
        exec(compile(c, str(APP_PY), "exec"), ns)
    return ns


def main():
    ns = extraer_pipeline()
    procesar = ns["procesar_con_tiling"]

    model = YOLO(str(MODELO))
    print("Modelo cargado:", MODELO.name, "| clases:", model.names)

    imgs = sorted(p for p in MUESTRA.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    print(f"Imagenes a procesar: {len(imgs)}")
    SALIDA.mkdir(parents=True, exist_ok=True)

    pendientes = []
    for p in imgs:
        rel = p.relative_to(MUESTRA)
        if (SALIDA / rel.parent / f"{rel.stem}.json").exists() and \
           (SALIDA / rel.parent / f"{rel.stem}_anotada.jpg").exists():
            continue
        pendientes.append(p)
    if pendientes:
        print(f"Pendientes (no procesadas aun): {len(pendientes)}")
        for p in pendientes:
            print("  -", p.relative_to(MUESTRA))
    imgs = pendientes
    if not imgs:
        print("Ya estan todas procesadas. El resumen se regenera abajo.")
    else:
        print(f"Imagenes a procesar ahora: {len(imgs)}")

    resumen = {"modelo": str(MODELO), "umbrales": UMBRALES, "iou": IOU,
               "gsd_cm_px": GSD, "tile": TILE, "overlap": OVERLAP,
               "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
               "imagenes": []}
    totals = {"crack": 0, "humidity": 0, "spalling": 0, "imgs_con_dano": 0}
    t_total = time.time()

    for i, imgp in enumerate(imgs, 1):
        t0 = time.time()
        image = Image.open(imgp).convert("RGB")
        res = procesar(model, image, UMBRALES, IOU, GSD, TILE, OVERLAP)

        rel = imgp.relative_to(MUESTRA)
        base = rel.stem
        anotadas_dir = SALIDA / rel.parent
        anotadas_dir.mkdir(parents=True, exist_ok=True)
        ruta_anotada = anotadas_dir / f"{base}_anotada.jpg"
        ruta_json = anotadas_dir / f"{base}.json"
        try:
            res["annotated"].save(ruta_anotada, quality=92)
        except Exception as e:
            print(f"  AVISO: no pude guardar {ruta_anotada}: {e}")

        dets = res["detections"]
        cc = res["class_counts"]
        entry = {
            "archivo": imgp.name, "grupo": str(rel.parent),
            "tamano_px": image.size, "n_detections": res["n_detections"],
            "class_counts": cc, "total_area_cm2": res["total_area_cm2"],
            "total_area_m2": res["total_area_m2"],
            "avg_confidence": res.get("avg_confidence", 0),
            "detections": dets, "tiempo_s": round(time.time() - t0, 2),
            "anotada": str(ruta_anotada),
        }
        resumen["imagenes"].append(entry)
        with open(ruta_json, "w", encoding="utf-8") as fh:
            json.dump(entry, fh, ensure_ascii=False, indent=1)

        for k in totals:
            if k != "imgs_con_dano":
                totals[k] += cc.get(k, 0)
        if res["n_detections"] > 0:
            totals["imgs_con_dano"] += 1
        print(f"[{i}/{len(imgs)}] {rel} | det={res['n_detections']} "
              f"crack={cc['crack']} hum={cc['humidity']} spal={cc['spalling']} "
              f"| {time.time()-t0:.1f}s")

    resumen["totales"] = totals
    resumen["tiempo_total_s"] = round(time.time() - t_total, 1)

    # Reconstruir resumen global desde TODOS los JSON de cada imagen
    todos = []
    for jp in sorted(SALIDA.rglob("*.json")):
        if jp.name == "_resumen.json":
            continue
        try:
            with open(jp, encoding="utf-8") as fh:
                todos.append(json.load(fh))
        except Exception:
            continue
    if todos:
        g_totals = {k: 0 for k in ["crack", "humidity", "spalling"]}
        g_imgs = sum(1 for e in todos if e["n_detections"] > 0)
        for e in todos:
            for k in g_totals:
                g_totals[k] += e["class_counts"].get(k, 0)
        resumen["imagenes"] = todos
        resumen["totales"] = {"imgs_procesadas": len(todos), "imgs_con_dano": g_imgs, **g_totals}
        resumen["totales"]["n_detections"] = sum(e["n_detections"] for e in todos)
        resumen["totales"]["total_area_m2"] = round(sum(float(e["total_area_m2"]) for e in todos), 4)

    with open(SALIDA / "_resumen.json", "w", encoding="utf-8") as fh:
        json.dump(resumen, fh, ensure_ascii=False, indent=1)

    print("\n===== RESUMEN =====")
    for k in ["crack", "humidity", "spalling"]:
        print(f"{k}: {totals[k]}")
    print(f"imagenes con dano: {totals['imgs_con_dano']}/{len(imgs)}")
    print("Resultados en:", SALIDA)
    print("JSON resumen: ", SALIDA / "_resumen.json")


if __name__ == "__main__":
    main()
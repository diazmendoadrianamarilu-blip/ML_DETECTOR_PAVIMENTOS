# -*- coding: utf-8 -*-
"""Muestreo estratificado y descarga de fotos de dron para probar best.pt."""
import json
import os
import re
import shutil
import time
from pathlib import Path

import gdown

BASE = Path(r"C:\Users\niurka\Downloads\sillar\datos_dron")
LISTA = BASE / "_lista_files.json"
MUESTRA = BASE / "muestra"

INT_PER_FLIGHT = 5   # imagenes por vuelo interior (6 vuelos -> 30)
EXT_PER_GROUP = {
    "DJI_202510121501_008_iglesia": 3,
    "DJI_202510121501_009_iglesia": 3,
    "DJI_202510121501_010_NuevarutadeCapturainteligente3D1": 6,
    "DJI_202510121548_011": 2,
    "DJI_202510121646_012": 4,
}


def seleccionar_espaciado(items, n):
    """Selecciona n items repartidos uniformemente a lo largo de la lista ordenada."""
    n = min(n, len(items))
    if n <= 0:
        return []
    idx = set()
    if n == 1:
        idx.add(len(items) // 2)
    else:
        for i in range(n):
            idx.add(round(i * (len(items) - 1) / (n - 1)))
    return [items[i] for i in sorted(idx)]


def main():
    with open(LISTA, encoding="utf-8") as fh:
        data = json.load(fh)

    seleccion = []  # (grupo, file_id, path_original, destino)

    # --- INT: 6 vuelos ---
    for vuelo in ["VUELO INTERIOR 1", "VUELO INTERIOR 2", "VUELO INTERIOR 3",
                  "VUELO INTERIOR 4", "VUELO INTERIOR 5", "VUELO INTERIOR 6"]:
        items = [p for p in data["INT"] if p["path"].startswith(vuelo)
                 and p["path"].lower().endswith(".jpg")]
        items.sort(key=lambda p: p["path"])
        for p in seleccionar_espaciado(items, INT_PER_FLIGHT):
            nombre = p["path"].replace("\\", "/").split("/")[-1]
            destino = MUESTRA / "INT" / vuelo.replace(" ", "_") / nombre
            seleccion.append((vuelo, p["id"], p["path"], destino))

    # --- EXT: 5 subcarpetas DJI ---
    ext = data["EXT"]
    for grupo, n in EXT_PER_GROUP.items():
        items = [p for p in ext if p["path"].split("\\")[1] == grupo
                 and p["path"].lower().endswith(".jpg")]
        items.sort(key=lambda p: p["path"])
        for p in seleccionar_espaciado(items, n):
            nombre = p["path"].replace("\\", "/").split("/")[-1]
            destino = MUESTRA / "EXT" / grupo[:20] / nombre
            seleccion.append((grupo, p["id"], p["path"], destino))

    print(f"Total seleccionadas: {len(seleccion)}")

    manfiesto = {"MUESTRA": str(MUESTRA)}
    for grupo, fid, path, destino in seleccion:
        manfiesto[path] = {"id": fid, "destino": str(destino)}
    with open(BASE / "_seleccion.json", "w", encoding="utf-8") as fh:
        json.dump(manfiesto, fh, ensure_ascii=False, indent=1)

    ok, fail = 0, 0
    t0 = time.time()
    for i, (grupo, fid, path, destino) in enumerate(seleccion, 1):
        if destino.exists() and destino.stat().st_size > 1000:
            ok += 1
            print(f"[{i}/{len(seleccion)}] ya existe: {destino.name}")
            continue
        destino.parent.mkdir(parents=True, exist_ok=True)
        tmp = destino.with_suffix(destino.suffix + ".part")
        try:
            id_limpio = fid.split("&")[0].split("?")[0]
            patched = f"https://drive.google.com/uc?id={id_limpio}"
            gdown.download(patched, str(tmp), quiet=True)
            if tmp.exists() and tmp.stat().st_size > 1000:
                shutil.move(str(tmp), destino)
                ok += 1
                print(f"[{i}/{len(seleccion)}] OK {path} -> {destino.name} ({destino.stat().st_size/1e6:.1f} MB)")
            else:
                fail += 1
                print(f"[{i}/{len(seleccion)}] FALLO descarga vacia: {path}")
        except Exception as e:
            fail += 1
            print(f"[{i}/{len(seleccion)}] ERROR {path}: {e}")
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    print(f"\nDescargadas OK: {ok}, fallidas: {fail}  (tiempo {time.time()-t0:.0f}s)")
    print(f"Directorios de salida: {MUESTRA}")


if __name__ == "__main__":
    main()
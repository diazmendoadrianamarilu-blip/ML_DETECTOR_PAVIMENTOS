# -*- coding: utf-8 -*-
"""
preparar_dataset_v2.py - Convierte la exportación YOLOv8 de Roboflow (v2) a las
3 clases de la app: crack, humidity, spalling.

- Usa la MISMA tabla de alias que la app (motor_deteccion.normalizar_clase):
  grieta/fisura -> crack, humedad/humeldad -> humidity, desprendimiento -> spalling.
- Excluye las imágenes que contienen clases sin patología definida
  (Patologias, 3, ...): tienen daño real, así que no pueden usarse como fondo.
- Conserva las imágenes de fondo (null, sin etiquetas).
- Compensa el desbalance humidity/crack duplicando en TRAIN (nunca en valid ni
  test) las imágenes que contienen grietas.
- Guarda la distribución de clases en distribucion_dataset_v2.csv (tesis) y
  crea un .zip listo para subir a Google Drive / Colab.

Uso (desde Heritage_Detector/):
    python preparar_dataset_v2.py "../Deteccion de Sillar.v2i.yolov8"
    python preparar_dataset_v2.py "../Deteccion de Sillar.v2i.yolov8" --sobremuestreo-crack 1
"""
import argparse
import csv
import os
import shutil
import sys
from collections import Counter

import yaml

from motor_deteccion import CLASES_VALIDAS, normalizar_clase

SPLITS = ("train", "valid", "test")


def preparar(origen, destino, factor_crack):
    with open(os.path.join(origen, "data.yaml"), encoding="utf-8") as f:
        nombres = yaml.safe_load(f)["names"]
    if isinstance(nombres, dict):
        nombres = [nombres[k] for k in sorted(nombres)]

    mapa = {}
    print("Clases de Roboflow -> clases de la app:")
    for i, n in enumerate(nombres):
        clase = normalizar_clase(n)
        mapa[i] = CLASES_VALIDAS.index(clase) if clase else None
        print(f"  {i:2d} {n!r:20s} -> {clase or 'EXCLUIR imagen'}")

    shutil.rmtree(destino, ignore_errors=True)
    resumen, excluidas = {}, []
    for split in SPLITS:
        dir_img = os.path.join(origen, split, "images")
        if not os.path.isdir(dir_img):
            continue
        out_img = os.path.join(destino, split, "images")
        out_lbl = os.path.join(destino, split, "labels")
        os.makedirs(out_img)
        os.makedirs(out_lbl)
        inst, n_img, nulas, duplicadas = Counter(), 0, 0, 0
        for img in sorted(os.listdir(dir_img)):
            base, ext = os.path.splitext(img)
            ruta_lbl = os.path.join(origen, split, "labels", base + ".txt")
            lineas = open(ruta_lbl).read().splitlines() if os.path.exists(ruta_lbl) else []
            nuevas, excluir = [], False
            for ln in lineas:
                p = ln.split()
                if not p:
                    continue
                nuevo = mapa.get(int(p[0]))
                if nuevo is None:
                    excluir = True
                    break
                nuevas.append((nuevo, " ".join([str(nuevo)] + p[1:])))
            if excluir:
                excluidas.append(f"{split}/{img}")
                continue

            texto = "\n".join(t for _, t in nuevas) + ("\n" if nuevas else "")
            tiene_crack = any(c == CLASES_VALIDAS.index("crack") for c, _ in nuevas)
            copias = factor_crack if (split == "train" and tiene_crack) else 1
            for k in range(copias):
                sufijo = "" if k == 0 else f"_dup{k}"
                shutil.copy2(os.path.join(dir_img, img), os.path.join(out_img, base + sufijo + ext))
                with open(os.path.join(out_lbl, base + sufijo + ".txt"), "w") as f:
                    f.write(texto)
                inst.update(CLASES_VALIDAS[c] for c, _ in nuevas)
                n_img += 1
                nulas += not nuevas
                duplicadas += k > 0
        resumen[split] = {"imagenes": n_img, "fondo_null": nulas, "duplicadas_crack": duplicadas,
                          **{c: inst[c] for c in CLASES_VALIDAS}}

    with open(os.path.join(destino, "data.yaml"), "w", encoding="utf-8") as f:
        # Sin "path": ultralytics toma la carpeta del propio yaml, asi que
        # funciona igual en local y en Colab tras descomprimir.
        yaml.safe_dump({"train": "train/images", "val": "valid/images",
                        "test": "test/images", "names": dict(enumerate(CLASES_VALIDAS))},
                       f, sort_keys=False)
    return resumen, excluidas


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("origen", help="carpeta exportada de Roboflow (contiene data.yaml)")
    parser.add_argument("--destino", default=None, help="carpeta de salida (por defecto: dataset_3clases junto al origen)")
    parser.add_argument("--sobremuestreo-crack", type=int, default=3,
                        help="copias en train de cada imagen con grietas (1 = sin sobremuestreo)")
    parser.add_argument("--sin-zip", action="store_true", help="no crear el .zip para Colab")
    args = parser.parse_args()

    origen = os.path.abspath(args.origen)
    destino = os.path.abspath(args.destino or os.path.join(os.path.dirname(origen), "dataset_3clases"))
    resumen, excluidas = preparar(origen, destino, max(1, args.sobremuestreo_crack))

    columnas = ["split", "imagenes", "fondo_null", "duplicadas_crack", *CLASES_VALIDAS]
    print("\n" + "  ".join(f"{c:>16s}" for c in columnas))
    for split, fila in resumen.items():
        print("  ".join(f"{str(v):>16s}" for v in [split, *(fila[c] for c in columnas[1:])]))
    tr = resumen.get("train", {})
    print(f"\nRelacion humidity/crack en train: {tr.get('humidity', 0) / max(tr.get('crack', 0), 1):.1f} : 1")
    print(f"Imagenes excluidas (Patologias / clases sin patologia): {len(excluidas)}")

    csv_path = os.path.join(destino, "distribucion_dataset_v2.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(columnas)
        for split, fila in resumen.items():
            w.writerow([split, *(fila[c] for c in columnas[1:])])
        w.writerow([])
        w.writerow(["imagenes_excluidas", len(excluidas)])
        for e in excluidas:
            w.writerow([e])
    print(f"Distribucion guardada en: {csv_path}")

    if not args.sin_zip:
        zip_path = shutil.make_archive(destino, "zip", destino)
        print(f"Zip para Colab: {zip_path} ({os.path.getsize(zip_path) / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

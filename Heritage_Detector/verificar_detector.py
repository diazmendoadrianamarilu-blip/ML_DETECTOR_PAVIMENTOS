# -*- coding: utf-8 -*-
"""
verificar_detector.py - Verificación sin interfaz del motor de desktop_app.py

1) Pruebas de la lógica (no necesitan el modelo):
   clases, umbrales, área sin duplicar, agrupación y filtro anti-cornisa.
2) Si se pasan imágenes (o una carpeta), las procesa con best.pt en modo
   "Recomendado" y "Alta confiabilidad (85%)" y guarda las imágenes anotadas.

Uso (desde Heritage_Detector/):
    python verificar_detector.py
    python verificar_detector.py ../02_Evidencia_y_Datos/datos_dron/muestra/INT
    python verificar_detector.py foto_cornisa.jpg --salida verificacion_salida
"""
import argparse
import os
import sys

import cv2
import numpy as np

import desktop_app as da
import motor_deteccion as md

UMBRALES_RECOMENDADO = {"crack": 0.20, "humidity": 0.30, "spalling": 0.60}
UMBRALES_ALTO = {c: da.UMBRAL_ALTA_CONFIABILIDAD for c in da.CLASES_VALIDAS}
EXTENSIONES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

_fallos = []


def comprobar(nombre, condicion, detalle=""):
    estado = "OK   " if condicion else "FALLA"
    print(f"  [{estado}] {nombre}" + (f"  ({detalle})" if detalle else ""))
    if not condicion:
        _fallos.append(nombre)


def _caja(clase, conf, x1, y1, x2, y2):
    return {"Clase": clase, "Confianza": conf, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "Ancho_px": x2 - x1, "Alto_px": y2 - y1}


def _muro_sillar(h, w, semilla=0):
    """Superficie beige con textura, parecida al sillar."""
    rng = np.random.default_rng(semilla)
    base = np.array([150, 175, 200], dtype=np.float32)  # BGR beige
    ruido = rng.normal(0, 14, (h, w, 1)).astype(np.float32)
    img = np.clip(base + ruido, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (3, 3), 0)


def pruebas_logica():
    print("\n== 1. Clases (Fase 1) ==")
    esperado = {
        "crack": "crack", "grieta": "crack", "fisura": "crack", "Fisura": "crack",
        "humedad": "humidity", "humeldad": "humidity", "humidity": "humidity",
        "desprendimiento": "spalling", "spalling": "spalling",
        "Patologias": None, "3": None, "67": None, "78": None, "h4": None,
    }
    for nombre, clase in esperado.items():
        comprobar(f"normalizar_clase({nombre!r}) -> {clase}", md.normalizar_clase(nombre) == clase)

    print("\n== 2. Umbrales mín./máx. (Fase 2) ==")
    comprobar("umbral bajo se acota a 5%", md.umbral_clase({"crack": 0.0}, "crack") == da.UMBRAL_CONF_MIN)
    comprobar("umbral alto se acota a 90%", md.umbral_clase({"crack": 0.99}, "crack") == da.UMBRAL_CONF_MAX)
    dets = [_caja("crack", 0.54, 10, 10, 60, 60), _caja("humidity", 0.90, 100, 100, 200, 200)]
    quedan = da.filtrar_falsos_positivos([dict(d) for d in dets], (400, 400, 3), UMBRALES_ALTO)
    comprobar("modo 85%: la caja al 54% se descarta y la del 90% se conserva",
              [d["Confianza"] for d in quedan] == [0.90], f"quedan {[d['Confianza'] for d in quedan]}")

    print("\n== 3. Filtro anti-cornisa (calibración geométrica) ==")
    img = _muro_sillar(400, 800)
    # Sombra de cornisa: banda oscura recta, horizontal y de ancho uniforme.
    # Contraste moderado (textura std < 50) para que no la elimine el filtro de
    # textura previo y la decida la calibración geométrica.
    img[190:204, 40:760] = (img[190:204, 40:760] * 0.65).astype(np.uint8)
    sombra = _caja("crack", 0.54, 30, 180, 770, 214)
    quedan = da.filtrar_falsos_positivos([dict(sombra)], img.shape, UMBRALES_RECOMENDADO, img)
    comprobar("sombra de cornisa al 54% (modo recomendado) se descarta", len(quedan) == 0)
    quedan = da.filtrar_falsos_positivos([dict(sombra, Confianza=0.90)], img.shape, UMBRALES_RECOMENDADO, img)
    comprobar("la misma caja con confianza >= 85% se respeta (decide la red)", len(quedan) == 1)

    img2 = _muro_sillar(400, 800, semilla=1)
    xs = np.arange(60, 740)
    ys = (200 + 18 * np.sin(xs / 23.0) + 9 * np.sin(xs / 7.0)).astype(np.int32)
    pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
    grosores = 1 + (np.abs(np.sin(xs / 31.0)) * 3).astype(int)
    for k in range(len(xs) - 1):
        cv2.line(img2, tuple(pts[k, 0]), tuple(pts[k + 1, 0]), (55, 60, 70), int(grosores[k]))
    grieta = _caja("crack", 0.54, 50, 165, 750, 235)
    quedan = da.filtrar_falsos_positivos([dict(grieta)], img2.shape, UMBRALES_RECOMENDADO, img2)
    comprobar("grieta sinuosa sintética al 54% se conserva", len(quedan) == 1)

    print("\n== 4. NMS y área sin duplicar (Fase 3) ==")
    comprobar("IoU de NMS = 0.45", da.IOU_NMS == 0.45)
    a = md._area_union_rectangulos([(0, 0, 10, 10), (5, 5, 15, 15)])
    comprobar("unión de 2 cajas solapadas = 175 px² (no 200)", a == 175, f"obtenido {a}")
    a = md._area_union_rectangulos([(0, 0, 10, 2), (0, 0, 2, 10)])
    comprobar("grupo en L = 36 px² (no 100 del recuadro envolvente)", a == 36, f"obtenido {a}")
    # A y B se tocan; su recuadro envolvente alcanza a C, que no tocaba a ninguna
    zonas = md.agrupar_zonas([
        _caja("spalling", 0.70, 0, 0, 100, 20),
        _caja("spalling", 0.95, 95, 0, 115, 100),
        _caja("spalling", 0.65, 10, 60, 40, 90),
        _caja("crack", 0.80, 10, 60, 40, 90),
    ])
    sp = [z for z in zonas if z["Clase"] == "spalling"]
    comprobar("fusión repetida: 1 sola zona de spalling", len(sp) == 1, f"zonas: {len(sp)}")
    comprobar("la zona conserva la confianza más alta (95%)", sp and sp[0]["Confianza"] == 0.95)
    area_esperada = md._area_union_rectangulos([(0, 0, 100, 20), (95, 0, 115, 100), (10, 60, 40, 90)])
    comprobar("área de la zona = unión de sus cajas", sp and sp[0]["Area_px"] == area_esperada,
              f"{sp[0]['Area_px'] if sp else '-'} px² vs recuadro {115 * 100} px²")
    comprobar("crack no se fusiona con spalling", any(z["Clase"] == "crack" for z in zonas))
    comprobar("no quedan campos internos (_rects/_polys)",
              all("_rects" not in z and "_polys" not in z for z in zonas))
    poly = np.array([[0, 0], [20, 0], [0, 20]], dtype=np.int32)
    zonas = md.agrupar_zonas([dict(_caja("humidity", 0.9, 0, 0, 20, 20), _poly=poly)])
    comprobar("con máscara (segmentación) el área es la del polígono, no la caja",
              zonas[0]["Area_metodo"] == "mascara" and zonas[0]["Area_px"] < 400,
              f"{zonas[0]['Area_px']} px² vs caja 400 px²")


def listar_imagenes(rutas):
    imagenes = []
    for ruta in rutas:
        if os.path.isdir(ruta):
            for raiz, _, archivos in os.walk(ruta):
                imagenes += [os.path.join(raiz, a) for a in sorted(archivos)
                             if a.lower().endswith(EXTENSIONES)]
        elif os.path.isfile(ruta):
            imagenes.append(ruta)
        else:
            print(f"  (no existe: {ruta})")
    return imagenes


def pruebas_modelo(imagenes, salida, gsd):
    from ultralytics import YOLO

    modelo = YOLO(da._resolver_ruta_recurso(da.MODEL_PATH))
    print(f"\n== 5. Modelo: tarea={modelo.task}, clases={modelo.names} ==")
    no_validas = [n for n in modelo.names.values() if md.normalizar_clase(n) is None]
    comprobar("todas las clases del modelo se reconocen", not no_validas,
              f"se descartarán: {no_validas}" if no_validas else "")
    os.makedirs(salida, exist_ok=True)

    for ruta in imagenes:
        nombre = os.path.splitext(os.path.basename(ruta))[0]
        print(f"\n  {ruta}")
        for modo, umbrales in (("recomendado", UMBRALES_RECOMENDADO), ("alto85", UMBRALES_ALTO)):
            r = da.procesar_imagen_completa(modelo, ruta, umbrales, da.IOU_NMS, gsd)
            if not r.get("success"):
                print(f"    {modo:12s} ERROR: {r.get('error')}")
                continue
            comprobar(f"{modo}: solo clases válidas",
                      set(d["Clase"] for d in r["detections"]) <= set(da.CLASES_VALIDAS))
            if modo == "alto85":
                comprobar("alto85: ninguna detección < 85%",
                          all(d["Confianza"] >= da.UMBRAL_ALTA_CONFIABILIDAD for d in r["detections"]))
            conteo = ", ".join(f"{c}={n}" for c, n in r["class_counts"].items())
            print(f"    {modo:12s} {r['n_detections']:3d} det. ({conteo}) | "
                  f"área {r['total_area_cm2']:.1f} cm² | conf. media {r['avg_confidence']}%")
            for d in r["detections"]:
                print(f"        - {d['Clase']:9s} {d['Confianza']*100:5.1f}%  "
                      f"caja ({d['x1']},{d['y1']})-({d['x2']},{d['y2']})  "
                      f"{d['Area_cm2']:.1f} cm² [{d.get('Area_metodo', '-')}]")
            destino = os.path.join(salida, f"{nombre}_{modo}.jpg")
            cv2.imwrite(destino, cv2.cvtColor(r["annotated_image"], cv2.COLOR_RGB2BGR))
    print(f"\n  Imágenes anotadas en: {os.path.abspath(salida)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rutas", nargs="*", help="imágenes o carpetas a procesar con best.pt")
    parser.add_argument("--salida", default="verificacion_salida", help="carpeta de imágenes anotadas")
    parser.add_argument("--gsd", type=float, default=0.13, help="cm por píxel (por defecto 0.13)")
    args = parser.parse_args()

    pruebas_logica()
    if args.rutas:
        imagenes = listar_imagenes(args.rutas)
        if imagenes:
            pruebas_modelo(imagenes, args.salida, args.gsd)

    print("\n" + ("TODO CORRECTO" if not _fallos else f"{len(_fallos)} COMPROBACIÓN(ES) FALLIDA(S):"))
    for f in _fallos:
        print(f"  - {f}")
    return 1 if _fallos else 0


if __name__ == "__main__":
    sys.exit(main())

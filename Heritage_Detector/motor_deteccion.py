# -*- coding: utf-8 -*-
"""
motor_deteccion.py - Núcleo común del Heritage Damage Detector v6.5
Universidad Católica de Santa María - Arequipa, Perú

Lógica compartida por la app de escritorio (desktop_app.py) y la web
(app.py), para que ambas den el mismo resultado sobre la misma imagen:
clases válidas, umbrales, tiling, NMS, agrupación de zonas con área sin
duplicar y confirmación geométrica de grietas (anti-cornisa).
Solo depende de OpenCV y NumPy.
"""
import cv2
import numpy as np

# =============================================================================
# CLASES Y CALIBRACIÓN DEL DETECTOR
# =============================================================================

# Únicas clases que el sistema procesa, dibuja y guarda en Supabase
# (crack_count / humidity_count / spalling_count).
CLASES_VALIDAS = ("crack", "humidity", "spalling")

# Cualquier variante de nombre (dataset de Roboflow, español, plurales) se
# agrupa en una de las 3 clases. Toda fisura/microfisura cuenta como crack.
# Un nombre no listado aquí se descarta: nunca se genera una clase extra.
ALIAS_CLASES = {
    "crack": "crack", "cracks": "crack", "grieta": "crack", "grietas": "crack",
    "fisura": "crack", "fisuras": "crack", "fissure": "crack", "fissures": "crack",
    "microcrack": "crack", "microfisura": "crack", "microfisuras": "crack",
    "humidity": "humidity", "humedad": "humidity", "moisture": "humidity",
    "humeldad": "humidity",  # error de tipeo presente en el dataset de Roboflow
    "damp": "humidity", "dampness": "humidity",
    "spalling": "spalling", "spall": "spalling", "desprendimiento": "spalling",
    "desprendimientos": "spalling",
}
# "patologias", "3", "67", "78", "h4" (clases del proyecto de Roboflow) NO se
# mapean a propósito: normalizar_clase() devuelve None y la caja se descarta.

# Modo "Alta confiabilidad": piso de confianza que ningún modo (ni el recorte
# de foto lejana) puede rebajar. Descarta, p. ej., sombras de cornisa al 54%.
UMBRAL_ALTA_CONFIABILIDAD = 0.85
# Límites admitidos para cualquier umbral por clase (mismo rango que los sliders).
UMBRAL_CONF_MIN = 0.05
UMBRAL_CONF_MAX = 0.90
# IoU único de la Supresión de No Máximos para las 3 clases.
IOU_NMS = 0.45
# Una caja de crack con lado mayor / lado menor >= este valor es "lineal":
# puede ser una grieta, pero también una cornisa, junta o borde de sombra.
RELACION_ASPECTO_LINEAL = 4.0


def normalizar_clase(nombre):
    """Devuelve 'crack', 'humidity' o 'spalling', o None si la clase no es válida."""
    clave = str(nombre).strip().lower().replace(" ", "_").replace("-", "_")
    return ALIAS_CLASES.get(clave)


def umbral_clase(umbrales_clase, cls_name, por_defecto=0.25):
    """Umbral de confianza de una clase acotado a [UMBRAL_CONF_MIN, UMBRAL_CONF_MAX]."""
    valor = float(umbrales_clase.get(cls_name, por_defecto))
    return min(max(valor, UMBRAL_CONF_MIN), UMBRAL_CONF_MAX)


def split_image_into_tiles(image_np, tile_size, overlap):
    """Divide imagen en tiles"""
    h, w = image_np.shape[:2]
    stride = int(tile_size * (1 - overlap))
    tiles = []
    
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            if y + tile_size > h:
                y = h - tile_size
            if x + tile_size > w:
                x = w - tile_size
            
            tile = image_np[y:y+tile_size, x:x+tile_size]
            tiles.append((tile, x, y))
            
            if x + tile_size == w:
                break
        if y + tile_size == h:
            break
    
    return tiles

def nms_boxes(boxes, scores, iou_threshold=0.5):
    """Non-Maximum Suppression por clase"""
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes).astype(np.float32)
    scores = np.array(scores)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    order = scores.argsort()[::-1]
    keep = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep


def _area_union_rectangulos(rects):
    """Area (px^2) de la union de rectangulos (x1, y1, x2, y2): las zonas
    solapadas se cuentan UNA sola vez (barrido por franjas verticales)."""
    if not rects:
        return 0
    xs = sorted({r[0] for r in rects} | {r[2] for r in rects})
    total = 0
    for xa, xb in zip(xs[:-1], xs[1:]):
        if xb <= xa:
            continue
        tramos = sorted((r[1], r[3]) for r in rects if r[0] <= xa and r[2] >= xb)
        cubierto = 0
        cur_y1 = cur_y2 = None
        for y1, y2 in tramos:
            if cur_y2 is None or y1 > cur_y2:
                if cur_y2 is not None:
                    cubierto += cur_y2 - cur_y1
                cur_y1, cur_y2 = y1, y2
            else:
                cur_y2 = max(cur_y2, y2)
        if cur_y2 is not None:
            cubierto += cur_y2 - cur_y1
        total += (xb - xa) * cubierto
    return total


def _poligono_mascara(result, i, x_off=0, y_off=0, escala=1):
    """Polígono de la máscara i (modelo de segmentación) en coordenadas de la
    imagen completa, o None si el modelo solo entrega cajas."""
    masks = getattr(result, "masks", None)
    if masks is None:
        return None
    try:
        poly = np.asarray(masks.xy[i], dtype=np.float32)
    except (IndexError, AttributeError):
        return None
    if poly.ndim != 2 or len(poly) < 3:
        return None
    return (poly / escala + np.array([x_off, y_off], dtype=np.float32)).astype(np.int32)


def _area_union_poligonos(polys, x1, y1, x2, y2):
    """Area (px^2) de la union de las mascaras de una zona: cada pixel afectado
    se cuenta una sola vez, sin el fondo que queda dentro de la caja."""
    mask = np.zeros((int(y2 - y1) + 1, int(x2 - x1) + 1), dtype=np.uint8)
    origen = np.array([x1, y1], dtype=np.int32)
    cv2.fillPoly(mask, [p - origen for p in polys], 1)
    return int(np.count_nonzero(mask))


def agrupar_zonas(detections, umbral_toc = 0.10):
    """Fusiona cuadros de la misma clase que pertenecen a la misma zona afectada.
    Dos cuadros se agrupan si, al expandir cada uno un 10% de su tamano, sus
    rectangulos se siguen tocando. El resultado es UN recuadro que abarca toda
    el area afectada, con la confianza maxima del grupo.
    Se repite hasta que ninguna zona de la misma clase se toque (un recuadro
    fusionado puede alcanzar a otro), y "Area_px" es el area de la UNION de los
    cuadros originales: el solape no se suma dos veces ni se cuenta el hueco
    del recuadro envolvente. Con un modelo de segmentacion (cada deteccion trae
    "_poly") el area se mide sobre la union de las MASCARAS reales."""
    zonas = []
    for d in detections:
        z = dict(d)
        z.setdefault("_rects", [(d["x1"], d["y1"], d["x2"], d["y2"])])
        z.setdefault("_polys", [d["_poly"]] if d.get("_poly") is not None else [])
        z.pop("_poly", None)
        zonas.append(z)
    while True:
        n_antes = len(zonas)
        zonas = _agrupar_zonas_pasada(zonas, umbral_toc)
        if len(zonas) == n_antes:
            break
    for z in zonas:
        rects = z.pop("_rects")
        polys = z.pop("_polys")
        if polys and len(polys) == len(rects):
            z["Area_px"] = _area_union_poligonos(polys, z["x1"], z["y1"], z["x2"], z["y2"])
            z["Area_metodo"] = "mascara"
        else:
            z["Area_px"] = _area_union_rectangulos(rects)
            z["Area_metodo"] = "cajas"
    return zonas


def _agrupar_zonas_pasada(detections, umbral_toc):
    """Una pasada de union-find de agrupar_zonas (conserva los cuadros originales en _rects)."""
    detections = list(detections)
    n = len(detections)
    if n == 0:
        return []

    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # unir por clase y cercania (rectangulos expandidos que se tocan)
    for i in range(n):
        di = detections[i]
        wi = di["Ancho_px"]; hi = di["Alto_px"]
        exi = max(wi * umbral_toc, 4); eyi = max(hi * umbral_toc, 4)
        ix1, iy1, ix2, iy2 = di["x1"]-exi, di["y1"]-eyi, di["x2"]+exi, di["y2"]+eyi
        for j in range(i+1, n):
            dj = detections[j]
            if di["Clase"] != dj["Clase"]:
                continue
            wj = dj["Ancho_px"]; hj = dj["Alto_px"]
            exj = max(wj * umbral_toc, 4); eyj = max(hj * umbral_toc, 4)
            jx1, jy1, jx2, jy2 = dj["x1"]-exj, dj["y1"]-eyj, dj["x2"]+exj, dj["y2"]+eyj
            if ix1 <= jx2 and jx1 <= ix2 and iy1 <= jy2 and jy1 <= iy2:
                union(i, j)

    grupos = {}
    for k in range(n):
        grupos.setdefault(find(k), []).append(detections[k])

    zonas = []
    for miembros in grupos.values():
        x1 = min(m["x1"] for m in miembros)
        y1 = min(m["y1"] for m in miembros)
        x2 = max(m["x2"] for m in miembros)
        y2 = max(m["y2"] for m in miembros)
        conf = max(m["Confianza"] for m in miembros)
        zonas.append({
            "Clase": miembros[0]["Clase"],
            "Confianza": conf,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "Ancho_px": x2 - x1,
            "Alto_px": y2 - y1,
            "_rects": [r for m in miembros for r in m["_rects"]],
            "_polys": [p for m in miembros for p in m["_polys"]],
        })

    zonas.sort(key=lambda z: (z["Clase"], -z["Confianza"]))
    return zonas

def deteccion_lineas_finas(gray_img, sigmas=(1, 2, 3)):
    """Detecta líneas oscuras filiformes (grietas finas sobre pintura) mediante
    análisis multiescala del Hessiano, sin depender de la red neuronal.
    Una grieta = valle oscuro sobre fondo claro: la curvatura dominante del
    Hessiano es POSITIVA (lambda1 > 0) y la perpendicular es ~0 (lambda2 ~ 0).
    Una mancha oscura (blob) tiene ambas curvaturas positivas grandes; se rechaza
    exigiendo |lambda2| << |lambda1|. Devuelve mapa de respuesta 0-255."""
    H = gray_img.astype(np.float32) / 255.0
    response = np.zeros_like(H, dtype=np.float32)

    for s in sigmas:
        if s > 1:
            blur = cv2.GaussianBlur(H, (0, 0), sigmaX=s, sigmaY=s)
        else:
            blur = H

        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        hxx = cv2.Sobel(gx, cv2.CV_32F, 1, 0, ksize=3)
        hyy = cv2.Sobel(gy, cv2.CV_32F, 0, 1, ksize=3)
        hxy = cv2.Sobel(gx, cv2.CV_32F, 0, 1, ksize=3)

        trace = hxx + hyy
        disc = np.sqrt(np.maximum((hxx - hyy) ** 2 / 4.0 + hxy ** 2, 0))
        lambda1 = trace / 2 + disc
        lambda2 = trace / 2 - disc

        # Ojo: lambda1 es el de mayor valor (puede ser positivo para valle).
        # Línea oscura: lambda1 > 0 grande, |lambda2| pequeño.
        lam1_pos = np.maximum(lambda1, 0)
        lam2_abs = np.abs(lambda2)
        lam1_abs = np.maximum(np.abs(lambda1), 1e-6)
        # Radio de anisotropía: penaliza blobs (ambas curvaturas grandes)
        anisotropy = 1.0 - np.minimum(lam2_abs / lam1_abs, 1.0)
        sigma_resp = lam1_pos * anisotropy
        response = np.maximum(response, sigma_resp)

    # Normalizar a 0-255
    rmax = float(np.max(response))
    if rmax <= 0:
        return np.zeros_like(gray_img)
    return (response / rmax * 255.0).astype(np.uint8)


def _metricas_filamento(mask, resp_sub, gx_sub, gy_sub, contour):
    """Metricas de forma de un filamento: tortuosidad, dispersion angular,
    constancia de anchura y orientacion dominante.
    Devuelve None si el componente no es medible (pocos pixeles)."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 30:
        return None
    pts = np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))
    mean = pts.mean(axis=0)
    pts_c = pts - mean
    cov = np.cov(pts_c.T)
    try:
        evals, evecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None
    if evals[1] <= 1e-6:
        return None
    main_axis = evecs[:, 1]
    t = pts_c @ main_axis
    tmin, tmax = float(t.min()), float(t.max())
    # Longitud de arco approximada (mitad del perimetro de la linea)
    arc = float(cv2.arcLength(contour, True)) / 2.0
    # Distancia recta entre los extremos proyectados en el eje principal
    p1 = mean + main_axis * tmin
    p2 = mean + main_axis * tmax
    straight = float(np.linalg.norm(p2 - p1))
    tortuosidad = arc / max(straight, 1e-6)
    # Anchura y constancia de anchura: distance transform a lo largo del eje
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    span = (tmax - tmin) if tmax - tmin > 0 else 1.0
    nbins = int(max(5, min(30, span / 3.0)))
    if nbins < 3:
        return None
    anchos = []
    for b in range(nbins):
        lo = tmin + span * b / nbins
        hi = tmin + span * (b + 1) / nbins
        sel = (t >= lo) & (t < hi)
        if int(sel.sum()) >= 3:
            anchos.append(2.0 * float(dt[ys[sel], xs[sel]].max()))
    if len(anchos) < 3:
        return None
    std_ancho = float(np.std(anchos))
    # Orientacion local en los pixeles de la linea (gradiente del mapa de respuesta)
    ang = np.arctan2(gy_sub[mask > 0], gx_sub[mask > 0])
    if ang.size < 30:
        return None
    r = float(np.hypot(np.cos(2 * ang).mean(), np.sin(2 * ang).mean()))
    r = max(min(r, 1.0), 1e-12)
    std_ang = float(np.degrees(np.sqrt(-2.0 * np.log(r))) / 1.0)
    orient = float(np.degrees(np.arctan2(main_axis[1], main_axis[0])) % 180.0)
    return {
        "tortuosidad": tortuosidad,
        "std_ang_deg": std_ang,
        "std_ancho_px": std_ancho,
        "ancho_medio_px": float(np.mean(anchos)),
        "orientacion_deg": orient,
        "arco_px": arc,
        "longitud_px": tmax - tmin,
    }


def _orientacion_dominante(binary_edges):
    """Orienta cion dominante (en grados 0-180) de las lineas del crop, usando
    Hough. Devuelve None si no hay lineas significativas (no se penaliza)."""
    lines = cv2.HoughLines(binary_edges, 1, np.pi / 180, threshold=90)
    if lines is None or len(lines) == 0:
        return None
    bins = np.zeros(36, dtype=np.float64)
    for l in lines:
        theta_deg = ((l[0][1] * 180.0 / np.pi) % 180.0) + 90.0
        idx = int(theta_deg // 5) % 36
        bins[idx] += float(l[0][0])
    return float((int(np.argmax(bins)) * 5) % 180.0)


def _parche_para_filiforme(patch, lado_max=1200):
    """Adapta un parche de la vista global a la escala en que se calibro
    _es_grieta_filiforme (recorte ampliado x2, Lanczos). Parches ya grandes
    se dejan igual para no disparar el tiempo de proceso."""
    patch = np.clip(patch, 0, 255).astype(np.uint8)
    h, w = patch.shape[:2]
    if max(h, w) <= lado_max:
        patch = cv2.resize(patch, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    return patch


def _es_grieta_filiforme(bgr_patch, tort_min=1.25, elong_min=4.0, largo_min=15.0,
                         std_ancho_min=0.3, dom_min=2.0):
    """Confirma si la caja YOLO de crack (modo recorte) contiene una grieta real.

    Ya descartado el patron con textura excesiva (std alto: techo corrugado,
    suelo con sombras) y la vegetacion/cielo, aqui se exige que entre los
    filamentos GRANDES de la caja (area de al menos 1/2 del mayor) exista una
    linea con geometria de grieta: sinuosa (tortuosidad alta), delgada
    (elongacion alta), de anchura VARIABLE (std_ancho; un cable/cuerda recta y
    de anchura constante falla aqui) y de longitud suficiente.

    Calibrado en la imagen real (escala 2x del recorte): la grieta real tiene
    tortuosidad 1.30-1.35 en sus ramas grandes; el techo corrugado, el cable de
    campana, arcos y muros lejanos no superan 1.16 en ningun componente
    dominante (sus fragmentos sinuosos son pequenos, < area del mayor/2, y
    quedan descartados junto con el ruido). El filtro de textura (std<=15)
    elimina aparte los FPs estructurales periodicos aunque sus ramas rocen la
    tortuosidad. El uso de "todos los grandes" (no solo el mayor absoluto) hace
    la decision robusta al recorte de la caja por tiles (bordes).

    Devuelve True si es una grieta plausible."""
    if bgr_patch is None or bgr_patch.size == 0:
        return False
    gray = cv2.cvtColor(bgr_patch.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    resp = deteccion_lineas_finas(gray)
    if resp.max() == 0:
        return False
    thr = max(int(cv2.threshold(resp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]), 40)
    binary = cv2.threshold(resp, max(40, thr - 15), 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    contornos, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contornos:
        return False

    gx = cv2.Sobel(resp, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(resp, cv2.CV_32F, 0, 1, ksize=3)

    componentes = []
    for c in contornos:
        area = cv2.contourArea(c)
        if area < 30:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        mask = np.zeros((bh, bw), dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1, offset=(-bx, -by))
        met = _metricas_filamento(mask, resp[by:by + bh, bx:bx + bw],
                                  gx[by:by + bh, bx:bx + bw], gy[by:by + bh, bx:bx + bw], c)
        if not met:
            continue
        met["area"] = float(area)
        met["elongacion"] = met["longitud_px"] / max(met["ancho_medio_px"], 1e-6)
        componentes.append(met)
    if not componentes:
        return False

    area_mayor = max(m["area"] for m in componentes)
    umbral_area = area_mayor / dom_min
    grandes = [m for m in componentes if m["area"] >= umbral_area]
    return any(m["tortuosidad"] >= tort_min and
               m["elongacion"] >= elong_min and
               m["longitud_px"] >= largo_min and
               m["std_ancho_px"] >= std_ancho_min
               for m in grandes)

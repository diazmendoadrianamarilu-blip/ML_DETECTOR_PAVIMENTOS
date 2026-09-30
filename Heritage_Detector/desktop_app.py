# -*- coding: utf-8 -*-
"""
desktop_app.py - Heritage Damage Detector v6.5 DESKTOP EDITION
Universidad Católica de Santa María - Arequipa, Perú
Aplicación de escritorio nativa con CustomTkinter + Supabase
Motor: Tiling + NMS + Filtros Geométricos + Anti-sombras
v6.5: Hover interactivo + Clasificación de gravedad + Priorización
"""
import customtkinter as ctk
from tkinter import filedialog, messagebox, ttk
import tkinter as tk
from PIL import Image, ImageTk
from alerts_system import AlertSystem
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from datetime import datetime
import os
import sys
import json
import re
import io
import time
import torch
import requests
import threading
from pathlib import Path
from dotenv import load_dotenv
from motor_deteccion import (
    CLASES_VALIDAS, UMBRAL_ALTA_CONFIABILIDAD, UMBRAL_CONF_MIN, UMBRAL_CONF_MAX,
    IOU_NMS, RELACION_ASPECTO_LINEAL, normalizar_clase, umbral_clase,
    _poligono_mascara, agrupar_zonas,
    deteccion_lineas_finas, _metricas_filamento, _orientacion_dominante, _parche_para_filiforme,
    _es_grieta_filiforme, split_image_into_tiles, nms_boxes,
)

def _cargar_config():
    """Carga .env desde el directorio empaquetado (PyInstaller) o el directorio actual."""
    try:
        base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
        env_path = os.path.join(base, ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path, override=True)
            return
    except Exception:
        pass
    load_dotenv()

_cargar_config()

# Intentar importar detection_gravity (si existe)
try:
    from detection_gravity import enrich_detections_with_gravity, generate_priority_report, GRAVITY_COLORS
    GRAVITY_MODULE_AVAILABLE = True
except ImportError:
    GRAVITY_MODULE_AVAILABLE = False
    # Valores por defecto si el módulo no existe aún
    GRAVITY_COLORS = {
        "leve": "#10B981",
        "moderada": "#F59E0B",
        "severa": "#F97316",
        "critica": "#DC2626"
    }
    
    def enrich_detections_with_gravity(detections):
        """Fallback: añade gravedad básica si no hay módulo"""
        for d in detections:
            area = d.get("Area_cm2", 0)
            if area > 100:
                gravity = "critica"
            elif area > 30:
                gravity = "severa"
            elif area > 5:
                gravity = "moderada"
            else:
                gravity = "leve"
            d["gravedad"] = gravity
            d["gravedad_score"] = {"leve": 1, "moderada": 2, "severa": 3, "critica": 4}[gravity]
            d["gravedad_color"] = GRAVITY_COLORS[gravity]
            d["gravedad_label"] = {"leve": "🟢 Leve", "moderada": "🟡 Moderada", "severa": "🟠 Severa", "critica": "🔴 CRÍTICA"}[gravity]
        return sorted(detections, key=lambda x: x["gravedad_score"], reverse=True)
    
    def generate_priority_report(detections, filename):
        """Fallback: genera reporte básico"""
        enriched = enrich_detections_with_gravity(detections)
        criticas = [d for d in enriched if d.get("gravedad") == "critica"]
        severas = [d for d in enriched if d.get("gravedad") == "severa"]
        total_score = sum(d.get("gravedad_score", 1) for d in enriched)
        
        if len(criticas) > 0:
            recommendation = "🚨 INTERVENCIÓN INMEDIATA requerida"
            urgency = "CRÍTICA"
        elif len(severas) > 2:
            recommendation = "⚠️ Programar intervención en < 30 días"
            urgency = "ALTA"
        elif total_score > 15:
            recommendation = "📋 Programar intervención en < 90 días"
            urgency = "MEDIA"
        else:
            recommendation = "✅ Monitoreo preventivo"
            urgency = "BAJA"
        
        return {
            "filename": filename,
            "total_priority_score": total_score,
            "urgency": urgency,
            "recommendation": recommendation,
            "criticas_count": len(criticas),
            "severas_count": len(severas),
            "top_3_detections": enriched[:3],
            "all_detections": enriched
        }

# =============================================================================
# CONFIGURACIÓN GLOBAL
# =============================================================================

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

MODEL_PATH = "best.pt"
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()

def _resolver_ruta_recurso(nombre):
    """Devuelve la ruta de un archivo empaquetado (PyInstaller _MEIPASS) o local."""
    try:
        base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
        ruta = os.path.join(base, nombre)
        if os.path.exists(ruta):
            return ruta
    except Exception:
        pass
    return nombre

COLORS = {
    "bg_primary":      "#F8FAFC",
    "bg_secondary":    "#FFFFFF",
    "bg_surface":      "#F1F5F9",
    "bg_panel":        "#F1F5F9",
    "bg_card":         "#FFFFFF",
    "primary":         "#1D4ED8",
    "primary_hover":   "#1E40AF",
    "accent_gold":     "#D97706",
    "success":         "#059669",
    "success_light":   "#D1FAE5",
    "crack":           "#DC2626",
    "humidity":        "#0891B2",
    "spalling":        "#B45309",
    "text_primary":    "#1E293B",
    "text_secondary":  "#475569",
    "text_muted":      "#64748B",
    "border":          "#E2E8F0",
    "danger":          "#DC2626",
}

FONT_FAMILY = "Segoe UI"


# =============================================================================
# CLIENTE SUPABASE
# =============================================================================

class SupabaseClient:
    """Cliente REST para Supabase"""
    
    def __init__(self, url: str, key: str):
        self.url = url.rstrip("/")
        # Las nuevas publishable/secret keys (sb_publishable_/sb_secret_) se
        # envian SOLO en el header 'apikey'. El header 'Authorization: Bearer'
        # queda reservado para JWTs (legacy anon/service_role).
        is_jwt = not key.startswith("sb_publishable_") and not key.startswith("sb_secret_")
        self.headers = {
            "apikey": key,
            "Content-Type": "application/json",
        }
        if is_jwt:
            self.headers["Authorization"] = f"Bearer {key}"
    
    def ping(self) -> tuple:
        try:
            r = requests.get(
                f"{self.url}/rest/v1/inspection_results",
                headers={**self.headers, "Prefer": "count=exact"},
                params={"select": "id", "limit": "1"},
                timeout=8,
            )
            if 200 <= r.status_code < 300:
                return True, "Conectado a Supabase"
            return False, f"Error HTTP {r.status_code} al conectar con Supabase"
        except Exception as e:
            return False, "Sin conexión con Supabase"
    
    def insert(self, table: str, data) -> dict:
        try:
            headers = {**self.headers, "Prefer": "return=representation"}
            r = requests.post(
                f"{self.url}/rest/v1/{table}",
                headers=headers,
                json=data,
                timeout=15,
            )
            if r.status_code in (200, 201):
                return {"data": r.json(), "error": None, "success": True, "status": r.status_code}
            return {"data": None, "error": f"Error HTTP {r.status_code} en Supabase", "success": False, "status": r.status_code}
        except Exception as e:
            return {"data": None, "error": "Error de conexión con Supabase", "success": False, "status": 0}
    
    def upload_image(self, bucket: str, path: str, data) -> dict:
        """Sube un archivo (bytes) a Supabase Storage y devuelve la URL publica."""
        try:
            headers = {**self.headers, "Content-Type": "application/octet-stream"}
            r = requests.post(
                f"{self.url}/storage/v1/object/{bucket}/{path}",
                headers=headers,
                data=data,
                timeout=30,
            )
            if r.status_code in (200, 201):
                public_url = f"{self.url}/storage/v1/object/public/{bucket}/{path}"
                return {"success": True, "url": public_url, "status": r.status_code}
            return {"success": False, "error": f"Error HTTP {r.status_code} al subir imagen", "url": None, "status": r.status_code}
        except Exception as e:
            return {"success": False, "error": "Error de conexión al subir imagen", "url": None, "status": 0}
    
    def select(self, table: str, query: str = "*", order: str = None, 
               limit: int = None, filters: dict = None) -> dict:
        try:
            params = {"select": query}
            if order:
                params["order"] = order
            if limit:
                params["limit"] = limit
            if filters:
                params.update(filters)
            
            r = requests.get(
                f"{self.url}/rest/v1/{table}",
                headers=self.headers,
                params=params,
                timeout=30,
            )
            if 200 <= r.status_code < 300:
                return {"data": r.json(), "error": None, "success": True, "status": r.status_code}
            return {"data": None, "error": f"Error HTTP {r.status_code} en Supabase", "success": False, "status": r.status_code}
        except Exception as e:
            return {"data": None, "error": "Error de conexión con Supabase", "success": False, "status": 0}
    
    def delete_inspection(self, inspection_id: int) -> dict:
        try:
            r = requests.delete(
                f"{self.url}/rest/v1/inspection_results",
                headers=self.headers,
                params={"id": f"eq.{inspection_id}"},
                timeout=10,
            )
            if 200 <= r.status_code < 300:
                return {"success": True, "status": r.status_code}
            return {"success": False, "status": r.status_code, "error": f"Error HTTP {r.status_code} en Supabase"}
        except Exception as e:
            return {"success": False, "error": "Error de conexión con Supabase"}
    
    def check_duplicate_files(self, filenames: list) -> dict:
        if not filenames:
            return {"duplicates": [], "new_files": []}
        
        try:
            safe_pattern = re.compile(r"^[\w.\- áéíóúÁÉÍÓÚñÑüÜ]+$")
            filenames_clean = []
            for f in filenames:
                if safe_pattern.match(f):
                    filenames_clean.append(f.replace("'", "''"))
            if not filenames_clean:
                return {"duplicates": [], "new_files": filenames}

            filter_value = f"in.({','.join(filenames_clean)})"
            
            result = self.select(
                "inspection_results",
                "filename",
                filters={"filename": filter_value}
            )
            
            if result["success"] and result["data"]:
                existing_files = {row["filename"] for row in result["data"]}
                duplicates = [f for f in filenames if f in existing_files]
                new_files = [f for f in filenames if f not in existing_files]
                return {"duplicates": duplicates, "new_files": new_files}
            
            return {"duplicates": [], "new_files": filenames}
        except Exception as e:
            print(f"Error verificando duplicados: {e}")
            return {"duplicates": [], "new_files": filenames}
    
    def _upload_inspection_images(self, result, filename):
        """Sube la imagen original y la anotada a Storage.
        Devuelve dict con 'original' y 'anotada' (URLs publicas o None)."""
        try:
            base = os.path.splitext(str(filename))[0]
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe = re.sub(r"[^\w\-]+", "_", base)[:60] or "inspeccion"

            img_urls = {"original": None, "anotada": None}

            # 1) Imagen anotada (desde el array numpy RGB -> JPG en memoria)
            if result.get("annotated_image") is not None:
                img_bgr = cv2.cvtColor(result["annotated_image"], cv2.COLOR_RGB2BGR)
                ok_enc, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                if ok_enc:
                    path = f"{safe}_{ts}_anotada.jpg"
                    resp = self.upload_image("inspecciones", path, buf.tobytes())
                    if resp["success"]:
                        img_urls["anotada"] = resp["url"]

            # 2) Imagen original limpia (desde la ruta local)
            orig_path = result.get("image_path")
            if orig_path and os.path.exists(orig_path):
                with open(orig_path, "rb") as f:
                    data = f.read()
                path = f"{safe}_{ts}_original.jpg"
                resp = self.upload_image("inspecciones", path, data)
                if resp["success"]:
                    img_urls["original"] = resp["url"]

            return img_urls
        except Exception as e:
            print(f"Error subiendo imagenes: {e}")
            return {"original": None, "anotada": None}

    def save_inspection(self, result, filename):
        inspection_id = None
        try:
            # Subir las imagenes (original y anotada) a Storage para el historial
            image_urls = self._upload_inspection_images(result, filename)

            inspection_payload = {
                "filename": str(filename),
                "crack_count": int(result["class_counts"].get("crack", 0)),
                "humidity_count": int(result["class_counts"].get("humidity", 0)),
                "spalling_count": int(result["class_counts"].get("spalling", 0)),
                "total_detections": int(result["n_detections"]),
                "total_area_cm2": float(result["total_area_cm2"]),
                "total_area_m2": float(result["total_area_m2"]),
                "avg_confidence": float(result.get("avg_confidence", 0)),
                "imagen_original_url": image_urls["original"],
                "imagen_anotada_url": image_urls["anotada"],
            }
            
            resp = self.insert("inspection_results", inspection_payload)
            
            if not resp["success"]:
                return False, resp.get("error", "Error desconocido")
            
            if resp.get("data") and len(resp["data"]) > 0:
                inspection_id = resp["data"][0].get("id")
            
            if not inspection_id:
                return False, "No se recibió ID"
            
            detections = result.get("detections", [])
            if detections:
                details_payload = []
                for d in detections:
                    conf_val = float(d["Confianza"])
                    if conf_val > 1.0:
                        conf_val = conf_val / 100.0
                    
                    details_payload.append({
                        "inspection_id": int(inspection_id),
                        "class_name": str(d["Clase"]),
                        "confidence": float(conf_val),
                        "x1": int(d["x1"]),
                        "y1": int(d["y1"]),
                        "x2": int(d["x2"]),
                        "y2": int(d["y2"]),
                        "width_px": int(d["Ancho_px"]),
                        "height_px": int(d["Alto_px"]),
                        "area_cm2": float(d["Area_cm2"]),
                        "area_m2": float(d["Area_m2"]),
                        "gravity_level": d.get("gravedad", "leve"),
                        "gravity_score": d.get("gravedad_score", 1),
                    })
                
                if details_payload:
                    resp2 = self.insert("detection_details", details_payload)
                    if not resp2["success"]:
                        if inspection_id:
                            self.delete_inspection(inspection_id)
                        return False, f"Detalles: {resp2.get('error')} - Rollback ejecutado"
            
            return True, inspection_id
        except Exception as e:
            if inspection_id:
                self.delete_inspection(inspection_id)
            return False, str(e)
    
    def load_historical_data(self, limit=100):
        result = self.select(
            "inspection_results",
            "*",
            order="created_at.desc",
            limit=limit
        )
        if result["success"]:
            return result["data"]
        return []
    
    def load_detection_details(self, inspection_ids):
        if not inspection_ids:
            return {}
        
        ids_str = ", ".join(str(int(id)) for id in inspection_ids if id is not None)
        if not ids_str:
            return {}
        
        result = self.select(
            "detection_details",
            "inspection_id, class_name, area_m2, area_cm2, confidence, x1, y1, x2, y2, width_px, height_px, gravity_level, gravity_score",
            filters={"inspection_id": f"in.({ids_str})"},
            limit=10000
        )
        
        if not result["success"] or not result["data"]:
            return {}
        
        details_by_inspection = {}
        for detail in result["data"]:
            insp_id = detail.get("inspection_id")
            if insp_id not in details_by_inspection:
                details_by_inspection[insp_id] = []
            details_by_inspection[insp_id].append(detail)
        
        return details_by_inspection
    
    def get_stats(self):
        data = self.select("inspection_results", "*", limit=10000)
        if not data["success"] or not data["data"]:
            return {
                "total_inspections": 0,
                "total_detections": 0,
                "total_area_m2": 0,
                "avg_confidence": 0,
                "total_cracks": 0,
                "total_humidity": 0,
                "total_spalling": 0,
            }
        
        rows = data["data"]
        return {
            "total_inspections": len(rows),
            "total_detections": sum(r.get("total_detections", 0) for r in rows),
            "total_area_m2": sum(float(r.get("total_area_m2", 0)) for r in rows),
            "avg_confidence": sum(float(r.get("avg_confidence", 0)) for r in rows) / max(len(rows), 1),
            "total_cracks": sum(r.get("crack_count", 0) for r in rows),
            "total_humidity": sum(r.get("humidity_count", 0) for r in rows),
            "total_spalling": sum(r.get("spalling_count", 0) for r in rows),
        }

# =============================================================================
# MOTOR DE PROCESAMIENTO
# =============================================================================

def filtrar_falsos_positivos(detections, image_shape, umbrales_clase, img_bgr=None, modo_recorte=False):
    """Filtro geométrico + anti-sombras + anti-vegetación/cielo.
    modo_recorte=True (herramienta Foto Lejana): se relaja el filtro de material
    para crack, porque la zona recortada está enfocada por el usuario y las
    grietas reales sobre pintura gris neutra tienen bajo rango de color."""
    filtered_detections = []
    height, width = image_shape[:2]
    image_area = width * height
    
    for d in detections:
        cls_name = d["Clase"]
        conf = d["Confianza"]
        if conf > 1.0:
            conf = conf / 100.0
        
        box_width = d["Ancho_px"]
        box_height = d["Alto_px"]
        box_area = box_width * box_height
        aspect_ratio = box_width / max(box_height, 1)
        inverse_aspect_ratio = box_height / max(box_width, 1)
        
        reject = False
        
        if cls_name == "humidity":
            if box_area > (image_area * 0.05) and (aspect_ratio > 3.0 or inverse_aspect_ratio > 3.0):
                reject = True
            elif box_area < (image_area * 0.002):
                reject = True
            elif aspect_ratio > 4.5 or inverse_aspect_ratio > 4.5:
                umbral_min_crack = umbral_clase(umbrales_clase, "crack", 0.25)
                if conf > umbral_min_crack:
                    d["Clase"] = "crack"
                    cls_name = "crack"
                else:
                    reject = True

        elif cls_name == "spalling":
            umbral_min_spalling = umbral_clase(umbrales_clase, "spalling", 0.50)
            if conf < umbral_min_spalling:
                reject = True

        elif cls_name == "crack":
            umbral_min_crack = umbral_clase(umbrales_clase, "crack", 0.25)
            if conf < umbral_min_crack:
                reject = True

        # Control final de mínimo: ninguna detección por debajo del umbral de su
        # clase (ya reclasificada) llega a pantalla ni a Supabase.
        if cls_name not in CLASES_VALIDAS or conf < umbral_clase(umbrales_clase, cls_name, 0.25):
            reject = True
        
        # --- FILTRO ANTI-VEGETACION / ANTI-CIELO (el sillar no es verde ni azul cielo) ---
        if not reject and img_bgr is not None:
            h_img, w_img = img_bgr.shape[:2]
            ix1 = max(0, int(d["x1"])); iy1 = max(0, int(d["y1"]))
            ix2 = min(w_img - 1, int(d["x2"])); iy2 = min(h_img - 1, int(d["y2"]))
            if ix2 > ix1 and iy2 > iy1:
                patch = img_bgr[iy1:iy2, ix1:ix2].astype(np.float32)
                b_m = float(np.mean(patch[..., 0]))
                g_m = float(np.mean(patch[..., 1]))
                r_m = float(np.mean(patch[..., 2]))
                # vegetación viva: verde brillante dominante (sillar nunca es verde vivo)
                if g_m > 115 and (g_m - r_m) > 15 and (g_m - b_m) > 15 and (2*g_m - r_m - b_m) > 40:
                    reject = True
                # cielo azul dominante (B claramente mayor que R y G)
                elif b_m > max(r_m, g_m) + 25 and (b_m - r_m) > 40 and (b_m - g_m) > 25:
                    reject = True
                # superficie casi uniforme (techo liso, vidrio, pared sin daño): no es spalling
                elif cls_name == "spalling" and float(np.std(patch)) < 12.0:
                    reject = True
                # humedad identica a su entorno y de material neutro (grisaceo, p. ej. calamina
                # oxidada que cubre toda la superficie): textura de material, no mancha localizada.
                # La humedad real (madera, piedra) tiene color saturado y/o contrasta con su entorno.
                elif cls_name == "humidity":
                    rango = max(b_m, g_m, r_m) - min(b_m, g_m, r_m)
                    mean_lum = (b_m + g_m + r_m) / 3.0
                    # Sombre del suelo / arbolado en modo recorte: la humedad real
                    # del sillar (validada) tiene brillo medio alto (~110+) y un
                    # color beige neutro; una sombra es gris muy oscura (mean<90)
                    # y casi neutral. Se descarta para no pintar suelo en sombra.
                    if modo_recorte and mean_lum < 90.0 and rango < 15.0:
                        reject = True
                    margin = max(int(box_width * 0.5), int(box_height * 0.5), 60)
                    rx1 = max(0, ix1 - margin); ry1 = max(0, iy1 - margin)
                    rx2 = min(w_img - 1, ix2 + margin); ry2 = min(h_img - 1, iy2 + margin)
                    if rx2 > rx1 and ry2 > ry1:
                        big = img_bgr[ry1:ry2, rx1:rx2].astype(np.float32)
                        n_big = big.shape[0] * big.shape[1]
                        n_inner = patch.shape[0] * patch.shape[1]
                        n_ring = n_big - n_inner
                        if n_ring > box_area:
                            big_sum = big.sum(axis=(0, 1))
                            big_ss = (big * big).sum(axis=(0, 1))
                            inner_sum = patch.sum(axis=(0, 1))
                            inner_ss = (patch * patch).sum(axis=(0, 1))
                            ring_mean = (big_sum - inner_sum) / n_ring
                            ring_var = (big_ss - inner_ss) / n_ring - ring_mean ** 2
                            ring_var = np.clip(ring_var, 0, None)
                            ring_std = np.sqrt(ring_var)
                            box_mean = np.array([b_m, g_m, r_m])
                            box_std = np.std(patch, axis=(0, 1))
                            color_diff = float(np.mean(np.abs(ring_mean - box_mean)))
                            tex_diff = float(np.mean(np.abs(ring_std - box_std)))
                            if color_diff < 12.0 and tex_diff < 12.0 and rango < 40:
                                reject = True
                # madera (techo/vigas): marron-anaranjado muy rojizo (R domina G y B),
                # con textura (vetas). El sillar es beige neutro (R-G ~ 10-16).
                # Solo para crack: las vetas lineales parecen grietas, pero la
                # humedad sobre madera es daño real y NO debe filtrarse.
                elif cls_name == "crack" and r_m > 60 and (r_m - g_m) > 55 and (r_m - b_m) > 55 and float(np.std(patch)) > 12.0:
                    reject = True
                # FILTRO DE MATERIAL:
                # - Spalling (global y recorte): el sillar es beige calido (rango 25-28,
                #   textura std ~30). Calamina corrugada/muros lejanos (gris/oscuro
                #   rango<15 o textura excesiva std>50) NO son sillar.
                # - Crack GLOBAL: se mantiene estricto para no reintroducir los FPs
                #   de calamina/techo/edificio lejano validados por el usuario.
                # - Crack EN RECORTE (modo_recorte): a una grieta real sobre pared
                #   pintada gris le basta contener una LINEA OSCURA (percentil 1 muy
                #   bajo vs la media); si no la hay, es superficie plana = se descarta.
                elif cls_name == "spalling":
                    rango_mat = max(b_m, g_m, r_m) - min(b_m, g_m, r_m)
                    if rango_mat < 15.0:
                        reject = True
                    elif float(np.std(patch)) > 50.0:
                        reject = True
                elif cls_name == "crack":
                    rango_mat = max(b_m, g_m, r_m) - min(b_m, g_m, r_m)
                    if float(np.std(patch)) > 50.0:
                        reject = True
                    elif rango_mat < 15.0 and not modo_recorte:
                        reject = True
                    elif not modo_recorte:
                        # CALIBRACION AUTOMATICA POR GEOMETRIA (vista global):
                        # una caja muy alargada (relacion de aspecto >= 4) puede ser
                        # una grieta, pero tambien la sombra de una cornisa, una
                        # junta o un borde recto, que YOLO puntua ~50-60%. Si la
                        # red no esta segura (< 85%), se exige la misma confirmacion
                        # filiforme del modo recorte: la grieta real es sinuosa y de
                        # anchura variable; la sombra de cornisa es recta y uniforme.
                        if (conf < UMBRAL_ALTA_CONFIABILIDAD
                                and max(aspect_ratio, inverse_aspect_ratio) >= RELACION_ASPECTO_LINEAL
                                and not _es_grieta_filiforme(_parche_para_filiforme(patch))):
                            reject = True
                    else:
                        # En modo recorte el umbral de confianza se relaja (crack 0.10) y el
                        # filtro de material solo revisaba rango<15; con eso volvieron a pasar
                        # falsos positivos estructurales (techo corrugado, cables, suelo con
                        # sombras) que YOLO marca como crack con confianza fuerte. Se exige
                        # ahora la confirmacion filiforme SIEMPRE: una grieta real es sinuosa
                        # (tortuosidad alta), de anchura variable (no cable) y suficientemente
                        # larga; el techo/cable/sombra no la tienen. Contra-luz del recorte:
                        # una grieta sobre pintura se reconoce ademas por su linea oscura
                        # (percentil 1 claramente bajo la media).
                        # Pre-filtro de textura: la grieta real vive sobre pintura casi
                        # uniforme (std ~5-8); el techo corrugado / arco lejano tienen
                        # textura periodica alta (std 20-45). Si el parche es "texturado",
                        # no puede ser una grieta real aislada.
                        if float(np.std(patch)) > 15.0:
                            reject = True
                        elif not _es_grieta_filiforme(patch):
                            reject = True
        
        if not reject:
            d["Confianza"] = round(float(conf), 4)
            filtered_detections.append(d)
    
    return filtered_detections



def candidatos_grietas_finas(gray_img, area_min=60, ampliar=1.5,
                             filtro_forma=True, ang_max_alineacion=7.0,
                             discretos=True):
    """Binariza el mapa de respuesta del Hessiano y extrae cajas candidatas a
    "posible grieta" (líneas oscuras finas).

    Filtros geometricos de forma (criterio del diagnostico):
    - area_min / relacion de aspecto: descarta ruido y blobs.
    - tortuosidad > 1.05: una grieta es sinuosa; una junta o cable es rectilinea.
    - desviacion de orientacion local > 10 grados: misma idea.
    - constancia de anchura: si la anchura es demasiado constante (std < 0.3 px)
      es un cable/ranura artificial, no una grieta real.
    - alineacion +-7 grados con la orientacion dominante de la silleria: se
      penaliza (junta/molduras pintadas), no una grieta estructural.

    Devuelve lista de dicts con x1,y1,x2,y2 y metricas en coords de gray_img.
    """
    resp = deteccion_lineas_finas(gray_img)
    if resp.max() == 0:
        return []

    thr, _ = cv2.threshold(resp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(int(thr), 40)
    binary = cv2.threshold(resp, max(40, thr - 15), 255, cv2.THRESH_BINARY)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Orientacion dominante de la silleria (para penalizar juntas/moldura pintadas)
    dom_orient = None
    if filtro_forma:
        edges = cv2.Canny(binary, 20, 60)
        dom_orient = _orientacion_dominante(edges)

    # Derivadas del mapa de respuesta (para orientacion local)
    gx_full = cv2.Sobel(resp, cv2.CV_32F, 1, 0, ksize=3)
    gy_full = cv2.Sobel(resp, cv2.CV_32F, 0, 1, ksize=3)

    contornos, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    crudos = []
    for c in contornos:
        area = float(cv2.contourArea(c))
        if area < area_min:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        ratio_dim = max(cw, ch) / max(1, min(cw, ch))
        if ratio_dim < 2.0:
            continue
        crudos.append({"contour": c, "x": x, "y": y, "cw": cw, "ch": ch,
                       "area_px": int(area), "contours": [c]})

    # Fusion por proximidad (misma grieta fragmentada)
    fusionados = []
    for c in crudos:
        nearest = None
        best_dist = float("inf")
        for f in fusionados:
            d = ((c["x"] - f["x"]) ** 2 + (c["y"] - f["y"]) ** 2) ** 0.5
            cw = max(c["cw"], f["cw"])
            ch = max(c["ch"], f["ch"])
            dist_rel = d / (cw + ch + 1e-6)
            if dist_rel < 0.6 and dist_rel < best_dist:
                nearest = f
                best_dist = dist_rel
        if nearest is not None:
            nearest["x"] = min(nearest["x"], c["x"])
            nearest["y"] = min(nearest["y"], c["y"])
            nearest["cw"] = max(nearest["x"] + nearest["cw"],
                                c["x"] + c["cw"]) - nearest["x"]
            nearest["ch"] = max(nearest["y"] + nearest["ch"],
                                c["y"] + c["ch"]) - nearest["y"]
            nearest["area_px"] += c["area_px"]
            nearest["contours"].append(c["contour"])
        else:
            fusionados.append(c)

    # Por cada componente fusionado: mascara + metricas + filtros de forma
    H, W = gray_img.shape[:2]
    candidatos = []
    for f in fusionados:
        x, y = max(0, f["x"]), max(0, f["y"])
        cw = min(W - x, f["cw"])
        ch = min(H - y, f["ch"])
        if cw <= 0 or ch <= 0:
            continue
        mask = np.zeros((ch, cw), dtype=np.uint8)
        for cc in f["contours"]:
            cv2.drawContours(mask, [cc], -1, 255, -1,
                             offset=(-f["x"], -f["y"]))
        res_sub = resp[y:y + ch, x:x + cw]
        gx_sub = gx_full[y:y + ch, x:x + cw]
        gy_sub = gy_full[y:y + ch, x:x + cw]
        f["x1"], f["y1"], f["x2"], f["y2"] = x, y, x + cw, y + ch

        if not filtro_forma:
            cand = {"x1": x, "y1": y, "x2": x + cw, "y2": y + ch,
                    "area_px": f["area_px"]}
            candidatos.append(cand)
            continue

        met = _metricas_filamento(mask, res_sub, gx_sub, gy_sub, f["contour"])
        if met is None:
            continue
        # 1) Tortuosidad: grieta real es sinuosa (>1.05). Juntas/cables ~1.0.
        if met["tortuosidad"] < 1.05:
            continue
        # 2) Dispersion angular local > 10 grados (no rectilinea).
        if met["std_ang_deg"] < 10.0:
            continue
        # 3) Anchura casi constante (<0.3 px) = cable/ranura artificial.
        if met["std_ancho_px"] < 0.3:
            continue
        # 4) Alineado +-7 grados con orientacion dominante de la silleria.
        if dom_orient is not None:
            diff = abs((met["orientacion_deg"] - dom_orient + 90.0) % 180.0 - 90.0)
            if diff < ang_max_alineacion:
                continue
        cand = {
            "x1": x, "y1": y, "x2": x + cw, "y2": y + ch,
            "area_px": f["area_px"],
            "tortuosidad": round(met["tortuosidad"], 3),
            "std_ang_deg": round(met["std_ang_deg"], 2),
            "std_ancho_px": round(met["std_ancho_px"], 3),
            "ancho_medio_px": round(met["ancho_medio_px"], 2),
            "longitud_px": round(met["longitud_px"], 1),
            "arco_px": round(met["arco_px"], 1),
            "orientacion_deg": round(met["orientacion_deg"], 1),
            "orientacion_dom": None if dom_orient is None else round(dom_orient, 1),
        }
        candidatos.append(cand)

    # Ampliar levemente la caja para que sea visible con el grosor del cuadro
    for f in candidatos:
        cw = f["x2"] - f["x1"]
        ch = f["y2"] - f["y1"]
        padx = int(max(cw * (ampliar - 1) / 2, 6))
        pady = int(max(ch * (ampliar - 1) / 2, 6))
        f["x1"] = max(0, f["x1"] - padx)
        f["y1"] = max(0, f["y1"] - pady)
        f["x2"] = f["x2"] + padx
        f["y2"] = f["y2"] + pady

    return candidatos


def dibujar_cajas(img_cv2, detections):
    """Dibuja bounding boxes con etiquetas y colores de gravedad"""
    color_map = {"crack": (0, 0, 220), "humidity": (240, 100, 30), "spalling": (0, 140, 235)}
    
    for d in detections:
        color = color_map.get(d["Clase"], (128, 128, 128))
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        
        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (255, 255, 255), 12)
        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color, 8)
        
        conf_porcentaje = d["Confianza"] * 100
        gravity_label = d.get("gravedad_label", "")
        label = f"{d['Clase']} {conf_porcentaje:.1f}% {gravity_label}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 3)[0]
        label_rect_origin = (x1, max(y1 - label_size[1] - 10, label_size[1] + 10))
        label_text_origin = (x1, max(y1, label_size[1] + 10))
        
        cv2.rectangle(img_cv2, 
                      (label_rect_origin[0], label_rect_origin[1] - label_size[1]),
                      (label_rect_origin[0] + label_size[0], label_rect_origin[1]), 
                      (255, 255, 255), -1)
        cv2.putText(img_cv2, label, label_text_origin, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    
    return img_cv2


def procesar_imagen_completa(model, image_path, class_thresholds, iou_threshold,
                              cm_per_pixel, tile_size=640, overlap=0.2, use_tiling=True,
                              progress_callback=None, modo_recorte=False, zoom_pass=False):
    """Motor completo con tiling + filtros geométricos.
    image_path puede ser una ruta de archivo (str) o un array BGR en memoria (np.ndarray)."""
    if isinstance(image_path, str):
        img_cv2 = cv2.imread(image_path)
    else:
        img_cv2 = np.array(image_path)
    if img_cv2 is None:
        return {"success": False, "error": "No se pudo cargar la imagen"}
    
    h, w = img_cv2.shape[:2]
    all_detections_raw = []
    inference_log = []
    
    if use_tiling:
        tiles = split_image_into_tiles(img_cv2, tile_size, overlap)
        total_tiles = len(tiles)
        tiles_sin_hits = []
        
        for idx, (tile, x_off, y_off) in enumerate(tiles):
            if progress_callback:
                progress_callback(idx / (total_tiles + 1))
            
            # ultralytics asume BGR en arreglos numpy: se pasa el tile BGR tal cual
            
            try:
                t0 = time.time()
                results = model.predict(
                    source=tile, imgsz=tile_size, conf=0.05, iou=iou_threshold,
                    verbose=False, show=False, augment=True,
                    agnostic_nms=False, retina_masks=True,
                    half=torch.cuda.is_available()
                )
                elapsed = (time.time() - t0) * 1000
                n_dets = len(results[0].boxes) if results[0].boxes is not None else 0
                inference_log.append({
                    "tile": (x_off, y_off),
                    "detections": n_dets,
                    "time_ms": round(elapsed, 1)
                })
                
                # Teselas sin hits se registran para el segundo pase con zoom x2
                if n_dets == 0:
                    tiles_sin_hits.append((tile, x_off, y_off))
                    continue
                
                if results[0].boxes is None:
                    continue
                
                for i, box in enumerate(results[0].boxes):
                    cls_id = int(box.cls[0])
                    cls_name = normalizar_clase(model.names.get(cls_id, ""))
                    if cls_name is None:
                        continue
                    conf = float(box.conf[0])

                    if conf < umbral_clase(class_thresholds, cls_name, 0.05):
                        continue
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1 += x_off
                    y1 += y_off
                    x2 += x_off
                    y2 += y_off
                    
                    x1 = max(0, min(x1, w-1))
                    y1 = max(0, min(y1, h-1))
                    x2 = max(0, min(x2, w-1))
                    y2 = max(0, min(y2, h-1))
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    all_detections_raw.append({
                        "Clase": cls_name,
                        "Confianza": conf,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "Ancho_px": x2 - x1,
                        "Alto_px": y2 - y1,
                        "_poly": _poligono_mascara(results[0], i, x_off, y_off),
                    })
            except Exception as e:
                inference_log.append({
                    "tile": (x_off, y_off),
                    "error": str(e),
                    "status": "failed"
                })
        
        # SEGUNDO PASE (opcional, zoom_pass=True): zoom x2 (Lanczos) en las teselas
        # sin detecciones del primer pase. Una grieta de 1-2 px sobre pintura gris
        # queda invisible en el primer pase (stride 8 de YOLO); ampliando x2 gana
        # espesor y se vuelve detectable. Es ADAPTATIVO A NIVEL IMAGEN: solo se
        # ejecuta si NINGUNA tesela del primer pase encontró daño (foto lejana
        # limpia). En fotos normales (con daños detectados) no se ejecuta, así se
        # preservan los resultados validados y no se duplica el tiempo de proceso.
        if zoom_pass and not all_detections_raw and tiles_sin_hits:
            n_pass2 = len(tiles_sin_hits)
            for idx, (tile, x_off, y_off) in enumerate(tiles_sin_hits):
                if progress_callback:
                    progress_callback((total_tiles + idx + 1) / (total_tiles + n_pass2))
                
                tile_z = cv2.resize(tile, (tile_size * 2, tile_size * 2),
                                    interpolation=cv2.INTER_LANCZOS4)
                # ultralytics asume BGR en arreglos numpy (no convertir a RGB)
                
                try:
                    t0 = time.time()
                    results = model.predict(
                        source=tile_z, imgsz=tile_size * 2, conf=0.05, iou=iou_threshold,
                        verbose=False, show=False, augment=True,
                        agnostic_nms=False, retina_masks=True,
                        half=torch.cuda.is_available()
                    )
                    elapsed = (time.time() - t0) * 1000
                    n_dets = len(results[0].boxes) if results[0].boxes is not None else 0
                    inference_log.append({
                        "tile": (x_off, y_off),
                        "zoom_x2": True,
                        "detections": n_dets,
                        "time_ms": round(elapsed, 1)
                    })
                    
                    if n_dets == 0 or results[0].boxes is None:
                        continue
                    
                    for i, box in enumerate(results[0].boxes):
                        cls_id = int(box.cls[0])
                        cls_name = normalizar_clase(model.names.get(cls_id, ""))
                        if cls_name is None:
                            continue
                        conf = float(box.conf[0])

                        if conf < umbral_clase(class_thresholds, cls_name, 0.05):
                            continue
                        
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        # Coordenadas en el espacio ampliado x2: se reducen y se suma el offset
                        x1 = x1 // 2 + x_off
                        y1 = y1 // 2 + y_off
                        x2 = x2 // 2 + x_off
                        y2 = y2 // 2 + y_off
                        
                        x1 = max(0, min(x1, w-1))
                        y1 = max(0, min(y1, h-1))
                        x2 = max(0, min(x2, w-1))
                        y2 = max(0, min(y2, h-1))
                        
                        if x2 <= x1 or y2 <= y1:
                            continue
                        
                        all_detections_raw.append({
                            "Clase": cls_name,
                            "Confianza": conf,
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "Ancho_px": x2 - x1,
                            "Alto_px": y2 - y1,
                            "zoom_x2": True,
                            "_poly": _poligono_mascara(results[0], i, x_off, y_off, escala=2),
                        })
                except Exception as e:
                    inference_log.append({
                        "tile": (x_off, y_off),
                        "zoom_x2": True,
                        "error": str(e),
                        "status": "failed"
                    })
        
        if progress_callback:
            progress_callback(1.0)
        
        if all_detections_raw:
            boxes_by_class = {}
            scores_by_class = {}
            polys_by_class = {}

            for d in all_detections_raw:
                cls = d["Clase"]
                boxes_by_class.setdefault(cls, []).append([d["x1"], d["y1"], d["x2"], d["y2"]])
                scores_by_class.setdefault(cls, []).append(d["Confianza"])
                polys_by_class.setdefault(cls, []).append(d.get("_poly"))
            
            final_detections = []
            # NMS por clase con el mismo IoU para las 3 clases (0.45 por defecto).
            # Los duplicados de spalling en bloques contiguos los resuelve despues
            # agrupar_zonas, sin contar dos veces el area.
            for cls, boxes in boxes_by_class.items():
                scores = scores_by_class[cls]
                keep = nms_boxes(boxes, scores, iou_threshold)
                
                for idx in keep:
                    d = {
                        "Clase": cls,
                        "Confianza": scores[idx],
                        "x1": boxes[idx][0], "y1": boxes[idx][1],
                        "x2": boxes[idx][2], "y2": boxes[idx][3],
                        "Ancho_px": boxes[idx][2] - boxes[idx][0],
                        "Alto_px": boxes[idx][3] - boxes[idx][1],
                        "_poly": polys_by_class[cls][idx],
                    }
                    final_detections.append(d)
        else:
            final_detections = []
    else:
        # ultralytics asume BGR en arreglos numpy: se pasa la imagen BGR
        imgsz_options = [1536, 1280, 960, 640] if max(w, h) > 800 else [1280, 640]
        
        results = None
        for imgsz in imgsz_options:
            try:
                t0 = time.time()
                results = model.predict(
                    source=img_cv2, imgsz=imgsz, conf=0.05, iou=iou_threshold,
                    verbose=False, show=False, augment=True,
                    agnostic_nms=False, retina_masks=True,
                    half=torch.cuda.is_available()
                )
                n = len(results[0].boxes) if results[0].boxes is not None else 0
                inference_log.append({
                    "imgsz": imgsz,
                    "time_ms": round((time.time() - t0) * 1000, 1),
                    "detections": n,
                    "status": "success"
                })
                if n > 0:
                    break
            except Exception as e:
                inference_log.append({
                    "imgsz": imgsz,
                    "error": str(e),
                    "status": "failed"
                })
        
        final_detections = []
        if results is not None and results[0].boxes is not None:
            for i, box in enumerate(results[0].boxes):
                cls_id = int(box.cls[0])
                cls_name = normalizar_clase(model.names.get(cls_id, ""))
                if cls_name is None:
                    continue
                conf = float(box.conf[0])

                if conf < umbral_clase(class_thresholds, cls_name, 0.25):
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                final_detections.append({
                    "Clase": cls_name,
                    "Confianza": conf,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "Ancho_px": x2 - x1,
                    "Alto_px": y2 - y1,
                    "_poly": _poligono_mascara(results[0], i),
                })
    
    detections_filtradas = filtrar_falsos_positivos(final_detections, img_cv2.shape, class_thresholds, img_cv2, modo_recorte=modo_recorte)
    
    # Fusionar cuadros de la misma clase que forman parte de la misma zona afectada:
    # UN recuadro por zona cubriendo toda el area danada (reduce duplicados y hace
    # el resultado mas legible).
    detections_filtradas = agrupar_zonas(detections_filtradas)
    
    for d in detections_filtradas:
        # Area real de la zona (union sin solapes), no la del recuadro envolvente
        area_px = d.get("Area_px", d["Ancho_px"] * d["Alto_px"])
        area_cm2 = area_px * (cm_per_pixel ** 2)
        d["Area_cm2"] = round(area_cm2, 2)
        d["Area_m2"] = round(area_cm2 / 10000, 5)
    
    # Enriquecer con gravedad
    detections_filtradas = enrich_detections_with_gravity(detections_filtradas)
    
    class_counts = {"crack": 0, "humidity": 0, "spalling": 0}
    total_area_cm2 = 0
    confidences = []
    
    for d in detections_filtradas:
        class_counts[d["Clase"]] += 1
        total_area_cm2 += d["Area_cm2"]
        confidences.append(d["Confianza"])
    
    img_con_cajas = dibujar_cajas(img_cv2.copy(), detections_filtradas)
    img_rgb = cv2.cvtColor(img_con_cajas, cv2.COLOR_BGR2RGB)
    
    preprocessing = f"Tiling (tile={tile_size}px, overlap={int(overlap*100)}%) + filtro geometrico" if use_tiling else "Nativa con filtro geometrico"
    
    return {
        "success": True,
        "annotated_image": img_rgb,
        "detections": detections_filtradas,
        "class_counts": class_counts,
        "total_area_cm2": round(total_area_cm2, 2),
        "total_area_m2": round(total_area_cm2 / 10000, 4),
        "n_detections": len(detections_filtradas),
        "avg_confidence": round(np.mean(confidences) * 100, 1) if confidences else 0,
        "preprocessing": preprocessing,
        "inference_log": inference_log,
    }

# =============================================================================
# APLICACIÓN PRINCIPAL
# =============================================================================

class HeritageDetectorDesktop(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Heritage Damage Detector v6.5 - Desktop Edition | UCSM")
        self.geometry("1500x900")
        self.minsize(1200, 700)
        
        self.model = None
        self.db = SupabaseClient(SUPABASE_URL, SUPABASE_KEY)
        if SUPABASE_URL and SUPABASE_KEY:
            ok, msg = self.db.ping()
            if not ok:
                messagebox.showwarning("Supabase", f"No se pudo conectar a Supabase:\n{msg}\n\nLa app funcionará pero no guardará datos.")
        else:
            messagebox.showwarning("Supabase", "Sin credenciales de Supabase.\nConfigura SUPABASE_URL y SUPABASE_KEY en .env\n\nLa app funcionará pero no guardará datos.")
        
        self.current_image_path = None
        self.last_result = None
        self.current_module = "Analisis"
        
        # Variables para el recorte de zona (fotos lejanas)
        self.crop_mode = False
        self.crop_start = None
        self.crop_rect_id = None
        self._last_crop_result = None
        
        # Variables para procesamiento por lotes
        self.processing_queue = []
        self.current_queue_index = -1
        self.all_results = []
        self.current_view_index = 0
        self.last_queued_paths = []
        
        self.class_thresholds = {"crack": 0.20, "humidity": 0.30, "spalling": 0.60}
        self.iou_threshold = IOU_NMS
        self.cm_per_pixel = 0.13
        self.use_tiling = True
        self.tile_size = 640
        self.overlap = 0.20
        
        self.load_model()
        self.create_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.alert_system = AlertSystem(thresholds={
            "max_damage_area_m2": 0.5,
            "min_new_cracks": 10,
            "growth_percentage_threshold": 20,
            "min_avg_confidence": 70.0
        })
    
    def load_model(self):
        try:
            ruta_modelo = _resolver_ruta_recurso(MODEL_PATH)
            if not os.path.exists(ruta_modelo):
                messagebox.showerror("Error", f"No se encontró {MODEL_PATH}\n\nColoca best.pt en la misma carpeta.")
                return
            
            self.model = YOLO(ruta_modelo)
            _ = self.model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False, imgsz=640)
        except Exception as e:
            messagebox.showerror("Error al cargar modelo",
                                 f"No se pudo cargar el modelo best.pt. Verifica que el archivo sea válido.\n\nDetalle: {type(e).__name__}")
    
    def create_ui(self):
        self.configure(fg_color=COLORS["bg_primary"])
        
        self.sidebar = ctk.CTkFrame(self, fg_color=COLORS["bg_secondary"], width=260)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        
        brand_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        brand_frame.pack(fill="x", padx=15, pady=(20, 10))
        
        ctk.CTkLabel(brand_frame, text="HERITAGE", font=(FONT_FAMILY, 22, "bold"), text_color=COLORS["text_primary"]).pack()
        ctk.CTkLabel(brand_frame, text="DETECTOR", font=(FONT_FAMILY, 22, "bold"), text_color=COLORS["accent_gold"]).pack()
        ctk.CTkLabel(brand_frame, text="v6.5 · Desktop · UCSM", font=(FONT_FAMILY, 9), text_color=COLORS["text_muted"]).pack(pady=(5, 0))
        
        separator = ctk.CTkFrame(self.sidebar, height=2, fg_color=COLORS["accent_gold"], corner_radius=1)
        separator.pack(fill="x", padx=20, pady=15)
        
        self.nav_buttons = {}
        modules = [
            ("Analisis", "Análisis en Campo"),
            ("Dashboard", "Cuadro de Mandos"),
            ("Historial", "Historial"),
            ("Especificacion", "Especificación Técnica"),
        ]
        
        for key, label in modules:
            btn = ctk.CTkButton(
                self.sidebar, text=label,
                command=lambda k=key: self.switch_module(k),
                font=(FONT_FAMILY, 12, "bold"),
                fg_color="transparent",
                text_color=COLORS["text_primary"],
                hover_color=COLORS["bg_surface"],
                anchor="w", height=40
            )
            btn.pack(fill="x", padx=15, pady=3)
            self.nav_buttons[key] = btn
        
        separator2 = ctk.CTkFrame(self.sidebar, height=2, fg_color=COLORS["accent_gold"], corner_radius=1)
        separator2.pack(fill="x", padx=20, pady=15)
        
        ok, msg = self.db.ping()
        status_text = "Supabase conectado" if ok else "Supabase offline"
        status_color = COLORS["success"] if ok else COLORS["danger"]
        status_bg = COLORS["success_light"] if ok else COLORS["bg_surface"]
        db_status = ctk.CTkFrame(self.sidebar, fg_color=status_bg, corner_radius=8, border_width=1, border_color=COLORS["border"])
        db_status.pack(fill="x", padx=15, pady=5)
        
        ctk.CTkLabel(db_status, text=status_text, font=(FONT_FAMILY, 10, "bold"), text_color=status_color).pack(pady=8)
        
        ctk.CTkLabel(self.sidebar, text="Santuario de San Agustin\nArequipa, Peru", font=(FONT_FAMILY, 9), text_color=COLORS["text_muted"], justify="center").pack(side="bottom", pady=15)
        
        self.content = ctk.CTkFrame(self, fg_color=COLORS["bg_primary"])
        self.content.pack(side="right", fill="both", expand=True)
        
        self.show_module_analisis()
    
    def switch_module(self, module_name):
        self.current_module = module_name
        
        for widget in self.content.winfo_children():
            widget.destroy()
        
        for key, btn in self.nav_buttons.items():
            if key == module_name:
                btn.configure(fg_color=COLORS["bg_surface"], text_color=COLORS["text_primary"], border_width=3, border_color=COLORS["primary"])
            else:
                btn.configure(fg_color="transparent", text_color=COLORS["text_primary"], border_width=0)
        
        if module_name == "Analisis":
            self.show_module_analisis()
        elif module_name == "Dashboard":
            self.show_module_dashboard()
        elif module_name == "Historial":
            self.show_module_historial()
        elif module_name == "Especificacion":
            self.show_module_especificacion()
    
    def show_module_analisis(self):
        header = ctk.CTkFrame(self.content, fg_color=COLORS["bg_secondary"], height=90, corner_radius=8)
        header.pack(fill="x", padx=15, pady=(15, 5))
        header.pack_propagate(False)
        
        ctk.CTkLabel(header, text="INSPECCIÓN ESTRUCTURAL", font=(FONT_FAMILY, 10, "bold"), text_color=COLORS["accent_gold"]).pack(pady=(15, 2))
        ctk.CTkLabel(header, text="Análisis de Sillar Volcánico", font=(FONT_FAMILY, 22, "bold"), text_color=COLORS["text_primary"]).pack()
        ctk.CTkLabel(header, text="Filtrado geométrico anti-porosidad + anti-sombras + micro-inferencia por tiling + HOVER INTERACTIVO", font=(FONT_FAMILY, 11), text_color=COLORS["text_secondary"], justify="center").pack(pady=(6, 12))
        
        main = ctk.CTkFrame(self.content, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=15, pady=15)
        
        left = ctk.CTkFrame(main, fg_color=COLORS["bg_card"], width=380, corner_radius=12, border_width=1, border_color=COLORS["border"])
        left.pack(side="left", fill="both", padx=(0, 10))
        left.pack_propagate(False)
        
        right = ctk.CTkFrame(main, fg_color=COLORS["bg_card"], corner_radius=12, border_width=1, border_color=COLORS["border"])
        right.pack(side="right", fill="both", expand=True)
        
        self.create_control_panel(left)
        self.create_visualization_panel(right)
    
    def create_control_panel(self, parent):
        scroll = ctk.CTkScrollableFrame(parent, fg_color=COLORS["bg_card"])
        scroll.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(scroll, text="CARGA DE IMAGEN", font=(FONT_FAMILY, 11, "bold"), text_color=COLORS["accent_gold"]).pack(pady=(10, 5))
        
        self.btn_load = ctk.CTkButton(
            scroll, text="Cargar Imagen(es) del Monumento",
            command=self.load_image,
            font=(FONT_FAMILY, 11, "bold"),
            fg_color=COLORS["primary"], hover_color=COLORS["primary_hover"],
            height=40
        )
        self.btn_load.pack(pady=5, fill="x")
        
        self.btn_cancel = ctk.CTkButton(
            scroll, text="CANCELAR PROCESAMIENTO",
            command=self.cancel_processing,
            font=(FONT_FAMILY, 10, "bold"),
            fg_color="transparent", border_width=2, border_color=COLORS["danger"],
            text_color=COLORS["danger"], hover_color=COLORS["bg_surface"],
            height=35
        )
        self.btn_cancel.pack(pady=5, fill="x")
        
        self.lbl_file = ctk.CTkLabel(scroll, text="Ningún archivo cargado", font=(FONT_FAMILY, 9), text_color=COLORS["text_muted"])
        self.lbl_file.pack(pady=(0, 10))
        
        separator = ctk.CTkFrame(scroll, height=2, fg_color=COLORS["accent_gold"], corner_radius=1)
        separator.pack(fill="x", pady=8)
        
        ctk.CTkLabel(scroll, text="CALIBRACIÓN PARA SILLAR", font=(FONT_FAMILY, 11, "bold"), text_color=COLORS["accent_gold"]).pack(pady=(10, 5))
        
        # ---- Preset de calibración (por defecto óptimo o alta confiabilidad 85%) ----
        self.preset_var = ctk.StringVar(value="recomendado")
        preset_row = ctk.CTkFrame(scroll, fg_color="transparent")
        preset_row.pack(fill="x", pady=(2, 4))
        ctk.CTkLabel(preset_row, text="Modo de detección:", font=(FONT_FAMILY, 10), text_color=COLORS["text_primary"], anchor="w").pack(anchor="w")
        self.opt_recomendado = ctk.CTkRadioButton(
            preset_row, text="Recomendado (detecta más daños)", variable=self.preset_var, value="recomendado",
            command=self.apply_preset, font=(FONT_FAMILY, 10), text_color=COLORS["text_secondary"], fg_color=COLORS["primary"])
        self.opt_recomendado.pack(anchor="w", pady=(3, 0))
        self.opt_85 = ctk.CTkRadioButton(
            preset_row, text="Alta confiabilidad (85%)", variable=self.preset_var, value="alto",
            command=self.apply_preset, font=(FONT_FAMILY, 10), text_color=COLORS["text_secondary"], fg_color=COLORS["primary"])
        self.opt_85.pack(anchor="w", pady=(2, 0))
        
        # ---- Umbrales simples (siempre visibles) ----
        ctk.CTkLabel(scroll, text="Grietas (Crack):", font=(FONT_FAMILY, 10), text_color=COLORS["text_primary"], anchor="w").pack(fill="x", pady=(5, 0))
        self.slider_crack = ctk.CTkSlider(scroll, from_=0.05, to=0.90, number_of_steps=17, command=lambda v: self.on_threshold_change(self.lbl_crack, v))
        self.slider_crack.set(0.20)
        self.slider_crack.pack(fill="x", pady=2)
        self.lbl_crack = ctk.CTkLabel(scroll, text="20%", font=(FONT_FAMILY, 9, "bold"), text_color=COLORS["crack"])
        self.lbl_crack.pack()
        
        ctk.CTkLabel(scroll, text="Humedad (Humidity):", font=(FONT_FAMILY, 10), text_color=COLORS["text_primary"], anchor="w").pack(fill="x", pady=(5, 0))
        self.slider_humidity = ctk.CTkSlider(scroll, from_=0.05, to=0.90, number_of_steps=17, command=lambda v: self.on_threshold_change(self.lbl_humidity, v))
        self.slider_humidity.set(0.30)
        self.slider_humidity.pack(fill="x", pady=2)
        self.lbl_humidity = ctk.CTkLabel(scroll, text="30%", font=(FONT_FAMILY, 9, "bold"), text_color=COLORS["humidity"])
        self.lbl_humidity.pack()
        
        ctk.CTkLabel(scroll, text="Desprendimiento (Spalling):", font=(FONT_FAMILY, 10), text_color=COLORS["text_primary"], anchor="w").pack(fill="x", pady=(5, 0))
        self.slider_spalling = ctk.CTkSlider(scroll, from_=0.05, to=0.90, number_of_steps=17, command=lambda v: self.on_threshold_change(self.lbl_spalling, v))
        self.slider_spalling.set(0.60)
        self.slider_spalling.pack(fill="x", pady=2)
        self.lbl_spalling = ctk.CTkLabel(scroll, text="60%", font=(FONT_FAMILY, 9, "bold"), text_color=COLORS["spalling"])
        self.lbl_spalling.pack()
        
        ctk.CTkButton(
            scroll, text="↺ Restaurar calibración recomendada",
            command=self.restore_recommended_calibration,
            font=(FONT_FAMILY, 10, "bold"),
            fg_color=COLORS["primary"], hover_color=COLORS["primary_hover"], height=28
        ).pack(fill="x", pady=(8, 2))
        
        separator2 = ctk.CTkFrame(scroll, height=2, fg_color=COLORS["accent_gold"], corner_radius=1)
        separator2.pack(fill="x", pady=8)
        
        # ---- Configuración avanzada (plegable, inicia cerrada para evitar confusiones al scroll) ----
        self.adv_open = False
        self.btn_advanced = ctk.CTkButton(
            scroll, text="▸ Configuración avanzada",
            command=self.toggle_advanced,
            font=(FONT_FAMILY, 10, "bold"),
            fg_color="transparent", border_width=1, border_color=COLORS["border"],
            text_color=COLORS["text_secondary"], hover_color=COLORS["bg_surface"], height=30
        )
        self.btn_advanced.pack(fill="x", pady=(6, 2))
        
        self.adv_frame = ctk.CTkFrame(scroll, fg_color=COLORS["bg_panel"], corner_radius=8)
        # (No se empaqueta hasta que se abra, para que no interfiera con el scroll)
        
        ctk.CTkLabel(self.adv_frame, text="IoU Threshold:", font=(FONT_FAMILY, 10), text_color=COLORS["text_primary"], anchor="w").pack(fill="x", pady=(8, 0), padx=10)
        self.slider_iou = ctk.CTkSlider(self.adv_frame, from_=0.1, to=0.9, number_of_steps=16, command=lambda v: self.update_label(self.lbl_iou, v))
        self.slider_iou.set(IOU_NMS)
        self.slider_iou.pack(fill="x", pady=2, padx=10)
        self.lbl_iou = ctk.CTkLabel(self.adv_frame, text="45%", font=(FONT_FAMILY, 9, "bold"), text_color=COLORS["text_primary"])
        self.lbl_iou.pack()
        
        ctk.CTkLabel(self.adv_frame, text="GSD (cm/pixel):", font=(FONT_FAMILY, 10), text_color=COLORS["text_primary"], anchor="w").pack(fill="x", pady=(5, 0), padx=10)
        self.entry_gsd = ctk.CTkEntry(self.adv_frame, width=100, font=(FONT_FAMILY, 11), fg_color=COLORS["bg_surface"], border_color=COLORS["border"], text_color=COLORS["text_primary"])
        self.entry_gsd.insert(0, "0.13")
        self.entry_gsd.pack(pady=2, padx=10)
        
        self.chk_tiling = ctk.CTkCheckBox(self.adv_frame, text="Activar Tiling (detección de detalles pequeños)", command=self.toggle_tiling, font=(FONT_FAMILY, 10), text_color=COLORS["text_primary"], fg_color=COLORS["primary"], hover_color=COLORS["primary_hover"])
        self.chk_tiling.select()
        self.chk_tiling.pack(anchor="w", pady=(5, 2), padx=10)
        
        ctk.CTkLabel(self.adv_frame, text="Tamaño del tile (px):", font=(FONT_FAMILY, 10), text_color=COLORS["text_primary"], anchor="w").pack(fill="x", pady=(5, 0), padx=10)
        self.slider_tile = ctk.CTkSlider(self.adv_frame, from_=320, to=1280, number_of_steps=15, command=lambda v: self.update_label(self.lbl_tile, v, is_int=True))
        self.slider_tile.set(640)
        self.slider_tile.pack(fill="x", pady=2, padx=10)
        self.lbl_tile = ctk.CTkLabel(self.adv_frame, text="640 px", font=(FONT_FAMILY, 9, "bold"), text_color=COLORS["text_primary"])
        self.lbl_tile.pack()
        
        ctk.CTkLabel(self.adv_frame, text="Solapamiento (%):", font=(FONT_FAMILY, 10), text_color=COLORS["text_primary"], anchor="w").pack(fill="x", pady=(5, 0), padx=10)
        self.slider_overlap = ctk.CTkSlider(self.adv_frame, from_=0, to=50, number_of_steps=10, command=lambda v: self.update_label(self.lbl_overlap, v, is_int=True))
        self.slider_overlap.set(20)
        self.slider_overlap.pack(fill="x", pady=2, padx=10)
        self.lbl_overlap = ctk.CTkLabel(self.adv_frame, text="20%", font=(FONT_FAMILY, 9, "bold"), text_color=COLORS["text_primary"])
        self.lbl_overlap.pack(pady=(0, 10))
        
        separator3 = ctk.CTkFrame(scroll, height=2, fg_color=COLORS["accent_gold"], corner_radius=1)
        separator3.pack(fill="x", pady=8)
        
        self.btn_process = ctk.CTkButton(
            scroll, text="INICIAR PROCESAMIENTO",
            command=self.process_image,
            font=(FONT_FAMILY, 12, "bold"),
            fg_color=COLORS["primary"], hover_color=COLORS["primary_hover"],
            height=45, state="disabled"
        )
        self.btn_process.pack(pady=10, fill="x")
        
        self.progress_bar = ctk.CTkProgressBar(scroll, height=8, fg_color=COLORS["bg_surface"], progress_color=COLORS["accent_gold"])
        self.progress_bar.pack(fill="x", pady=5)
        self.progress_bar.set(0)
        
        self.btn_save = ctk.CTkButton(
            scroll, text="GUARDAR EN BASE DE DATOS",
            command=self.save_results,
            font=(FONT_FAMILY, 11, "bold"),
            fg_color=COLORS["success"], hover_color=COLORS["primary_hover"],
            height=40, state="disabled"
        )
        self.btn_save.pack(pady=5, fill="x")
        
        self.nav_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        self.nav_frame.pack(fill="x", pady=10)
        
        self.btn_previous = ctk.CTkButton(
            self.nav_frame, text="◀ Anterior",
            command=self.show_previous_image,
            font=(FONT_FAMILY, 10, "bold"),
            fg_color=COLORS["bg_surface"], hover_color=COLORS["border"],
            text_color=COLORS["text_primary"],
            height=35, state="disabled", width=120
        )
        self.btn_previous.pack(side="left", padx=(0, 5))
        
        self.lbl_image_counter = ctk.CTkLabel(
            self.nav_frame, text="Imagen 0 de 0",
            font=(FONT_FAMILY, 10, "bold"),
            text_color=COLORS["text_primary"]
        )
        self.lbl_image_counter.pack(side="left", padx=5)
        
        self.btn_next = ctk.CTkButton(
            self.nav_frame, text="Siguiente ▶",
            command=self.show_next_image,
            font=(FONT_FAMILY, 10, "bold"),
            fg_color=COLORS["bg_surface"], hover_color=COLORS["border"],
            text_color=COLORS["text_primary"],
            height=35, state="disabled", width=120
        )
        self.btn_next.pack(side="left", padx=(5, 0))
        
        self.btn_export = ctk.CTkButton(
            scroll, text="EXPORTAR CSV",
            command=self.export_csv,
            font=(FONT_FAMILY, 11, "bold"),
            fg_color="transparent", border_width=2, border_color=COLORS["border"],
            text_color=COLORS["text_primary"], hover_color=COLORS["bg_surface"],
            height=40, state="disabled"
        )
        self.btn_export.pack(pady=5, fill="x")
        
        # Recorte/foco: procesar SOLO una zona ampliada de la imagen (fotos lejanas)
        self.btn_crop = ctk.CTkButton(
            scroll, text="✂️ ANALIZAR ZONA (FOTO LEJANA)",
            command=self.start_crop_selection,
            font=(FONT_FAMILY, 10, "bold"),
            fg_color="transparent", border_width=2, border_color=COLORS["accent_gold"],
            text_color=COLORS["accent_gold"], hover_color=COLORS["bg_surface"],
            height=35, state="disabled"
        )
        self.btn_crop.pack(pady=(0, 5), fill="x")
        
        separator4 = ctk.CTkFrame(scroll, height=2, fg_color=COLORS["accent_gold"], corner_radius=1)
        separator4.pack(fill="x", pady=8)
        
        ctk.CTkLabel(scroll, text="RESULTADOS", font=(FONT_FAMILY, 11, "bold"), text_color=COLORS["accent_gold"]).pack(pady=(10, 5))
        
        self.results_frame = ctk.CTkFrame(scroll, fg_color=COLORS["bg_surface"], corner_radius=8, border_width=1, border_color=COLORS["border"])
        self.results_frame.pack(fill="both", expand=True, pady=5)
        
        self.lbl_waiting = ctk.CTkLabel(self.results_frame, text="Esperando imagen...", font=(FONT_FAMILY, 10), text_color=COLORS["text_muted"])
        self.lbl_waiting.pack(pady=20)
    
    def create_visualization_panel(self, parent):
        """Panel de visualización con Canvas interactivo para hover"""
        self.viz_frame = ctk.CTkFrame(parent, fg_color=COLORS["bg_surface"], corner_radius=8)
        self.viz_frame.pack(expand=True, fill="both", padx=20, pady=20)
        
        self.canvas = tk.Canvas(
            self.viz_frame, 
            bg=COLORS["bg_surface"], 
            highlightthickness=0
        )
        self.canvas.pack(expand=True, fill="both", padx=10, pady=10)
        
        self.tooltip = tk.Toplevel(self.canvas)
        self.tooltip.withdraw()
        self.tooltip.overrideredirect(True)
        self.tooltip.configure(bg="#1E293B", padx=8, pady=6)
        
        self.tooltip_label = tk.Label(
            self.tooltip, 
            text="", 
            bg="#1E293B", 
            fg="#FFFFFF", 
            font=("Segoe UI", 10, "bold"),
            justify="left"
        )
        self.tooltip_label.pack()
        
        self.canvas.create_text(
            400, 300,
            text="Carga una imagen del monumento para comenzar\n\n• Pasa el cursor sobre las detecciones para ver medidas\n• Colores indican gravedad: 🟢 Leve → 🔴 Crítica",
            fill=COLORS["text_muted"],
            font=("Segoe UI", 14),
            justify="center"
        )
        
        self.drawn_boxes = []
    
    def display_image(self, path_or_array):
        """Muestra imagen con detecciones INTERACTIVAS (hover = medidas)"""
        if isinstance(path_or_array, str):
            img = cv2.imread(path_or_array)
            if img is None:
                return
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = path_or_array
    
        canvas_w = self.canvas.winfo_width() or 1000
        canvas_h = self.canvas.winfo_height() or 700
        h, w = img.shape[:2]
        ratio = min(canvas_w / w, canvas_h / h, 1.0)
        new_w, new_h = int(w * ratio), int(h * ratio)
    
        img_resized = cv2.resize(img, (new_w, new_h))
        img_pil = Image.fromarray(img_resized)
    
        # ✅ CORRECCIÓN: Usar ImageTk.PhotoImage en lugar de CTkImage
        self.tk_image = ImageTk.PhotoImage(img_pil)
    
        self.canvas.delete("all")
        self.drawn_boxes = []
    
        self.canvas.update_idletasks()
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        offset_x = (cw - new_w) // 2
        offset_y = (ch - new_h) // 2
    
        # ✅ Ahora sí funciona con tkinter Canvas
        self.canvas.create_image(offset_x, offset_y, anchor="nw", image=self.tk_image)
    
        self._current_display_ratio = ratio
        self._current_display_offset = (offset_x, offset_y)
    
        if self.last_result and self.last_result.get("detections"):
            for d in self.last_result["detections"]:
                x1 = int(d["x1"] * ratio) + offset_x
                y1 = int(d["y1"] * ratio) + offset_y
                x2 = int(d["x2"] * ratio) + offset_x
                y2 = int(d["y2"] * ratio) + offset_y
            
                gravity = d.get("gravedad", "leve")
                color_hex = GRAVITY_COLORS.get(gravity, "#DC2626")
                color_rgb = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                color_str = f"#{color_rgb[0]:02x}{color_rgb[1]:02x}{color_rgb[2]:02x}"
            
                rect_id = self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline=color_str, width=3, fill=""
                )
            
                label_text = f"{d['Clase']} {d.get('gravedad_label', '')} | {d['Area_cm2']:.1f}cm²"
                label_id = self.canvas.create_text(
                x1, max(y1 - 12, 10),
                text=label_text,
                anchor="sw",
                fill=color_str,
                font=("Segoe UI", 9, "bold")
               )   
            
                self.drawn_boxes.append({
                "coords": (x1, y1, x2, y2),
                "detection": d,
                "rect_id": rect_id,
                "label_id": label_id,
                "color": color_str
                })
    
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<Leave>", self._on_canvas_leave)
    def _on_canvas_motion(self, event):
        """Muestra tooltip con medidas exactas al pasar el cursor"""
        if self.crop_mode:
            return
        if not self.drawn_boxes:
            return
        
        hovered_box = None
        for box in self.drawn_boxes:
            x1, y1, x2, y2 = box["coords"]
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                hovered_box = box
                break
        
        if hovered_box:
            d = hovered_box["detection"]
            
            self.canvas.itemconfig(hovered_box["rect_id"], width=5)
            
            tooltip_text = (
                f"📐 {d['Clase'].upper()}\n"
                f"🎯 Gravedad: {d.get('gravedad_label', 'N/A')}\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"📏 Área: {d['Area_cm2']:.2f} cm²\n"
                f"      ({d['Area_m2']:.5f} m²)\n"
                f"↔️  Ancho: {d['Ancho_px']} px\n"
                f"↕️  Alto:  {d['Alto_px']} px\n"
                f"🎯 Confianza: {d['Confianza']*100:.1f}%"
            )
            
            self.tooltip_label.configure(text=tooltip_text)
            
            canvas_root_x = self.canvas.winfo_rootx()
            canvas_root_y = self.canvas.winfo_rooty()
            self.tooltip.geometry(f"+{canvas_root_x + event.x + 15}+{canvas_root_y + event.y + 15}")
            self.tooltip.deiconify()
        else:
            for box in self.drawn_boxes:
                self.canvas.itemconfig(box["rect_id"], width=3)
            self.tooltip.withdraw()
    
    def _on_canvas_leave(self, event):
        """Oculta tooltip al salir del canvas"""
        self.tooltip.withdraw()
        for box in self.drawn_boxes:
            self.canvas.itemconfig(box["rect_id"], width=3)
    
    def _safe_image_source(self):
        """Devuelve la imagen BGR original de la foto actual (si existe) para recortar."""
        result = self.last_result
        if not result:
            return None
        image_path = result.get("image_path") or result.get("filename")
        if image_path and os.path.exists(image_path):
            return cv2.imread(image_path)
        # Si la foto no esta en disco, intentar la anotada (no ideal pero funcional)
        annotated = result.get("annotated_image")
        if annotated is not None:
            return cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR)
        return None

    def start_crop_selection(self):
        """Activa el modo seleccion de zona sobre la imagen actual."""
        if self.crop_mode:
            self._cancel_crop_mode()
            return
        if self.last_result is None:
            messagebox.showwarning("Recorte", "Primero procesa una imagen para poder recortar una zona.")
            return
        if getattr(self, "processing_crop", False):
            messagebox.showwarning("Recorte", "Ya hay un recorte en proceso. Espera a que termine.")
            return
        self.crop_mode = True
        self.crop_start = None
        self.crop_rect_id = None
        self.canvas.configure(cursor="crosshair")
        self.btn_crop.configure(text="✂️ CANCELAR SELECCIÓN")
        self.canvas.bind("<ButtonPress-1>", self._on_crop_press)
        self.canvas.bind("<B1-Motion>", self._on_crop_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_crop_release)

    def _cancel_crop_mode(self):
        self.crop_mode = False
        self.crop_start = None
        if self.crop_rect_id is not None:
            self.canvas.delete(self.crop_rect_id)
            self.crop_rect_id = None
        self.canvas.configure(cursor="")
        self.btn_crop.configure(text="✂️ ANALIZAR ZONA (FOTO LEJANA)")
        self.canvas.bind("<ButtonPress-1>", "break")
        self.canvas.bind("<B1-Motion>", "break")
        self.canvas.bind("<ButtonRelease-1>", "break")
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<Leave>", self._on_canvas_leave)

    def _on_crop_press(self, event):
        self.crop_start = (event.x, event.y)
        if self.crop_rect_id is not None:
            self.canvas.delete(self.crop_rect_id)
        self.crop_rect_id = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline="#FBBF24", width=2, dash=(4, 2)
        )

    def _on_crop_drag(self, event):
        if self.crop_start is None or self.crop_rect_id is None:
            return
        evx = max(0, min(event.x, self.canvas.winfo_width()))
        evy = max(0, min(event.y, self.canvas.winfo_height()))
        sx, sy = self.crop_start
        self.canvas.coords(self.crop_rect_id, sx, sy, evx, evy)

    def _on_crop_release(self, event):
        if self.crop_start is None or self.crop_rect_id is None:
            self._cancel_crop_mode()
            return
        evx = max(0, min(event.x, self.canvas.winfo_width()))
        evy = max(0, min(event.y, self.canvas.winfo_height()))
        sx, sy = self.crop_start
        x1s, y1s = min(sx, evx), min(sy, evy)
        x2s, y2s = max(sx, evx), max(sy, evy)

        if (x2s - x1s) < 15 or (y2s - y1s) < 15:
            messagebox.showwarning("Recorte", "La selección es demasiado pequeña. Arrastra un rectángulo sobre la zona a analizar.")
            self._cancel_crop_mode()
            return

        source_bgr = self._safe_image_source()
        if source_bgr is None:
            messagebox.showerror("Recorte", "No se encontró la imagen original para recortar.")
            self._cancel_crop_mode()
            return

        ratio = getattr(self, "_current_display_ratio", None)
        offset_x, offset_y = getattr(self, "_current_display_offset", (0, 0))
        if not ratio:
            messagebox.showerror("Recorte", "No se puede calcular el punto de recorte.")
            self._cancel_crop_mode()
            return

        h_i, w_i = source_bgr.shape[:2]
        ix1 = int(round((x1s - offset_x) / ratio)); ix1 = max(0, min(ix1, w_i - 1))
        iy1 = int(round((y1s - offset_y) / ratio)); iy1 = max(0, min(iy1, h_i - 1))
        ix2 = int(round((x2s - offset_x) / ratio)); ix2 = max(0, min(ix2, w_i - 1))
        iy2 = int(round((y2s - offset_y) / ratio)); iy2 = max(0, min(iy2, h_i - 1))

        if ix2 <= ix1 or iy2 <= iy1:
            messagebox.showwarning("Recorte", "Selección inválida. Intenta nuevamente.")
            self._cancel_crop_mode()
            return

        self._cancel_crop_mode()
        self._process_crop_zone(source_bgr, ix1, iy1, ix2, iy2)

    def _process_crop_zone(self, source_bgr, ix1, iy1, ix2, iy2):
        """Amplia el recorte (foto lejana) y lo procesa con el motor completo."""
        crop = source_bgr[iy1:iy2, ix1:ix2].copy()
        c_h, c_w = crop.shape[:2]

        # Escala de ampliación: mínimo x2 (para dar espesor a grietas finas) y de ahí
        # hasta llevar el lado mayor a 2400 px si la zona seleccionada es pequeña.
        # Lanczos conserva mejor los bordes finos que el bicúbico.
        escala = max(2.0, 2400.0 / max(c_w, c_h))
        if escala > 1.0:
            crop = cv2.resize(crop, (int(c_w * escala), int(c_h * escala)), interpolation=cv2.INTER_LANCZOS4)

        try:
            self.cm_per_pixel = float(self.entry_gsd.get())
        except:
            self.cm_per_pixel = 0.13
        cm_per_pixel_eff = self.cm_per_pixel / escala

        # En modo "foto lejana" se relajan los umbrales de crack y humedad: los
        # rasgos finos lejanos bajan su score, y al estar el recorte enfocado por
        # el usuario hay menos superficie disponible para falsos positivos.
        # En "Alta confiabilidad (85%)" no se relaja nada: el piso del 85% se
        # mantiene también en la zona recortada.
        alta_conf = self._modo_alta_confiabilidad()
        thresholds = self._leer_umbrales()
        if not alta_conf:
            thresholds["crack"] = min(thresholds.get("crack", 0.20), 0.10)
            thresholds["humidity"] = min(thresholds.get("humidity", 0.30), 0.20)
        tile_size = self.tile_size
        overlap = self.overlap
        iou_thr = self.iou_threshold
        use_tiling = self.use_tiling

        self.processing_crop = True
        self.btn_crop.configure(state="disabled", text="⏳ PROCESANDO ZONA...")
        self.lbl_file.configure(text=f"Analizando zona ({c_w}x{c_h} px, zoom x{escala:.1f}...", text_color=COLORS["accent_gold"])

        def worker():
            result = procesar_imagen_completa(
                self.model, crop, thresholds, iou_thr,
                cm_per_pixel_eff, tile_size, overlap,
                use_tiling=use_tiling,
                modo_recorte=True,
                zoom_pass=False,
                progress_callback=lambda p: self.after(0, lambda: self.progress_bar.set(p))
            )

            # Módulo complementario "posible grieta": busca líneas oscuras finas
            # (grietas sobre pintura) que la red neuronal no detecta por desfase
            # de dominio o falta de resolución. Solo se muestra como AYUDA en la
            # zona recortada, nunca sustituye a la red: la grieta requiere
            # confirmación en campo.
            # En modo 85% no se muestran: su confianza (50%) es fija y no
            # proviene de la red, por lo que serían detecciones dudosas.
            posibles = []
            if result.get("success") is True:
                try:
                    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    gray_clahe = clahe.apply(gray_crop)
                    candidatos = [] if alta_conf else candidatos_grietas_finas(gray_clahe)
                    for cand in candidatos:
                        # Solo reportar candidatos al menos tan largos como un
                        # sisma real apreciable (~1 cm) en escala de recorte
                        posibles.append({
                            "Clase": "posible_grieta",
                            "Confianza": 0.5,
                            "x1": cand["x1"], "y1": cand["y1"],
                            "x2": cand["x2"], "y2": cand["y2"],
                            "Ancho_px": cand["x2"] - cand["x1"],
                            "Alto_px": cand["y2"] - cand["y1"],
                            "Area_px": cand.get("area_px", 0),
                            "Longitud_px": cand.get("longitud_px", 0),
                            "Arco_px": cand.get("arco_px", 0),
                            "Tortuosidad": cand.get("tortuosidad", 0),
                            "StdAnguloDeg": cand.get("std_ang_deg", 0),
                            "StdAnchoPx": cand.get("std_ancho_px", 0),
                            "AnchoMedioPx": cand.get("ancho_medio_px", 0),
                            "OrientacionDeg": cand.get("orientacion_deg", 0),
                        })
                except Exception:
                    posibles = []

                # Dibujar las posibles grietas sobre la imagen anotada
                if posibles:
                    img_bgr = cv2.cvtColor(result["annotated_image"], cv2.COLOR_RGB2BGR)
                    for p in posibles:
                        cv2.rectangle(img_bgr, (p["x1"], p["y1"]), (p["x2"], p["y2"]), (245, 192, 40), 2)
                    result["annotated_image"] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    result["n_posibles"] = len(posibles)
                    result["posibles_grietas"] = posibles
                else:
                    result["n_posibles"] = 0
                    result["posibles_grietas"] = []

            self.after(0, lambda: self._on_crop_done(result))

        threading.Thread(target=worker, daemon=True).start()

    def _on_crop_done(self, result):
        self.processing_crop = False
        self.progress_bar.stop()
        self.btn_crop.configure(state="normal", text="✂️ ANALIZAR ZONA (FOTO LEJANA)")
        if not result["success"]:
            self.lbl_file.configure(text="Error al procesar la zona", text_color=COLORS["danger"])
            messagebox.showerror("Recorte", result.get("error", "Error desconocido"))
            return

        base = os.path.basename(self.last_result.get("filename", "imagen"))
        result["filename"] = f"Zona recortada de {base}"
        result["image_path"] = self.last_result.get("image_path")
        result["is_crop"] = True
        self._last_crop_result = result

        self.all_results.append(result)
        self.current_view_index = len(self.all_results) - 1
        self.display_image(result["annotated_image"])
        self.show_results_visual(result)
        self.update_image_counter()

        if self.current_view_index > 0:
            self.btn_previous.configure(state="normal")
        else:
            self.btn_previous.configure(state="disabled")
        self.btn_next.configure(state="disabled")

        self.lbl_file.configure(
            text=f"Zona analizada: {result['n_detections']} detección(es)",
            text_color=COLORS["success"]
        )

    def update_label(self, label, value, is_int=False):
        if is_int:
            label.configure(text=f"{int(value)} px" if "tile" in str(label) else f"{int(value)}%")
        else:
            label.configure(text=f"{int(value*100)}%")
    
    def toggle_tiling(self):
        self.use_tiling = self.chk_tiling.get()

    def on_threshold_change(self, label, value):
        """Actualiza el texto del umbral cuando el usuario mueve el slider.
        Bajar un umbral por debajo del 85% saca la app del modo de alta
        confiabilidad (el radio deja de estar marcado) para no ocultar el cambio."""
        self.update_label(label, value)
        if self._modo_alta_confiabilidad() and value < UMBRAL_ALTA_CONFIABILIDAD - 1e-6:
            self.preset_var.set("personalizado")

    def _modo_alta_confiabilidad(self):
        return self.preset_var.get() == "alto"

    def _leer_umbrales(self):
        """Umbrales por clase de los sliders, acotados a [mín, máx]. En modo
        'Alta confiabilidad' ninguna clase queda por debajo del 85%."""
        umbrales = {
            "crack": self.slider_crack.get(),
            "humidity": self.slider_humidity.get(),
            "spalling": self.slider_spalling.get(),
        }
        piso = UMBRAL_ALTA_CONFIABILIDAD if self._modo_alta_confiabilidad() else UMBRAL_CONF_MIN
        return {cls: min(max(v, piso), UMBRAL_CONF_MAX) for cls, v in umbrales.items()}

    def toggle_advanced(self):
        """Muestra u oculta la configuracion avanzada (para evitar confusiones al hacer scroll)."""
        if self.adv_open:
            self.adv_frame.pack_forget()
            self.btn_advanced.configure(text="▸ Configuración avanzada")
            self.adv_open = False
        else:
            self.adv_frame.pack(fill="x", pady=(2, 4))
            self.btn_advanced.configure(text="▾ Configuración avanzada")
            self.adv_open = True

    def apply_preset(self):
        """Aplica el preset de calibracion segun el modo seleccionado."""
        modo = self.preset_var.get()
        if modo == "recomendado":
            self.slider_crack.set(0.20)
            self.slider_humidity.set(0.30)
            self.slider_spalling.set(0.60)
            self.update_label(self.lbl_crack, 0.20)
            self.update_label(self.lbl_humidity, 0.30)
            self.update_label(self.lbl_spalling, 0.60)
        elif modo == "alto":
            for slider, lbl in ((self.slider_crack, self.lbl_crack),
                                (self.slider_humidity, self.lbl_humidity),
                                (self.slider_spalling, self.lbl_spalling)):
                slider.set(UMBRAL_ALTA_CONFIABILIDAD)
                self.update_label(lbl, UMBRAL_ALTA_CONFIABILIDAD)

    def restore_recommended_calibration(self):
        """Restaura la calibracion optima recomendada para sillar."""
        self.preset_var.set("recomendado")
        self.apply_preset()
        self.slider_iou.set(IOU_NMS)
        self.update_label(self.lbl_iou, IOU_NMS)
        messagebox.showinfo("Calibración", "Calibración restablecida al modo recomendado para sillar (Crack 20%, Humedad 30%, Desprendimiento 60%). Esto detecta más daños de forma confiable.\n\nSi la evaluación exige confiabilidad alta, elija el modo 'Alta confiabilidad (85%)'.")
    
    def load_image(self):
        file_paths = filedialog.askopenfilenames(
            title="Seleccionar Imágenes del Monumento",
            filetypes=[
                ("Imágenes", "*.jpg *.jpeg *.png *.webp"),
                ("Todos los archivos", "*.*")
            ]
        )
        
        if not file_paths:
            return
        
        # Limite de tamano por imagen (20 MB)
        MAX_IMAGE_MB = 20
        oversized = [os.path.basename(fp) for fp in file_paths if os.path.getsize(fp) > MAX_IMAGE_MB * 1024 * 1024]
        if oversized:
            messagebox.showerror(
                "Imagenes Demasiado Grandes",
                f"El peso maximo por imagen es de {MAX_IMAGE_MB} MB.\n\n"
                f"Archivo(s) excedido(s):\n{chr(10).join(oversized)}"
            )
            return
        
        filenames = [os.path.basename(fp) for fp in file_paths]
        
        try:
            validation = self.db.check_duplicate_files(filenames)
            duplicates = validation.get("duplicates", [])
            new_files = validation.get("new_files", [])
            
            if duplicates:
                dup_list = "\n".join(duplicates)
                continuar = messagebox.askyesno(
                    "Archivos Duplicados Detectados",
                    f"Los siguientes archivos YA EXISTEN en la base de datos:\n\n{dup_list}\n\n"
                    f"Se ignorarán y se procesarán los {len(new_files)} archivos nuevos.\n\n"
                    "¿Deseas continuar?"
                )
                
                if not continuar:
                    return
            
            if not new_files:
                messagebox.showinfo(
                    "Sin Archivos Nuevos",
                    "Todos los archivos seleccionados ya existen en la base de datos."
                )
                return
            
            self.processing_queue = [
                fp for fp in file_paths
                if os.path.basename(fp) in new_files
            ]
            self.last_queued_paths = list(self.processing_queue)
            
            self.current_queue_index = -1
            self.all_results = []
            self.current_view_index = 0
            
            self.lbl_file.configure(
                text=f"{len(self.processing_queue)} archivos listos para procesar",
                text_color=COLORS["success"]
            )
            
            self.btn_process.configure(state="normal", text="INICIAR PROCESAMIENTO")
            self.btn_save.configure(state="disabled", text="GUARDAR EN BASE DE DATOS")
            self.btn_export.configure(state="disabled")
            self.btn_previous.configure(state="disabled")
            self.btn_next.configure(state="disabled")
            self.lbl_image_counter.configure(text="Imagen 0 de 0")
            
            messagebox.showinfo(
                "Cola Preparada",
                f"✅ {len(self.processing_queue)} archivos nuevos listos.\n\n"
                f"Haz clic en 'INICIAR PROCESAMIENTO' para comenzar.\n"
                f"El botón GUARDAR se habilitará al terminar todo el lote."
            )
        
        except Exception:
            messagebox.showerror("Error de Conexión", "No se pudo verificar duplicados con Supabase.")
    
    def _start_batch_processing(self):
        if not self.processing_queue:
            return
        
        self.current_queue_index += 1
        
        if self.current_queue_index >= len(self.processing_queue):
            self._on_batch_finished()
            return
        
        next_path = self.processing_queue[self.current_queue_index]
        filename = os.path.basename(next_path)
        total = len(self.processing_queue)
        actual = self.current_queue_index + 1
        
        self.lbl_file.configure(
            text=f"Procesando {filename} ({actual}/{total})",
            text_color=COLORS["accent_gold"]
        )
        self.btn_process.configure(state="disabled", text=f"PROCESANDO {actual}/{total}...")
        self.btn_load.configure(state="disabled")
        self.btn_save.configure(state="disabled", text="GUARDAR EN BASE DE DATOS")
        
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        self.lbl_waiting = ctk.CTkLabel(
            self.results_frame,
            text=f"Procesando imagen {actual} de {total}...\n{filename}",
            font=(FONT_FAMILY, 12),
            text_color=COLORS["text_muted"]
        )
        self.lbl_waiting.pack(pady=20)
        
        self.class_thresholds = self._leer_umbrales()
        self.iou_threshold = self.slider_iou.get()
        
        try:
            self.cm_per_pixel = float(self.entry_gsd.get())
        except:
            self.cm_per_pixel = 0.13
        
        self.tile_size = int(self.slider_tile.get())
        self.overlap = self.slider_overlap.get() / 100.0
        
        def worker():
            result = procesar_imagen_completa(
                self.model, next_path,
                self.class_thresholds, self.iou_threshold,
                self.cm_per_pixel, self.tile_size, self.overlap,
                use_tiling=self.use_tiling,
                progress_callback=lambda p: self.after(0, lambda: self.progress_bar.set(p))
            )
            self.after(0, lambda: self._on_single_process_done(result, next_path))
        
        threading.Thread(target=worker, daemon=True).start()
    
    def _on_single_process_done(self, result, image_path):
        self.progress_bar.stop()
        self.progress_bar.set(1.0)
        
        if not result["success"]:
            messagebox.showerror(
                "Error de Procesamiento",
                f"No se pudo procesar {os.path.basename(image_path)}:\n{result.get('error', 'Error desconocido')}\n\n"
                "Se continuará con la siguiente imagen."
            )
            self.after(500, self._start_batch_processing)
            return
        
        # Enriquecer con reporte de priorización
        if result.get("detections"):
            result["priority_report"] = generate_priority_report(
                result["detections"], 
                os.path.basename(image_path)
            )
        
        last_history = None
        filename = os.path.basename(image_path)
        hist = self.db.select("inspection_results", "*", filters={"filename": f"eq.{filename}"}, order="created_at.desc", limit=1)
        if hist["success"] and hist["data"]:
            last_history = hist["data"][0]
        
        alerts = self.alert_system.evaluate_current_result(result, last_history)
        if alerts:
            self.alert_system.log_alerts(alerts)
        
        result["filename"] = os.path.basename(image_path)
        result["timestamp"] = datetime.now().isoformat()
        result["image_path"] = image_path
        
        self.all_results.append(result)
        self.last_result = result
        
        self.display_image(result["annotated_image"])
        self.show_results_visual(result)
        self.btn_crop.configure(state="normal")
        
        self.after(500, self._start_batch_processing)
    
    def _on_batch_finished(self):
        """Se llama cuando termina toda la cola - NO guarda automáticamente"""
        total = len(self.processing_queue)
        processed = len(self.all_results)
        
        self.processing_queue.clear()
        self.current_queue_index = -1
        
        self.btn_process.configure(state="normal", text="INICIAR PROCESAMIENTO")
        self.btn_load.configure(state="normal")
        
        if processed > 0:
            self.btn_save.configure(
                state="normal",
                text=f"GUARDAR {processed} RESULTADOS"
            )
            self.btn_export.configure(state="normal")
            
            self.current_view_index = 0
            
            if processed > 1:
                self.btn_previous.configure(state="normal")
                self.btn_next.configure(state="normal")
            else:
                self.btn_previous.configure(state="disabled")
                self.btn_next.configure(state="disabled")
            
            self.update_image_counter()
            
            self.last_result = self.all_results[0]
            self.display_image(self.last_result["annotated_image"])
            self.show_results_visual(self.last_result)
            self.btn_crop.configure(state="normal")
            
            messagebox.showinfo(
                "Procesamiento Completado",
                f"✅ Se procesaron {processed} imágenes exitosamente.\n\n"
                f"📊 Usa los botones ◀ Anterior y Siguiente ▶ para navegar.\n\n"
                f"💡 Pasa el cursor sobre las detecciones para ver medidas exactas.\n\n"
                f"Haz clic en 'GUARDAR {processed} RESULTADOS' para subirlos a Supabase."
            )
        else:
            self.btn_save.configure(state="disabled")
            self.btn_export.configure(state="disabled")
            messagebox.showwarning(
                "Sin Resultados",
                "No se pudo procesar ninguna imagen de la cola."
            )
    
    def process_image(self):
        if not self.processing_queue:
            if self.last_queued_paths:
                reprocesar = messagebox.askyesno(
                    "Volver a Procesar",
                    f"Las {len(self.last_queued_paths)} imágenes ya cargadas se procesaron con la calibración anterior.\n\n"
                    "¿Deseas volver a procesarlas con la calibración actual (sliders y modo visible)?",
                    icon="question"
                )
                if not reprocesar:
                    return
                self.processing_queue = list(self.last_queued_paths)
                self.current_queue_index = -1
                self.all_results = []
                self.current_view_index = 0
            else:
                messagebox.showwarning("Sin cola", "No hay imágenes en la cola.\nCarga imágenes primero.")
                return
        
        if self.current_queue_index >= 0:
            messagebox.showwarning("En progreso", "Ya hay un procesamiento en curso.")
            return
        
        self.progress_bar.set(0)
        self._start_batch_processing()
    
    def show_previous_image(self):
        if not self.all_results:
            return
        
        if self.current_view_index > 0:
            self.current_view_index -= 1
            self.update_display_for_current_image()
    
    def show_next_image(self):
        if not self.all_results:
            return
        
        if self.current_view_index < len(self.all_results) - 1:
            self.current_view_index += 1
            self.update_display_for_current_image()
    
    def update_display_for_current_image(self):
        if not self.all_results or self.current_view_index >= len(self.all_results):
            return
        
        result = self.all_results[self.current_view_index]
        self.last_result = result
        
        self.display_image(result["annotated_image"])
        self.show_results_visual(result)
        self.update_image_counter()
        
        if self.current_view_index == 0:
            self.btn_previous.configure(state="disabled")
        else:
            self.btn_previous.configure(state="normal")
        
        if self.current_view_index == len(self.all_results) - 1:
            self.btn_next.configure(state="disabled")
        else:
            self.btn_next.configure(state="normal")
    
    def update_image_counter(self):
        total = len(self.all_results)
        current = self.current_view_index + 1
        self.lbl_image_counter.configure(text=f"Imagen {current} de {total}")
    
    def cancel_processing(self):
        """Cancela el procesamiento de la cola restante"""
        if not self.processing_queue or self.current_queue_index == -1:
            messagebox.showinfo("Sin cola", "No hay procesamiento en curso.")
            return
        
        pendientes = len(self.processing_queue) - self.current_queue_index - 1
        
        if pendientes <= 0:
            messagebox.showinfo("Sin pendientes", "No hay imágenes pendientes en la cola.")
            return
        
        confirmar = messagebox.askyesno(
            "Cancelar Procesamiento",
            f"Se cancelarán {pendientes} imágenes pendientes.\n"
            f"Las {self.current_queue_index + 1} ya procesadas se mantendrán.\n\n"
            "¿Deseas continuar?"
        )
        
        if confirmar:
            self.processing_queue = self.processing_queue[:self.current_queue_index + 1]
            self._on_batch_finished()
    
    def show_results_visual(self, result):
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        ctk.CTkLabel(
            self.results_frame,
            text=f"Archivo: {result['filename']}",
            font=(FONT_FAMILY, 10, "bold"),
            text_color=COLORS["text_primary"]
        ).pack(anchor="w", padx=10, pady=(10, 5))
        
        metrics_grid = ctk.CTkFrame(self.results_frame, fg_color="transparent")
        metrics_grid.pack(fill="x", padx=10, pady=5)
        
        metric_card = ctk.CTkFrame(metrics_grid, fg_color=COLORS["bg_card"], corner_radius=8, border_width=1, border_color=COLORS["border"])
        metric_card.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        metric_card.columnconfigure(0, weight=1)
        ctk.CTkLabel(metric_card, text=str(result['n_detections']), font=(FONT_FAMILY, 24, "bold"), text_color=COLORS["primary"]).pack(pady=(10, 2))
        ctk.CTkLabel(metric_card, text="Total Detecciones", font=(FONT_FAMILY, 9), text_color=COLORS["text_muted"]).pack(pady=(0, 10))
        
        metric_card2 = ctk.CTkFrame(metrics_grid, fg_color=COLORS["bg_card"], corner_radius=8, border_width=1, border_color=COLORS["border"])
        metric_card2.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        metric_card2.columnconfigure(0, weight=1)
        ctk.CTkLabel(metric_card2, text=f"{result['total_area_cm2']:.1f} cm²", font=(FONT_FAMILY, 24, "bold"), text_color=COLORS["accent_gold"]).pack(pady=(10, 2))
        ctk.CTkLabel(metric_card2, text="Área Afectada", font=(FONT_FAMILY, 9), text_color=COLORS["text_muted"]).pack(pady=(0, 10))
        
        metric_card3 = ctk.CTkFrame(metrics_grid, fg_color=COLORS["bg_card"], corner_radius=8, border_width=1, border_color=COLORS["border"])
        metric_card3.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        metric_card3.columnconfigure(0, weight=1)
        ctk.CTkLabel(metric_card3, text=f"{result['avg_confidence']:.1f}%", font=(FONT_FAMILY, 24, "bold"), text_color=COLORS["success"]).pack(pady=(10, 2))
        ctk.CTkLabel(metric_card3, text="Confianza Promedio", font=(FONT_FAMILY, 9), text_color=COLORS["text_muted"]).pack(pady=(0, 10))
        
        separator = ctk.CTkFrame(self.results_frame, height=2, fg_color=COLORS["border"], corner_radius=1)
        separator.pack(fill="x", padx=10, pady=10)
        
        # Aviso inteligente: sin daños confirmados por la red.
        # Caso A: hay candidatos del análisis de borde (amarillo) -> mensaje de
        #     candidatura a confirmar en campo (sin contradicción visual).
        # Caso B: tampoco hay candidatos -> mensaje de captura lejana + recorte.
        n_det = result.get("n_detections", 0)
        n_pos = result.get("n_posibles", 0)
        if n_det == 0 and n_pos == 0:
            warning_frame = ctk.CTkFrame(
                self.results_frame,
                fg_color=COLORS["bg_card"],
                corner_radius=8,
                border_width=2,
                border_color="#F59E0B"
            )
            warning_frame.pack(fill="x", padx=10, pady=(0, 10))
            ctk.CTkLabel(
                warning_frame,
                text="⚠️ NO SE DETECTARON DAÑOS EN ESTA IMAGEN",
                font=(FONT_FAMILY, 11, "bold"),
                text_color="#F59E0B"
            ).pack(pady=(10, 4))
            ctk.CTkLabel(
                warning_frame,
                text=(
                    "Posible causa: la foto está tomada desde lejos y los daños ocupan "
                    "pocos píxeles (grietas finas o desprendimientos pequeños quedan ocultos).\n\n"
                    "Recomendación: usa el botón verde '✂️ ANALIZAR ZONA (FOTO LEJANA)' "
                    "y arrastra un recuadro sobre el muro de sillar que quieres inspeccionar. "
                    "El sistema analizará esa zona ampliada."
                ),
                font=(FONT_FAMILY, 10),
                text_color=COLORS["text_secondary"],
                wraplength=620,
                justify="left"
            ).pack(padx=12, pady=(0, 10))
        elif n_det == 0 and n_pos > 0:
            warning_frame = ctk.CTkFrame(
                self.results_frame,
                fg_color=COLORS["bg_card"],
                corner_radius=8,
                border_width=2,
                border_color="#FBBF24"
            )
            warning_frame.pack(fill="x", padx=10, pady=(0, 10))
            ctk.CTkLabel(
                warning_frame,
                text="⚠️ SIN DAÑOS CONFIRMADOS POR EL MODELO",
                font=(FONT_FAMILY, 11, "bold"),
                text_color="#FBBF24"
            ).pack(pady=(10, 4))
            ctk.CTkLabel(
                warning_frame,
                text=(
                    "El modelo no detectó daños confirmados en esta zona. El análisis "
                    f"complementario de bordes señaló {n_pos} zonas candidatas (marcadas en "
                    "amarillo) que requieren verificación en campo. No constituyen un diagnóstico."
                ),
                font=(FONT_FAMILY, 10),
                text_color=COLORS["text_secondary"],
                wraplength=620,
                justify="left"
            ).pack(padx=12, pady=(0, 10))
        
        # Reporte de priorización si existe
        if result.get("priority_report"):
            pr = result["priority_report"]
            
            priority_frame = ctk.CTkFrame(
                self.results_frame, 
                fg_color=COLORS["bg_card"], 
                corner_radius=8, 
                border_width=2,
                border_color=GRAVITY_COLORS.get(
                    "critica" if pr["urgency"] == "CRÍTICA" else 
                    "severa" if pr["urgency"] == "ALTA" else "moderada",
                    "#10B981"
                )
            )
            priority_frame.pack(fill="x", padx=10, pady=10)
            
            ctk.CTkLabel(
                priority_frame,
                text=f"🏛️ PRIORIZACIÓN DE INTERVENCIÓN",
                font=(FONT_FAMILY, 11, "bold"),
                text_color=COLORS["accent_gold"]
            ).pack(pady=(10, 5))
            
            ctk.CTkLabel(
                priority_frame,
                text=pr["recommendation"],
                font=(FONT_FAMILY, 12, "bold"),
                text_color=COLORS["text_primary"]
            ).pack(pady=5)
            
            info_text = (
                f"Score prioridad: {pr['total_priority_score']} | "
                f"Urgencia: {pr['urgency']} | "
                f"Críticas: {pr['criticas_count']} | Severas: {pr['severas_count']}"
            )
            ctk.CTkLabel(
                priority_frame,
                text=info_text,
                font=(FONT_FAMILY, 10),
                text_color=COLORS["text_secondary"]
            ).pack(pady=(0, 10))
        
        ctk.CTkLabel(
            self.results_frame,
            text="Desglose por tipo:",
            font=(FONT_FAMILY, 9, "bold"),
            text_color=COLORS["text_muted"]
        ).pack(anchor="w", padx=10, pady=(0, 5))
        
        crack_badge = ctk.CTkFrame(self.results_frame, fg_color=COLORS["bg_surface"], corner_radius=20, border_width=1, border_color=COLORS["crack"])
        crack_badge.pack(fill="x", padx=10, pady=2)
        crack_inner = ctk.CTkFrame(crack_badge, fg_color=COLORS["bg_surface"], corner_radius=20)
        crack_inner.pack(fill="x", padx=2, pady=2)
        ctk.CTkLabel(crack_inner, text=f"🔴 Grietas: {result['class_counts']['crack']}", font=(FONT_FAMILY, 10, "bold"), text_color=COLORS["crack"]).pack(pady=5)
        
        humidity_badge = ctk.CTkFrame(self.results_frame, fg_color=COLORS["bg_surface"], corner_radius=20, border_width=1, border_color=COLORS["humidity"])
        humidity_badge.pack(fill="x", padx=10, pady=2)
        humidity_inner = ctk.CTkFrame(humidity_badge, fg_color=COLORS["bg_surface"], corner_radius=20)
        humidity_inner.pack(fill="x", padx=2, pady=2)
        ctk.CTkLabel(humidity_inner, text=f"🔵 Humedad: {result['class_counts']['humidity']}", font=(FONT_FAMILY, 10, "bold"), text_color=COLORS["humidity"]).pack(pady=5)
        
        spalling_badge = ctk.CTkFrame(self.results_frame, fg_color=COLORS["bg_surface"], corner_radius=20, border_width=1, border_color=COLORS["spalling"])
        spalling_badge.pack(fill="x", padx=10, pady=2)
        spalling_inner = ctk.CTkFrame(spalling_badge, fg_color=COLORS["bg_surface"], corner_radius=20)
        spalling_inner.pack(fill="x", padx=2, pady=2)
        ctk.CTkLabel(spalling_inner, text=f"🟠 Desprendimiento: {result['class_counts']['spalling']}", font=(FONT_FAMILY, 10, "bold"), text_color=COLORS["spalling"]).pack(pady=5)

        # Candidatas a grieta por análisis de borde (módulo complementario)
        n_posibles = result.get("n_posibles", 0)
        if n_posibles > 0:
            posible_badge = ctk.CTkFrame(self.results_frame, fg_color=COLORS["bg_surface"], corner_radius=20, border_width=1, border_color="#FBBF24")
            posible_badge.pack(fill="x", padx=10, pady=2)
            posible_inner = ctk.CTkFrame(posible_badge, fg_color=COLORS["bg_surface"], corner_radius=20)
            posible_inner.pack(fill="x", padx=2, pady=2)
            ctk.CTkLabel(
                posible_inner,
                text=f"⚡ Posibles Grietas (análisis de borde): {n_posibles} — confirmar en campo",
                font=(FONT_FAMILY, 10, "bold"),
                text_color="#FBBF24"
            ).pack(pady=5)
    
    def save_results(self):
        """Guarda TODOS los resultados acumulados en Supabase"""
        if not self.all_results:
            messagebox.showwarning("Sin datos", "No hay resultados para guardar.")
            return
        
        confirmar = messagebox.askyesno(
            "Confirmar Guardado",
            f"Se guardarán {len(self.all_results)} inspecciones en Supabase.\n\n"
            "¿Deseas continuar?"
        )
        if not confirmar:
            return
        
        success_count = 0
        error_count = 0
        errores = []
        
        for result in self.all_results:
            success, info = self.db.save_inspection(result, result["filename"])
            if success:
                success_count += 1
                self.last_result = result
                self.save_local_files()
            else:
                error_count += 1
                errores.append(f"{result['filename']}: {info}")
        
        if success_count > 0:
            messagebox.showinfo(
                "Guardado Exitoso",
                f"✅ {success_count} inspecciones guardadas en Supabase.\n"
                f"❌ {error_count} errores." + 
                (f"\n\nDetalles de errores:\n" + "\n".join(errores[:5]) if errores else "")
            )
            self.btn_save.configure(state="disabled", text="YA GUARDADO")
            self.btn_export.configure(state="disabled")
            self.all_results.clear()
        else:
            messagebox.showerror(
                "Error al guardar",
                "No se pudo guardar ninguna inspección.\n\n" + "\n".join(errores)
            )
    
    def save_local_files(self):
        filename_base = os.path.splitext(self.last_result["filename"])[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("resultados")
        output_dir.mkdir(exist_ok=True)
        
        try:
            img_path = output_dir / f"resultado_{filename_base}_{timestamp}.jpg"
            img_bgr = cv2.cvtColor(self.last_result["annotated_image"], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(img_path), img_bgr)
            
            json_path = output_dir / f"resultado_{filename_base}_{timestamp}.json"
            json_data = {
                "filename": self.last_result["filename"],
                "timestamp": self.last_result["timestamp"],
                "parameters": {
                    "crack_threshold": self.class_thresholds["crack"],
                    "humidity_threshold": self.class_thresholds["humidity"],
                    "spalling_threshold": self.class_thresholds["spalling"],
                    "iou_threshold": self.iou_threshold,
                    "gsd_cm_per_pixel": self.cm_per_pixel,
                    "use_tiling": self.use_tiling,
                    "tile_size": self.tile_size,
                    "overlap": self.overlap
                },
                "summary": {
                    "total_detections": self.last_result["n_detections"],
                    "crack_count": self.last_result["class_counts"]["crack"],
                    "humidity_count": self.last_result["class_counts"]["humidity"],
                    "spalling_count": self.last_result["class_counts"]["spalling"],
                    "total_area_cm2": self.last_result["total_area_cm2"],
                    "total_area_m2": self.last_result["total_area_m2"],
                    "avg_confidence": self.last_result["avg_confidence"]
                },
                "detections": [
                    {
                        "class": d["Clase"],
                        "confidence": round(d["Confianza"] * 100, 1),
                        "x1": d["x1"], "y1": d["y1"],
                        "x2": d["x2"], "y2": d["y2"],
                        "width_px": d["Ancho_px"],
                        "height_px": d["Alto_px"],
                        "area_cm2": d["Area_cm2"],
                        "area_m2": d["Area_m2"],
                        "gravedad": d.get("gravedad", "leve"),
                        "gravedad_score": d.get("gravedad_score", 1)
                    }
                    for d in self.last_result["detections"]
                ]
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        except Exception as e:
            print(f"Error guardando archivos locales: {e}")
    
    def export_csv(self):
        if not self.last_result:
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Exportar Reporte CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile=f"reporte_sillar_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        )
        
        if not file_path:
            return
        
        try:
            data = [{
                "Archivo": self.last_result["filename"],
                "Cracks": self.last_result["class_counts"].get("crack", 0),
                "Humedad": self.last_result["class_counts"].get("humidity", 0),
                "Spalling": self.last_result["class_counts"].get("spalling", 0),
                "Total_Detecciones": self.last_result["n_detections"],
                "Area_cm2": self.last_result["total_area_cm2"],
                "Area_m2": self.last_result["total_area_m2"],
                "Confianza_Promedio": self.last_result.get("avg_confidence", 0),
            }]
            
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False, sep=";", decimal=",", encoding="utf-8-sig")
            
            messagebox.showinfo("Exportación Exitosa", f"CSV guardado en:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def show_module_dashboard(self):
        header = ctk.CTkFrame(self.content, fg_color=COLORS["bg_secondary"], height=90, corner_radius=8)
        header.pack(fill="x", padx=15, pady=(15, 5))
        header.pack_propagate(False)
        
        ctk.CTkLabel(header, text="DASHBOARD EJECUTIVO", font=(FONT_FAMILY, 10, "bold"), text_color=COLORS["accent_gold"]).pack(pady=(15, 2))
        ctk.CTkLabel(header, text="Cuadro de Mandos Consolidado", font=(FONT_FAMILY, 22, "bold"), text_color=COLORS["text_primary"]).pack()
        ctk.CTkLabel(header, text="Métricas clave, distribución por tipología y superficie afectada", font=(FONT_FAMILY, 11), text_color=COLORS["text_secondary"], justify="center").pack(pady=(6, 12))
        
        content = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=15, pady=15)
        
        stats = self.db.get_stats()
        
        if stats["total_inspections"] == 0:
            ctk.CTkLabel(content, text="No hay datos procesados.\nEjecute primero 'Análisis en Campo' y guarde resultados.", font=(FONT_FAMILY, 14), text_color=COLORS["text_muted"]).pack(pady=50)
            return
        
        metrics_frame = ctk.CTkFrame(content, fg_color="transparent")
        metrics_frame.pack(fill="x", pady=(0, 20))
        
        for label, value, color in [
            ("Total Inspecciones", str(stats["total_inspections"]), COLORS["primary"]),
            ("Total Daños", str(stats["total_detections"]), COLORS["crack"]),
            ("Superficie Afectada", f"{stats['total_area_m2']:.4f} m²", COLORS["accent_gold"]),
            ("Confianza Promedio", f"{stats['avg_confidence']:.1f}%", COLORS["success"]),
        ]:
            card = ctk.CTkFrame(metrics_frame, fg_color=COLORS["bg_card"], corner_radius=12, border_width=1, border_color=COLORS["border"])
            card.pack(side="left", fill="both", expand=True, padx=5)
            
            bar = ctk.CTkFrame(card, height=4, fg_color=color, corner_radius=2)
            bar.pack(fill="x", padx=15, pady=(15, 5))
            
            ctk.CTkLabel(card, text=label.upper(), font=(FONT_FAMILY, 9, "bold"), text_color=COLORS["text_muted"]).pack(pady=(5, 5))
            ctk.CTkLabel(card, text=value, font=(FONT_FAMILY, 28, "bold"), text_color=color).pack(pady=(0, 15))
        
        charts_frame = ctk.CTkFrame(content, fg_color="transparent")
        charts_frame.pack(fill="both", expand=True, pady=10)
        
        chart1_frame = ctk.CTkFrame(charts_frame, fg_color=COLORS["bg_card"], corner_radius=12, border_width=1, border_color=COLORS["border"])
        chart1_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        ctk.CTkLabel(chart1_frame, text="Distribución de Daños por Tipo", font=(FONT_FAMILY, 13, "bold"), text_color=COLORS["text_primary"]).pack(pady=10)
        
        fig1 = Figure(figsize=(5, 4), dpi=100, facecolor=COLORS["bg_card"])
        ax1 = fig1.add_subplot(111)
        ax1.set_facecolor(COLORS["bg_card"])
        
        types = ["Grietas", "Humedad", "Desprendimiento"]
        counts = [stats["total_cracks"], stats["total_humidity"], stats["total_spalling"]]
        colors_bar = [COLORS["crack"], COLORS["humidity"], COLORS["spalling"]]
        
        bars = ax1.bar(types, counts, color=colors_bar, edgecolor=COLORS["bg_surface"], linewidth=2)
        ax1.set_ylabel("Cantidad", fontweight='bold', color=COLORS["text_primary"])
        ax1.set_title("")
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.tick_params(colors=COLORS["text_muted"])
        
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(count), ha='center', fontweight='bold', fontsize=11, color=COLORS["text_primary"])
        
        canvas1 = FigureCanvasTkAgg(fig1, master=chart1_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        
        chart2_frame = ctk.CTkFrame(charts_frame, fg_color=COLORS["bg_card"], corner_radius=12, border_width=1, border_color=COLORS["border"])
        chart2_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))
        
        ctk.CTkLabel(chart2_frame, text="Distribución de Área Afectada (m²)", font=(FONT_FAMILY, 13, "bold"), text_color=COLORS["text_primary"]).pack(pady=10)
        
        historical = self.db.load_historical_data(1000)
        inspection_ids = [r["id"] for r in historical]
        details_by_insp = self.db.load_detection_details(inspection_ids)
        
        crack_area = humidity_area = spalling_area = 0.0
        for insp_id, details in details_by_insp.items():
            for d in details:
                if d["class_name"] == "crack":
                    crack_area += d["area_m2"]
                elif d["class_name"] == "humidity":
                    humidity_area += d["area_m2"]
                elif d["class_name"] == "spalling":
                    spalling_area += d["area_m2"]
        
        fig2 = Figure(figsize=(5, 4), dpi=100, facecolor=COLORS["bg_card"])
        ax2 = fig2.add_subplot(111)
        ax2.set_facecolor(COLORS["bg_card"])
        
        areas = [crack_area, humidity_area, spalling_area]
        labels_pie = ["Grietas", "Humedad", "Desprendimiento"]
        
        filtered = [(l, a, c) for l, a, c in zip(labels_pie, areas, colors_bar) if a > 0]
        
        if filtered:
            labels_f, areas_f, colors_f = zip(*filtered)
            ax2.pie(areas_f, labels=labels_f, colors=colors_f, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold', 'color': COLORS["text_primary"]})
            ax2.axis('equal')
        else:
            ax2.text(0.5, 0.5, "Sin datos de área", ha='center', va='center', fontsize=12, color=COLORS["text_muted"])
            ax2.axis('off')
        
        canvas2 = FigureCanvasTkAgg(fig2, master=chart2_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    
    def show_module_historial(self):
        header = ctk.CTkFrame(self.content, fg_color=COLORS["bg_secondary"], height=90, corner_radius=8)
        header.pack(fill="x", padx=15, pady=(15, 5))
        header.pack_propagate(False)
        
        ctk.CTkLabel(header, text="REGISTRO HISTÓRICO", font=(FONT_FAMILY, 10, "bold"), text_color=COLORS["accent_gold"]).pack(pady=(15, 2))
        ctk.CTkLabel(header, text="Historial de Inspecciones", font=(FONT_FAMILY, 22, "bold"), text_color=COLORS["text_primary"]).pack()
        ctk.CTkLabel(header, text="Consulta todas las inspecciones guardadas en la base de datos", font=(FONT_FAMILY, 11), text_color=COLORS["text_secondary"], justify="center").pack(pady=(6, 12))
        
        content = ctk.CTkFrame(self.content, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=15, pady=15)
        
        data = self.db.load_historical_data(1000)
        
        if not data:
            ctk.CTkLabel(content, text="Aún no hay inspecciones guardadas.\nVe a 'Análisis en Campo' para procesar imágenes.", font=(FONT_FAMILY, 14), text_color=COLORS["text_muted"]).pack(pady=50)
            return
        
        df = pd.DataFrame(data)
        
        metrics_frame = ctk.CTkFrame(content, fg_color="transparent")
        metrics_frame.pack(fill="x", pady=(0, 15))
        
        total_dets = int(df["total_detections"].sum())
        total_area = float(df["total_area_m2"].sum())
        avg_conf = float(df["avg_confidence"].mean()) if "avg_confidence" in df.columns else 0
        
        for label, value, color in [
            ("Imágenes", str(len(df)), COLORS["primary"]),
            ("Detecciones", str(total_dets), COLORS["crack"]),
            ("Superficie", f"{total_area:.3f} m²", COLORS["accent_gold"]),
            ("Confianza", f"{avg_conf:.1f}%", COLORS["success"]),
        ]:
            card = ctk.CTkFrame(metrics_frame, fg_color=COLORS["bg_card"], corner_radius=12, border_width=1, border_color=COLORS["border"])
            card.pack(side="left", fill="both", expand=True, padx=5)
            
            bar = ctk.CTkFrame(card, height=4, fg_color=color, corner_radius=2)
            bar.pack(fill="x", padx=15, pady=(15, 5))
            
            ctk.CTkLabel(card, text=label.upper(), font=(FONT_FAMILY, 9, "bold"), text_color=COLORS["text_muted"]).pack(pady=(5, 5))
            ctk.CTkLabel(card, text=value, font=(FONT_FAMILY, 22, "bold"), text_color=color).pack(pady=(0, 15))
        
        table_frame = ctk.CTkFrame(content, fg_color=COLORS["bg_card"], corner_radius=12, border_width=1, border_color=COLORS["border"])
        table_frame.pack(fill="both", expand=True, pady=10)
        
        ctk.CTkLabel(table_frame, text="Tabla de Inspecciones", font=(FONT_FAMILY, 14, "bold"), text_color=COLORS["text_primary"]).pack(pady=(12, 2), padx=15, anchor="w")
        ctk.CTkLabel(table_frame, text="PISTA: Haz clic en cualquier fila para ver la imagen ANTES / DESPUÉS (original y con detecciones)", font=(FONT_FAMILY, 10, "italic"), text_color=COLORS["accent_gold"]).pack(pady=(0, 8), padx=15, anchor="w")
        
        scroll_frame = ctk.CTkScrollableFrame(table_frame, fg_color="transparent", width=1000)
        scroll_frame.pack(fill="x", padx=15, pady=(0, 12))
        
        header_row = ctk.CTkFrame(scroll_frame, fg_color=COLORS["bg_secondary"], corner_radius=6, height=36)
        header_row.pack(fill="x", pady=(0, 6))
        
        columns = ["Fecha", "Archivo", "Total", "Grietas", "Humedad", "Despr.", "Área (m²)", "Conf. (%)"]
        col_widths = [120, 220, 60, 60, 60, 60, 100, 80]
        
        for i, (col, w) in enumerate(zip(columns, col_widths)):
            ctk.CTkLabel(header_row, text=col, font=(FONT_FAMILY, 10, "bold"), text_color=COLORS["text_primary"], width=w, anchor="w").pack(side="left", padx=(0 if i == 0 else 2, 2), pady=6)
        
        for idx, (_, row) in enumerate(df.iterrows()):
            row_color = COLORS["bg_surface"] if idx % 2 == 0 else COLORS["bg_card"]
            row_frame = ctk.CTkFrame(scroll_frame, fg_color=row_color, corner_radius=4, height=36, cursor="hand2")
            row_frame.pack(fill="x", pady=2)
            
            created_at = row.get("created_at", "")
            if created_at:
                try:
                    dt = datetime.fromisoformat(str(created_at).replace('Z', '+00:00'))
                    created_at = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    created_at = "—"
            
            values = [
                created_at,
                row.get("filename", "—"),
                str(row.get("total_detections", 0)),
                str(row.get("crack_count", 0)),
                str(row.get("humidity_count", 0)),
                str(row.get("spalling_count", 0)),
                f"{row.get('total_area_m2', 0):.4f}",
                f"{row.get('avg_confidence', 0):.1f}"
            ]
            
            row_labels = []
            for i, (val, w) in enumerate(zip(values, col_widths)):
                lbl = ctk.CTkLabel(row_frame, text=val, font=(FONT_FAMILY, 10), text_color=COLORS["text_primary"], width=w, anchor="w", justify="left", cursor="hand2")
                lbl.pack(side="left", padx=(0 if i == 0 else 2, 2), pady=4)
                row_labels.append(lbl)

            # Fila clicable -> abrir vista antes/después
            def _on_click_row(e, r=row):
                self._open_inspeccion_images(r)
            row_frame.bind("<Button-1>", _on_click_row)
            for lbl in row_labels:
                lbl.bind("<Button-1>", _on_click_row)
        
        btn_frame = ctk.CTkFrame(content, fg_color="transparent")
        btn_frame.pack(fill="x", pady=10)
        
        ctk.CTkButton(
            btn_frame, text="Exportar Historial Completo (CSV)",
            command=lambda: self.export_historical_csv(df),
            font=(FONT_FAMILY, 11, "bold"),
            fg_color=COLORS["primary"], hover_color=COLORS["primary_hover"],
            height=40
        ).pack(side="right")
    
    def export_historical_csv(self, df):
        file_path = filedialog.asksaveasfilename(
            title="Exportar Historial CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile=f"historial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        
        if not file_path:
            return
        
        try:
            df.to_csv(file_path, index=False, sep=";", decimal=",", encoding="utf-8-sig")
            messagebox.showinfo("Exportación Exitosa", f"CSV guardado en:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def _download_image_to_tk(self, url, max_width=420, max_height=520):
        """Descarga una imagen de una URL publica y la convierte a PhotoImage escalado."""
        try:
            r = requests.get(url, timeout=30)
            if r.status_code != 200:
                return None
            img = Image.open(io.BytesIO(r.content))
            img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(img)
        except Exception as e:
            print(f"Error descargando imagen {url}: {e}")
            return None

    def _open_inspeccion_images(self, row):
        """Abre una ventana con la imagen original y la anotada (antes/despues)."""
        original_url = row.get("imagen_original_url")
        anotada_url = row.get("imagen_anotada_url")

        if not original_url and not anotada_url:
            messagebox.showinfo(
                "Sin imágenes",
                "Esta inspección no tiene imágenes guardadas.\n\n"
                "Solo las inspecciones guardadas a partir de ahora incluyen las imágenes\n"
                "para la vista ANTES / DESPUÉS."
            )
            return

        win = ctk.CTkToplevel(self)
        win.title(f"Inspección: {row.get('filename', '')}")
        win.geometry("980x680")
        win.transient(self)
        win.grab_set()

        content = ctk.CTkScrollableFrame(win, fg_color=COLORS["bg_secondary"])
        content.pack(fill="both", expand=True, padx=15, pady=15)

        ctk.CTkLabel(
            content,
            text=f"Inspección: {row.get('filename', '—')}",
            font=(FONT_FAMILY, 16, "bold"),
            text_color=COLORS["accent_gold"]
        ).pack(pady=(5, 2))

        created_at = row.get("created_at", "")
        if created_at:
            try:
                dt = datetime.fromisoformat(str(created_at).replace('Z', '+00:00'))
                created_at = dt.strftime("%Y-%m-%d %H:%M")
            except:
                created_at = "—"
        ctk.CTkLabel(
            content,
            text=f"Fecha: {created_at}  |  Detecciones: {row.get('total_detections', 0)}  |  "
                 f"Grietas: {row.get('crack_count', 0)}  Humedad: {row.get('humidity_count', 0)}  "
                 f"Despr.: {row.get('spalling_count', 0)}  |  Área: {row.get('total_area_m2', 0):.4f} m²  |  "
                 f"Conf.: {row.get('avg_confidence', 0):.1f}%",
            font=(FONT_FAMILY, 11),
            text_color=COLORS["text_secondary"]
        ).pack(pady=(0, 12))

        # Panel con ambas imagenes lado a lado
        images_frame = ctk.CTkFrame(content, fg_color="transparent")
        images_frame.pack(fill="both", expand=True)

        # Variable para mantener referencias a las imagenes (evitar GC)
        keep = []

        def _build_panel(parent, url, titulo, color):
            panel = ctk.CTkFrame(parent, fg_color=COLORS["bg_card"], corner_radius=12,
                                 border_width=1, border_color=COLORS["border"])
            panel.pack(side="left", fill="both", expand=True, padx=6, pady=6)
            ctk.CTkLabel(panel, text=titulo, font=(FONT_FAMILY, 12, "bold"),
                         text_color=color).pack(pady=(12, 6))
            if not url:
                ctk.CTkLabel(panel, text="Imagen no disponible", font=(FONT_FAMILY, 11),
                             text_color=COLORS["text_muted"]).pack(pady=40)
                return
            img = self._download_image_to_tk(url)
            if img is None:
                ctk.CTkLabel(panel, text="No se pudo cargar la imagen", font=(FONT_FAMILY, 11),
                             text_color=COLORS["text_muted"]).pack(pady=40)
                return
            keep.append(img)
            img_lbl = ctk.CTkLabel(panel, image=img, text="")
            img_lbl.pack(padx=12, pady=(0, 12))

        _build_panel(images_frame, original_url, "ANTES - Imagen Original", COLORS["text_primary"])
        _build_panel(images_frame, anotada_url, "DESPUÉS - Con Detecciones", COLORS["success"])

        ctk.CTkButton(
            content, text="Cerrar", command=win.destroy,
            font=(FONT_FAMILY, 11, "bold"), fg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"], width=120, height=36
        ).pack(pady=12)

    def show_module_especificacion(self):
        header = ctk.CTkFrame(self.content, fg_color=COLORS["bg_secondary"], height=90, corner_radius=8)
        header.pack(fill="x", padx=15, pady=(15, 5))
        header.pack_propagate(False)
        
        ctk.CTkLabel(header, text="DOCUMENTACIÓN", font=(FONT_FAMILY, 10, "bold"), text_color=COLORS["accent_gold"]).pack(pady=(15, 2))
        ctk.CTkLabel(header, text="Especificación Técnica del Sistema", font=(FONT_FAMILY, 22, "bold"), text_color=COLORS["text_primary"]).pack()
        ctk.CTkLabel(header, text="Arquitectura, parametrización y estrategia de filtrado para sillar volcánico", font=(FONT_FAMILY, 11), text_color=COLORS["text_secondary"], justify="center").pack(pady=(6, 12))
        
        content = ctk.CTkScrollableFrame(self.content, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=15, pady=15)
        
        model_frame = ctk.CTkFrame(content, fg_color=COLORS["bg_card"], corner_radius=12, border_width=1, border_color=COLORS["border"])
        model_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(model_frame, text="CONFIGURACIÓN DEL MODELO", font=(FONT_FAMILY, 12, "bold"), text_color=COLORS["accent_gold"]).pack(pady=(15, 10), anchor="w", padx=20)
        
        model_info = [
            ("Arquitectura:", "YOLOv8 Small"),
            ("Clases:", "3 (crack, humidity, spalling)"),
            ("mAP@50-95 validado:", "63.7%"),
            ("Pesos:", "best.pt (21.47 MB)"),
        ]
        
        for label, value in model_info:
            row = ctk.CTkFrame(model_frame, fg_color="transparent")
            row.pack(fill="x", padx=20, pady=2)
            ctk.CTkLabel(row, text=label, font=(FONT_FAMILY, 10, "bold"), text_color=COLORS["text_muted"], width=180, anchor="w").pack(side="left")
            ctk.CTkLabel(row, text=value, font=(FONT_FAMILY, 10), text_color=COLORS["text_primary"], anchor="w").pack(side="left")
        
        env_frame = ctk.CTkFrame(content, fg_color=COLORS["bg_card"], corner_radius=12, border_width=1, border_color=COLORS["border"])
        env_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(env_frame, text="ENTORNO DE EJECUCIÓN", font=(FONT_FAMILY, 12, "bold"), text_color=COLORS["accent_gold"]).pack(pady=(15, 10), anchor="w", padx=20)
        
        env_info = [
            ("Framework:", "CustomTkinter (Desktop nativo)"),
            ("Backend ML:", "Ultralytics YOLO v8.x"),
            ("Base de Datos:", "Supabase PostgreSQL (nube)"),
            ("Visualización:", "Matplotlib + OpenCV + Canvas Interactivo"),
            ("Aceleración:", "CUDA si disponible"),
        ]
        
        for label, value in env_info:
            row = ctk.CTkFrame(env_frame, fg_color="transparent")
            row.pack(fill="x", padx=20, pady=2)
            ctk.CTkLabel(row, text=label, font=(FONT_FAMILY, 10, "bold"), text_color=COLORS["text_muted"], width=180, anchor="w").pack(side="left")
            ctk.CTkLabel(row, text=value, font=(FONT_FAMILY, 10), text_color=COLORS["text_primary"], anchor="w").pack(side="left")
        
        filters_frame = ctk.CTkFrame(content, fg_color=COLORS["bg_card"], corner_radius=12, border_width=1, border_color=COLORS["border"])
        filters_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(filters_frame, text="ESTRATEGIA DE FILTRADO PARA SILLAR VOLCÁNICO", font=(FONT_FAMILY, 12, "bold"), text_color=COLORS["accent_gold"]).pack(pady=(15, 10), anchor="w", padx=20)
        
        filters_info = [
            ("Grietas (Crack)", "Umbral: 0.15", "Alta sensibilidad para fisuras delgadas"),
            ("Humedad (Humidity)", "Umbral: 0.60", "Anti-sombras + reclasifica a grieta si aspect ratio > 4.5"),
            ("Desprendimiento (Spalling)", "Umbral: 0.70", "Filtra textura rugosa natural con umbral alto"),
        ]
        
        for patologia, umbral, justificacion in filters_info:
            row = ctk.CTkFrame(filters_frame, fg_color="transparent")
            row.pack(fill="x", padx=20, pady=5)
            ctk.CTkLabel(row, text=f"- {patologia}", font=(FONT_FAMILY, 10, "bold"), text_color=COLORS["text_primary"], anchor="w").pack(side="left")
            ctk.CTkLabel(row, text=umbral, font=(FONT_FAMILY, 9), text_color=COLORS["accent_gold"], anchor="w", padx=10).pack(side="left")
            ctk.CTkLabel(row, text=justificacion, font=(FONT_FAMILY, 9), text_color=COLORS["text_muted"], anchor="w").pack(side="left")
        
        db_frame = ctk.CTkFrame(content, fg_color=COLORS["bg_card"], corner_radius=12, border_width=1, border_color=COLORS["border"])
        db_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(db_frame, text="ESQUEMA DE BASE DE DATOS (Supabase)", font=(FONT_FAMILY, 12, "bold"), text_color=COLORS["accent_gold"]).pack(pady=(15, 10), anchor="w", padx=20)
        
        db_text = """
Tabla: inspection_results
id (INTEGER PRIMARY KEY)
filename (TEXT UNIQUE)
crack_count, humidity_count, spalling_count (INTEGER)
total_detections (INTEGER)
total_area_cm2, total_area_m2 (REAL)
avg_confidence (REAL)
created_at (TIMESTAMP)

Tabla: detection_details
id (INTEGER PRIMARY KEY)
inspection_id (FK -> inspection_results.id)
class_name (TEXT)
confidence (REAL)
x1, y1, x2, y2 (INTEGER)
width_px, height_px (INTEGER)
area_cm2, area_m2 (REAL)
gravity_level (TEXT) - NUEVO v6.5
gravity_score (INTEGER) - NUEVO v6.5
"""
        ctk.CTkLabel(db_frame, text=db_text, font=(FONT_FAMILY, 9), text_color=COLORS["text_primary"], justify="left").pack(padx=20, pady=(0, 15), anchor="w")
    
    def on_closing(self):
        self.destroy()

if __name__ == "__main__":
    app = HeritageDetectorDesktop()
    app.mainloop()
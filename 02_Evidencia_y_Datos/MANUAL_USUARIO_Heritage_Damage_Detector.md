# PORTADA

---

**HERITAGE DAMAGE DETECTOR v6.5**

**MANUAL DE USUARIO Y DOCUMENTACIÓN TÉCNICA**

Sistema de detección de patologías en mampostería de sillar mediante visión computacional con YOLOv8

---

Universidad Católica de Santa María — Arequipa, Perú

Escuela Profesional de Ingeniería de Sistemas

Autor: Niurka Guevara

Versión del documento: 1.0

Fecha: 16 de septiembre de 2026

---

[CAPTURA: Figura de portada - (archivo sugerido: portada_sillar.png) - Imagen de portada (pendiente de incorporar).]

---

# ÍNDICE GENERAL

---

# ÍNDICE DE FIGURAS

---

# ÍNDICE DE TABLAS

---

# 1. INTRODUCCIÓN

## 1.1 Propósito

Este documento describe de forma completa el sistema **Heritage Damage Detector**, desarrollado en el marco de un proyecto de investigación de la Universidad Católica de Santa María (UCSM) de Arequipa. El sistema automatiza la detección de patologías —grietas, humedad y desprendimiento— en mampostería de sillar volcánico mediante visión computacional, empleando el modelo de detección de objetos YOLOv8.

El manual está dirigido a dos grupos de lectores:

- **Operadores técnicos** (ingenieros, arquitectos, restauradores y personal de mantenimiento) que utilizarán el sistema para inspeccionar el patrimonio; encontrarán las guías de instalación y de uso.
- **Evaluadores y jurados de sustentación** que requieran comprender la arquitectura, el método de procesamiento y los parámetros del modelo; encontrarán la descripción general y las referencias.

## 1.2 Alcance

El documento cubre la **aplicación de escritorio** del sistema (`desktop_app.py`, basada en CustomTkinter), que integra el análisis en campo, la visualización interactiva, el recorte de zonas lejanas, el cuadro de mandos, el historial y la especificación técnica.

No forma parte del alcance de este manual la construcción del conjunto de datos ni el entrenamiento del modelo, que se documenta en la especificación técnica del módulo *Especificación Técnica* del propio sistema.

## 1.3 A quién va dirigido

- Personal de inspección y mantenimiento del Santuario de San Agustín (Arequipa) y de otros inmuebles con fachadas de sillar.
- Estudiantes e investigadores del área de visión computacional aplicada al patrimonio.
- Docentes y jurados de evaluación del proyecto de investigación.

## 1.4 Convenciones tipográficas

| Elemento | Formato usado |
|----------|---------------|
| Comandos en terminal | `Courier New` 10 pt (p. ej. `pip install -r REQUIREMENTS.TXT`) |
| Controles de la interfaz | **Negrita** (p. ej. el botón **INICIAR PROCESAMIENTO**) |
| Rutas de archivo | `Courier New` 10 pt (p. ej. `resultados\`) |
| Notas y advertencias | Recuadros con fondo gris y borde, precedidos por `>` |

En las guías de uso, los pasos se numeran y cada paso finaliza con la captura de pantalla correspondiente, referenciada como *Figura N.M*.

---

# 2. DESCRIPCIÓN GENERAL DEL SISTEMA

## 2.1 Qué es el sistema

Heritage Damage Detector es un sistema de inspección asistida que toma fotografías o capturas de dron de muros de sillar y devuelve:

- una imagen anotada con cajas sobre cada patología detectada;
- el número de detecciones por tipo (grieta, humedad, desprendimiento);
- el área afectada en cm² y m²;
- la confianza promedio del modelo;
- la clasificación de gravedad de cada daño y un reporte de priorización de intervención;
- la opción de guardar el historial en una base de datos en la nube (Supabase) y de exportar un reporte CSV.

## 2.2 Patologías detectadas

El modelo identifica tres clases de patología:

Tabla 2.1. Clases de patología detectadas.

| Clase | Nombre en la interfaz | Descripción |
|-------|------------------------|-------------|
| `crack` | Grietas (Crack) | Fisuras y grietas en la superficie del sillar, incluidas fisuras capilares delgadas |
| `humidity` | Humedad (Humidity) | Manchas de humedad, filtraciones y zonas húmedas |
| `spalling` | Desprendimiento (Spalling) | Pérdida de material, descascarado y desprendimiento superficial |

## 2.3 Modelo de visión computacional

- **Arquitectura:** YOLOv8 Small (YOLOv8s) de Ultralytics.
- **Número de clases:** 3 (crack, humidity, spalling).
- **Métrica validada:** mAP@50-95 ≈ 63,5 %.
- **Archivo de pesos:** `best.pt` (≈ 21,47 MB).
- **Motor de inferencia:** Ultralytics YOLO v8.x con PyTorch, acelerado por CUDA cuando está disponible.

## 2.4 Estrategia de procesamiento

El sistema combina la red neuronal con una cadena de filtros y post-procesado diseñada específicamente para el sillar volcánico arequipeño:

1. **Micro-inferencia por tiling:** la imagen se divide en mosaicos (*tiles*) de 640 px con un solapamiento del 20 %, de modo que los daños pequeños en tomas lejanas no se pierden.
2. **Multi-escala:** se ejecuta la inferencia a varias resoluciones (1536, 1280, 960 y 640 px) para cubrir daños de distintos tamaños.
3. **Supresión de no máximos (NMS):** se eliminan las cajas duplicadas por clase con un umbral de IoU de 0,45.
4. **Filtro geométrico anti-porosidad:** descarta cajas de humedad demasiado alargadas o demasiado pequeñas, propias de la porosidad del sillar.
5. **Reclasificación:** una humedad delgada y alargada (relación de aspecto mayor de 4,5) con confianza suficiente se reclasifica como grieta; en caso contrario se descarta.
6. **Filtros anti-vegetación, anti-cielo y anti-sombra:** el sillar nunca es verde vivo ni azul cielo, por lo que dichas regiones se descartan; las sombras alargadas no se confunden con humedad.
7. **Filtro de material:** descarta calamina corrugada, techos lisos, vidrio y muros lejanos (rangos de color y textura que no corresponden al sillar beige).
8. **Confirmación de grietas filiformes:** en el análisis de zonas recortadas, una caja de grieta solo se acepta si el trazo es sinuoso, de anchura variable y con línea oscura propia de una fisura real; así se evitan falsos positivos estructurales (techos corrugados, cables y sombras).
9. **Agrupación de zonas:** las cajas de la misma clase que se tocan se fusionan en una única zona afectada, conservando la confianza más alta del grupo.
10. **Clasificación de gravedad y priorización:** cada detección se clasifica en *leve, moderada, severa o crítica* según su área, y se genera un reporte de urgencia con recomendación de intervención.
11. **Generación de resultados:** imagen anotada, métricas, detalle de detecciones, CSV e historial.

Figura 2.1. Flujo general de procesamiento.

[CAPTURA: Figura 2.1 – (archivo sugerido: diagrama_flujo_procesamiento.png) – Diagrama de bloques: entrada de imagen, tiling, inferencia YOLOv8s, NMS, filtros, agrupación, gravedad, salidas.]

## 2.5 Interfaz de la aplicación

El sistema se ejecuta como una **aplicación de escritorio** (`desktop_app.py`, basada en CustomTkinter) que integra todos los módulos en una única ventana:

| Módulo | Descripción |
|--------|-------------|
| Análisis en Campo | Carga de imágenes, calibración, procesamiento, revisión interactiva con *hover*, recorte de zonas lejanas y exportación |
| Cuadro de Mandos | Métricas globales y gráficos del historial almacenado |
| Historial | Listado de inspecciones guardadas y vista ANTES / DESPUÉS |
| Especificación Técnica | Documentación del modelo y del esquema de datos integrada en la aplicación |

Todos los módulos comparten el mismo motor de procesamiento y los mismos umbrales de configuración.

## 2.6 Persistencia de datos

Los resultados se almacenan en **Supabase** (PostgreSQL en la nube) a través de su API REST:

- Tabla `inspection_results`: una fila por imagen procesada (conteos, área total, confianza promedio, URLs de imágenes original y anotada).
- Tabla `detection_details`: una fila por cada caja detectada (clase, confianza, coordenadas, área y gravedad).
- Bucket de almacenamiento `inspecciones`: imágenes originales y anotadas en formato JPG.

La conexión se configura mediante el archivo `.env` (ver sección 4).

## 2.7 Sistema de alertas

El sistema evalúa cada resultado contra umbrales configurables y genera alertas clasificadas por nivel (ALTA, MEDIA):

- Área de daño excesiva (por defecto, mayor de 0,5 m²).
- Alta cantidad de grietas (por defecto, más de 10).
- Baja confianza promedio (por defecto, menor de 70 %).
- Crecimiento significativo de daño respecto al último historial (por defecto, mayor de 20 %).

Las alertas se registran en el archivo de registro de la aplicación y pueden enviarse, de forma opcional, por correo electrónico.

---

# 3. REQUISITOS DE INSTALACIÓN

## 3.1 Requisitos de software

Tabla 3.1. Software requerido.

| Software | Versión mínima | Observaciones |
|----------|----------------|---------------|
| Sistema operativo | Windows 10/11 (también compatible con Linux/macOS) | Se recomienda Windows para la aplicación de escritorio |
| Python | 3.8 o superior | Instalar marcando la opción "Add Python to PATH" |
| pip | Incluido con Python | Gestor de paquetes |
| Conexión a internet | Requerida | Para Supabase (nube) y la primera carga de dependencias |

## 3.2 Paquete comprimido

El sistema se entrega como un paquete comprimido (ZIP/RAR). El receptor debe **descomprimirlo** en una carpeta de su computadora antes de instalar. El contenido del paquete es el siguiente:

Tabla 3.2. Contenido del paquete comprimido.

| Archivo | Descripción |
|---------|-------------|
| `desktop_app.py` | Aplicación de escritorio (CustomTkinter) |
| `best.pt` | Modelo entrenado (pesos YOLOv8s) |
| `detection_gravity.py` | Módulo de clasificación de gravedad y priorización |
| `alerts_system.py` | Módulo de alertas automáticas |
| `REQUIREMENTS.TXT` | Lista de dependencias de Python |
| `.env.example` | Plantilla del archivo de configuración |
| `config_storage_historial.sql` | Script SQL para habilitar el historial con imágenes en Supabase |
| `INSTRUCCIONES_DE_INSTALACION.txt` | Instrucciones rápidas |

> Nota: el archivo `best.pt` debe permanecer **en la misma carpeta que `desktop_app.py`**. Si no está presente, la aplicación muestra el error "Modelo no encontrado".

## 3.3 Pasos de instalación

### 3.3.1 Descomprimir el paquete

1. Copie el paquete comprimido a la computadora donde se instalará el sistema.
2. Descomprímalo en una carpeta sin espacios ni caracteres especiales en la ruta, por ejemplo `C:\Heritage_Detector`.
3. Verifique que el contenido coincida con la Tabla 3.2.

### 3.3.2 Verificar Python

1. Abra una terminal (en Windows, CMD o PowerShell) en la carpeta del proyecto.
2. Ejecute:

```
python --version
```

3. El sistema debe responder con una versión 3.8 o superior.

### 3.3.3 Crear un entorno virtual (recomendado)

Para evitar conflictos con otros programas de la computadora, se recomienda crear un entorno virtual:

```
python -m venv venv
```

En Windows, actívelo con:

```
venv\Scripts\activate
```

En Linux/macOS:

```
source venv/bin/activate
```

### 3.3.4 Instalar las dependencias

En la terminal, dentro de la carpeta del proyecto, ejecute:

```
pip install -r REQUIREMENTS.TXT
```

La instalación puede tardar de 5 a 15 minutos según la conexión, ya que incluye PyTorch, OpenCV, Ultralytics, etc.

Figura 3.1. Consola instalando las dependencias.

[CAPTURA: Figura 3.1 – (archivo sugerido: CAP_51_1_ConsolaPipInstall.png) – Ventana de consola ejecutando `pip install -r REQUIREMENTS.TXT` con barras de progreso.]

Figura 3.2. Carpeta del proyecto con `best.pt` visible.

[CAPTURA: Figura 3.2 – (archivo sugerido: CAP_51_2_CarpetaProyectoBestPth.png) – Carpeta del proyecto descomprimida mostrando desktop_app.py, best.pt, REQUIREMENTS.TXT y el archivo .env.]

### 3.3.5 Comprobar la instalación

Para verificar que todo quedó instalado correctamente:

1. Ejecute la aplicación de escritorio (sección 4).
2. La primera ejecución carga el modelo y, si no hay errores en la consola, la instalación fue exitosa.

Figura 3.3. Primera ejecución de la aplicación de escritorio.

[CAPTURA: Figura 3.3 – (archivo sugerido: CAP_51_3_PrimeraEjecucion.png) – Consola mostrando la ejecución de la aplicación de escritorio sin errores, con el mensaje de carga del modelo y la apertura de la ventana principal.]

---

# 4. PUESTA EN MARCHA INICIAL

## 4.1 Configurar Supabase (base de datos en la nube)

El sistema guarda las inspecciones en Supabase. Si no desea guardar historial, puede omitir este paso; la aplicación funcionará en modo local sin conectar la base de datos.

1. Cree una cuenta gratuita en `https://supabase.com`.
2. Cree un nuevo proyecto y anote la **URL del proyecto** y la **clave publicable (`sb_publishable_...`)**.
3. Abra el **SQL Editor** del panel de Supabase.
4. Pegue el contenido completo del archivo `config_storage_historial.sql` y pulse **Run**. Este script crea el bucket `inspecciones` para las imágenes y habilita las columnas de URLs.

## 4.2 Configurar las variables de entorno

1. En la carpeta del proyecto, localice el archivo `.env.example`.
2. Cópielo o créelo con el nombre `.env`.
3. Ábralo con un editor de texto y complete los valores:

Tabla 4.1. Variables de entorno requeridas.

| Variable | Descripción | Ejemplo |
|----------|-------------|---------|
| `SUPABASE_URL` | URL del proyecto Supabase | `https://<proyecto>.supabase.co` |
| `SUPABASE_KEY` | Clave publicable del proyecto | `sb_publishable_XXXX...` |

Figura 4.1. Archivo `.env` configurado (valores ocultos).

[CAPTURA: Figura 4.1 – (archivo sugerido: CAP_61_1_ArchivoEnv.png) – Captura del archivo .env abierto en un editor con SUPABASE_URL y SUPABASE_KEY y los valores cubiertos con recuadro.]

> Importante: no comparta la clave API con terceros. El sistema de la UCSM ha sido publicado con claves de demostración que deben reemplazarse por las propias de cada proyecto.

## 4.3 Iniciar la aplicación de escritorio

1. En la terminal, dentro de la carpeta del proyecto:

```
python desktop_app.py
```

2. Se abrirá la ventana principal de la aplicación.
3. En el panel inferior de la barra lateral de la ventana se indica el estado de conexión con Supabase ("Supabase conectado" o "Supabase offline").

Figura 4.2. Indicador de conexión con Supabase.

[CAPTURA: Figura 4.2 – (archivo sugerido: CAP_61_2_IndicadorSupabase.png) – Recorte de la barra lateral de la aplicación de escritorio mostrando el estado "Supabase conectado" (verde).]

> Si las variables de entorno no están configuradas, la aplicación muestra el aviso "Sin credenciales de Supabase. Configura SUPABASE_URL y SUPABASE_KEY en .env" y continúa funcionando en modo local (no guarda datos).

---

# 5. GUÍA DE USO POR MÓDULOS

Esta sección presenta el uso de las cuatro áreas del sistema: **Análisis en Campo**, **Cuadro de Mandos**, **Historial** y **Especificación Técnica**.

## 5.1 Módulo de Análisis en Campo (aplicación de escritorio)

### 5.1.1 Acceso

Al iniciar `desktop_app.py`, la ventana principal se organiza en:

- **Barra lateral** con el logotipo HERITAGE DETECTOR, los módulos (Análisis en Campo, Cuadro de Mandos, Historial, Especificación Técnica), el estado de Supabase y el pie "Santuario de San Agustin, Arequipa, Perú".
- **Panel izquierdo** con la carga de imagen, la calibración y los resultados.
- **Panel derecho** con la imagen y las detecciones interactivas.

Figura 5.1. Pantalla principal de la aplicación de escritorio.

[CAPTURA: Figura 5.1 – (archivo sugerido: CAP_71_1_PantallaPrincipalDesktop.png) – Ventana principal de la aplicación de escritorio con la barra lateral, el panel de controles y la zona de imagen vacía con el mensaje "Carga una imagen del monumento para comenzar".]

### 5.1.2 Cargar imágenes

1. En el panel izquierdo, haga clic en **"Cargar Imagen(es) del Monumento"**.
2. Seleccione una o varias imágenes (JPG, JPEG, PNG, WEBP), de hasta 20 MB cada una.
3. Si alguna supera 20 MB, la aplicación muestra el error "Imagenes Demasiado Grandes" y no la acepta.
4. Si algún archivo ya existe en la base de datos, se muestra la ventana "Archivos Duplicados Detectados" y solo se procesan los nuevos.
5. El campo inferior confirma con "N archivos listos para procesar".

### 5.1.3 Configurar la calibración

En el panel izquierdo, sección **CALIBRACIÓN PARA SILLAR**:

1. **Modo de detección:** elija entre
   - **Recomendado (detecta más daños)** — umbrales 20 %, 30 %, 60 %.
   - **Alta confiabilidad (85%)** — umbrales 85 % en las tres clases.
2. Ajuste los deslizadores **Grietas (Crack)**, **Humedad (Humidity)** y **Desprendimiento (Spalling)** (rango 5 %–90 %).
3. El botón **"↺ Restaurar calibración recomendada"** restablece los valores óptimos.
4. Haga clic en **"▸ Configuración avanzada"** para ajustar:
   - **IoU Threshold** (por defecto 45 %).
   - **GSD (cm/pixel)** (por defecto 0,13).
   - Casilla **Activar Tiling (detección de detalles pequeños)** (activada).
   - **Tamaño del tile (px)** (por defecto 640).
   - **Solapamiento (%)** (por defecto 20).

Tabla 5.1. Valores de calibración recomendados.

| Control | Formato | Valor recomendado | Descripción |
|---------|---------|-------------------|-------------|
| **Grietas (Crack)** | Deslizador 5–90 % | 20 % | Umbral bajo para capturar fisuras delgadas |
| **Humedad (Humidity)** | Deslizador 5–90 % | 30 % | Umbral medio para evitar confusiones con sombras |
| **Desprendimiento (Spalling)** | Deslizador 5–90 % | 60 % | Umbral medio para filtrar textura rugosa natural |
| **IoU Threshold** | Deslizador 0,1–0,9 | 0,45 | Superposición aceptada entre cajas |
| **GSD (cm/pixel)** | Campo numérico 0,01–1,0 | 0,13 | Conversión de píxeles a centímetros reales |
| **Tamaño del tile (px)** | Deslizador 320–1280 | 640 | Tamaño de cada mosaico |
| **Solapamiento (%)** | Deslizador 0–50 | 20 | Solape entre mosaicos |

Tabla 5.2. Controles de la interfaz de escritorio.

| Control | Tipo | Función |
|---------|------|---------|
| **Cargar Imagen(es) del Monumento** | Botón | Abre el selector de archivos |
| **Modo de detección** | Botón de opción | Recomendado o Alta confiabilidad |
| **Grietas / Humedad / Desprendimiento** | Deslizadores | Umbral de confianza por clase |
| **↺ Restaurar calibración recomendada** | Botón | Vuelve a los valores óptimos |
| **▸ Configuración avanzada** | Botón | Muestra IoU, GSD, tiling, tile y solapamiento |
| **INICIAR PROCESAMIENTO** | Botón | Procesa las imágenes cargadas |
| **CANCELAR PROCESAMIENTO** | Botón | Detiene las imágenes pendientes |
| **✂️ ANALIZAR ZONA (FOTO LEJANA)** | Botón | Activa el recorte manual sobre la imagen |
| **GUARDAR EN BASE DE DATOS** | Botón | Sube las inspecciones a Supabase |
| **EXPORTAR CSV** | Botón | Exporta el reporte en formato CSV |

### 5.1.4 Procesar las imágenes

1. Haga clic en **"INICIAR PROCESAMIENTO"**.
2. La barra de progreso muestra el avance; cada imagen procesada se añade a la cola de resultados.
3. Use **"◀ Anterior"** y **"Siguiente ▶"** para recorrer los resultados, junto con el contador "Imagen X de Y".
4. Si desea reprocesar con una calibración distinta, haga clic otra vez en *INICIAR PROCESAMIENTO* y confirme "Volver a Procesar".
5. Use **"CANCELAR PROCESAMIENTO"** para detener las imágenes pendientes (se conservan las ya procesadas).

### 5.1.5 Revisar resultados interactivos

1. Pase el **cursor** sobre cada caja de la imagen anotada: aparece una ventana emergente (*tooltip*) con las medidas de la detección (clase, confianza, área y tamaño).
2. Los colores de las cajas indican el **tipo de patología**; en el panel de resultados, las insignias resumen el conteo por clase (`🔴 Grietas`, `🔵 Humedad`, `🟠 Desprendimiento`, `⚡ Posibles Grietas`).
3. El panel central muestra tres tarjetas: **Total Detecciones**, **Área Afectada** y **Confianza Promedio**.
4. Si el modelo no detecta daños, aparece el aviso "NO SE DETECTARON DAÑOS EN ESTA IMAGEN" con la recomendación de usar la herramienta de zona lejana (5.1.6).
5. Si hay daños, se muestra el bloque **PRIORIZACIÓN DE INTERVENCIÓN** con la recomendación, el score de prioridad, la urgencia y los conteos de críticas y severas.

Figura 5.2. Carga de imagen y resultado anotado (escritorio).

[CAPTURA: Figura 5.2 – (archivo sugerido: CAP_71_2_ResultadoDesktop.png) – Ventana de la aplicación de escritorio con una imagen de sillar anotada con cajas y el panel de resultados con las métricas.]

### 5.1.6 Analizar una zona de una foto lejana

Cuando la fotografía está tomada desde lejos y los daños ocupan pocos píxeles:

1. Haga clic en **"✂️ ANALIZAR ZONA (FOTO LEJANA)"**.
2. Arrastre un rectángulo sobre la zona del muro que desea inspeccionar.
3. La aplicación amplía el recorte (mínimo 2x, hasta 2400 px en el lado mayor) y lo procesa con el motor completo, relajando los umbrales de grieta y humedad (10 % y 20 %) y exigiendo la confirmación filiforme de las grietas.
4. La zona recortada puede mostrar además **"Posibles Grietas"** (marcadas en amarillo) detectadas por análisis de bordes. Estas candidatas **requieren confirmación en campo** y no constituyen un diagnóstico.

Figura 5.3. Selección de zona en imágenes lejanas.

[CAPTURA: Figura 5.3 – (archivo sugerido: CAP_71_3_RecorteZona.png) – Imagen con un rectángulo amarillo punteado sobre una zona del muro y, a la derecha, la zona ampliada procesada con cajas.]

### 5.1.7 Guardar y exportar

1. Haga clic en **"GUARDAR EN BASE DE DATOS"** para guardar todas las inspecciones del lote en Supabase. Al confirmar, el botón cambia a **"YA GUARDADO"**.
2. La aplicación guarda además copias locales en la carpeta `resultados\`:
   - `resultado_<nombre>_<fecha-hora>.jpg` (imagen anotada).
   - `resultado_<nombre>_<fecha-hora>.json` (metadatos y parámetros usados).
3. Haga clic en **"EXPORTAR CSV"** para guardar el reporte con el nombre `reporte_sillar_AAAAMMDD_HHMM.csv`.

## 5.2 Cuadro de Mandos (escritorio)

1. En la barra lateral, seleccione **Cuadro de Mandos**.
2. Se muestran las tarjetas **Total Inspecciones**, **Total Daños**, **Superficie Afectada** y **Confianza Promedio**.
3. Los gráficos **Distribución de Daños por Tipo** (barras) y **Distribución de Área Afectada (m²)** (circular) resumen el historial almacenado.

Figura 5.4. Cuadro de mandos (escritorio).

[CAPTURA: Figura 5.4 – (archivo sugerido: CAP_71_4_DashboardDesktop.png) – Módulo Cuadro de Mandos con las cuatro tarjetas de métricas y los dos gráficos.]

## 5.3 Historial (escritorio)

1. Seleccione **Historial** en la barra lateral.
2. Se carga la lista de inspecciones guardadas con fecha, archivo, conteos, área y confianza.
3. Seleccione una fila y pulse el botón para abrir la vista **ANTES / DESPUÉS**: la imagen original y la anotada lado a lado.

Figura 5.5. Historial de inspecciones (escritorio).

[CAPTURA: Figura 5.5 – (archivo sugerido: CAP_71_5_HistorialDesktop.png) – Módulo Historial con la tabla de inspecciones y la ventana ANTES/DESPUÉS abierta.]

## 5.4 Especificación Técnica

El módulo **Especificación Técnica** documenta, dentro del propio sistema:

- Configuración del modelo (arquitectura YOLOv8s, clases, mAP@50-95 63,5 %, pesos best.pt).
- Entorno de ejecución (framework, backend, base de datos, aceleración CUDA).
- Estrategia de filtrado para sillar volcánico.
- Esquema de la base de datos en Supabase.

---

# 6. INTERPRETACIÓN DE RESULTADOS

## 6.1 Colores de las cajas en la imagen anotada

Cada detección se dibuja como una caja de color según la clase de patología, con una etiqueta de texto sobre ella que indica la clase y la confianza en porcentaje:

| Clase | Color de la caja | Ejemplo de etiqueta |
|-------|------------------|----------------------|
| `crack` (grieta) | **Rojo** | `crack 85.3%` |
| `humidity` (humedad) | **Azul / turquesa** | `humidity 78.9%` |
| `spalling` (desprendimiento) | **Naranja** | `spalling 91.2%` |

El color de las cajas depende del modo de visualización. En la **imagen anotada exportada** (y en las figuras de ejemplo de este manual) el color corresponde a la **clase** de patología, según la tabla anterior. En cambio, en el **lienzo de la aplicación de escritorio** las cajas se colorean según la **gravedad** del daño, para resaltar de un vistazo lo más urgente: verde (leve), ámbar (moderada), naranja (severa) y rojo (crítica).

Las **Posibles Grietas** (solo en el análisis de zona de fotos lejanas) se dibujan en **amarillo** y representan candidatas a confirmar en campo.

Figura 6.1. Imagen anotada de una fachada de sillar con cajas de colores según la clase detectada.

[CAPTURA: Figura 6.1 – (archivo sugerido: CAP_81_1_CajasPorColor.png) – Fotografía de fachada de sillar anotada por el sistema: las cajas rojas corresponden a grietas y las naranjas a desprendimiento, cada una con su etiqueta de clase y confianza.]

## 6.2 Métricas mostradas

| Métrica | Significado |
|---------|-------------|
| Total Detecciones | Número de cajas validadas tras los filtros |
| Área Afectada | Suma de las áreas de las cajas en cm² y m² (convierte píxeles usando el GSD) |
| Confianza Promedio | Promedio de las confianzas de todas las detecciones |
| Distribución por tipo | Conteo de grietas, humedad y desprendimiento (insignias) |

## 6.3 Clasificación de gravedad

Cada detección se clasifica según su área (en cm²) con los siguientes umbrales:

Tabla 6.1. Umbrales de gravedad por patología.

| Patología | Leve | Moderada | Severa | Crítica |
|-----------|------|----------|--------|---------|
| Grieta (crack) | 0 – 5 cm² | 5 – 30 cm² | 30 – 100 cm² | > 100 cm² |
| Humedad (humidity) | 0 – 50 cm² | 50 – 200 cm² | 200 – 500 cm² | > 500 cm² |
| Desprendimiento (spalling) | 0 – 20 cm² | 20 – 100 cm² | 100 – 300 cm² | > 300 cm² |

| Nivel | Color | Indicador |
|-------|-------|-----------|
| Leve | Verde | Símbolo 🟢 |
| Moderada | Ámbar | Símbolo 🟡 |
| Severa | Naranja | Símbolo 🟠 |
| Crítica | Rojo | Símbolo 🔴 |

## 6.4 Reporte de priorización de intervención

El sistema genera un **score de prioridad** (suma de puntuaciones: leve=1, moderada=2, severa=3, crítica=4) y un nivel de urgencia global:

Tabla 6.2. Niveles de urgencia y recomendaciones.

| Nivel | Condición | Recomendación |
|-------|-----------|----------------|
| CRÍTICA | Al menos 1 detección crítica | INTERVENCIÓN INMEDIATA requerida |
| ALTA | Más de 2 detecciones severas | Programar intervención en < 30 días |
| MEDIA | Score total > 15 | Programar intervención en < 90 días |
| BAJA | Resto de casos | Monitoreo preventivo |

Figura 6.2. Reporte de priorización de intervención.

[CAPTURA: Figura 6.2 – (archivo sugerido: CAP_81_2_Priorizacion.png) – Panel "PRIORIZACIÓN DE INTERVENCIÓN" con la recomendación, el score de prioridad, la urgencia y los conteos de críticas y severas.]

## 6.5 Reporte CSV

El CSV exportado usa punto y coma (`;`) como separador y coma como separador decimal. Sus columnas son:

Tabla 6.3. Columnas del reporte CSV.

| Columna | Descripción |
|---------|-------------|
| `Archivo` | Nombre del archivo de imagen |
| `Cracks` | Número de grietas |
| `Humedad` | Número de humedades |
| `Spalling` | Número de desprendimientos |
| `Total_Detecciones` | Total de cajas validadas |
| `Area_cm2` | Área total en centímetros cuadrados |
| `Area_m2` | Área total en metros cuadrados |
| `Confianza_Promedio` | Confianza promedio en porcentaje |

---

# 7. SOLUCIÓN DE PROBLEMAS FRECUENTES

Tabla 7.1. Errores frecuentes y soluciones.

| Síntoma / mensaje | Causa probable | Solución |
|--------------------|----------------|----------|
| "Modelo no encontrado: best.pt" o "Modelo no encontrado" | El archivo `best.pt` no está junto a la aplicación | Copie `best.pt` a la misma carpeta que `desktop_app.py` |
| "Sin conexión con Supabase" | Falta internet, credenciales erróneas o tablas no creadas | Verifique internet; revise `SUPABASE_URL` y `SUPABASE_KEY` en `.env`; ejecute el SQL de configuración en el SQL Editor |
| No se guardan inspecciones | Base de datos en modo offline | Configure el `.env` y reinicie la aplicación |
| `ModuleNotFoundError` | Dependencias incompletas | Ejecute de nuevo `pip install -r REQUIREMENTS.TXT` |
| La aplicación es lenta | Carga inicial del modelo, imágenes grandes o tiling activado | Es normal la primera carga; desactive tiling si el tiempo es excesivo; use imágenes de menor resolución |
| "IMAGEN(ES) DEMASIADO GRANDE(S)" / "Imagenes Demasiado Grandes" | Imagen mayor de 20 MB | Comprima o reduzca la imagen y vuelva a cargarla |
| "ARCHIVOS DUPLICADOS DETECTADOS" | Los archivos ya fueron procesados y guardados | Retire los duplicados del selector o procese solo archivos nuevos |
| "NO SE DETECTARON DAÑOS EN ESTA IMAGEN" | La foto está tomada desde lejos y los daños ocupan pocos píxeles | Use el botón "✂️ ANALIZAR ZONA (FOTO LEJANA)" y arrastre un recuadro sobre el muro |
| "Sin credenciales de Supabase" | Falta el archivo `.env` o está vacío | Cree `.env` a partir de `.env.example` y complete los valores |
| Se detectan muchas humedades falsas en interiores pintados | Paredes o techos interiores pintados en tonos cálidos | Descarte esas capturas o ajuste el umbral de humedad; la humedad real es más saturada y contrasta con su entorno |

Figura 7.1. Mensaje de error "Sin conexión con Supabase".

[CAPTURA: Figura 7.1 – (archivo sugerido: CAP_91_1_ErrorSupabase.png) – Estado "Supabase offline" en la barra lateral de la aplicación de escritorio.]

Figura 7.2. Mensaje de límite de tamaño de archivo.

[CAPTURA: Figura 7.2 – (archivo sugerido: CAP_91_2_ErrorLimiteArchivo.png) – Ventana o mensaje "Imagenes Demasiado Grandes" / "IMAGEN(ES) DEMASIADO GRANDE(S)" indicando que el peso máximo por imagen es de 20 MB.]

---

# 8. MANTENIMIENTO Y ACTUALIZACIÓN DEL MODELO

## 8.1 Recomendaciones de uso y conservación del historial

1. Mantenga `best.pt` respaldado y versionado (por ejemplo, en un repositorio Git).
2. Realice inspecciones periódicas comparando con el historial de Supabase; el sistema alerta ante crecimiento significativo de daño.
3. Guarde los resultados en la base de datos para alimentar los cuadros de mandos y la tendencia temporal.
4. No comparta el archivo `.env` en repositorios; úselo solo localmente.

## 8.2 Actualización del modelo

El modelo puede reentrenarse con nuevas imágenes etiquetadas para mejorar la detección en otros inmuebles:

1. Recopile un conjunto de imágenes nuevas y etiquételas en formato YOLO con las tres clases (crack, humidity, spalling).
2. Reentrene el modelo (los ajustes de entrenamiento se encuentran documentados en la especificación técnica del proyecto).
3. Reemplace `best.pt` por el nuevo archivo de pesos; no es necesario cambiar la aplicación.
4. Valide el nuevo modelo con un conjunto de prueba y verifique que el mAP@50-95 no disminuya.

## 8.3 Ajuste de umbrales de alertas

Los umbrales del sistema de alertas (`max_damage_area_m2`, `min_new_cracks`, `growth_percentage_threshold`, `min_avg_confidence`) se configuran en el módulo `alerts_system.py` y pueden ajustarse según el criterio del arquitecto supervisor.

## 8.4 Eliminación de datos

Las inspecciones pueden eliminarse desde el panel de Supabase (dashboard de la nube). El botón de guardado en la aplicación solo inserta nuevos registros; no elimina datos existentes.

---

# 9. REFERENCIAS

- Jocher, G., Chaurasia, A. y Qiu, J. (2026). *Ultralytics YOLOv8. Documentación oficial*. Disponible en: https://docs.ultralytics.com
- Redmon, J., Divvala, S., Girshick, R. y Farhadi, A. (2016). *You Only Look Once: Unified, Real-Time Object Detection*. IEEE CVPR.
- Supabase. (2026). *Supabase Documentation: REST API*. Disponible en: https://supabase.com/docs/guides/api
- OpenCV. (2026). *Open Source Computer Vision Library*. Disponible en: https://opencv.org
- Biblioteca de la Escuela Politécnica Superior, Universidad de Alicante. *Libro de estilo para los trabajos de fin de grado y máster*.
- Gutiérrez, G. (2023). *Patología de la construcción en mampostería de piedra volcánica*. Universidad Católica de Santa María, Arequipa.

---

# APÉNDICE A. TABLA DE CAPTURAS

| Nº Figura | Archivo sugerido | Sección | Descripción | Estado |
|-----------|------------------|---------|-------------|--------|
| Figura 2.1 | diagrama_flujo_procesamiento.png | 2.4 | Diagrama de bloques del flujo de procesamiento | Hecho |
| Figura 3.1 | CAP_51_1_ConsolaPipInstall.png | 3.3.4 | Consola instalando dependencias | Hecho |
| Figura 3.2 | CAP_51_2_CarpetaProyectoBestPth.png | 3.3.1 | Carpeta del proyecto con best.pt | Hecho |
| Figura 3.3 | CAP_51_3_PrimeraEjecucion.png | 3.3.5 | Primera ejecución sin errores | Hecho |
| Figura 4.1 | CAP_61_1_ArchivoEnv.png | 4.2 | Archivo .env (valores ocultos) | Hecho |
| Figura 4.2 | CAP_61_2_IndicadorSupabase.png | 4.3 | Indicador "Supabase conectado" | Hecho |
| Figura 5.1 | CAP_71_1_PantallaPrincipalDesktop.png | 5.1.1 | Ventana principal escritorio | Hecho |
| Figura 5.2 | CAP_71_2_ResultadoDesktop.png | 5.1.5 | Resultado anotado escritorio | Hecho |
| Figura 5.3 | CAP_71_3_RecorteZona.png | 5.1.6 | Selección de zona foto lejana | Hecho |
| Figura 5.4 | CAP_71_4_DashboardDesktop.png | 5.2 | Cuadro de mandos escritorio | Hecho |
| Figura 5.5 | CAP_71_5_HistorialDesktop.png | 5.3 | Historial escritorio | Hecho |
| Figura 6.1 | CAP_81_1_CajasPorColor.png | 6.1 | Cajas rojas (grieta) y naranjas (desprendimiento) sobre fachada de sillar | Hecho |
| Figura 6.2 | CAP_81_2_Priorizacion.png | 6.4 | Reporte de priorización | Hecho |
| Figura 7.1 | CAP_91_1_ErrorSupabase.png | 7 | Error "Sin conexión con Supabase" | Hecho |
| Figura 7.2 | CAP_91_2_ErrorLimiteArchivo.png | 7 | Mensaje de límite de tamaño | Hecho |

---

*Fin del manual.*

"""
Convección Chile — satelite.metotec.cl
Autor: Orlando Álvarez Bustamante

Cada ejecución:
  1. Busca la imagen más reciente del GOES-19 (banda 13, disco completo) en el
     repositorio público de la NOAA en AWS, y los rayos GLM de esos 10 minutos.
  2. Reproyecta la banda 13 a un mapa (Mercator) de Chile y el Pacífico vecino.
  3. Detecta nubes de tope frío, las clasifica (CB, moderada, alta, frontal) y
     las sigue respecto de la imagen anterior (creciendo / estable / disipando).
  4. Escribe la imagen PNG y el archivo data/index.json que lee la página.

Uso:
  python procesar.py                       # datos reales
  python procesar.py --sitio-url URL       # recupera el loop ya publicado
  python procesar.py --demo                # datos sintéticos, para probar sin internet
"""

import argparse, io, json, math, os, re, sys, time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import numpy as np
import sat_estaciones as SE
import requests
from PIL import Image
from scipy import ndimage
from shapely import contains_xy
from shapely.geometry import shape
from skimage.measure import find_contours, approximate_polygon

# ---------------- Configuración ----------------
LON_MIN, LON_MAX = -118.0, -32.0      # ancho suficiente para llenar una pantalla horizontal con todo Chile
LAT_MIN, LAT_MAX = -62.0, -10.0
ANCHO = 2000                           # píxeles de la imagen (~4 km por píxel)
LON_VECINOS = -53.0                    # al este de esto no se detecta (Atlántico, Brasil)
UMBRAL = -40.0                         # °C: tope frío que define un objeto
UMBRAL_SOMERO = -10.0                  # °C: convección somera (cúmulos en desarrollo)
ENFRIA = -1.5                          # °C/10 min: enfriamiento mínimo en cada uno de los dos últimos intervalos
ENFRIA_TOTAL = -4.0                    # °C: enfriamiento mínimo sumando esos dos intervalos
AREA_MIN = 250.0                       # km²
ZONA_MAR_KM = 300.0                    # Pacífico cercano que también cuenta
FRAMES = 12                            # imágenes del loop (2 horas)
BUCKET = "https://noaa-goes19.s3.amazonaws.com"
R_TIERRA = 6371.0

BASE = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "conveccion-chile/1.0 (proyecto personal)"}

LUGARES = [  # referencias para nombrar las celdas
    ("Arica", -18.48, -70.31), ("Putre", -18.2, -69.56), ("Iquique", -20.21, -70.15), ("Colchane", -19.28, -68.64),
    ("Calama", -22.46, -68.93), ("San Pedro de Atacama", -22.91, -68.2), ("Antofagasta", -23.65, -70.4),
    ("Taltal", -25.4, -70.48), ("Copiapó", -27.37, -70.33), ("Vallenar", -28.58, -70.76), ("La Serena", -29.9, -71.25),
    ("Ovalle", -30.6, -71.2), ("Illapel", -31.63, -71.17), ("Los Andes", -32.83, -70.6), ("Valparaíso", -33.05, -71.62),
    ("Santiago", -33.45, -70.67), ("San Antonio", -33.59, -71.61), ("Rancagua", -34.17, -70.74), ("Pichilemu", -34.39, -72.0),
    ("Curicó", -34.98, -71.24), ("Talca", -35.43, -71.66), ("Linares", -35.85, -71.6), ("Cauquenes", -35.97, -72.32),
    ("Chillán", -36.61, -72.1), ("Concepción", -36.83, -73.05), ("Los Ángeles", -37.47, -72.35), ("Lebu", -37.61, -73.65),
    ("Angol", -37.8, -72.71), ("Temuco", -38.74, -72.6), ("Villarrica", -39.28, -72.23), ("Valdivia", -39.81, -73.25),
    ("Osorno", -40.57, -73.13), ("Puerto Montt", -41.47, -72.94), ("Ancud", -41.87, -73.83), ("Castro", -42.48, -73.76),
    ("Chaitén", -42.92, -72.71), ("Futaleufú", -43.19, -71.87), ("Puerto Aysén", -45.4, -72.69), ("Coyhaique", -45.57, -72.07),
    ("Cochrane", -47.25, -72.57), ("Caleta Tortel", -47.8, -73.54), ("Villa O'Higgins", -48.47, -72.56),
    ("Puerto Natales", -51.73, -72.5), ("Punta Arenas", -53.16, -70.91), ("Porvenir", -53.3, -70.37),
    ("Puerto Williams", -54.93, -67.6), ("Juan Fernández", -33.64, -78.83),
]
VECINOS = [  # referencias fuera de Chile
    ("Salta", -24.79, -65.41), ("San Miguel de Tucumán", -26.82, -65.22), ("Santiago del Estero", -27.79, -64.26),
    ("Catamarca", -28.47, -65.78), ("La Rioja", -29.41, -66.86), ("San Juan", -31.54, -68.53), ("Mendoza", -32.89, -68.83),
    ("San Luis", -33.3, -66.34), ("Córdoba", -31.42, -64.18), ("Rosario", -32.95, -60.64), ("Santa Fe", -31.63, -60.7),
    ("Resistencia", -27.45, -58.99), ("Formosa", -26.18, -58.17), ("Corrientes", -27.47, -58.83), ("Posadas", -27.37, -55.9),
    ("Buenos Aires", -34.6, -58.38), ("Santa Rosa", -36.62, -64.29), ("Bahía Blanca", -38.72, -62.27), ("Neuquén", -38.95, -68.06),
    ("San Carlos de Bariloche", -41.13, -71.31), ("Esquel", -42.91, -71.32), ("Trelew", -43.25, -65.31),
    ("Comodoro Rivadavia", -45.86, -67.48), ("Río Gallegos", -51.62, -69.22), ("Ushuaia", -54.8, -68.3),
    ("Malargüe", -35.47, -69.58), ("Mar del Plata", -38.0, -57.56), ("Montevideo", -34.9, -56.16),
    ("Asunción", -25.26, -57.58), ("Tarija", -21.53, -64.73), ("Uyuni", -20.46, -66.83), ("Tacna", -18.01, -70.25),
]

# ---------------- Proyección geoestacionaria ----------------
class Geo:
    def __init__(self, H, req, rpol, lon0):
        self.H, self.req, self.rpol, self.lon0 = H, req, rpol, math.radians(lon0)
        self.e2 = 1 - (rpol * rpol) / (req * req)

    def scan(self, lat, lon):
        """lat/lon (grados) -> ángulos de escaneo x, y (radianes). NaN si no es visible."""
        la, lo = np.radians(lat), np.radians(lon)
        pc = np.arctan((self.rpol ** 2 / self.req ** 2) * np.tan(la))
        rc = self.rpol / np.sqrt(1 - self.e2 * np.cos(pc) ** 2)
        sx = self.H - rc * np.cos(pc) * np.cos(lo - self.lon0)
        sy = -rc * np.cos(pc) * np.sin(lo - self.lon0)
        sz = rc * np.sin(pc)
        visible = self.H * (self.H - sx) >= sy ** 2 + (self.req ** 2 / self.rpol ** 2) * sz ** 2
        x = np.arcsin(-sy / np.sqrt(sx ** 2 + sy ** 2 + sz ** 2))
        y = np.arctan(sz / sx)
        return np.where(visible, x, np.nan), np.where(visible, y, np.nan)


# ---------------- Grilla de salida (Mercator, igual que el mapa web) ----------------
def merc_y(lat):
    return np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))

def grilla():
    ym0, ym1 = merc_y(LAT_MIN), merc_y(LAT_MAX)
    xr = math.radians(LON_MAX - LON_MIN)
    alto = int(round(ANCHO * (ym1 - ym0) / xr))
    lon = LON_MIN + (np.arange(ANCHO) + 0.5) * (LON_MAX - LON_MIN) / ANCHO
    ym = ym1 - (np.arange(alto) + 0.5) * (ym1 - ym0) / alto           # fila 0 = norte
    lat = np.degrees(2 * np.arctan(np.exp(ym)) - np.pi / 2)
    LON, LAT = np.meshgrid(lon, lat)
    px_km = R_TIERRA * math.radians((LON_MAX - LON_MIN) / ANCHO) * np.cos(np.radians(lat))  # lado del píxel por fila
    return LON, LAT, px_km, alto

def fila_a_lat(fila, alto):
    ym0, ym1 = merc_y(LAT_MIN), merc_y(LAT_MAX)
    ym = ym1 - (np.asarray(fila) + 0.5) * (ym1 - ym0) / alto
    return np.degrees(2 * np.arctan(np.exp(ym)) - np.pi / 2)

def col_a_lon(col):
    return LON_MIN + (np.asarray(col) + 0.5) * (LON_MAX - LON_MIN) / ANCHO


# ---------------- Máscaras geográficas ----------------
def mascaras(LON, LAT, px_km):
    chile = shape(json.load(open(os.path.join(BASE, "geo", "chile.geojson")))["geometry"])
    paises = [shape(f["geometry"]) for f in json.load(open(os.path.join(BASE, "geo", "paises.geojson")))["features"]]
    m_chile = contains_xy(chile, LON, LAT)
    m_tierra = np.zeros_like(m_chile)
    m_pais = np.zeros(m_chile.shape, "i2")
    for i, p in enumerate(paises, start=1):
        dentro = contains_xy(p, LON, LAT)
        m_tierra |= dentro; m_pais[dentro] = i
    dist_px = ndimage.distance_transform_edt(~m_chile)
    dist_km = dist_px * px_km[:, None]
    # solo el lado del Pacífico: oeste del punto más oriental de Chile en cada fila
    cols = np.arange(m_chile.shape[1])[None, :]
    borde = np.where(m_chile.any(1), np.where(m_chile, cols, -1).max(1), -1).astype(float)
    filas = np.arange(m_chile.shape[0]); ok = borde >= 0
    borde = np.interp(filas, filas[ok], borde[ok])
    pacifico = cols <= borde[:, None]
    zona = m_chile | (~m_tierra & pacifico & (dist_km <= ZONA_MAR_KM))
    nombres = [None] + [f["properties"]["nombre"] for f in json.load(open(os.path.join(BASE, "geo", "paises.geojson")))["features"]]
    return m_chile, m_tierra, zona, m_pais, nombres


# ---------------- Descarga desde AWS (NOAA Open Data) ----------------
def listar(prefijo):
    claves, token = [], None
    while True:
        p = {"list-type": "2", "prefix": prefijo}
        if token: p["continuation-token"] = token
        r = requests.get(BUCKET + "/", params=p, headers=UA, timeout=60)
        r.raise_for_status()
        raiz = ET.fromstring(r.content)
        ns = {"s3": raiz.tag.split("}")[0].strip("{")}
        claves += [c.find("s3:Key", ns).text for c in raiz.findall("s3:Contents", ns)]
        if raiz.findtext("s3:IsTruncated", namespaces=ns) == "true":
            token = raiz.findtext("s3:NextContinuationToken", namespaces=ns)
        else:
            return claves

def inicio_archivo(clave):
    m = re.search(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})", clave)
    y, d, H, M, S = map(int, m.groups())
    return datetime(y, 1, 1, H, M, S, tzinfo=timezone.utc) + timedelta(days=d - 1)

def archivos(producto, patron, desde, hasta):
    res, t = [], desde.replace(minute=0, second=0, microsecond=0)
    while t <= hasta:
        pref = f"{producto}/{t:%Y}/{t.timetuple().tm_yday:03d}/{t:%H}/"
        try:
            res += [k for k in listar(pref) if patron in k]
        except Exception as ex:
            print("  aviso: no se pudo listar", pref, ex)
        t += timedelta(hours=1)
    res = [(inicio_archivo(k), k) for k in res]
    return sorted([x for x in res if desde <= x[0] <= hasta])

def bajar(clave):
    for intento in range(3):
        try:
            r = requests.get(f"{BUCKET}/{clave}", headers=UA, timeout=180)
            r.raise_for_status()
            return r.content
        except Exception as ex:
            print("  reintento", intento + 1, ex); time.sleep(5)
    raise RuntimeError("No se pudo descargar " + clave)


# ---------------- Lectura de la banda 13 ----------------
def banda13(contenido, LON, LAT):
    import netCDF4
    ds = netCDF4.Dataset("c13.nc", memory=contenido)
    pr = ds.variables["goes_imager_projection"]
    geo = Geo(pr.perspective_point_height + pr.semi_major_axis, pr.semi_major_axis,
              pr.semi_minor_axis, pr.longitude_of_projection_origin)
    xs = ds.variables["x"][:].astype("f8"); ys = ds.variables["y"][:].astype("f8")
    sx, sy = geo.scan(LAT, LON)
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    ix = np.rint((sx - xs[0]) / dx); iy = np.rint((sy - ys[0]) / dy)
    ok = np.isfinite(ix) & np.isfinite(iy) & (ix >= 0) & (ix < xs.size) & (iy >= 0) & (iy < ys.size)
    i0, i1 = int(np.nanmin(ix[ok])), int(np.nanmax(ix[ok])) + 1
    j0, j1 = int(np.nanmin(iy[ok])), int(np.nanmax(iy[ok])) + 1
    ventana = ds.variables["CMI"][j0:j1, i0:i1]                 # Kelvin, ya escalado
    ventana = np.ma.filled(ventana.astype("f4"), np.nan)
    T = np.full(LON.shape, np.nan, "f4")
    T[ok] = ventana[(iy[ok] - j0).astype(int), (ix[ok] - i0).astype(int)]
    ds.close()
    return T - 273.15


def rayos_glm(t_ini):
    """Destellos GLM entre t_ini y t_ini+10 min. Devuelve (lat, lon) o None si falla."""
    import netCDF4
    try:
        lista = archivos("GLM-L2-LCFA", "OR_GLM-L2-LCFA", t_ini, t_ini + timedelta(minutes=10))
        if not lista:
            return None
        las, los = [], []
        for _, k in lista:
            ds = netCDF4.Dataset("glm.nc", memory=bajar(k))
            if "flash_lat" in ds.variables and ds.dimensions["number_of_flashes"].size:
                las.append(np.asarray(ds.variables["flash_lat"][:], "f8"))
                los.append(np.asarray(ds.variables["flash_lon"][:], "f8"))
            ds.close()
        la = np.concatenate(las) if las else np.zeros(0); lo = np.concatenate(los) if los else np.zeros(0)
        m = (la >= LAT_MIN) & (la <= LAT_MAX) & (lo >= LON_MIN) & (lo <= LON_MAX)
        print(f"  GLM: {len(lista)} archivos, {int(m.sum())} destellos en la región")
        return la[m], lo[m]
    except Exception as ex:
        print("  aviso: sin rayos GLM:", ex)
        return None


# ---------------- Colores de la imagen IR ----------------
ESCALA = [(40, (8, 8, 8)), (-30, (215, 215, 215)), (-30.01, (120, 205, 240)), (-40, (40, 95, 220)),
          (-48, (40, 190, 90)), (-55, (240, 225, 60)), (-62, (235, 90, 40)), (-70, (255, 255, 255)), (-85, (130, 70, 170))]

def colorear(T):
    t = np.nan_to_num(T, nan=50.0)
    xs = np.array([e[0] for e in ESCALA][::-1]); cs = np.array([e[1] for e in ESCALA][::-1], "f4")
    rgb = np.stack([np.interp(t, xs, cs[:, k]) for k in range(3)], -1)
    a = np.where(np.isfinite(T), 255, 0)
    return np.dstack([rgb, a]).astype("u1")


# ---------------- Detección y clasificación ----------------
def lugar(lat, lon, m_tierra_pt, en_chile, pais=None):
    def dkm(a, b, c, d):
        p = math.radians
        return 2 * R_TIERRA * math.asin(math.sqrt(math.sin(p(c - a) / 2) ** 2 + math.cos(p(a)) * math.cos(p(c)) * math.sin(p(d - b) / 2) ** 2))
    lista = LUGARES if (en_chile or not m_tierra_pt) else VECINOS + LUGARES
    n, la, lo = min(lista, key=lambda L: dkm(lat, lon, L[1], L[2]))
    d = dkm(lat, lon, la, lo)
    if not m_tierra_pt:
        return f"Pacífico frente a {n}" if lon < -70 else "Atlántico"
    if en_chile:
        return n if d < 60 else f"Cerca de {n}"
    base = n if d < 60 else f"Cerca de {n}"
    return f"{base} ({pais})" if pais and pais not in n else base

def detectar(T, LON, LAT, px_km, m_chile, m_tierra, zona, rayos, m_pais=None, nombres_pais=None):
    alto = T.shape[0]
    fria = np.isfinite(T) & (T <= UMBRAL)
    fria = ndimage.binary_opening(fria, iterations=1)
    etq, n = ndimage.label(fria, structure=np.ones((3, 3)))
    area_px = (px_km ** 2)[:, None] * np.ones_like(T)
    # rayos -> píxel
    conteo_rayos = np.zeros(n + 1, int)
    if rayos is not None and len(rayos[0]):
        la, lo = rayos
        col = np.clip(((lo - LON_MIN) / (LON_MAX - LON_MIN) * ANCHO).astype(int), 0, ANCHO - 1)
        ym0, ym1 = merc_y(LAT_MIN), merc_y(LAT_MAX)
        fil = np.clip(((ym1 - merc_y(la)) / (ym1 - ym0) * alto).astype(int), 0, alto - 1)
        etq_dil = ndimage.maximum_filter(etq, size=5)        # incluye destellos en el borde de la nube
        np.add.at(conteo_rayos, etq_dil[fil, col], 1)
    celdas = []
    for i, sl in enumerate(ndimage.find_objects(etq), start=1):
        if sl is None: continue
        m = etq[sl] == i
        area = float(area_px[sl][m].sum())
        if area < AREA_MIN: continue
        Ti = T[sl][m]
        en_zona = bool(zona[sl][m].any())
        frac_mar = 1 - float(m_tierra[sl][m].mean())
        lon_c = float(LON[sl][m].mean()); lat_c = float(LAT[sl][m].mean())
        region = "chile"
        if not en_zona:
            if frac_mar > 0.6 and lon_c < -74:
                region = "lejano"
            elif frac_mar <= 0.6 and lon_c <= LON_VECINOS:
                region = "vecinos"
            else:
                continue                                       # Atlántico o Brasil: no interesa
        tmin = float(np.nanmin(Ti)); tmed = float(np.nanmean(Ti)); desv = float(np.nanstd(Ti))
        nucleo = float(area_px[sl][m & (T[sl] <= -55)].sum())
        # forma (PCA en km)
        yy, xx = np.nonzero(m)
        la = LAT[sl][m]; lo = LON[sl][m]
        X = np.stack([np.radians(lo) * R_TIERRA * math.cos(math.radians(lat_c)), np.radians(la) * R_TIERRA])
        cov = np.cov(X) if X.shape[1] > 2 else np.eye(2)
        ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
        ev = np.maximum(ev, 1.0)
        razon = float(math.sqrt(ev[0] / ev[1])); largo = float(4 * math.sqrt(ev[0]))
        ry = int(conteo_rayos[i]) if rayos is not None else None
        # núcleo: centro del 10 % más frío
        frio = Ti <= np.nanpercentile(Ti, 10)
        lat_n, lon_n = float(la[frio].mean()), float(lo[frio].mean())
        r0, c0 = yy[frio].mean() + sl[0].start, xx[frio].mean() + sl[1].start
        tierra_n = bool(m_tierra[int(r0), int(c0)]); chile_n = bool(m_chile[sl][m].any())
        pais = None
        if m_pais is not None and region == "vecinos":
            vals = m_pais[sl][m]; vals = vals[vals > 0]
            pais = nombres_pais[int(np.bincount(vals).argmax())] if vals.size else None
        # clasificación
        rayo_ok = ry or 0
        hay_glm = ry is not None
        grande = area >= 25000
        alargada = razon >= 3.0 and largo >= 350
        if grande or alargada:
            if rayo_ok >= 5:
                tipo = "frontal_conv" if alargada else "sistema"
            elif not hay_glm and tmin <= -60 and desv >= 4:
                tipo = "frontal_conv"
            else:
                tipo = "frontal"
        elif rayo_ok >= 2:
            tipo = "cb"
        elif not hay_glm and tmin <= -55 and desv >= 3.5:
            tipo = "cb"
        elif tmin <= -50 and desv >= 3.0:
            tipo = "moderada"
        else:
            tipo = "alta"
        # contorno
        sub = np.pad(m.astype("f4"), 1)
        polis = []
        for cnt in sorted(find_contours(sub, 0.5), key=len, reverse=True)[:3]:
            if len(cnt) < 6: continue
            cnt = approximate_polygon(cnt, tolerance=0.8)
            fil = cnt[:, 0] - 1 + sl[0].start; col = cnt[:, 1] - 1 + sl[1].start
            polis.append([[round(float(a), 3), round(float(b), 3)] for a, b in zip(fila_a_lat(fil, alto), col_a_lon(col))])
        celdas.append({
            "tipo": tipo, "lejano": region == "lejano", "region": region,
            "lugar": lugar(lat_n, lon_n, tierra_n, chile_n and region == "chile", pais),
            "lat": round(lat_n, 3), "lon": round(lon_n, 3),
            "tmin": round(tmin, 1), "tmed": round(tmed, 1), "desv": round(desv, 1),
            "area": int(area), "nucleo": int(nucleo), "largo": int(largo), "razon": round(razon, 1),
            "rayos": ry, "contorno": polis,
        })
    return celdas


# ---------------- Convección somera y rayos sin nube fría ----------------
def a_pixel(la, lo, alto):
    col = np.clip(((np.asarray(lo) - LON_MIN) / (LON_MAX - LON_MIN) * ANCHO).astype(int), 0, ANCHO - 1)
    ym0, ym1 = merc_y(LAT_MIN), merc_y(LAT_MAX)
    fil = np.clip(((ym1 - merc_y(np.asarray(la))) / (ym1 - ym0) * alto).astype(int), 0, alto - 1)
    return fil, col

def contorno_de(m, sl, alto):
    sub = np.pad(m.astype("f4"), 1); polis = []
    for cnt in sorted(find_contours(sub, 0.5), key=len, reverse=True)[:2]:
        if len(cnt) < 4: continue
        cnt = approximate_polygon(cnt, tolerance=0.6)
        fil = cnt[:, 0] - 1 + sl[0].start; col = cnt[:, 1] - 1 + sl[1].start
        polis.append([[round(float(a), 3), round(float(b), 3)] for a, b in zip(fila_a_lat(fil, alto), col_a_lon(col))])
    return polis

def detectar_somera(T, LON, LAT, px_km, m_chile, m_tierra, zona, rayos):
    """Nubes compactas con tope entre −10 y −40 °C sobre Chile (candidatas), y grupos de rayos sin nube detectada."""
    alto = T.shape[0]
    area_px = (px_km ** 2)[:, None] * np.ones_like(T)
    tibia = np.isfinite(T) & (T <= UMBRAL_SOMERO)
    etq, n = ndimage.label(tibia, structure=np.ones((3, 3)))
    conteo = np.zeros(n + 1, int)
    if rayos is not None and len(rayos[0]):
        fil, col = a_pixel(rayos[0], rayos[1], alto)
        np.add.at(conteo, ndimage.maximum_filter(etq, size=5)[fil, col], 1)
    res = []
    for i, sl in enumerate(ndimage.find_objects(etq), start=1):
        if sl is None: continue
        m = etq[sl] == i
        Ti = T[sl][m]
        if np.nanmin(Ti) <= UMBRAL: continue                 # ya la cubre la detección principal
        if not zona[sl][m].any(): continue                     # solo Chile y Pacífico cercano
        area = float(area_px[sl][m].sum())
        if area < 30 or area > 15000: continue
        yy, xx = np.nonzero(m); la = LAT[sl][m]; lo = LON[sl][m]
        if yy.size > 3:
            X = np.stack([np.radians(lo) * R_TIERRA * math.cos(math.radians(float(la.mean()))), np.radians(la) * R_TIERRA])
            ev = np.maximum(np.sort(np.linalg.eigvalsh(np.cov(X)))[::-1], 1.0)
            if math.sqrt(ev[0] / ev[1]) >= 3.0: continue      # alargada: no es un cúmulo
        frio = Ti <= np.nanpercentile(Ti, 15)
        lat_n, lon_n = float(la[frio].mean()), float(lo[frio].mean())
        res.append({"tipo": "somera", "lejano": False, "region": "chile",
                    "lugar": lugar(lat_n, lon_n, bool(m_tierra[sl][m].mean() > .5), True),
                    "lat": round(lat_n, 3), "lon": round(lon_n, 3),
                    "tmin": round(float(np.nanmin(Ti)), 1), "tmed": round(float(np.nanmean(Ti)), 1), "desv": round(float(np.nanstd(Ti)), 1),
                    "area": int(area), "nucleo": 0, "largo": 0, "razon": 1.0,
                    "rayos": int(conteo[i]) if rayos is not None else None, "contorno": contorno_de(m, sl, alto)})
    # rayos sobre Chile que no caen en ninguna nube de −10 °C o menos
    if rayos is not None and len(rayos[0]):
        fil, col = a_pixel(rayos[0], rayos[1], alto)
        sueltos = zona[fil, col] & (ndimage.maximum_filter(etq, size=5)[fil, col] == 0)
        if sueltos.any():
            la, lo = rayos[0][sueltos], rayos[1][sueltos]
            grupos = {}
            for x_, y_ in zip(la, lo):
                grupos.setdefault((round(x_ * 4) / 4, round(y_ * 4) / 4), []).append((x_, y_))
            for pts in grupos.values():
                if len(pts) < 2: continue
                plat = float(np.mean([p[0] for p in pts])); plon = float(np.mean([p[1] for p in pts]))
                r = 15 / 111.0
                circ = [[round(plat + r * math.sin(a), 3), round(plon + r * math.cos(a) / math.cos(math.radians(plat)), 3)]
                        for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)]
                ff, cc = a_pixel(plat, plon, alto)
                res.append({"tipo": "tormenta", "lejano": False, "region": "chile", "lugar": lugar(plat, plon, bool(m_tierra[ff, cc]), True),
                            "lat": round(plat, 3), "lon": round(plon, 3), "tmin": None, "tmed": None, "desv": None,
                            "area": 0, "nucleo": 0, "largo": 0, "razon": 1.0, "rayos": len(pts), "contorno": [circ], "sin_nube": True})
    return res

def clasificar_somera(celdas):
    """Después del seguimiento: decide si cada candidata es tormenta, cúmulo en desarrollo o se oculta."""
    for c in celdas:
        if c["tipo"] != "somera": continue
        tr = [p for p in (c.get("traza") or []) if len(p) >= 4 and p[3] is not None]
        enfr = (len(tr) >= 3 and (tr[-1][3] - tr[-2][3]) <= ENFRIA and (tr[-2][3] - tr[-3][3]) <= ENFRIA
                and (tr[-1][3] - tr[-3][3]) <= ENFRIA_TOTAL)
        if (c.get("rayos") or 0) >= 1:
            c["tipo"] = "tormenta"
        elif enfr:
            c["tipo"] = "desarrollo"
        else:
            c["oculta"] = True; c["contorno"] = []

# ---------------- Seguimiento entre imágenes ----------------
def dkm(a, b):
    p = math.radians
    return 2 * R_TIERRA * math.asin(math.sqrt(math.sin(p(b[0] - a[0]) / 2) ** 2 + math.cos(p(a[0])) * math.cos(p(b[0])) * math.sin(p(b[1] - a[1]) / 2) ** 2))

def seguir(celdas, previas, dt_min, sig_id, t_iso):
    usadas = set()
    for c in sorted(celdas, key=lambda c: c["tmin"] if c["tmin"] is not None else 99):
        fam = lambda x: "frontal" if x["tipo"].startswith("frontal") else "conv"
        cand = [(dkm((c["lat"], c["lon"]), (p["lat"], p["lon"])), j, p) for j, p in enumerate(previas)
                if j not in usadas and fam(p) == fam(c) and p.get("tmin") is not None and c.get("tmin") is not None]
        lim = (180.0 if c["tipo"].startswith("frontal") else 70.0) * max(dt_min, 10) / 10.0
        cand = [x for x in cand if x[0] <= lim]
        if c.get("tmin") is None:                             # grupo de rayos sin nube: no se sigue
            c["id"] = sig_id; sig_id += 1; c["tendencia"] = "nueva"; c["dT10"] = None; c["dA10"] = None
            c["traza"] = [[c["lat"], c["lon"], t_iso, None, 0, c["rayos"]]]
            continue
        if cand and dt_min > 0:
            _, j, p = min(cand, key=lambda x: x[0]); usadas.add(j)
            c["id"] = p["id"]
            dT = (c["tmin"] - p["tmin"]) * 10.0 / dt_min
            dA = (c["area"] - p["area"]) / max(p["area"], 1) * 10.0 / dt_min
            c["dT10"] = round(dT, 1); c["dA10"] = int(round(dA * 100))
            c["tendencia"] = "creciendo" if (dT <= -2 or dA >= 0.25) else "disipando" if (dT >= 2 or dA <= -0.25) else "estable"
            c["traza"] = (p.get("traza") or [])[-11:] + [[c["lat"], c["lon"], t_iso, c["tmin"], c["area"], c["rayos"]]]
        else:
            c["id"] = sig_id; sig_id += 1
            c["tendencia"] = "nueva"; c["dT10"] = None; c["dA10"] = None
            c["traza"] = [[c["lat"], c["lon"], t_iso, c["tmin"], c["area"], c["rayos"]]]
    return sig_id


# ---------------- Datos de demostración ----------------
def demo(LON, LAT, m_tierra, k):
    """Escena sintética k (0..5): frente en el sur, CB que crece sobre Rancagua, etc."""
    rng = np.random.default_rng(10 + k)
    T = np.where(m_tierra, 22.0, 13.0) - (np.abs(LAT) - 20) * 0.45
    T = T + ndimage.gaussian_filter(rng.normal(0, 6, LON.shape), 3)
    def gota(la, lo, r, t, alarg=1.0, ang=0.0):
        a = math.radians(ang); dx = (LON - lo) * math.cos(math.radians(la)); dy = LAT - la
        u = dx * math.cos(a) + dy * math.sin(a); v = -dx * math.sin(a) + dy * math.cos(a)
        return np.exp(-((u / (r * alarg)) ** 2 + (v / r) ** 2)), t
    ruido = ndimage.gaussian_filter(rng.normal(0, 1, LON.shape), 2)
    capas = [
        gota(-44 + 0.15 * k, -80 + 0.5 * k, 1.1, -50, 7, 35),          # banda frontal Pacífico sur
        gota(-34.2, -70.9 + 0.12 * k, 0.28 + 0.03 * k, -48 - 3.5 * k),  # CB que crece cerca de Rancagua
        gota(-30.3, -70.6, 0.35, -44),                                    # convección moderada en el norte chico
        gota(-27, -63, 1.2, -66),                                         # convección en Argentina (se ignora)
        gota(-31, -86, 0.9, -45, 2.5, -20),                               # sistema en el Pacífico lejano
    ]
    for (w, t) in capas:
        T = T + (t + 4 * ruido - T) * np.clip(w * 1.6, 0, 1) * (w > 0.12)
    n = rng.poisson(12 + 9 * k); m = 120
    rayos = (np.concatenate([rng.normal(-34.2, 0.15, n), rng.normal(-27, 0.7, m)]),
             np.concatenate([rng.normal(-70.9 + 0.12 * k, 0.18, n), rng.normal(-63, 0.8, m)]))
    return T.astype("f4"), rayos


# ---------------- Programa principal ----------------
def cargar_previo(dir_datos, sitio_url):
    idx = os.path.join(dir_datos, "index.json")
    if os.path.exists(idx):
        return json.load(open(idx))
    if not sitio_url:
        return None
    try:
        base = sitio_url.rstrip("/") + "/data/"
        r = requests.get(base + "index.json", headers=UA, timeout=30); r.raise_for_status()
        prev = json.loads(r.content)
        os.makedirs(os.path.join(dir_datos, "frames"), exist_ok=True)
        for f in prev.get("frames", []):
            rr = requests.get(base + f["img"], headers=UA, timeout=60)
            if rr.ok:
                open(os.path.join(dir_datos, f["img"]), "wb").write(rr.content)
        print(f"  loop anterior recuperado: {len(prev.get('frames', []))} imágenes")
        return prev
    except Exception as ex:
        print("  sin loop anterior (primera ejecución o sitio no disponible):", ex)
        return None

def procesar_escena(T, rayos, t, prev, LON, LAT, px_km, mascs, dir_datos):
    m_chile, m_tierra, zona, m_pais, nombres = mascs
    celdas = detectar(T, LON, LAT, px_km, m_chile, m_tierra, zona, rayos, m_pais, nombres)
    celdas += detectar_somera(T, LON, LAT, px_km, m_chile, m_tierra, zona, rayos)
    lim = [[LAT_MIN, LON_MIN], [LAT_MAX, LON_MAX]]
    frames = [f for f in (prev or {}).get("frames", [])
              if f["t"] != t.strftime("%Y-%m-%dT%H:%MZ") and f.get("lim") == lim]   # descarta imágenes con otra cobertura
    previas = frames[-1]["celdas"] if frames else []
    dt = (t - datetime.strptime(frames[-1]["t"], "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)).total_seconds() / 60 if frames else 0
    if dt > 40: previas = []                                          # hueco grande: no seguir
    sig = seguir(celdas, previas, dt, (prev or {}).get("sig_id", 1), t.strftime("%H:%M"))
    clasificar_somera(celdas)
    nombre = f"frames/ir_{t:%Y%m%d%H%M}.png"
    os.makedirs(os.path.join(dir_datos, "frames"), exist_ok=True)
    img = Image.fromarray(colorear(T), "RGBA").quantize(colors=200, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)
    img.save(os.path.join(dir_datos, nombre), optimize=True)
    rp = []
    if rayos is not None and len(rayos[0]):
        idx = np.arange(len(rayos[0]))
        if idx.size > 2500:
            idx = np.random.default_rng(0).choice(idx, 2500, replace=False)
        rp = [[round(float(rayos[0][i]), 2), round(float(rayos[1][i]), 2)] for i in idx]
    frames.append({"t": t.strftime("%Y-%m-%dT%H:%MZ"), "img": nombre, "rayos": rayos is not None, "lim": lim,
                   "rp": rp, "celdas": celdas})
    frames = frames[-FRAMES:]
    vigentes = {f["img"] for f in frames}
    for arch in os.listdir(os.path.join(dir_datos, "frames")):
        if "frames/" + arch not in vigentes:
            os.remove(os.path.join(dir_datos, "frames", arch))
    return {"generado": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
            "limites": [[LAT_MIN, LON_MIN], [LAT_MAX, LON_MAX]],
            "umbral": UMBRAL, "sig_id": sig, "frames": frames,
            "lugares": [[n, la, lo, "CL"] for n, la, lo in LUGARES] + [[n, la, lo, "X"] for n, la, lo in VECINOS]}

# ---------------- Análisis de horas específicas ----------------
def analizar(horas, LON, LAT, px_km, mascs, dir_out, punto=("Santiago", -33.45, -70.67), radio_km=60.0):
    """Procesa imágenes de horas puntuales (UTC) y deja un informe y recortes de la zona central."""
    from PIL import ImageDraw
    os.makedirs(dir_out, exist_ok=True)
    alto = LON.shape[0]
    informe = []
    previas, prev_t = [], None
    for h in horas:
        t0 = datetime.strptime(h, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
        lista = archivos("ABI-L2-CMIPF", "M6C13_G19", t0 - timedelta(minutes=6), t0 + timedelta(minutes=6))
        if not lista:
            informe.append(f"\n### {h} UTC\nNo se encontró imagen."); continue
        t, clave = min(lista, key=lambda x: abs((x[0] - t0).total_seconds()))
        t = t.replace(second=0, microsecond=0)
        T = banda13(bajar(clave), LON, LAT)
        rayos = rayos_glm(t)
        m_chile, m_tierra, zona, m_pais, nombres = mascs
        celdas = detectar(T, LON, LAT, px_km, m_chile, m_tierra, zona, rayos, m_pais, nombres)
        celdas += detectar_somera(T, LON, LAT, px_km, m_chile, m_tierra, zona, rayos)
        dt = (t - prev_t).total_seconds() / 60 if prev_t else 0
        seguir(celdas, previas, dt, 1000, t.strftime("%H:%M")); clasificar_somera(celdas); previas, prev_t = celdas, t
        # diagnóstico alrededor del punto
        nom, plat, plon = punto
        d = 2 * R_TIERRA * np.arcsin(np.sqrt(np.sin(np.radians(LAT - plat) / 2) ** 2 + np.cos(np.radians(plat)) * np.cos(np.radians(LAT)) * np.sin(np.radians(LON - plon) / 2) ** 2))
        cerca = d <= radio_km
        Tc = T[cerca]
        n_ray = 0
        if rayos is not None and len(rayos[0]):
            la, lo = rayos
            dd = 2 * R_TIERRA * np.arcsin(np.sqrt(np.sin(np.radians(la - plat) / 2) ** 2 + np.cos(np.radians(plat)) * np.cos(np.radians(la)) * np.sin(np.radians(lo - plon) / 2) ** 2))
            n_ray = int((dd <= radio_km).sum())
        hl = t.astimezone(timezone(timedelta(hours=-3))).strftime("%H:%M")
        r = [f"\n### {t:%H:%M} UTC ({hl} Chile) — {clave.split('/')[-1][:40]}…",
             f"En {radio_km:.0f} km alrededor de {nom}: tope mínimo {np.nanmin(Tc):.1f} °C · "
             f"percentil 10: {np.nanpercentile(Tc,10):.1f} °C · mediana {np.nanmedian(Tc):.1f} °C · "
             f"píxeles ≤ −40 °C: {100*np.mean(Tc<=-40):.0f} % · ≤ −20 °C: {100*np.mean(Tc<=-20):.0f} % · "
             f"rayos GLM: {n_ray if rayos is not None else 's/d'}"]
        cz = [c for c in celdas if c["region"] == "chile" and -38.5 <= c["lat"] <= -29.5 and not c.get("oculta")]
        if cz:
            r.append("Objetos detectados en la zona central (−29,5° a −38,5°):")
            for c in sorted(cz, key=lambda c: c["tmin"] if c["tmin"] is not None else 99):
                r.append(f"- {c['tipo']} · {c['lugar']} · tope mín {c['tmin']} °C · medio {c['tmed']} °C · desv {c['desv']} · "
                         f"{c['area']} km² · rayos {c['rayos']} · {c.get('tendencia')} ({c.get('dT10')} °C/10 min)")
        else:
            r.append("Ningún objeto marcado en la zona central.")
        for c in [c for c in celdas if c.get("oculta") and abs(c["lat"] - plat) < 1 and abs(c["lon"] - plon) < 1.2]:
            tr = [p[3] for p in c.get("traza", []) if len(p) >= 4]
            r.append(f"  (candidata no marcada · {c['lugar']} · tope {c['tmin']} °C · {c['area']} km² · historia {tr})")
        informe += r
        # recorte de la zona central con contornos y rayos
        ym0, ym1 = merc_y(LAT_MIN), merc_y(LAT_MAX)
        fila = lambda la_: int((ym1 - merc_y(la_)) / (ym1 - ym0) * alto)
        col = lambda lo_: int((lo_ - LON_MIN) / (LON_MAX - LON_MIN) * ANCHO)
        f0, f1, c0, c1 = fila(-29.5), fila(-38.0), col(-75.0), col(-68.5)
        img = Image.fromarray(colorear(T)[f0:f1, c0:c1]).convert("RGB")
        k = 5; img = img.resize((img.width * k, img.height * k), Image.NEAREST)
        dr = ImageDraw.Draw(img)
        X = lambda lo_: (col(lo_) - c0) * k; Y = lambda la_: (fila(la_) - f0) * k
        for cel in cz:
            colr = {"cb": (255, 90, 95), "sistema": (255, 90, 95), "frontal_conv": (255, 159, 67), "moderada": (242, 180, 65),
                    "frontal": (92, 200, 230), "alta": (169, 184, 255), "tormenta": (255, 90, 95), "desarrollo": (255, 209, 102)}[cel["tipo"]]
            for anillo in cel["contorno"]:
                dr.line([(X(p[1]), Y(p[0])) for p in anillo] + [(X(anillo[0][1]), Y(anillo[0][0]))], fill=colr, width=3)
        if rayos is not None:
            for la_, lo_ in zip(*rayos):
                if -38 <= la_ <= -29.5 and -75 <= lo_ <= -68.5:
                    dr.ellipse([X(lo_) - 4, Y(la_) - 4, X(lo_) + 4, Y(la_) + 4], outline=(200, 235, 255), width=2)
        for n_, la_, lo_ in LUGARES:
            if -38 <= la_ <= -29.5 and -75 <= lo_ <= -68.5:
                dr.ellipse([X(lo_) - 3, Y(la_) - 3, X(lo_) + 3, Y(la_) + 3], fill=(255, 255, 255))
                dr.text((X(lo_) + 6, Y(la_) - 6), n_, fill=(255, 255, 255))
        rr = radio_km / 111.0
        dr.ellipse([X(plon - rr / math.cos(math.radians(plat))), Y(plat + rr), X(plon + rr / math.cos(math.radians(plat))), Y(plat - rr)], outline=(255, 255, 0), width=2)
        dr.text((10, 10), f"GOES-19 banda 13 · {t:%Y-%m-%d %H:%M} UTC ({hl} Chile)", fill=(255, 255, 255))
        img.save(os.path.join(dir_out, f"centro_{t:%H%M}.png"))
    texto = "\n".join(informe)
    open(os.path.join(dir_out, "informe.md"), "w").write(texto)
    print(texto)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--analisis", nargs="+", metavar="AAAA-MM-DDTHH:MM", help="horas UTC a analizar")
    ap.add_argument("--sitio-url", default=os.environ.get("SITIO_URL", ""))
    ap.add_argument("--salida", default=os.path.join(BASE, "site", "data"))
    a = ap.parse_args()
    os.makedirs(a.salida, exist_ok=True)
    LON, LAT, px_km, alto = grilla()
    mascs = mascaras(LON, LAT, px_km)
    print(f"Grilla {ANCHO}x{alto}")

    if a.analisis:
        analizar(a.analisis, LON, LAT, px_km, mascs, os.path.join(BASE, "analisis"))
        return

    if a.demo:
        prev = None
        for k in range(6):
            ahora = datetime.now(timezone.utc).replace(second=0, microsecond=0)
            t = ahora - timedelta(minutes=ahora.minute % 10 + 10 * (5 - k))
            T, rayos = demo(LON, LAT, mascs[1], k)
            prev = procesar_escena(T, rayos, t, prev, LON, LAT, px_km, mascs, a.salida)
        prev["demo"] = True
    else:
        prev = cargar_previo(a.salida, a.sitio_url)
        sat = SE.cargar(a.salida, a.sitio_url, UA)
        ahora = datetime.now(timezone.utc)
        lista = archivos("ABI-L2-CMIPF", "M6C13_G19", ahora - timedelta(hours=2), ahora)
        if not lista:
            sys.exit("No se encontraron imágenes recientes del GOES-19 (banda 13).")
        t, clave = lista[-1]
        t = t.replace(second=0, microsecond=0)
        if prev and prev.get("frames") and prev["frames"][-1]["t"] == t.strftime("%Y-%m-%dT%H:%MZ"):
            print("Sin imagen nueva; se mantiene la publicación anterior.")
            SE.guardar(a.salida, sat)
            json.dump(prev, open(os.path.join(a.salida, "index.json"), "w"), ensure_ascii=False, separators=(",", ":"))
            return
        print("Imagen:", clave)
        c13 = bajar(clave)
        T = banda13(c13, LON, LAT)
        rayos = rayos_glm(t)
        prev = procesar_escena(T, rayos, t, prev, LON, LAT, px_km, mascs, a.salida)
        try:                                   # resumen por estación para el control de calidad (no detiene el visor si falla)
            sat = SE.actualizar(sat, t, c13, rayos, BASE, Geo, archivos, bajar)
        except Exception as ex:
            print("  aviso: resumen satelital falló:", ex)
        SE.guardar(a.salida, sat)

    json.dump(prev, open(os.path.join(a.salida, "index.json"), "w"), ensure_ascii=False, separators=(",", ":"))
    ult = prev["frames"][-1]["celdas"]
    print(f"Listo: {len(ult)} objetos ({sum(c['tipo']=='cb' for c in ult)} CB)")

if __name__ == "__main__":
    main()

"""
Resumen satelital por estación (para el control de calidad de metotec.cl).

En cada imagen nueva del GOES-19 calcula, alrededor de cada estación de la DMC
(estaciones.json), lo que ve el satélite en radios de 10 y 30 km:

  t13      temperatura de brillo banda 13 en la estación (°C)
  t13n10   mínima banda 13 a ≤ 10 km (°C)                 → topes fríos encima
  t13sd30  desviación estándar banda 13 a ≤ 30 km (°C)    → textura (cumuliforme alta, estratiforme baja)
  frio30   % de píxeles < −40 °C a ≤ 30 km
  nub10    % nublado (máscara de nubes ACM) a ≤ 10 km
  nub30    % nublado a ≤ 30 km
  rr10     máxima tasa de lluvia RRQPE a ≤ 10 km (mm/h)
  rrp10    % de píxeles con lluvia (≥ 0,5 mm/h) a ≤ 10 km
  ht10     altura mediana del tope (ACHA) a ≤ 10 km (km)
  ray10    destellos GLM a ≤ 10 km en los 10 minutos
  ray30    destellos GLM a ≤ 30 km en los 10 minutos

Publica data/sat_estaciones.json con las últimas 12 horas (una fila por imagen).
"""

import json, os
from datetime import datetime, timedelta, timezone

import numpy as np
import requests

CAMPOS = ["t13", "t13n10", "t13sd30", "frio30", "nub10", "nub30", "rr10", "rrp10", "ht10", "ray10", "ray30"]
HORAS = 12
PASO_KM = 2.0
RADIO = 30.0
LAT_MIN_VISIBLE = -70.0          # más al sur el GOES-19 ve demasiado rasante: no se usa
ARCHIVO = "sat_estaciones.json"
PRODUCTOS = {   # clave: (producto en AWS, patrón del nombre, variable)
    "acm": ("ABI-L2-ACMF", "ACMF-M6_G19", "BCM"),
    "rr": ("ABI-L2-RRQPEF", "RRQPEF-M6_G19", "RRQPE"),
    "acha": ("ABI-L2-ACHAF", "ACHAF-M6_G19", "HT"),
}


def estaciones(base):
    E = json.load(open(os.path.join(base, "estaciones.json"), encoding="utf-8"))
    return [e for e in E if e[2] >= LAT_MIN_VISIBLE]


def puntos(E):
    """Puntos de muestreo cada 2 km dentro de 30 km de cada estación."""
    k = np.arange(-RADIO, RADIO + 0.1, PASO_KM)
    dx, dy = np.meshgrid(k, k)
    d = np.hypot(dx, dy)
    m = d <= RADIO
    dx, dy, d = dx[m], dy[m], d[m]
    la, lo, di, est = [], [], [], []
    for i, (_, _, lat, lon) in enumerate(E):
        la.append(lat + dy / 111.2)
        lo.append(lon + dx / (111.2 * np.cos(np.radians(lat))))
        di.append(d)
        est.append(np.full(d.size, i))
    return np.concatenate(la), np.concatenate(lo), np.concatenate(di), np.concatenate(est)


def muestrear(contenido, var, la, lo, Geo):
    """Valor del píxel más cercano de una variable de disco completo en cada punto."""
    import netCDF4
    ds = netCDF4.Dataset("p.nc", memory=contenido)
    pr = ds.variables["goes_imager_projection"]
    geo = Geo(pr.perspective_point_height + pr.semi_major_axis, pr.semi_major_axis, pr.semi_minor_axis, pr.longitude_of_projection_origin)
    xs = ds.variables["x"][:].astype("f8"); ys = ds.variables["y"][:].astype("f8")
    sx, sy = geo.scan(la, lo)
    ix = np.rint((sx - xs[0]) / (xs[1] - xs[0])); iy = np.rint((sy - ys[0]) / (ys[1] - ys[0]))
    ok = np.isfinite(ix) & np.isfinite(iy) & (ix >= 0) & (ix < xs.size) & (iy >= 0) & (iy < ys.size)
    out = np.full(la.size, np.nan, "f4")
    if ok.any():
        i0, i1 = int(ix[ok].min()), int(ix[ok].max()) + 1
        j0, j1 = int(iy[ok].min()), int(iy[ok].max()) + 1
        w = np.ma.filled(ds.variables[var][j0:j1, i0:i1].astype("f4"), np.nan)
        out[ok] = w[(iy[ok] - j0).astype(int), (ix[ok] - i0).astype(int)]
    ds.close()
    return out


def _r(x, n=1):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def _pct(m, valido):
    return _r(100.0 * m.sum() / valido.sum(), 0) if valido.sum() >= 5 else None


def resumir(E, d, est, V, rayos):
    """V: dict con arrays por punto (t13, acm, rr, acha) o None si el producto no estuvo."""
    fila = {}
    if rayos is not None and len(rayos[0]):
        rla, rlo = np.radians(rayos[0]), np.radians(rayos[1])
    for i, (cod, _, lat, lon) in enumerate(E):
        s = est == i
        di = d[s]
        c10 = di <= 10
        g = lambda k: None if V.get(k) is None else V[k][s]
        t, acm, rr, ht = g("t13"), g("acm"), g("rr"), g("acha")
        v = dict.fromkeys(CAMPOS)
        if t is not None and np.isfinite(t).sum() >= 5:
            v["t13"] = _r(t[np.argmin(di)])
            v["t13n10"] = _r(np.nanmin(t[c10])) if np.isfinite(t[c10]).any() else None
            v["t13sd30"] = _r(np.nanstd(t))
            v["frio30"] = _pct(t < -40, np.isfinite(t))
        if acm is not None:
            val = np.isfinite(acm)
            v["nub10"] = _pct((acm == 1) & c10, val & c10)
            v["nub30"] = _pct(acm == 1, val)
        if rr is not None:
            val = np.isfinite(rr) & c10
            if val.sum() >= 5:
                v["rr10"] = _r(np.nanmax(rr[val]))
                v["rrp10"] = _pct((rr >= 0.5) & c10, val)
        if ht is not None:
            h = ht[c10 & np.isfinite(ht)]
            v["ht10"] = _r(np.median(h) / 1000, 1) if h.size >= 3 else None
        if rayos is not None:
            if len(rayos[0]):
                a, b = np.radians(lat), np.radians(lon)
                dd = 6371 * np.arccos(np.clip(np.sin(a) * np.sin(rla) + np.cos(a) * np.cos(rla) * np.cos(rlo - b), -1, 1))
                v["ray10"], v["ray30"] = int((dd <= 10).sum()), int((dd <= 30).sum())
            else:
                v["ray10"] = v["ray30"] = 0
        fila[cod] = [v[k] for k in CAMPOS]
    return fila


def cargar(dir_datos, sitio_url, UA):
    p = os.path.join(dir_datos, ARCHIVO)
    if os.path.exists(p):
        return json.load(open(p))
    if sitio_url:
        try:
            r = requests.get(sitio_url.rstrip("/") + "/data/" + ARCHIVO, headers=UA, timeout=30)
            if r.ok:
                return r.json()
        except Exception as ex:
            print("  sin resumen satelital anterior:", ex)
    return None


def guardar(dir_datos, js):
    if js:
        json.dump(js, open(os.path.join(dir_datos, ARCHIVO), "w"), ensure_ascii=False, separators=(",", ":"))


def producto_reciente(clave, t, archivos, bajar):
    """Archivo L2 más reciente con inicio ≤ t (hasta 40 min antes). Devuelve (momento, bytes) o (None, None)."""
    prod, patron, _ = PRODUCTOS[clave]
    try:
        lista = archivos(prod, patron, t - timedelta(minutes=40), t + timedelta(minutes=1))
        if lista:
            tt, k = lista[-1]
            return tt, bajar(k)
        print(f"  aviso: {clave} sin archivos entre {t - timedelta(minutes=40):%H:%M} y {t:%H:%M} ({prod}/*{patron}*)")
    except Exception as ex:
        print(f"  aviso: sin {clave}:", ex)
    return None, None


def actualizar(prev, t, c13, rayos, base, Geo, archivos, bajar):
    """Agrega la fila de la imagen t al historial y recorta a 12 h."""
    E = estaciones(base)
    la, lo, d, est = puntos(E)
    V, tp = {}, {}
    V["t13"] = muestrear(c13, "CMI", la, lo, Geo) - 273.15
    for clave in PRODUCTOS:
        tt, cont = producto_reciente(clave, t, archivos, bajar)
        if cont is None:
            continue
        try:
            V[clave] = muestrear(cont, PRODUCTOS[clave][2], la, lo, Geo)
            tp[clave] = tt.strftime("%Y-%m-%dT%H:%MZ")
        except Exception as ex:
            print(f"  aviso: no se pudo leer {clave}:", ex)
    fila = resumir(E, d, est, V, rayos)
    ts = t.strftime("%Y-%m-%dT%H:%MZ")
    js = prev if prev and prev.get("campos") == CAMPOS else {"campos": CAMPOS, "frames": []}
    limite = (t - timedelta(hours=HORAS)).strftime("%Y-%m-%dT%H:%MZ")
    js["frames"] = [f for f in js["frames"] if f["t"] != ts and f["t"] > limite] + [{"t": ts, "prod": tp, "glm": rayos is not None, "d": fila}]
    js["generado"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    js["estaciones"] = {e[0]: [e[1], e[2], e[3]] for e in E}
    print(f"  resumen satelital: {len(fila)} estaciones, productos {sorted(tp)}")
    return js

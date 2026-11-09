"""
Niebla y nubes bajas de noche (producto propio de METOTEC).

De noche, las gotitas de agua de la niebla y los estratos emiten menos en 3,9 µm (banda 7) que en 10,3 µm (banda 13):
la diferencia T13 − T7 es positiva (2 a 6 K o más) sobre niebla/estratos y cercana a 0 sobre suelo despejado.
Es la base de la «microfísica nocturna» y del producto de niebla de la NOAA. Sirve para ver la camanchaca y la niebla
de los valles, que el infrarrojo común (banda 13) no distingue del suelo.

Publica site/niebla/AAAAMMDDHHMM.png (últimas 3 horas) y site/niebla/lista.json. De día queda transparente.
"""
import json, math, os
from datetime import timedelta

import numpy as np
import requests
from PIL import Image

LAT_S, LAT_N, LON_O, LON_E = -56.0, -17.0, -80.0, -66.0
CUADROS = 18


def grilla(ancho=520):
    my = lambda la: np.log(np.tan(np.pi / 4 + np.radians(la) / 2))
    alto = int(round(ancho * (my(LAT_N) - my(LAT_S)) / np.radians(LON_E - LON_O)))
    lon = np.linspace(LON_O, LON_E, ancho)
    lat = np.degrees(2 * np.arctan(np.exp(np.linspace(my(LAT_N), my(LAT_S), alto))) - np.pi / 2)
    return np.meshgrid(lon, lat)


def muestrear(contenido, LO, LA, Geo):
    import netCDF4
    ds = netCDF4.Dataset("x.nc", memory=contenido)
    pr = ds.variables["goes_imager_projection"]
    geo = Geo(pr.perspective_point_height + pr.semi_major_axis, pr.semi_major_axis, pr.semi_minor_axis, pr.longitude_of_projection_origin)
    sx, sy = geo.scan(LA.ravel(), LO.ravel())
    xs = ds.variables["x"][:].astype("f8"); ys = ds.variables["y"][:].astype("f8")
    ix = np.rint((sx - xs[0]) / (xs[1] - xs[0])); iy = np.rint((sy - ys[0]) / (ys[1] - ys[0]))
    ok = np.isfinite(ix) & np.isfinite(iy) & (ix >= 0) & (ix < xs.size) & (iy >= 0) & (iy < ys.size)
    i0, i1, j0, j1 = int(ix[ok].min()), int(ix[ok].max()) + 1, int(iy[ok].min()), int(iy[ok].max()) + 1
    W = np.ma.filled(ds.variables["CMI"][j0:j1, i0:i1].astype("f4"), np.nan)
    v = np.full(LA.size, np.nan, "f4")
    v[ok] = W[(iy[ok] - j0).astype(int), (ix[ok] - i0).astype(int)]
    ds.close()
    return v.reshape(LA.shape)


def elevacion_sol(LA, LO, t):
    doy = t.timetuple().tm_yday
    g = 2 * math.pi / 365 * (doy - 1 + (t.hour - 12) / 24)
    decl = 0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g) - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    hs = t.hour + t.minute / 60 + (eqt + 4 * LO) / 60
    ha = np.radians(15 * (hs - 12)); la = np.radians(LA)
    return np.degrees(np.arcsin(np.sin(la) * math.sin(decl) + np.cos(la) * math.cos(decl) * np.cos(ha)))


def imagen(t13, t07, sol):
    d = t13 - t07
    img = np.zeros(t13.shape + (4,), "u1")
    noche = np.clip((-sol - 3) / 7, 0, 1)                       # transparente con sol > −3°, completo bajo −10°
    baja = np.isfinite(d) & (d >= 1.5) & (t13 > 255)              # gotitas de agua y tope no muy frío (no cirros)
    k = np.clip((d - 1.5) / 4.5, 0, 1)
    img[..., 0] = (150 + 105 * k).astype("u1"); img[..., 1] = (215 + 40 * k).astype("u1"); img[..., 2] = 255
    img[..., 3] = np.where(baja, (110 + 140 * k) * noche, 0).astype("u1")
    return img


def actualizar(t, c13, salida_site, sitio_url, Geo):
    import incendios as IN
    d = os.path.join(salida_site, "niebla"); os.makedirs(d, exist_ok=True)
    lista = []
    if sitio_url:
        try:
            r = requests.get(sitio_url.rstrip("/") + "/niebla/lista.json", timeout=30)
            if r.ok:
                lista = r.json().get("cuadros", [])
        except Exception:
            pass
    if IN.ULTIMO_C07 is not None and abs((IN.ULTIMO_C07[0] - t).total_seconds()) < 900:
        LO, LA = grilla()
        t13 = muestrear(c13, LO, LA, Geo); t07 = muestrear(IN.ULTIMO_C07[1], LO, LA, Geo)
        ts = t.strftime("%Y%m%d%H%M")
        Image.fromarray(imagen(t13, t07, elevacion_sol(LA, LO, t)), "RGBA").save(os.path.join(d, ts + ".png"), optimize=True)
        lista.append(ts)
    lista = sorted(set(lista))[-CUADROS:]
    quedan = []
    for ts in lista:
        p = os.path.join(d, ts + ".png")
        if not os.path.exists(p) and sitio_url:
            try:
                r = requests.get(f"{sitio_url.rstrip('/')}/niebla/{ts}.png", timeout=30)
                if r.ok:
                    open(p, "wb").write(r.content)
            except Exception:
                pass
        if os.path.exists(p):
            quedan.append(ts)
    json.dump({"cuadros": quedan, "limites": [[LAT_S, LON_O], [LAT_N, LON_E]]}, open(os.path.join(d, "lista.json"), "w"))
    print(f"  niebla: {len(quedan)} cuadros")


def mantener(salida_site, sitio_url):
    d = os.path.join(salida_site, "niebla"); os.makedirs(d, exist_ok=True)
    try:
        r = requests.get(sitio_url.rstrip("/") + "/niebla/lista.json", timeout=30)
        js = r.json() if r.ok else None
    except Exception:
        js = None
    if not js:
        return
    for ts in js.get("cuadros", []):
        try:
            r = requests.get(f"{sitio_url.rstrip('/')}/niebla/{ts}.png", timeout=30)
            if r.ok:
                open(os.path.join(d, ts + ".png"), "wb").write(r.content)
        except Exception:
            pass
    json.dump(js, open(os.path.join(d, "lista.json"), "w"))

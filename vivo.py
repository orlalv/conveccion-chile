"""
Color real propio de la zona central, casi en vivo (10 a 20 min de atraso, en vez de ~1 h de NASA GIBS).

De día: bandas 1 (azul), 2 (rojo, 0,5 km) y 3 (infrarrojo cercano) del GOES-19, con el verde sintético habitual
(0,45·rojo + 0,1·IR cercano + 0,45·azul), corrección por altura del sol y gamma. Se ve el humo, el fuego grande
y las cicatrices oscuras de lo quemado tal como son, sin colores agregados.
De noche: nubes en infrarrojo (banda 13) sobre fondo oscuro, y transición suave al amanecer y atardecer.
Cuadros JPEG cada 10 min en incendios/vivo/ (últimas 6 h).
"""
import os
from datetime import timedelta

import numpy as np
import requests
from PIL import Image

import incendios as IN
import niebla as NB

CUADROS = 36
ANCHO = 900                       # ~0,45 km por píxel en la zona: aprovecha la banda 2 de 0,5 km


def _banda(archivos, bajar, banda, t):
    c = archivos("ABI-L2-CMIPF", f"M6C{banda:02d}_G19", t - timedelta(minutes=12), t + timedelta(minutes=1))
    return bajar(c[-1][1]) if c else None


def cuadro(t, c13, Geo, archivos, bajar):
    LO, LA = IN.grilla_zona(ANCHO)
    sol = NB.elevacion_sol(LA, LO, t)
    t13 = NB.muestrear(c13, LO, LA, Geo)
    fondo = np.array([12, 20, 36], "f4")
    k = np.clip((288 - np.nan_to_num(t13, nan=300)) / (288 - 215), 0, 1)[..., None] ** .8
    noche = fondo * (1 - k) + np.array([225, 232, 245], "f4") * k
    img = noche
    if sol.max() > -4:
        bs = [_banda(archivos, bajar, b, t) for b in (1, 2, 3)]
        if all(b is not None for b in bs):
            A, R, N = (NB.muestrear(b, LO, LA, Geo) for b in bs)
            del bs
            G = .45 * R + .1 * N + .45 * A
            mu = np.clip(np.sin(np.radians(sol)), .25, 1)              # corrección por sol bajo (limitada)
            rgb = np.stack([R, G, A], -1) / mu[..., None]
            rgb = np.clip(np.nan_to_num(rgb), 0, 1) ** (1 / 2.2)
            rgb = np.clip((rgb - .03) * 1.08, 0, 1) * 255
            w = np.clip((sol + 3) / 10, 0, 1)[..., None]               # transición con el sol entre −3° y 7°
            img = rgb * w + noche * (1 - w)
        else:
            print("  vivo: faltan bandas 1–3 para", t)
    return Image.fromarray(np.clip(img, 0, 255).astype("u1"), "RGB")


def cuadros(t, c13, prev, dir_inc, sitio_url, Geo, archivos, bajar):
    d = os.path.join(dir_inc, "vivo"); os.makedirs(d, exist_ok=True)
    lista = list(prev.get("vivo", []))
    ts = t.strftime("%Y%m%d%H%M")
    if ts not in lista and c13 is not None:
        try:
            cuadro(t, c13, Geo, archivos, bajar).save(os.path.join(d, ts + ".jpg"), quality=84, optimize=True)
            lista.append(ts)
        except Exception as ex:
            print("  vivo: falló:", ex)
    return restaurar(sorted(set(lista))[-CUADROS:], d, sitio_url)


def restaurar(lista, d, sitio_url):
    os.makedirs(d, exist_ok=True)
    quedan = []
    for ts in lista:                                  # el sitio se arma de cero: se traen los publicados antes
        p = os.path.join(d, ts + ".jpg")
        if not os.path.exists(p) and sitio_url:
            try:
                r = requests.get(f"{sitio_url.rstrip('/')}/incendios/vivo/{ts}.jpg", timeout=30)
                if r.ok:
                    open(p, "wb").write(r.content)
            except Exception:
                pass
        if os.path.exists(p):
            quedan.append(ts)
    return quedan

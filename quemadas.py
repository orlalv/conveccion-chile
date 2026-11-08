"""
Áreas quemadas con Sentinel-2 (etapa 2 del producto de incendios).

1. Registro de la temporada: cada incendio detectado (GOES-19 / FIRMS) queda guardado con su zona y sus fechas.
2. Cuando un incendio lleva al menos 1 día sin detecciones, se busca una imagen Sentinel-2 anterior al fuego
   y una posterior (Earth Search, AWS, sin clave) y se calcula el índice de quemado:
       NBR = (NIR − SWIR2) / (NIR + SWIR2)   (bandas B08 y B12, 20 m)
       dNBR = NBR antes − NBR después
   Quemado = dNBR ≥ 0,27 en píxeles sin nubes (capa SCL), solo en manchas conectadas a las detecciones de fuego
   (así no se confunde con cosechas u otros cambios). Severidad según USGS: 0,27–0,44 moderada baja,
   0,44–0,66 moderada alta, > 0,66 alta.
3. Se publica site/incendios/quemadas.geojson (polígonos, hectáreas, fechas de las imágenes).
"""

import json, math, os
from datetime import datetime, timedelta, timezone

import numpy as np
import requests

STAC = "https://earth-search.aws.element84.com/v1/search"
UMBRAL = 0.27
MAX_POR_PASADA = 2
DIAS_TEMPORADA = 180
SCL_OK = {4, 5, 7}            # vegetación, suelo desnudo, sin clasificar (excluye nubes, sombras, agua, nieve)


def _t(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)


def _km(a1, o1, a2, o2):
    a, b = math.radians(a1), math.radians(a2)
    return 6371 * math.acos(max(-1, min(1, math.sin(a) * math.sin(b) + math.cos(a) * math.cos(b) * math.cos(math.radians(o2 - o1)))))


# ------------------------------------------------------------------ registro de la temporada
def registrar(temporada, focos, incendios, ahora):
    """Une los incendios actuales con el registro (mismo incendio si está a ≤ 8 km y no lleva más de 2 días apagado)."""
    T = [r for r in (temporada or []) if _t(r["ultima"]) >= ahora - timedelta(days=DIAS_TEMPORADA)]
    vivos = [f for f in focos if not f.get("industrial")]
    for i in incendios:
        pts = [[round(f["lat"], 4), round(f["lon"], 4)] for f in vivos if _km(i["lat"], i["lon"], f["lat"], f["lon"]) <= 8]
        r = next((r for r in T if _km(i["lat"], i["lon"], r["lat"], r["lon"]) <= 8 and _t(i["primera"]) <= _t(r["ultima"]) + timedelta(days=2)), None)
        if r is None:
            r = dict(id=f"{i['primera'][:10]}_{i['lat']:.3f}_{i['lon']:.3f}", lat=i["lat"], lon=i["lon"], localidad=i["localidad"],
                     dist=i["dist"], rumbo=i["rumbo"], primera=i["primera"], ultima=i["ultima"], frp=i["frp"], puntos=[])
            T.append(r)
        r["primera"] = min(r["primera"], i["primera"]); r["ultima"] = max(r["ultima"], i["ultima"])
        r["frp"] = max(r.get("frp") or 0, i["frp"] or 0)
        conocidos = {tuple(p) for p in r["puntos"]}
        r["puntos"] = (r["puntos"] + [p for p in pts if tuple(p) not in conocidos])[-400:]
    return T


# ------------------------------------------------------------------ Sentinel-2
def buscar(lat, lon, desde, hasta, nubes=60):
    body = {"collections": ["sentinel-2-l2a"], "intersects": {"type": "Point", "coordinates": [lon, lat]},
            "datetime": f"{desde:%Y-%m-%dT%H:%M:%SZ}/{hasta:%Y-%m-%dT%H:%M:%SZ}", "query": {"eo:cloud_cover": {"lt": nubes}},
            "limit": 40, "sortby": [{"field": "properties.datetime", "direction": "desc"}]}
    r = requests.post(STAC, json=body, timeout=60)
    r.raise_for_status()
    return r.json().get("features", [])


def _escala(asset):
    b = (asset.get("raster:bands") or [{}])[0]
    return b.get("scale", 1e-4), b.get("offset", 0.0)


def leer(item, caja):
    """Devuelve NIR, SWIR2, SCL (grilla 20 m de B12) y la transformación, para la caja (oeste, sur, este, norte)."""
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    from rasterio.enums import Resampling
    A = item["assets"]
    env = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif")
    with rasterio.Env(**env):
        with rasterio.open(A["swir22"]["href"]) as ds:
            b = transform_bounds("EPSG:4326", ds.crs, *caja)
            w = from_bounds(*b, transform=ds.transform).round_offsets().round_lengths()
            sw = ds.read(1, window=w, boundless=True, fill_value=0).astype("f4")
            tr, crs = ds.window_transform(w), ds.crs
        h, wd = sw.shape
        with rasterio.open(A["nir"]["href"]) as ds:
            w2 = from_bounds(*b, transform=ds.transform)
            ni = ds.read(1, window=w2, out_shape=(h, wd), resampling=Resampling.average, boundless=True, fill_value=0).astype("f4")
        with rasterio.open(A["scl"]["href"]) as ds:
            w3 = from_bounds(*b, transform=ds.transform)
            scl = ds.read(1, window=w3, out_shape=(h, wd), resampling=Resampling.nearest, boundless=True, fill_value=0)
    s1, o1 = _escala(A["swir22"]); s2, o2 = _escala(A["nir"])
    valido = (sw > 0) & (ni > 0) & np.isin(scl, list(SCL_OK))
    return ni * s2 + o2, sw * s1 + o1, valido, tr, crs


def nbr(ni, sw):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (ni - sw) / (ni + sw)


def caja_de(r, margen_km=3):
    la = [p[0] for p in r["puntos"]] or [r["lat"]]; lo = [p[1] for p in r["puntos"]] or [r["lon"]]
    dla = margen_km / 111.0; dlo = margen_km / (111.0 * math.cos(math.radians(r["lat"])))
    s, n = min(la) - dla, max(la) + dla; w, e = min(lo) - dlo, max(lo) + dlo
    lim = 0.2                                     # máximo ~22 km por lado
    cy, cx = (s + n) / 2, (w + e) / 2
    return (max(w, cx - lim), max(s, cy - lim), min(e, cx + lim), min(n, cy + lim))


def cicatriz(r):
    """Calcula el área quemada de un incendio del registro. Devuelve (feature GeoJSON | None, estado)."""
    from rasterio.features import shapes
    from rasterio.warp import transform_geom, transform
    from shapely.geometry import shape, mapping
    from shapely.ops import unary_union
    from skimage.measure import label
    pri, ult = _t(r["primera"]), _t(r["ultima"])
    post = [i for i in buscar(r["lat"], r["lon"], ult + timedelta(hours=12), datetime.now(timezone.utc), 50)]
    if not post:
        return None, "sin imagen posterior todavía"
    ip = post[0]
    tile = ip["properties"].get("grid:code")
    pre = [i for i in buscar(r["lat"], r["lon"], pri - timedelta(days=45), pri - timedelta(hours=2), 50) if i["properties"].get("grid:code") == tile]
    if not pre:
        return None, "sin imagen anterior del mismo cuadro"
    ia = pre[0]
    caja = caja_de(r)
    n1, s1, v1, tr, crs = leer(ia, caja)
    n2, s2, v2, tr2, _ = leer(ip, caja)
    if n1.shape != n2.shape:
        return None, "las imágenes no coinciden en grilla"
    d = nbr(n1, s1) - nbr(n2, s2)
    ok = v1 & v2
    if ok.mean() < 0.3:
        return None, "imagen con muchas nubes"
    q = ok & (d >= UMBRAL)
    # solo manchas conectadas a las detecciones de fuego (≤ 1 km)
    L = label(q, connectivity=2)
    if L.max() == 0:
        return None, "sin área quemada detectable"
    xs, ys = transform("EPSG:4326", crs, [p[1] for p in r["puntos"]] or [r["lon"]], [p[0] for p in r["puntos"]] or [r["lat"]])
    inv = ~tr
    marcas = set()
    rad = 50                                        # 50 píxeles de 20 m = 1 km
    for x, y in zip(xs, ys):
        c, f = inv * (x, y)
        c, f = int(c), int(f)
        f0, f1, c0, c1 = max(0, f - rad), min(L.shape[0], f + rad), max(0, c - rad), min(L.shape[1], c + rad)
        if f0 < f1 and c0 < c1:
            marcas |= set(np.unique(L[f0:f1, c0:c1]).tolist())
    marcas.discard(0)
    if not marcas:
        return None, "sin área quemada junto a las detecciones"
    m = np.isin(L, list(marcas))
    ha = float(m.sum() * 400 / 10000)
    if ha < 2:
        return None, "área quemada menor a 2 ha"
    sev = {"moderada_baja": float(((d >= .27) & (d < .44) & m).sum() * .04), "moderada_alta": float(((d >= .44) & (d < .66) & m).sum() * .04),
           "alta": float(((d >= .66) & m).sum() * .04)}
    geoms = [shape(transform_geom(crs, "EPSG:4326", g)) for g, v in shapes(m.astype("u1"), mask=m, transform=tr) if v == 1]
    geo = unary_union(geoms).simplify(0.0002, preserve_topology=True)
    redondear = lambda c: [redondear(x) for x in c] if isinstance(c, (list, tuple)) and c and isinstance(c[0], (list, tuple)) else [round(c[0], 5), round(c[1], 5)]
    gj = mapping(geo); gj = {"type": gj["type"], "coordinates": redondear(gj["coordinates"])}
    props = dict(id=r["id"], localidad=r["localidad"], hectareas=round(ha, 1), severidad={k: round(v, 1) for k, v in sev.items()},
                 imagen_antes=ia["properties"]["datetime"][:10], imagen_despues=ip["properties"]["datetime"][:10],
                 primera=r["primera"], ultima=r["ultima"], calculado=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"))
    return {"type": "Feature", "geometry": gj, "properties": props}, "ok"


def actualizar(temporada, dir_inc, sitio_url, ahora):
    """Calcula cicatrices pendientes (máx. 2 por pasada, cada incendio se reintenta cada 12 h hasta 30 días) y publica quemadas.geojson."""
    p = os.path.join(dir_inc, "quemadas.geojson")
    fc = None
    if os.path.exists(p):
        fc = json.load(open(p))
    elif sitio_url:
        try:
            rr = requests.get(sitio_url.rstrip("/") + "/incendios/quemadas.geojson", timeout=30)
            fc = rr.json() if rr.ok else None
        except Exception:
            fc = None
    fc = fc or {"type": "FeatureCollection", "features": []}
    hechos = {f["properties"]["id"] for f in fc["features"]}
    hechos_t = {f["properties"]["id"]: f["properties"]["calculado"] for f in fc["features"]}
    n = 0
    for r in sorted(temporada, key=lambda r: r["ultima"], reverse=True):
        if n >= MAX_POR_PASADA:
            break
        ult = _t(r["ultima"])
        if ahora - ult < timedelta(days=1) or ahora - ult > timedelta(days=30):
            continue
        if r["id"] in hechos and hechos_t[r["id"]] >= r["ultima"]:
            continue
        if r.get("intento") and _t(r["intento"]) > ahora - timedelta(hours=12):
            continue
        r["intento"] = ahora.strftime("%Y-%m-%dT%H:%MZ")
        n += 1
        try:
            f, estado = cicatriz(r)
        except Exception as ex:
            f, estado = None, f"error: {ex}"
        r["cicatriz"] = estado
        print(f"  quemadas: {r['localidad']} ({r['id']}): {estado}" + (f", {f['properties']['hectareas']} ha" if f else ""))
        if f:
            fc["features"] = [x for x in fc["features"] if x["properties"]["id"] != r["id"]] + [f]
    corte = (ahora - timedelta(days=DIAS_TEMPORADA)).strftime("%Y-%m-%dT%H:%MZ")
    fc["features"] = [x for x in fc["features"] if x["properties"]["ultima"] >= corte]
    json.dump(fc, open(p, "w"), ensure_ascii=False, separators=(",", ":"))
    return fc

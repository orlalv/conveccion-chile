"""
Incendios y humo en vivo · zona central de Chile (METOTEC).

En cada ejecución (~10 min):
  1. Focos del GOES-19 (producto FDC, disco completo, ~2 km) dentro de la zona.
  2. Humo del GOES-19 (producto ADP, solo de día) como imagen semitransparente.
  3. Focos de NASA FIRMS (VIIRS 375 m y MODIS) si existe la clave FIRMS_KEY.
  4. Viento a 10 m de Open-Meteo en una grilla, para la dirección del humo.
  5. Agrupa los focos en incendios, los nombra por la localidad más cercana y escribe un resumen en texto claro.
Se descartan los puntos calientes industriales (lista fija + los que aparecen casi todos los días).

Publica site/incendios/data.json (lo lee site/incendios/index.html) y site/incendios/humo.png.
"""

import json, math, os
from datetime import datetime, timedelta, timezone

import numpy as np
import requests
from PIL import Image

LAT_S, LAT_N, LON_O, LON_E = -38.0, -32.0, -74.0, -69.8
HORAS = 72                      # focos que se guardan (la página anima hasta 72 h)
B7_CUADROS = 36
ULTIMO_C07 = None                 # imágenes de banda 7 que se guardan (6 h)
DIAS_PERSIST = 7                # historial para detectar puntos industriales
CODIGOS = {10: "alta", 11: "alta", 30: "alta", 31: "alta", 13: "alta", 33: "alta",
           12: "media", 32: "media", 14: "media", 34: "media", 15: "baja", 35: "baja"}
INDUSTRIALES = [  # puntos calientes permanentes conocidos (fundiciones, refinerías, celulosas)
    ("Fundición Caletones", -34.105, -70.452), ("Fundición Ventanas", -32.745, -71.484), ("Fundición Chagres", -32.795, -70.957),
    ("Refinería Aconcagua (Concón)", -32.935, -71.515), ("Refinería Biobío (Hualpén)", -36.776, -73.125),
    ("Celulosa Nueva Aldea", -36.648, -72.455), ("Celulosa Constitución", -35.330, -72.410), ("Celulosa Licancel", -34.985, -72.005),
    ("Celulosa Laja", -37.275, -72.705), ("Celulosa Santa Fe (Nacimiento)", -37.470, -72.570), ("Celulosa Arauco (Horcones)", -37.200, -73.130),
]
LOCALIDADES = [
    ("Rancagua", -34.170, -70.744), ("Machalí", -34.180, -70.650), ("Graneros", -34.065, -70.727), ("Codegua", -34.036, -70.666),
    ("Mostazal", -33.980, -70.712), ("Doñihue", -34.226, -70.965), ("Coltauco", -34.287, -71.075), ("Requínoa", -34.286, -70.818),
    ("Rengo", -34.407, -70.857), ("Malloa", -34.446, -70.946), ("San Vicente de Tagua Tagua", -34.438, -71.077), ("Peumo", -34.396, -71.170),
    ("Las Cabras", -34.293, -71.307), ("Pichidegua", -34.358, -71.283), ("San Fernando", -34.585, -70.988), ("Chimbarongo", -34.711, -71.043),
    ("Nancagua", -34.663, -71.217), ("Santa Cruz", -34.639, -71.366), ("Lolol", -34.729, -71.645), ("Pichilemu", -34.387, -72.003),
    ("Marchigüe", -34.398, -71.623), ("Litueche", -34.110, -71.724), ("Navidad", -33.957, -71.832), ("Peralillo", -34.477, -71.484),
    ("Palmilla", -34.604, -71.358), ("Chépica", -34.731, -71.282), ("Placilla", -34.612, -71.118), ("Olivar", -34.212, -70.823),
    ("Quinta de Tilcoco", -34.354, -70.962), ("Coinco", -34.270, -70.957), ("Paredones", -34.648, -71.900), ("La Estrella", -34.204, -71.606),
    ("Pumanque", -34.607, -71.666),
    ("Santiago", -33.450, -70.660), ("Puente Alto", -33.610, -70.580), ("Melipilla", -33.690, -71.210), ("Talagante", -33.660, -70.930),
    ("Buin", -33.730, -70.740), ("Paine", -33.810, -70.740), ("San José de Maipo", -33.640, -70.350), ("Colina", -33.200, -70.670),
    ("Curacaví", -33.400, -71.130), ("Alhué", -34.030, -71.100),
    ("Valparaíso", -33.047, -71.612), ("Viña del Mar", -33.024, -71.552), ("Quilpué", -33.047, -71.442), ("Casablanca", -33.320, -71.410),
    ("San Antonio", -33.590, -71.610), ("Quillota", -32.880, -71.250), ("Los Andes", -32.830, -70.600), ("San Felipe", -32.750, -70.720),
    ("Limache", -33.000, -71.270),
    ("Curicó", -34.980, -71.240), ("Teno", -34.870, -71.170), ("Molina", -35.110, -71.280), ("Hualañé", -34.980, -71.800),
    ("Licantén", -34.980, -72.000), ("Talca", -35.430, -71.660), ("Constitución", -35.330, -72.410), ("Linares", -35.850, -71.600),
    ("Cauquenes", -35.970, -72.320), ("Chillán", -36.610, -72.100), ("Quirihue", -36.280, -72.540), ("Concepción", -36.830, -73.050),
    ("Los Ángeles", -37.470, -72.350),
    ("Arauco", -37.250, -73.320), ("Lebu", -37.610, -73.650), ("Curanilahue", -37.470, -73.350), ("Cañete", -37.800, -73.400),
    ("Lota", -37.090, -73.160), ("Coronel", -37.030, -73.140), ("Tomé", -36.620, -72.960), ("Nacimiento", -37.500, -72.670),
    ("Mulchén", -37.720, -72.240), ("Yumbel", -37.100, -72.560), ("Cabrero", -37.030, -72.400), ("Florida", -36.820, -72.660),
    ("Santa Juana", -37.170, -72.940), ("Quillón", -36.740, -72.470), ("Coelemu", -36.490, -72.700), ("San Carlos", -36.420, -71.960),
    ("Parral", -36.140, -71.830), ("San Javier", -35.600, -71.730), ("Empedrado", -35.600, -72.280), ("Pelluhue", -35.810, -72.570),
]
RUMBOS = ["norte", "noreste", "este", "sureste", "sur", "suroeste", "oeste", "noroeste"]


def km(la1, lo1, la2, lo2):
    a = np.radians(la1); b = np.radians(la2)
    return 6371 * np.arccos(np.clip(np.sin(a) * np.sin(b) + np.cos(a) * np.cos(b) * np.cos(np.radians(lo2) - np.radians(lo1)), -1, 1))


def rumbo(grados):
    return RUMBOS[int(((grados % 360) + 22.5) // 45) % 8]


def cercana(la, lo):
    d = [(km(la, lo, a, b), n, a, b) for n, a, b in LOCALIDADES]
    dist, n, a, b = min(d)
    az = math.degrees(math.atan2(math.radians(lo - b) * math.cos(math.radians(a)), math.radians(la - a))) % 360
    return n, float(dist), rumbo(az)


# ------------------------------------------------------------------ proyección inversa GOES (de ángulos de barrido a lat/lon)
def latlon(ds, ix, iy):
    pr = ds.variables["goes_imager_projection"]
    req, rpol = pr.semi_major_axis, pr.semi_minor_axis
    H = pr.perspective_point_height + req
    l0 = math.radians(pr.longitude_of_projection_origin)
    x = ds.variables["x"][:].astype("f8")[ix]; y = ds.variables["y"][:].astype("f8")[iy]
    a = np.sin(x) ** 2 + np.cos(x) ** 2 * (np.cos(y) ** 2 + (req ** 2 / rpol ** 2) * np.sin(y) ** 2)
    b = -2 * H * np.cos(x) * np.cos(y)
    c = H ** 2 - req ** 2
    rs = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    sx = rs * np.cos(x) * np.cos(y); sy = -rs * np.sin(x); sz = rs * np.cos(x) * np.sin(y)
    lat = np.degrees(np.arctan((req ** 2 / rpol ** 2) * (sz / np.sqrt((H - sx) ** 2 + sy ** 2))))
    lon = np.degrees(l0 - np.arctan(sy / (H - sx)))
    return lat, lon


def ventana(ds, Geo):
    """Índices de la ventana del disco que cubre la zona."""
    pr = ds.variables["goes_imager_projection"]
    geo = Geo(pr.perspective_point_height + pr.semi_major_axis, pr.semi_major_axis, pr.semi_minor_axis, pr.longitude_of_projection_origin)
    la, lo = np.meshgrid(np.linspace(LAT_S, LAT_N, 25), np.linspace(LON_O, LON_E, 25))
    sx, sy = geo.scan(la.ravel(), lo.ravel())
    xs = ds.variables["x"][:].astype("f8"); ys = ds.variables["y"][:].astype("f8")
    ix = (sx - xs[0]) / (xs[1] - xs[0]); iy = (sy - ys[0]) / (ys[1] - ys[0])
    i0, i1 = int(max(0, np.nanmin(ix) - 2)), int(min(xs.size, np.nanmax(ix) + 3))
    j0, j1 = int(max(0, np.nanmin(iy) - 2)), int(min(ys.size, np.nanmax(iy) + 3))
    return i0, i1, j0, j1


# ------------------------------------------------------------------ 1. focos GOES (FDC)
def focos_goes(contenido, t, Geo):
    import netCDF4
    ds = netCDF4.Dataset("fdc.nc", memory=contenido)
    i0, i1, j0, j1 = ventana(ds, Geo)
    M = np.ma.filled(ds.variables["Mask"][j0:j1, i0:i1], -1).astype(int)
    P = np.ma.filled(ds.variables["Power"][j0:j1, i0:i1].astype("f4"), np.nan) if "Power" in ds.variables else np.full(M.shape, np.nan)
    jj, ii = np.nonzero(np.isin(M, list(CODIGOS)))
    out = []
    if jj.size:
        lat, lon = latlon(ds, ii + i0, jj + j0)
        for k in range(jj.size):
            if LAT_S <= lat[k] <= LAT_N and LON_O <= lon[k] <= LON_E:
                frp = P[jj[k], ii[k]]
                out.append(dict(lat=round(float(lat[k]), 4), lon=round(float(lon[k]), 4), t=t.strftime("%Y-%m-%dT%H:%MZ"), fuente="GOES-19",
                                conf=CODIGOS[int(M[jj[k], ii[k]])], frp=None if not np.isfinite(frp) else round(float(frp), 1)))
    ds.close()
    print(f"  incendios: GOES {len(out)} píxeles con fuego en la zona")
    return out


# ------------------------------------------------------------------ 2. humo GOES (ADP)
def grilla_zona(ancho=420):
    my = lambda la: np.log(np.tan(np.pi / 4 + np.radians(la) / 2))
    alto = int(round(ancho * (my(LAT_N) - my(LAT_S)) / np.radians(LON_E - LON_O)))
    lon = np.linspace(LON_O, LON_E, ancho)
    yy = np.linspace(my(LAT_N), my(LAT_S), alto)
    lat = np.degrees(2 * np.arctan(np.exp(yy)) - np.pi / 2)
    LO, LA = np.meshgrid(lon, lat)
    return LO, LA


def humo_goes(contenido, Geo, salida):
    import netCDF4
    ds = netCDF4.Dataset("adp.nc", memory=contenido)
    nombre = next((v for v in ("Smoke", "smoke", "SMK") if v in ds.variables), None)
    if nombre is None:
        print("  incendios: ADP sin variable de humo; variables:", list(ds.variables)[:30])
        ds.close(); return None
    LO, LA = grilla_zona()
    pr = ds.variables["goes_imager_projection"]
    geo = Geo(pr.perspective_point_height + pr.semi_major_axis, pr.semi_major_axis, pr.semi_minor_axis, pr.longitude_of_projection_origin)
    sx, sy = geo.scan(LA.ravel(), LO.ravel())
    xs = ds.variables["x"][:].astype("f8"); ys = ds.variables["y"][:].astype("f8")
    ix = np.rint((sx - xs[0]) / (xs[1] - xs[0])); iy = np.rint((sy - ys[0]) / (ys[1] - ys[0]))
    ok = np.isfinite(ix) & np.isfinite(iy) & (ix >= 0) & (ix < xs.size) & (iy >= 0) & (iy < ys.size)
    i0, i1, j0, j1 = int(ix[ok].min()), int(ix[ok].max()) + 1, int(iy[ok].min()), int(iy[ok].max()) + 1
    W = np.ma.filled(ds.variables[nombre][j0:j1, i0:i1], 0).astype("f4")
    v = np.zeros(LA.size, "f4")
    v[ok] = W[(iy[ok] - j0).astype(int), (ix[ok] - i0).astype(int)]
    ds.close()
    m = (v.reshape(LA.shape) >= 1)
    img = np.zeros(LA.shape + (4,), "u1")
    img[m] = (205, 190, 160, 170)                     # humo: gris pardo semitransparente
    Image.fromarray(img, "RGBA").save(os.path.join(salida, "humo.png"), optimize=True)
    pct = 100 * m.mean()
    print(f"  incendios: humo en {pct:.1f} % de la zona")
    return {"img": "humo.png", "limites": [[LAT_S, LON_O], [LAT_N, LON_E]], "pixeles": int(m.sum())}


# ------------------------------------------------------------------ 2b. banda 7 (3,9 µm): el fuego como punto caliente
def paleta_b7(tb):
    """Gris para el paisaje (frío = claro, como el infrarrojo habitual) y amarillo→rojo→blanco para lo muy caliente."""
    img = np.zeros(tb.shape + (4,), "u1")
    ok = np.isfinite(tb)
    g = np.clip((330 - tb) / (330 - 200), 0, 1)                    # 200 K blanco, 330 K negro
    v = (40 + 200 * g).astype("u1")
    img[..., 0] = img[..., 1] = img[..., 2] = v
    img[..., 3] = np.where(ok, 255, 0)
    cal = ok & (tb >= 320)
    k = np.clip((tb - 320) / 60, 0, 1)                             # 320 K amarillo, 350 K rojo, 380+ K blanco
    rr = np.full(tb.shape, 255.0); gg = np.where(k < .5, 230 - 380 * k, 40 + 430 * (k - .5)); bb = np.where(k < .5, 40, 40 + 430 * (k - .5))
    img[cal, 0] = rr[cal]; img[cal, 1] = np.clip(gg[cal], 0, 255); img[cal, 2] = np.clip(bb[cal], 0, 255)
    return img


def banda7(contenido, Geo, ruta):
    import netCDF4
    ds = netCDF4.Dataset("c07.nc", memory=contenido)
    LO, LA = grilla_zona(560)
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
    Image.fromarray(paleta_b7(v.reshape(LA.shape)), "RGBA").save(ruta, optimize=True)


def cuadros_b7(t, prev, dir_inc, sitio_url, Geo, archivos, bajar):
    """Agrega la banda 7 de la imagen t y recupera las anteriores publicadas (las últimas 6 h)."""
    d7 = os.path.join(dir_inc, "b7"); os.makedirs(d7, exist_ok=True)
    lista = list(prev.get("b7", []))
    try:
        cand = archivos("ABI-L2-CMIPF", "M6C07_G19", t - timedelta(minutes=12), t + timedelta(minutes=1))
        if cand:
            tf, k = cand[-1]
            ts = tf.strftime("%Y%m%d%H%M")
            global ULTIMO_C07
            ULTIMO_C07 = (tf, bajar(k))                 # lo reutiliza el producto de niebla nocturna
            if ts not in lista:
                banda7(ULTIMO_C07[1], Geo, os.path.join(d7, ts + ".png"))
                lista.append(ts)
    except Exception as ex:
        print("  incendios: banda 7 falló:", ex)
    lista = sorted(set(lista))[-B7_CUADROS:]
    quedan = []
    for ts in lista:                                  # el sitio se arma de cero: se traen las publicadas antes
        p = os.path.join(d7, ts + ".png")
        if not os.path.exists(p) and sitio_url:
            try:
                rr = requests.get(f"{sitio_url.rstrip('/')}/incendios/b7/{ts}.png", timeout=30)
                if rr.ok:
                    open(p, "wb").write(rr.content)
            except Exception:
                pass
        if os.path.exists(p):
            quedan.append(ts)
    return quedan


# ------------------------------------------------------------------ 3. FIRMS
def focos_firms(clave):
    if not clave:
        return None
    out = []
    for fuente in ("VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "MODIS_NRT"):
        u = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{clave}/{fuente}/{LON_O},{LAT_S},{LON_E},{LAT_N}/4"
        try:
            r = requests.get(u, timeout=60); r.raise_for_status()
            filas = [l.split(",") for l in r.text.strip().splitlines()]
            if len(filas) < 2 or "latitude" not in filas[0]:
                continue
            c = {k: i for i, k in enumerate(filas[0])}
            for f in filas[1:]:
                hh = f[c["acq_time"]].zfill(4)
                conf = f[c["confidence"]]
                conf = {"h": "alta", "n": "media", "l": "baja"}.get(conf, "alta" if conf.isdigit() and int(conf) >= 80 else "media" if conf.isdigit() and int(conf) >= 30 else "baja")
                out.append(dict(lat=float(f[c["latitude"]]), lon=float(f[c["longitude"]]), t=f"{f[c['acq_date']]}T{hh[:2]}:{hh[2:]}Z",
                                fuente=fuente.replace("_NRT", "").replace("_", " "), conf=conf, frp=float(f[c["frp"]]) if f[c["frp"]] else None))
        except Exception as ex:
            print(f"  incendios: FIRMS {fuente} falló:", ex)
    print(f"  incendios: FIRMS {len(out)} focos")
    return out


# ------------------------------------------------------------------ 4. viento
def viento():
    lats = np.arange(LAT_S + .25, LAT_N, .5); lons = np.arange(LON_O + .25, LON_E, .5)
    LA, LO = [a.ravel() for a in np.meshgrid(lats, lons)]
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=60, params={
            "latitude": ",".join(f"{x:.2f}" for x in LA), "longitude": ",".join(f"{x:.2f}" for x in LO),
            "current": "wind_speed_10m,wind_direction_10m", "wind_speed_unit": "kmh"})
        r.raise_for_status()
        js = r.json(); js = js if isinstance(js, list) else [js]
        return [[round(float(a), 2), round(float(b), 2), j["current"]["wind_direction_10m"], j["current"]["wind_speed_10m"]] for a, b, j in zip(LA, LO, js)]
    except Exception as ex:
        print("  incendios: sin viento:", ex)
        return []


def viento_en(V, la, lo):
    if not V:
        return None
    d = min(V, key=lambda w: (w[0] - la) ** 2 + ((w[1] - lo) * math.cos(math.radians(la))) ** 2)
    return d[2], d[3]


# ------------------------------------------------------------------ 5. agrupar y resumir
def persistentes(hist):
    """Celdas de 0,02° con fuego en al menos 4 de los últimos 7 días: punto caliente fijo (industrial)."""
    return {c for c, dias in hist.items() if len(dias) >= 4}


def celda(la, lo):
    return f"{round(la / .02) * .02:.2f},{round(lo / .02) * .02:.2f}"


def es_industrial(f, fijos):
    if celda(f["lat"], f["lon"]) in fijos:
        return True
    return any(km(f["lat"], f["lon"], a, b) <= 3 for _, a, b in INDUSTRIALES)


def agrupar(focos, ahora, V):
    vivos = [f for f in focos if not f.get("industrial")]
    grupos = []
    for f in sorted(vivos, key=lambda f: f["t"]):
        g = next((g for g in grupos if min(km(f["lat"], f["lon"], h["lat"], h["lon"]) for h in g) <= 6), None)
        (g.append(f) if g is not None else grupos.append([f]))
    out = []
    for g in grupos:
        if all(f["conf"] == "baja" for f in g) and len(g) < 2:
            continue                                  # una sola detección de baja confianza: no se muestra como incendio
        la = float(np.mean([f["lat"] for f in g])); lo = float(np.mean([f["lon"] for f in g]))
        n, d, az = cercana(la, lo)
        ult = max(f["t"] for f in g); pri = min(f["t"] for f in g)
        edad = (ahora - datetime.strptime(ult, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)).total_seconds() / 60
        g1 = [f for f in g if f["fuente"] == "GOES-19" and f["t"] >= (ahora - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%MZ")]
        g0 = [f for f in g if f["fuente"] == "GOES-19" and (ahora - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%MZ") <= f["t"] < (ahora - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%MZ")]
        tendencia = "creciendo" if len(g1) > len(g0) + 1 else "disminuyendo" if len(g1) + 1 < len(g0) else "estable" if g1 else "sin detección reciente"
        w = viento_en(V, la, lo)
        frp = max([f["frp"] or 0 for f in g])
        donde = f"a {d:.0f} km al {az} de {n}" if d >= 3 else f"en {n}"
        txt = f"Incendio {donde}. "
        txt += f"Detectado desde las {hora_local(pri)}; última detección a las {hora_local(ult)} ({tendencia}). "
        if w:
            txt += f"Viento de {w[1]:.0f} km/h: el humo se dirige hacia el {rumbo(w[0] + 180)}."
        goes = any(f["fuente"] == "GOES-19" for f in g)
        relevante = goes or frp >= 20 or len(g) >= 3          # el GOES solo ve fuegos grandes; un foco VIIRS chico y suelto suele ser quema o fuego menor
        out.append(dict(relevante=relevante, lat=round(la, 4), lon=round(lo, 4), localidad=n, dist=round(d, 1), rumbo=az, primera=pri, ultima=ult, edad_min=round(edad),
                        focos=len(g), frp=round(frp, 1), tendencia=tendencia, viento=w, fuentes=sorted({f["fuente"] for f in g}), texto=txt))
    out.sort(key=lambda x: (not x["relevante"], x["edad_min"] > 180, -x["frp"]))
    return out


def hora_local(t):
    """Hora de Chile continental (UTC−3 en horario de verano; UTC−4 en invierno)."""
    d = datetime.strptime(t, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
    y = d.year
    # horario de verano en Chile: desde el primer domingo de septiembre hasta el primer domingo de abril (aprox.)
    def domingo(m):
        x = datetime(y, m, 1, 4, tzinfo=timezone.utc)
        return x + timedelta(days=(6 - x.weekday()) % 7)
    off = -3 if (d >= domingo(9) or d < domingo(4)) else -4
    return (d + timedelta(hours=off)).strftime("%H:%M")


def estado_localidades(incendios):
    out = {}
    for n, a, b in LOCALIDADES:
        cerca = [(km(a, b, i["lat"], i["lon"]), i) for i in incendios if i["edad_min"] <= 360 and i.get("relevante", True)]
        cerca = [(d, i) for d, i in cerca if d <= 40]
        if not cerca:
            out[n] = {"nivel": 0, "texto": f"{n}: sin incendios detectados a menos de 40 km en las últimas 6 horas."}
            continue
        d, i = min(cerca, key=lambda x: x[0])
        hacia = None
        if i.get("viento"):
            dir_humo = (i["viento"][0] + 180) % 360
            az = math.degrees(math.atan2(math.radians(b - i["lon"]) * math.cos(math.radians(a)), math.radians(a - i["lat"]))) % 360
            hacia = abs((az - dir_humo + 180) % 360 - 180) <= 45
        nivel = 2 if d <= 15 else 1
        txt = f"{n}: incendio a {d:.0f} km ({'en ' + i['localidad'] if i['dist'] < 3 else 'cerca de ' + i['localidad']}), {i['tendencia']}."
        if hacia:
            txt += " El viento lleva el humo hacia esta comuna."
        out[n] = {"nivel": nivel, "texto": txt, "humo": bool(hacia)}
    return out


# ------------------------------------------------------------------ ciclo
def cargar_previo(dir_inc, sitio_url):
    p = os.path.join(dir_inc, "data.json")
    if os.path.exists(p):
        return json.load(open(p))
    if sitio_url:
        try:
            r = requests.get(sitio_url.rstrip("/") + "/incendios/data.json", timeout=30)
            if r.ok:
                return r.json()
        except Exception as ex:
            print("  incendios: sin datos anteriores:", ex)
    return {}


def actualizar(t, salida_site, sitio_url, Geo, archivos, bajar, prev=None):
    dir_inc = os.path.join(salida_site, "incendios")
    os.makedirs(dir_inc, exist_ok=True)
    prev = prev if prev is not None else cargar_previo(dir_inc, sitio_url)
    ahora = datetime.now(timezone.utc)
    limite = (ahora - timedelta(hours=HORAS)).strftime("%Y-%m-%dT%H:%MZ")
    focos = [f for f in prev.get("focos", []) if f["t"] >= limite and f["fuente"] == "GOES-19"]
    hist = prev.get("historial", {})
    estado = {}

    # GOES FDC (mismo barrido o el más reciente disponible)
    try:
        lista = archivos("ABI-L2-FDCF", "FDCF-M6_G19", t - timedelta(minutes=30), t + timedelta(minutes=1))
        if lista:
            tf, k = lista[-1]
            ts = tf.strftime("%Y-%m-%dT%H:%MZ")
            if not any(f["t"] == ts for f in focos):
                nuevos = focos_goes(bajar(k), tf, Geo)
                focos += nuevos
                dia = tf.strftime("%Y-%m-%d")
                for f in nuevos:
                    c = hist.setdefault(celda(f["lat"], f["lon"]), [])
                    if dia not in c:
                        c.append(dia)
            estado["fdc"] = ts
        else:
            print("  incendios: sin archivo FDC reciente")
    except Exception as ex:
        print("  incendios: FDC falló:", ex)

    # historial de celdas: solo últimos 7 días
    corte = (ahora - timedelta(days=DIAS_PERSIST)).strftime("%Y-%m-%d")
    hist = {c: [d for d in ds if d >= corte] for c, ds in hist.items()}
    hist = {c: ds for c, ds in hist.items() if ds}
    fijos = persistentes(hist)

    # FIRMS
    firms = focos_firms(os.environ.get("FIRMS_KEY", "").strip())   # 4 días UTC para cubrir 72 h (con 1 día se vacía a las 00 UTC)
    if firms is not None:
        firms = [f for f in firms if f["t"] >= limite]
    if firms is None:
        firms = [f for f in prev.get("focos", []) if f["fuente"] != "GOES-19" and f["t"] >= limite]
        estado["firms"] = "sin clave" if not os.environ.get("FIRMS_KEY") else "sin datos"
    else:
        estado["firms"] = f"{len(firms)} focos"
    todos = focos + firms
    for f in todos:
        f["industrial"] = es_industrial(f, fijos)

    # humo (solo de día; el ADP no se calcula de noche)
    humo = prev.get("humo")
    try:
        lista = archivos("ABI-L2-ADPF", "ADPF-M6_G19", t - timedelta(minutes=40), t + timedelta(minutes=1))
        if lista:
            tf, k = lista[-1]
            h = humo_goes(bajar(k), Geo, dir_inc)
            if h:
                h["t"] = tf.strftime("%Y-%m-%dT%H:%MZ")
                humo = h
        elif humo and humo.get("t", "") < (ahora - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%MZ"):
            humo = None
    except Exception as ex:
        print("  incendios: ADP falló:", ex)
    if humo and not os.path.exists(os.path.join(dir_inc, "humo.png")) and sitio_url:
        try:
            r = requests.get(sitio_url.rstrip("/") + "/incendios/humo.png", timeout=30)
            if r.ok:
                open(os.path.join(dir_inc, "humo.png"), "wb").write(r.content)
            else:
                humo = None
        except Exception:
            humo = None

    b7 = cuadros_b7(t, prev, dir_inc, sitio_url, Geo, archivos, bajar)
    V = viento() or prev.get("viento", [])
    inc = agrupar(todos, ahora, V)
    temporada = prev.get("temporada", [])
    try:                                           # etapa 2: registro de la temporada y áreas quemadas (Sentinel-2)
        import quemadas as Q
        temporada = Q.registrar(temporada, todos, inc, ahora)
        Q.actualizar(temporada, dir_inc, sitio_url, ahora)
    except Exception as ex:
        print("  incendios: áreas quemadas falló:", ex)
    js = dict(generado=ahora.strftime("%Y-%m-%dT%H:%MZ"), goes=t.strftime("%Y-%m-%dT%H:%MZ"), zona=[[LAT_S, LON_O], [LAT_N, LON_E]],
              estado=estado, focos=todos, incendios=inc, humo=humo, viento=V, localidades=estado_localidades(inc),
              coords={n: [a, b] for n, a, b in LOCALIDADES}, industriales=INDUSTRIALES, historial=hist, temporada=temporada, b7=b7, zona_img=[[LAT_S, LON_O], [LAT_N, LON_E]])
    json.dump(js, open(os.path.join(dir_inc, "data.json"), "w"), ensure_ascii=False, separators=(",", ":"))
    print(f"  incendios: {len(inc)} incendios activos/recientes, {sum(f['industrial'] for f in todos)} puntos industriales descartados")
    return js


def mantener(salida_site, sitio_url):
    """Sin imagen nueva: vuelve a publicar lo anterior (el sitio se arma de cero en cada ejecución)."""
    dir_inc = os.path.join(salida_site, "incendios")
    os.makedirs(dir_inc, exist_ok=True)
    prev = cargar_previo(dir_inc, sitio_url)
    if prev:
        json.dump(prev, open(os.path.join(dir_inc, "data.json"), "w"), ensure_ascii=False, separators=(",", ":"))
        for ts in prev.get("b7", []):
            p = os.path.join(dir_inc, "b7", ts + ".png"); os.makedirs(os.path.dirname(p), exist_ok=True)
            if sitio_url and not os.path.exists(p):
                try:
                    r = requests.get(f"{sitio_url.rstrip('/')}/incendios/b7/{ts}.png", timeout=30)
                    if r.ok:
                        open(p, "wb").write(r.content)
                except Exception:
                    pass
        for nombre in ("humo.png", "quemadas.geojson"):
            if sitio_url and not os.path.exists(os.path.join(dir_inc, nombre)):
                try:
                    r = requests.get(sitio_url.rstrip("/") + "/incendios/" + nombre, timeout=30)
                    if r.ok:
                        open(os.path.join(dir_inc, nombre), "wb").write(r.content)
                except Exception:
                    pass

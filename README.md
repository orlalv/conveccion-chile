# Convección Chile — satelite.metotec.cl

Visor automático de nubes convectivas sobre Chile y el Pacífico cercano, con datos del satélite **GOES-19** (banda 13 y rayos GLM) de la NOAA.

Autor: **Orlando Álvarez Bustamante**

Cada 10 minutos, GitHub ejecuta `procesar.py`, que:

1. Descarga la última imagen de la banda 13 (disco completo) y los rayos GLM desde el repositorio público de la NOAA en AWS.
2. Reproyecta la imagen a un mapa de Chile y el Pacífico.
3. Detecta nubes con tope de −40 °C o menos y las clasifica en **CB**, **frontal con convección**, **convección moderada**, **banda frontal** o **nubosidad alta**. Usa forma, temperatura mínima, textura y rayos.
4. Compara con la imagen anterior: creciendo, estable o disipando, y dibuja la trayectoria.
5. Publica la página con un loop de las últimas 2 horas.

Solo se marcan las nubes que tocan Chile o el Pacífico hasta 300 km de la costa. Las que están completamente en Argentina se ignoran, y las del Pacífico lejano se muestran tenues.

---

## Puesta en marcha (una sola vez)

### 1. Crear el repositorio

1. Entra a <https://github.com> con tu cuenta.
2. Arriba a la derecha: **+ → New repository**.
3. Nombre: `conveccion-chile`. Marca **Public**. Esto es necesario para que la ejecución automática sea gratis e ilimitada.
4. Presiona **Create repository**.

### 2. Subir los archivos

1. En el repositorio recién creado, haz clic en **uploading an existing file**.
2. Arrastra **todo el contenido** de esta carpeta: `procesar.py`, `requirements.txt`, `README.md` y las carpetas `geo`, `site` y `.github`.
   - La carpeta `.github` empieza con punto y en Windows puede estar oculta. Actívala en el Explorador con **Vista → Mostrar → Elementos ocultos**.
   - Si GitHub no acepta la carpeta `.github` arrastrada, créala a mano: **Add file → Create new file**, escribe como nombre `.github/workflows/actualizar.yml` y pega el contenido del archivo.
3. Presiona **Commit changes**.

### 3. Activar GitHub Pages

1. En el repositorio: **Settings → Pages**.
2. En **Source**, elige **GitHub Actions**.

### 4. Primera ejecución

1. Pestaña **Actions**. Si aparece un aviso, presiona **I understand my workflows, go ahead and enable them**.
2. Elige **Actualizar visor** a la izquierda → **Run workflow** → **Run workflow**.
3. Espera 2 a 4 minutos. Si queda con ✓ verde, la página ya está publicada en `https://TU-USUARIO.github.io/conveccion-chile/`.
4. Si queda con ✗ rojo, abre la ejecución, haz clic en el paso que falló y copia el mensaje de error.

Desde ahí se ejecuta solo cada 10 minutos.

> GitHub pausa las tareas programadas de un repositorio público si pasan 60 días sin cambios en él. Si alguna vez el visor deja de actualizarse, entra a **Actions → Actualizar visor** y presiona **Enable workflow**, o haz cualquier cambio pequeño en el README.

### 5. Conectar satelite.metotec.cl

**En Netlify** (donde está el DNS de metotec.cl):

1. **Domains → metotec.cl → DNS settings → Add new record**.
2. Tipo **CNAME**, nombre **satelite**, valor **TU-USUARIO.github.io** (con tu usuario de GitHub, sin `https://` ni el nombre del repositorio).
3. Guarda.

> Si los servidores de nombre de metotec.cl no son los de Netlify, el registro se crea en el panel de quien sí los administre. Se revisan en nic.cl → tu dominio → Servidores de nombre.

**En GitHub:**

1. **Settings → Pages → Custom domain**: escribe `satelite.metotec.cl` y presiona **Save**.
2. Espera a que la verificación de DNS quede en verde. Puede tardar desde minutos hasta unas horas.
3. Marca **Enforce HTTPS** cuando se habilite.

Tu sitio principal metotec.cl no se ve afectado: sigue en Netlify.

---

## Uso

- **Pantalla de 75":** abre `https://satelite.metotec.cl/?tv` en pantalla completa (F11). El loop corre solo y se detiene unos segundos en la última imagen.
- **Teclado:** espacio reproduce o pausa, y las flechas avanzan o retroceden imagen por imagen.
- **Lista lateral:** al hacer clic en una celda, el mapa va a ella.
- La página revisa cada 2 minutos si hay datos nuevos y se actualiza sola.

## Probar sin internet

```bash
pip install -r requirements.txt
python procesar.py --demo
cd site && python -m http.server 8000
```

Abre <http://localhost:8000>. Verás una escena sintética marcada como "DATOS DE DEMOSTRACIÓN".

## Ajustes

Están al comienzo de `procesar.py`:

| Variable | Qué controla |
|---|---|
| `UMBRAL` | Temperatura de tope que define una nube fría (−40 °C) |
| `AREA_MIN` | Área mínima de una celda (km²) |
| `ZONA_MAR_KM` | Distancia de Pacífico que se considera "cercano" |
| `FRAMES` | Imágenes del loop (12 = 2 horas) |

## Fuentes y créditos

- NOAA GOES-19, ABI L2 CMIP (banda 13) y GLM L2 LCFA, vía NOAA Open Data Dissemination en AWS.
- Límites: Natural Earth. Mapa base: © OpenStreetMap, © CARTO.
- La detección es automática y de apoyo; no reemplaza el análisis del observador ni los productos oficiales.

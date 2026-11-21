import numpy as np
from osgeo import gdal
from sklearn.cluster import KMeans

INPUT = "question5_bands.tif"
N_CLUSTERS = 4

ds = gdal.Open(INPUT)
names = ["VH", "VV", "DPRI", "DPDI", "DPMI"]
bands = {n: ds.GetRasterBand(i + 1).ReadAsArray().astype(np.float32) for i, n in enumerate(names)}
rows, cols = ds.RasterYSize, ds.RasterXSize

for index in ["DPRI", "DPDI", "DPMI"]:
    combo = ["VV", "VH", index]
    X = np.stack([bands[n].ravel() for n in combo], axis=1)

    # gueltige Pixel (kein NaN, nicht alles 0)
    valid = np.all(np.isfinite(X), axis=1) & np.any(X != 0, axis=1)

    Xv = X[valid]
    # VV und VH in dB umrechnen (lineare Rueckstreuung ist extrem schief verteilt)
    Xv[:, :2] = 10 * np.log10(np.maximum(Xv[:, :2], 1e-6))
    # Ausreisser abschneiden, sonst bekommen sie eigene Cluster
    lo, hi = np.percentile(Xv, [1, 99], axis=0)
    Xv = np.clip(Xv, lo, hi)
    # Merkmale standardisieren, damit alle Kanaele gleich gewichtet sind
    Xv = (Xv - Xv.mean(axis=0)) / Xv.std(axis=0)

    # auf Stichprobe trainieren (schneller), dann alle Pixel klassifizieren
    rng = np.random.default_rng(0)
    sample = Xv[rng.choice(len(Xv), min(200_000, len(Xv)), replace=False)]
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=0).fit(sample)

    labels = np.zeros(rows * cols, dtype=np.uint8)  # 0 = NoData
    labels[valid] = km.predict(Xv) + 1              # Klassen 1..N
    labels = labels.reshape(rows, cols)

    out_name = f"question5_kmeans_{'_'.join(combo)}.tif"
    out = gdal.GetDriverByName("GTiff").Create(out_name, cols, rows, 1, gdal.GDT_Byte)
    out.SetGeoTransform(ds.GetGeoTransform())
    out.SetProjection(ds.GetProjection())
    out.GetRasterBand(1).WriteArray(labels)
    out.GetRasterBand(1).SetNoDataValue(0)
    out = None
    print("geschrieben:", out_name)

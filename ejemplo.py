import funciones_analisis as fa
import numpy as np

A_test = np.array([
    [0.0, 377.0,   0.0,   0.0],
    [-0.8,  -0.05,   0.8,   0.0],
    [ 0.0,   0.0,  0.0, 377.0],
    [ 0.2,   0.0, -0.2,   -0.01]
])

etiqueta = [
    'G1.delta',
    'G1.omega',
    'G2.delta',
    'G2.omega'
]

A_test, etiqueta = fa.configuracion_powerfactory()

vp, vpd, vpi = fa.procesamiento_matriz_estado(A=A_test)

resumen = fa.resumen_resultados(
    A=A_test,
    vp = vp,
    vpd = vpd,
    vpi = vpi,
    etiqueta = etiqueta,
    selector_estado = ['speed', 'phi']
)

fa.reporte_sistema(reporte=resumen)
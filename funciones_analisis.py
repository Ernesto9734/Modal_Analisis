import numpy as np
import pandas as pd
import math, cmath
from pathlib import Path
from datetime import datetime

# Directorio raíz donde reside este script
DIRECTORIO_SCRIPT = Path(__file__).resolve().parent

# Nombre de carpeta donde se contendrán los resultados generados
CARPETA_REPORTES = DIRECTORIO_SCRIPT / "reportes"

def procesamiento_matriz_estado(A: np.ndarray):
    vp, vpd = np.linalg.eig(A)
    vpi = np.linalg.inv(vpd)
    return (vp, vpd, vpi)

# Umbral numérico para considerar un valor como cero en operaciones en coma flotante
TOL = 1e-7

def determinar_tipo_estabilidad(autovalor: complex) -> str:
    parte_real = autovalor.real
    if parte_real < -TOL:
        return 'Estable'
    elif parte_real > TOL:
        return 'Inestable'
    else:
        return 'Estabilidad Marginal'

def determinar_tipo_oscilacion(autovalor: complex) -> str:
    if abs(autovalor.imag) > TOL:
        return 'Oscilatorio'
    else:
        return 'No oscilatorio'

def calcular_magnitud(autovalor: complex) -> float:
    return abs(autovalor)

def calcular_constante_tiempo(autovalor: complex) -> float:
    parte_real = autovalor.real
    # Tau solo tiene sentido físico práctico para la componente real
    if abs(parte_real) < TOL:
        return np.inf
    return abs(1.0 / parte_real)

def calcular_factor_amortiguamiento(autovalor: complex) -> float:
    mag = abs(autovalor)
    if mag < TOL:
        return 0.0
    # Convención estándar: zeta = -real / magnitud
    # Si alpha < 0 -> zeta > 0 (Amortiguado)
    # Si alpha > 0 -> zeta < 0 (Inestable / Desamortiguado)
    return -autovalor.real / mag

def calcular_frecuencia_Hz(autovalor: complex) -> float:
    parte_imaginaria = abs(autovalor.imag)
    if parte_imaginaria < TOL:
        return 0.0
    return parte_imaginaria / (2.0 * math.pi)

def calcular_periodo(autovalor: complex) -> float:
    parte_imaginaria = abs(autovalor.imag)
    if parte_imaginaria < TOL:
        return np.inf
    return (2.0 * math.pi) / parte_imaginaria

def analisis_vp(vp: complex) -> dict:
    """
    Analiza un autovalor (vp) y retorna un diccionario con sus propiedades físicas y dinámicas.
    """
    return {
        "autovalor": vp,
        "estabilidad": determinar_tipo_estabilidad(autovalor=vp),
        "tipo_oscilacion": determinar_tipo_oscilacion(autovalor=vp),
        "magnitud": calcular_magnitud(autovalor=vp),
        "constante_tiempo_tau": calcular_constante_tiempo(autovalor=vp),
        "factor_amortiguamiento_zeta": calcular_factor_amortiguamiento(autovalor=vp),
        "frecuencia_fd_hz": calcular_frecuencia_Hz(autovalor=vp),
        "periodo_T": calcular_periodo(autovalor=vp)
    }

def presentar_resultados_autovalores(dict_autovalor: dict) -> str:

    str_autovalores = (
        f"{'='*120}\n"
        f"----Propiedades del Autovalor: {dict_autovalor['autovalor']:.3f}----\n"
        f"{'='*120}\n"
        f"Tipo Autovalor (Estabilidad): - {dict_autovalor["estabilidad"]} -\n"
        f"Tipo Autovalor (Oscilacion): - {dict_autovalor["tipo_oscilacion"]} -\n"
        f"Magnitud: {dict_autovalor["magnitud"]:.3f}\n"
        f"Frecuencia: {dict_autovalor["frecuencia_fd_hz"]:.3f} Hz\n"
        f"Periodo: {dict_autovalor["periodo_T"]:.3f} s\n"
        f"Constante de Tiempo: {dict_autovalor["constante_tiempo_tau"]:.3f} s\n"
        f"Factor de Amortiguamiento: {dict_autovalor["factor_amortiguamiento_zeta"]*100:.3f} %\n"
    )

    return str_autovalores

def presentar_resultados_forma_modal(
        dict_autovalor: dict, 
        vector_derecho: list[complex], 
        etiqueta: list[str], 
        selector_estado: list[str] | None = None
    ) -> str:

    vd_magnitud = np.array([round(abs(i),2) for i in vector_derecho])
    vd_angulo =  np.array([round(cmath.phase(i)*180/math.pi,2) for i in vector_derecho])

    dict_modal = {
        'Estado' : etiqueta,
        'Magnitud Forma Modal' : vd_magnitud,
        'Angulo Forma Modal' : vd_angulo
    }

    tabla_forma_modal = pd.DataFrame(dict_modal).sort_values('Magnitud Forma Modal', ascending=False)

    tabla_forma_modal['Magnitud Relativa al Maximo'] = round(
        tabla_forma_modal['Magnitud Forma Modal']
        / tabla_forma_modal['Magnitud Forma Modal'].max()*100,3
    )

    tabla_forma_modal['Participacion Modal'] = round(
            tabla_forma_modal['Magnitud Forma Modal']
            / tabla_forma_modal['Magnitud Forma Modal'].sum()*100,3
    )

    str_forma_modal = (
        f"{'='*120}\n"
        f"----Manifestacion de los Modos en los Estados: {dict_autovalor['autovalor']:.3f}----\n"
        f"Significado: Si este modo se activa ¿Cómo se manifiesta físicamente en cada una de las variables de estado del sistema?\n"
        f"{'='*120}\n"
        f"{tabla_forma_modal}"
    )

    str_modal_estados = ''

    if selector_estado is not None:

        str_modal_estados += (
            f"\n{'-'*120}\n"
            f"----Caracterización de Forma Modal por Estados----\n"
            f"{'-'*120}\n"
        )

        for i in selector_estado:

            selectores = []
            for j in etiqueta:
                if i in j:
                    selectores.append(j)

            tabla_por_estado = tabla_forma_modal[tabla_forma_modal['Estado'].isin(selectores)]
            tabla_por_estado['Magnitud Relativa al Maximo'] = round(
                tabla_por_estado['Magnitud Forma Modal']
                / tabla_por_estado['Magnitud Forma Modal'].max()*100,3
            )
            tabla_por_estado['Participacion Modal'] = round(
                            tabla_por_estado['Magnitud Forma Modal']
                            / tabla_por_estado['Magnitud Forma Modal'].sum()*100,3
            )
            str_modal_estados += (
                f"Estado: {i} \n"
                f"{tabla_por_estado}\n"
                f"\n{'-'*120}\n"
            )

    str_forma_modal += str_modal_estados

    return str_forma_modal

def presentar_resultados_observabilidad(
        dict_autovalor: dict, 
        vector_izquierdo: list[complex], 
        etiqueta: list[str], 
        selector_estado: list[str] | None = None
    ) -> str:

    vd_magnitud = np.array([round(abs(i),2) for i in vector_izquierdo])

    dict_modal = {
        'Estado' : etiqueta,
        'Magnitud Observabilidad' : vd_magnitud
    }

    tabla_observabilidad = pd.DataFrame(dict_modal).sort_values('Magnitud Observabilidad', ascending=False)

    tabla_observabilidad['Magnitud Relativa al Maximo'] = round(
        tabla_observabilidad['Magnitud Observabilidad']
        / tabla_observabilidad['Magnitud Observabilidad'].max()*100,3
    )

    tabla_observabilidad['Participacion Observabilidad'] = round(
            tabla_observabilidad['Magnitud Observabilidad']
            / tabla_observabilidad['Magnitud Observabilidad'].sum()*100,3
    )

    str_observabilidad = (
        f"{'='*120}\n"
        f"----Observabilidad del Modo desde los Estados: {dict_autovalor['autovalor']:.3f}----\n"
        f"Significado: ¿Qué tan bien puedo detectar el modo desde este estado?\n"
        f"{'='*120}\n"
        f"{tabla_observabilidad}"
    )

    str_observabilidad_estados = ''

    if selector_estado is not None:

        str_observabilidad_estados += (
            f"\n{'-'*120}\n"
            f"----Caracterización de la Observabilidad por Estados----\n"
            f"{'-'*120}\n"
        )

        for i in selector_estado:

            selectores = []
            for j in etiqueta:
                if i in j:
                    selectores.append(j)

            tabla_por_estado = tabla_observabilidad[tabla_observabilidad['Estado'].isin(selectores)]
            tabla_por_estado['Magnitud Relativa al Maximo'] = round(
                tabla_por_estado['Magnitud Observabilidad']
                / tabla_por_estado['Magnitud Observabilidad'].max()*100,3
            )
            tabla_por_estado['Participacion Observabilidad'] = round(
                            tabla_por_estado['Magnitud Observabilidad']
                            / tabla_por_estado['Magnitud Observabilidad'].sum()*100,3
            )
            str_observabilidad_estados += (
                f"Estado: {i} \n"
                f"{tabla_por_estado}\n"
                f"\n{'-'*120}\n"
            )
    str_observabilidad += str_observabilidad_estados

    return str_observabilidad

def presentar_resultados_participacion(
        valores_propios : list[complex],
        vector_izquierdo : list[complex],
        vector_derecho : list[complex],
        estados : list[str],
        nombre_archivo : str | None = None
) -> str:

    # 2. Producto elemento a elemento (Hadamard) de V.T y U
    # V.T alinea la fila i (autovector izquierdo del modo i) con la columna i de U
    P_complejo = vector_derecho * vector_izquierdo.T

    # 3. Magnitud absoluta de los factores de participación
    P_abs = np.abs(P_complejo)

    # 4. Normalización porcentual por columna (por cada modo i)
    # np.sum(P_abs, axis=0) suma sobre todas las filas (estados) para cada columna (modo)
    P_porcentaje = (P_abs / np.sum(P_abs, axis=0)) * 100

    # 5. Redondeo a 3 decimales
    factor_participacion = np.round(P_porcentaje, 3)

    # Extracto de etiquetas para columnas
    autovalores = [f"Autovalor - {i+1}: {round(abs(valores_propios[i]),2)}<{round(cmath.phase(valores_propios[i])*180/math.pi,2)}" for i in range(len(valores_propios))]

    # 6. Construcción del DataFrame estructurado
    df_participacion = pd.DataFrame(
        factor_participacion, 
        index=estados, 
        columns=autovalores
    )

    # Encabezado explicativo
    str_participacion = (
    f"{'='*120}\n"
    f"---- Factor de Participación Asociado al Sistema (%) ----\n"
    f"Significado: Medida selectiva de la relación bidireccional entre estados y modos.\n"
    f"{'='*120}\n"
    f"{df_participacion}\n"
    )

    """Genera un archivo de texto plano (.txt) con un resumen del sistema."""
    CARPETA_REPORTES.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    
    if nombre_archivo is None:
        nombre_final = f"factores_participacion_{timestamp}.csv"
    else:
        p = Path(nombre_archivo)
        nombre_final = f"{p.stem}_{timestamp}.csv" if p.suffix == "" else f"{p.stem}_{timestamp}{p.suffix}"
                    
    """
    Genera un archivo de texto plano (.csv) con un resumen del sistema
    y la lista numerada de los estados registrados.
    """
    ruta_salida = CARPETA_REPORTES / nombre_final
            
    try:
        df_participacion.reset_index().rename(columns={'index': 'Estado'}).to_csv(ruta_salida, index=False)
        print(f"✅ Guardado correctamente en: {ruta_salida}")
    except PermissionError:
        print("❌ Error: Cierra 'resumen_autovalores.csv' si está abierto.")
    except Exception as e:
        print(f"❌ Error guardando CSV de autovalores: {e}")

    return str_participacion
    

def presentar_resultados_matriz_estado(A: np.ndarray, estados: list[str]) -> str:

    estados_etiquetas = ""
    for i in range(len(estados)):
        estados_etiquetas += f"{i+1} -> {estados[i]}\n"

    str_estado = (
        f"{'='*120}\n"
        f"----Propiedades de la Matriz de Estado:----\n"
        f"{'='*120}\n"
        f"Cantidad de Estados: {A.shape[0]}\n"
        f"Rango de la Matriz de Estado: {np.linalg.matrix_rank(A)}\n"
        f"Estados:\n"
        f"{estados_etiquetas}"
    )

    return str_estado

def resumen_resultados(
        A: np.ndarray, 
        vp: list[np.ndarray], 
        vpd: list[np.ndarray], 
        vpi: list[np.ndarray], 
        etiqueta: list[str],
        selector_estado: list[str]
    ) -> str:

    str_modos = ''
    str_modos += (presentar_resultados_matriz_estado(A=A, estados=etiqueta) + '\n')
    str_modos += (presentar_resultados_participacion(valores_propios=vp, vector_derecho=vpd, vector_izquierdo=vpi, estados=etiqueta) + '\n')
    for i in range(len(vp)):
        dict_vp = analisis_vp(vp[i])
        str_modos += presentar_resultados_autovalores(dict_autovalor=dict_vp)
        str_modos += presentar_resultados_forma_modal(dict_autovalor=dict_vp, vector_derecho=vpd[:,i], etiqueta=etiqueta, selector_estado=selector_estado)
        str_modos += (presentar_resultados_observabilidad(dict_autovalor=dict_vp, vector_izquierdo=vpi[i,:], etiqueta=etiqueta, selector_estado=selector_estado) + '\n')

    return str_modos

def reporte_sistema(reporte : str, nombre_archivo: str = "reporte_sistema.txt"
    ) -> None:

        """
        Genera un archivo de texto plano (.txt) con un resumen del sistema.
        """

        CARPETA_REPORTES.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if nombre_archivo is None:
            nombre_final = f"reporte_sistema_{timestamp}.txt"
        else:
            p = Path(nombre_archivo)
            nombre_final = f"{p.stem}_{timestamp}.txt" if p.suffix == "" else f"{p.stem}_{timestamp}{p.suffix}"

        """
        Genera un archivo de texto plano (.txt) con un resumen del sistema
        y la lista numerada de los estados registrados.
        """
        ruta_salida = CARPETA_REPORTES / nombre_final

        with open(ruta_salida, "w", encoding="utf-8") as f:
            f.write(reporte)

        print(f"✅ Reporte exportado exitosamente en: {ruta_salida}")

def configuracion_powerfactory( 
        nombre_archivo_matriz: str = 'Amat', 
        extension_matriz : str = '.mtl',
        nombre_archivo_etiquetas: str = 'VariableToIdx_Amat',
        extension_etiquetas : str = '.txt'
    ):

        # 1. Extraer nombres/etiquetas de los estados
        nombre_estados = extraer_nombre_estados_powerfactory(
            nombre_archivo = nombre_archivo_etiquetas, 
            extension= extension_etiquetas
        )

        # 2. Extraer y reconstruir la matriz de estado A
        matriz_estado = extraer_matriz_estados_powerfactory(
            nombre_archivo=nombre_archivo_matriz, 
            extension= extension_matriz
        )

        # 3. Retornar la clase instanciada con los datos procesados
        return (matriz_estado, nombre_estados)

def extraer_nombre_estados_powerfactory(
        nombre_archivo: str, 
        extension: str
    ) -> list[str]:

        ruta = DIRECTORIO_SCRIPT / f"{nombre_archivo}{extension}"
        
        datos = pd.read_csv(
            ruta,
            sep=r"\s{2,}",      # Separa por 2 o más espacios
            engine="python",
            skiprows=2,         # Omite los encabezados de PowerFactory
            header=None,
            names=["indice", "modelo", "estado"]
        )
        
        # Extraer el elemento después del último backslash o manejar strings simples
        datos["tipo_modelo"] = (
            datos["modelo"]
            .str.split(r"\\")
            .str[-1]  # Toma el último elemento por seguridad
            .str.split(r"\.")
            .str[0]
        )

        # Formatear la etiqueta final: "Modelo-Estado" (ej. "G1-speed")
        datos['estado'] = datos['tipo_modelo'] + '-' + datos['estado'].str.strip('"')
        
        return datos['estado'].to_list()

def extraer_matriz_estados_powerfactory(
        nombre_archivo: str, 
        extension: str
    ) -> np.ndarray:

        ruta = DIRECTORIO_SCRIPT / f"{nombre_archivo}{extension}"

        datos = pd.read_csv(
            ruta,
            sep=r"\s+",
            header=None,
            names=["fila", "columna", "valor"]
        )

        n_filas = int(datos["fila"].max())
        n_columnas = int(datos["columna"].max())

        A = np.zeros((n_filas, n_columnas), dtype=float)

        A[
            datos["fila"].to_numpy(dtype=int) - 1,
            datos["columna"].to_numpy(dtype=int) - 1
        ] = datos["valor"].to_numpy()

        return A

# -------------------------------------------------------------------------
# MÉTODOS DE CÁLCULO Y CONSTRUCTOR MANUAL
# -------------------------------------------------------------------------

def configuracion_manual( 
        matriz_estado: np.ndarray | list[list[float]], 
        lista_estados: list[str]
    ):
        """
        Crea una instancia de Modelo_Sistema directamente desde objetos en memoria
        (matriz de NumPy o lista 2D, y lista de etiquetas).

        Parametros
        ----------
        matriz_estado : np.ndarray o list[list[float]]
            Matriz de estado A del sistema linealizado (dx/dt = A*x).
        lista_estados : list[str]
            Lista de nombres o etiquetas correspondientes a cada variable de estado.
        """
        # Convertir a numpy.ndarray si se pasa una lista de Python
        if not isinstance(matriz_estado, np.ndarray):
            matriz_estado = np.array(matriz_estado, dtype=float)

        return (matriz_estado, lista_estados)
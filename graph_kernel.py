import numpy as np
import os
import pandas as pd

# def calcular_vector_caracteristicas(A: np.ndarray) -> np.ndarray:
#     A = A.copy()
#     np.fill_diagonal(A, 0)
#     n = A.shape[0]
#     u = np.ones((n, 1))
#     t = np.diag(np.linalg.matrix_power(A, 3)).reshape(-1, 1)
#     k = A @ u
#     k1 = k - 1
#     k2 = k - 2
#     m = np.sum(k) / 2

#     A2 = A @ A
#     Q = A2 * A
#     P = 0.5 * A2 * (A2 - 1)
#     A5 = np.linalg.matrix_power(A, 5)
#     q = np.diag(A5).reshape(-1, 1)
#     A3 = np.linalg.matrix_power(A, 3)

#     b = 0.5 * (q - 5 * t - 2 * (t * k2) - 2 * (Q @ k2) - 2 * ((0.5 * A @ t) - Q @ u))
#     R = (1 / 6) * A2 * (A2 - 1) * (A2 - 2)

#     scalars = [
#         float(n), float(m),
#         float(F1 := 0.5 * (k.T @ k1)),
#         float(F2 := (1 / 6) * np.sum(t)),
#         float(F3 := 0.5 * (k1.T @ A @ k1) - 3 * F2),
#         float(F4 := (1 / 6) * ((k * k1).T @ k2)),
#         float(F5 := (1 / 8) * (np.trace(np.linalg.matrix_power(A, 4)) - 4 * F1 - 2 * m)),
#         float(F6 := 0.5 * (t.T @ k2)),
#         float(F7 := 0.25 * (u.T @ (Q * (Q - A)) @ u)),
#         float(F8 := (1 / 10) * (np.trace(A5) - 10 * F6 - 30 * F2)),
#         float(F9 := 0.25 * ((k2 * (k2 - 1)).T @ t)),
#         float(F10 := (u.T @ (P - np.diag(np.diag(P))) @ k2) - 2 * F7),
#         float(F11 := 0.5 * (k2.T @ Q @ k2) - 2 * F7),
#         float(F12 := 0.5 * (u.T @ (A2 - np.diag(np.diag(A2))) @ t) - 6 * F2 - 2 * F6 - 4 * F7),
#         float(F13 := 0.25 * (t.T @ (0.5 * t - 1)) - 2 * F7),
#         float(F14 := 0.5 * (u.T @ (Q * (A3 * A)) @ u) - 9 * F2 - 2 * F6 - 4 * F7),
#         float(F15 := (1 / 12) * (np.trace(np.linalg.matrix_power(A, 6)) - 2 * m - 12 * F1 - 24 * F2 - 6 * F3 - 12 * F4 - 48 * F5 - 36 * F7 - 12 * F10 - 24 * F13)),
#         float(F16 := (k2.T @ b) - 2 * F14),
#         float(F17 := 0.5 * (u.T @ (R * A) @ u)),
#         float(F18 := 0.5 * (u.T @ (P - np.diag(np.diag(P))) @ t) - 6 * F7 - 2 * F14 - 6 * F17)
#     ]

#     return np.array(scalars)

def calcular_vector_caracteristicas(A: np.ndarray) -> np.ndarray:
    A = A.copy()
    np.fill_diagonal(A, 0)
    n = A.shape[0]
    u = np.ones((n, 1))
    t = np.diag(np.linalg.matrix_power(A, 3)).reshape(-1, 1)
    k = A @ u
    k1 = k - 1
    k2 = k - 2
    m = (np.sum(k) / 2).item()

    A2 = A @ A
    Q = A2 * A
    P = 0.5 * A2 * (A2 - 1)
    A3 = np.linalg.matrix_power(A, 3)
    A5 = np.linalg.matrix_power(A, 5)
    q = np.diag(A5).reshape(-1, 1)

    b = 0.5 * (q - 5 * t - 2 * (t * k2) - 2 * (Q @ k2) - 2 * ((0.5 * A @ t) - Q @ u))
    R = (1/6) * A2 * (A2 - 1) * (A2 - 2)

    # Definimos los F corregidos
    F1 = (0.5 * (k.T @ k1)).item()
    F2 = ((1/6) * np.sum(t)).item()
    F3 = (0.5 * (k1.T @ A @ k1) - 3 * F2).item()
    F4 = ((1/6) * ((k * k1).T @ k2)).item()
    F5 = ((1/8) * (np.trace(np.linalg.matrix_power(A, 4)) - 4 * F1 - 2 * m)).item()
    F6 = (0.5 * (t.T @ k2)).item()
    F7 = (0.25 * (u.T @ (Q * (Q - A)) @ u)).item()
    F8 = ((1/10) * (np.trace(A5) - 10 * F6 - 30 * F2)).item()
    F9 = (0.25 * ((k2 * (k2 - 1)).T @ t)).item()
    F10 = ((u.T @ (P - np.diag(np.diag(P))) @ k2) - 2 * F7).item()
    F11 = (0.5 * (k2.T @ Q @ k2) - 2 * F7).item()
    F12 = (0.5 * (u.T @ (A2 - np.diag(np.diag(A2))) @ t) - 6 * F2 - 2 * F6 - 4 * F7).item()
    F13 = (0.25 * (t.T @ (0.5 * t - 1)) - 2 * F7).item()
    F14 = (0.5 * (u.T @ (Q * (A3 * A)) @ u) - 9 * F2 - 2 * F6 - 4 * F7).item()
    F15 = ((1/12) * (np.trace(np.linalg.matrix_power(A, 6)) - 2 * m - 12 * F1 - 24 * F2 - 6 * F3 - 12 * F4 - 48 * F5 - 36 * F7 - 12 * F10 - 24 * F13)).item()
    F16 = ((k2.T @ b) - 2 * F14).item()
    F17 = (0.5 * (u.T @ (R * A) @ u)).item()
    F18 = (0.5 * (u.T @ (P - np.diag(np.diag(P))) @ t) - 6 * F7 - 2 * F14 - 6 * F17).item()

    scalars = [
        float(n), float(m),
        F1, F2, F3, F4, F5, F6, F7, F8, F9, F10,
        F11, F12, F13, F14, F15, F16, F17, F18
    ]

    return np.array(scalars)


def distancia_kernel(S1: np.ndarray, S2: np.ndarray) -> float:
    dist = np.sqrt(np.mean((S1 - S2) ** 2))
    dist_relativa = dist / (np.linalg.norm(S1) + 1e-12)  # Evitar división por cero
    return dist, dist_relativa
    # return np.sqrt(np.mean((S1 - S2) ** 2))

def cargar_matriz_desde_txt(filepath: str) -> np.ndarray:
    return np.loadtxt(filepath)

def clasificar_distancia_relativa(dist_rel: float) -> str:
    """
    Clasifica una distancia relativa en una categoría interpretativa.
    
    Parámetros:
    ------------
    dist_rel : float
        Distancia relativa normalizada.

    Retorna:
    ---------
    categoria : str
        Descripción textual del grado de preservación.
    """
    if dist_rel < 0.1:
        return "Excelente (casi idéntico)"
    elif dist_rel < 0.3:
        return "Buena (ligeros cambios)"
    elif dist_rel < 0.7:
        return "Moderada (cambios apreciables)"
    elif dist_rel < 1.0:
        return "Mala (estructura bastante afectada)"
    else:
        return "Muy mala (estructura fuertemente destruida)"


# main function for testing
def main_test():
    archivo_principal = "karate_coarseNet_reduced_adj_matrix.txt"
    path_principal = os.path.join("./subgrafos_txt", archivo_principal)
    if os.path.exists(path_principal):
        matriz = cargar_matriz_desde_txt(path_principal)
        print(f"\nSubgrafo extraído de: {archivo_principal}")
        print(matriz)
        S_vector = calcular_vector_caracteristicas(matriz)
        print(f"\nVector de características S para {archivo_principal}:")
        print(S_vector)
        print(f"\nLongitud del vector S: {len(S_vector)}")
    else:
        print(f"\nArchivo no encontrado: {archivo_principal}")

    directorio = "./subgrafos_txt"
    archivos = sorted([f for f in os.listdir(directorio) if f.endswith(".txt")])

    matriz_original = None
    for archivo in archivos:
        if "original" in archivo:
            matriz_original = cargar_matriz_desde_txt(os.path.join(directorio, archivo))
            break
    if matriz_original is None:
        raise FileNotFoundError("No se encontró un archivo con 'original' en el nombre.")
    S1 = calcular_vector_caracteristicas(matriz_original)

    distancias = []
    distancias_relativas = []
    nombres = []
    

    for archivo in archivos:
        if "original" in archivo:
            continue
        path = os.path.join(directorio, archivo)
        A_sub = cargar_matriz_desde_txt(path)
        Si = calcular_vector_caracteristicas(A_sub)
        dist, dist_rel = distancia_kernel(S1, Si)
        distancias.append(dist)
        distancias_relativas.append(dist_rel)
        nombres.append(archivo)

    df = pd.DataFrame({
        "Subgrafo": nombres, 
        "Distancia al original": distancias,
        "Distancia relativa al original": distancias_relativas
        })
    df = df.sort_values(by="Distancia al original")
    print("\nDistancias al grafo original:\n")
    print(df.to_string(index=False))

# principal function for processing multiple directories
def main():
    '''Main function to process multiple directories and calculate distances to the original graph.
    '''

    # directorio = "./bigsNetworks_final_test_RESULT"
    directorio = "./academicNetworks_final_test_RESULT"
    subdirectorios = [os.path.join(directorio, d) for d in os.listdir(directorio) if os.path.isdir(os.path.join(directorio, d))]

    for subdirectorio in subdirectorios:
        print(f"\nProcesando subdirectorio: {subdirectorio}")
        archivos = sorted([f for f in os.listdir(subdirectorio) if f.endswith(".txt")])

        matriz_original = None
        for archivo in archivos:
            if "original" in archivo:
                matriz_original = cargar_matriz_desde_txt(os.path.join(subdirectorio, archivo))
                break
        if matriz_original is None:
            print(f"No se encontró un archivo con 'original' en el nombre en {subdirectorio}.")
            continue

        S1 = calcular_vector_caracteristicas(matriz_original)

        distancias = []
        distancias_relativas = []
        nombres = []

        for archivo in archivos:
            if "original" in archivo:
                continue
            path = os.path.join(subdirectorio, archivo)
            A_sub = cargar_matriz_desde_txt(path)
            Si = calcular_vector_caracteristicas(A_sub)
            dist, dist_rel = distancia_kernel(S1, Si)
            distancias.append(dist)
            distancias_relativas.append(dist_rel)
            nombres.append(archivo)

        # Create a DataFrame to store distances
        df = pd.DataFrame({
        "Subgrafo": nombres, 
        "Distancia al original": distancias,
        "Distancia relativa al original": distancias_relativas
        })

        # Agregar clasificación
        df["Clasificación"] = df["Distancia relativa al original"].apply(clasificar_distancia_relativa)

        df = df.sort_values(by="Distancia al original")
        print(f"\nDistancias al grafo original en {subdirectorio}:\n")
        print(df.to_string(index=False))

        # Save distances to an Excel file
        output_file = os.path.join(subdirectorio, f"{os.path.basename(subdirectorio)}_distances.xlsx")
        df.to_excel(output_file, index=False)
        print(f"Distancias guardadas en: {output_file}")

if __name__ == "__main__":
    main()
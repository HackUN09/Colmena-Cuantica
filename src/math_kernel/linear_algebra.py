import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.linalg import svd

def apply_pca_reduction(df, n_components=0.95):
    """
    Aplica PCA para reducir la dimensionalidad conservando el % de varianza especificado.
    Ayuda a eliminar el ruido común entre los 100 activos.
    """
    try:
        data = df.dropna()
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(data)
        return {
            'n_components': pca.n_components_,
            'explained_variance': np.sum(pca.explained_variance_ratio_),
            'components': components
        }
    except:
        return None

def apply_svd_denoising(df):
    """
    Descomposición en Valores Singulares (SVD).
    Permite reconstruir la matriz de retornos filtrando los valores singulares más pequeños (ruido).
    """
    try:
        data = df.dropna().values
        U, s, Vt = svd(data, full_matrices=False)
        # Filtrar valores singulares (mantener solo los top 50%)
        k = len(s) // 2
        s_filtered = np.zeros_like(s)
        s_filtered[:k] = s[:k]
        
        # Reconstruir matriz denoiseada
        data_denoised = U @ np.diag(s_filtered) @ Vt
        return pd.DataFrame(data_denoised, index=df.dropna().index, columns=df.columns)
    except:
        return None

def apply_rmt_filtering(df):
    """
    Teoría de Matrices Aleatorias (Random Matrix Theory).
    Filtra la matriz de correlación eliminando los autovalores que caen dentro de la
    distribución de Marchenko-Pastur (autovalores debidos al puro azar).
    """
    try:
        data = df.dropna()
        corr = data.corr().values
        eigenvalues, eigenvectors = np.linalg.eigh(corr)
        
        # Teórico Marchenko-Pastur limit
        # q = T / N (observaciones / variables)
        T, N = data.shape
        q = T / N
        sigma2 = 1.0 # Varianza de la matriz de correlación
        max_lambda = sigma2 * (1 + (1/q)**0.5)**2
        
        # Filtrado: Mantener solo autovalores por encima del límite teórico de ruido
        eigenvalues_filtered = np.where(eigenvalues > max_lambda, eigenvalues, 0)
        
        # Reconstruir matriz de correlación filtrada
        corr_filtered = eigenvectors @ np.diag(eigenvalues_filtered) @ eigenvectors.T
        # Asegurar diagonal = 1
        np.fill_diagonal(corr_filtered, 1.0)
        
        return pd.DataFrame(corr_filtered, index=df.columns, columns=df.columns)
    except:
        return None

if __name__ == "__main__":
    # Test con matriz de correlación sintética con ruido
    data = pd.DataFrame(np.random.randn(200, 10))
    print("Aplicando PCA...")
    pca_res = apply_pca_reduction(data)
    print(f"Componentes necesarios para 95% varianza: {pca_res['n_components']}")
    
    print("\nAplicando RMT Filtering...")
    rmt_corr = apply_rmt_filtering(data)
    print(f"Suma de autovalores significativos: {np.trace(rmt_corr.values):.2f}")

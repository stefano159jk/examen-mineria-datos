import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA

def mostrar_fase2(df):
    st.header("🟡 Fase 2 — Segmentación (Clustering)")

    # ── Preparar datos numéricos ─────────────────────────────────
    df_num = df.select_dtypes(include=np.number).dropna()

    if df_num.shape[1] < 2:
        st.error("❌ El dataset necesita al menos 2 columnas numéricas para clustering.")
        return

    st.subheader("📋 Columnas numéricas usadas para clustering")
    st.write(list(df_num.columns))

    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_num)

    # Muestra para clustering jerárquico (costoso con 45k filas)
    muestra_n = min(2000, len(X_scaled))
    idx_muestra = np.random.choice(len(X_scaled), muestra_n, replace=False)
    X_muestra = X_scaled[idx_muestra]

    st.info(f"ℹ️ Se usan **{muestra_n} filas** como muestra para clustering jerárquico (rendimiento). K-Means usa todo el dataset.")

    # ════════════════════════════════════════════════════════════
    # SECCIÓN 1 — K-MEANS
    # ════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("🔵 K-Means Clustering")
    st.markdown("""
    **K-Means** agrupa los datos en **K clústeres** asignando cada punto
    al centroide más cercano e iterando hasta que los grupos sean estables.
    
    - Rápido y escalable
    - Requiere definir K de antemano
    - Funciona mejor con variables numéricas escaladas
    """)

    # Método del codo
    st.markdown("#### 📐 Método del Codo — ¿Cuántos clústeres elegir?")
    max_k = st.slider("Rango de K a evaluar", 2, 10, 6)

    if st.button("🔍 Calcular Método del Codo + Silhouette"):
        inercias = []
        silhouettes = []
        ks = list(range(2, max_k + 1))

        with st.spinner("Calculando..."):
            for k in ks:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(X_scaled)
                inercias.append(km.inertia_)
                sil = silhouette_score(X_scaled, labels, sample_size=3000, random_state=42)
                silhouettes.append(round(sil, 4))

        st.session_state['ks'] = ks
        st.session_state['inercias'] = inercias
        st.session_state['silhouettes'] = silhouettes

        # Gráfica codo
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(ks, inercias, 'bo-', linewidth=2, markersize=8)
        axes[0].set_title('Método del Codo', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Número de Clústeres (K)')
        axes[0].set_ylabel('Inercia')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(ks, silhouettes, 'gs-', linewidth=2, markersize=8)
        axes[1].set_title('Índice Silhouette por K', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Número de Clústeres (K)')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].grid(True, alpha=0.3)

        mejor_k = ks[np.argmax(silhouettes)]
        axes[1].axvline(x=mejor_k, color='red', linestyle='--', label=f'Mejor K={mejor_k}')
        axes[1].legend()

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Tabla silhouette
        df_sil = pd.DataFrame({'K': ks, 'Inercia': inercias, 'Silhouette': silhouettes})
        st.dataframe(df_sil, use_container_width=True)

        mejor_sil = max(silhouettes)
        st.success(f"✅ Mejor K sugerido: **{mejor_k}** con Silhouette = **{mejor_sil:.4f}**")
        st.session_state['mejor_k'] = mejor_k

    # ── Aplicar K-Means con K elegido ────────────────────────────
    st.markdown("#### ▶️ Aplicar K-Means")
    k_elegido = st.number_input("Elige el K para aplicar:", min_value=2, max_value=10,
                                 value=st.session_state.get('mejor_k', 3))

    if st.button("🚀 Ejecutar K-Means"):
        with st.spinner("Ejecutando K-Means..."):
            km_final = KMeans(n_clusters=int(k_elegido), random_state=42, n_init=10)
            labels_km = km_final.fit_predict(X_scaled)
            sil_final = silhouette_score(X_scaled, labels_km, sample_size=3000, random_state=42)

        df_resultado = df_num.copy()
        df_resultado['Cluster_KMeans'] = labels_km
        st.session_state['df_clusters'] = df_resultado
        st.session_state['labels_km'] = labels_km

        st.success(f"✅ K-Means ejecutado | Silhouette Score: **{sil_final:.4f}**")

        # Interpretar silhouette
        if sil_final >= 0.5:
            st.info("📊 Silhouette ≥ 0.5 → Clústeres **bien definidos**")
        elif sil_final >= 0.25:
            st.info("📊 Silhouette entre 0.25-0.5 → Clústeres **razonables**")
        else:
            st.warning("📊 Silhouette < 0.25 → Clústeres **débiles**, considera otro K")

        # Visualización PCA 2D
        st.markdown("#### 🗺️ Visualización de Clústeres (PCA 2D)")
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        fig2, ax = plt.subplots(figsize=(9, 5))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                             c=labels_km, cmap='tab10',
                             alpha=0.4, s=10)
        plt.colorbar(scatter, ax=ax, label='Clúster')
        ax.set_title(f'K-Means — K={k_elegido} (visualización PCA 2D)', fontsize=13, fontweight='bold')
        ax.set_xlabel('Componente Principal 1')
        ax.set_ylabel('Componente Principal 2')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig2)
        plt.close()

        # Perfiles por clúster
        st.markdown("#### 👥 Perfiles de Clústeres")
        perfil = df_resultado.groupby('Cluster_KMeans').mean().round(2)
        st.dataframe(perfil, use_container_width=True)

        # Tamaño de cada clúster
        st.markdown("#### 📊 Tamaño de cada Clúster")
        tam = df_resultado['Cluster_KMeans'].value_counts().sort_index()
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        ax3.bar(tam.index.astype(str), tam.values,
                color=plt.cm.tab10(np.linspace(0, 1, len(tam))))
        ax3.set_title('Cantidad de clientes por Clúster', fontweight='bold')
        ax3.set_xlabel('Clúster')
        ax3.set_ylabel('Cantidad')
        for i, v in enumerate(tam.values):
            ax3.text(i, v + 50, str(v), ha='center', fontweight='bold')
        st.pyplot(fig3)
        plt.close()

    # ════════════════════════════════════════════════════════════
    # SECCIÓN 2 — CLUSTERING JERÁRQUICO
    # ════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("🟣 Clustering Jerárquico")
    st.markdown("""
    El **Clustering Jerárquico** construye un árbol de fusiones (dendrograma)
    uniendo los puntos más similares progresivamente.
    
    - No requiere definir K de antemano
    - El dendrograma ayuda a decidir el número de clústeres visualmente
    - Más costoso computacionalmente (se usa muestra de datos)
    """)

    if st.button("🌳 Generar Dendrograma"):
        with st.spinner(f"Generando dendrograma con {muestra_n} muestras..."):
            Z = linkage(X_muestra, method='ward')

        fig4, ax4 = plt.subplots(figsize=(12, 5))
        dendrogram(Z, ax=ax4, truncate_mode='lastp', p=30,
                   leaf_rotation=90, leaf_font_size=10, show_contracted=True)
        ax4.set_title(f'Dendrograma — Clustering Jerárquico (muestra {muestra_n} filas)',
                      fontsize=13, fontweight='bold')
        ax4.set_xlabel('Clústeres')
        ax4.set_ylabel('Distancia (Ward)')
        ax4.grid(True, alpha=0.3)
        st.pyplot(fig4)
        plt.close()
        st.info("💡 Corta el dendrograma donde la distancia vertical sea mayor — ahí está el K óptimo.")

    # Aplicar jerárquico
    st.markdown("#### ▶️ Aplicar Clustering Jerárquico")
    k_hier = st.number_input("K para clustering jerárquico:", min_value=2, max_value=10,
                              value=st.session_state.get('mejor_k', 3), key='k_hier')

    if st.button("🚀 Ejecutar Clustering Jerárquico"):
        with st.spinner("Ejecutando clustering jerárquico..."):
            hier = AgglomerativeClustering(n_clusters=int(k_hier))
            labels_hier = hier.fit_predict(X_muestra)
            sil_hier = silhouette_score(X_muestra, labels_hier)

        st.success(f"✅ Clustering Jerárquico ejecutado | Silhouette Score: **{sil_hier:.4f}**")

        # Comparativa
        st.markdown("#### 📊 Comparativa K-Means vs Jerárquico")
        sil_km = st.session_state.get('silhouettes', [None])
        col1, col2 = st.columns(2)
        col1.metric("🔵 K-Means Silhouette", f"{silhouette_score(X_scaled, st.session_state.get('labels_km', labels_hier), sample_size=3000, random_state=42):.4f}" if 'labels_km' in st.session_state else "Ejecuta K-Means primero")
        col2.metric("🟣 Jerárquico Silhouette", f"{sil_hier:.4f}")

        # Perfil jerárquico
        df_hier = df_num.iloc[idx_muestra].copy()
        df_hier['Cluster_Hier'] = labels_hier
        st.markdown("#### 👥 Perfiles — Clustering Jerárquico")
        st.dataframe(df_hier.groupby('Cluster_Hier').mean().round(2), use_container_width=True)

        st.session_state['fase2_completa'] = True
        st.success("✅ Fase 2 completada. Puedes continuar con la Fase 3.")
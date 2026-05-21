import streamlit as st
import pandas as pd
from fases.fase1_particion   import mostrar_fase1
from fases.fase2_clustering  import mostrar_fase2
from fases.fase3_clasificacion import mostrar_fase3
from fases.fase4_evaluacion  import mostrar_fase4

st.set_page_config(
    page_title="Minería de Datos — Examen",
    page_icon="⛏️",
    layout="wide"
)

st.title("⛏️ App de Minería de Datos")
st.markdown("---")

st.sidebar.header("📁 Cargar Dataset")
archivo = st.sidebar.file_uploader("Sube tu CSV o Excel", type=["csv", "xlsx"])

if archivo:
    if archivo.name.endswith(".csv"):
        df = pd.read_csv(archivo, sep=';')
    else:
        df = pd.read_excel(archivo)

    st.sidebar.success(f"✅ {archivo.name} cargado")
    st.sidebar.markdown("---")
    st.sidebar.header("🗂️ Fases")

    fase = st.sidebar.radio("Selecciona la fase:", [
        "🔵 Fase 1 — Partición & Baseline",
        "🟡 Fase 2 — Clustering",
        "🟠 Fase 3 — Clasificación",
        "🔴 Fase 4 — Evaluación Comparativa"
    ])

    if   fase == "🔵 Fase 1 — Partición & Baseline":
        mostrar_fase1(df)
    elif fase == "🟡 Fase 2 — Clustering":
        mostrar_fase2(df)
    elif fase == "🟠 Fase 3 — Clasificación":
        mostrar_fase3()
    elif fase == "🔴 Fase 4 — Evaluación Comparativa":
        mostrar_fase4()
else:
    st.info("👈 Sube tu dataset desde el panel izquierdo para comenzar.")
    st.markdown("""
    ### ¿Cómo usar esta app?
    1. Sube un archivo **CSV o Excel** desde el sidebar
    2. Ejecuta la **Fase 1** primero siempre
    3. Navega por cada fase en orden
    4. La Fase 4 muestra la comparativa final completa
    """)
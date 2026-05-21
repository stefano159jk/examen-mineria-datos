# ⛏️ App de Minería de Datos

Aplicación interactiva desarrollada con **Python + Streamlit** para aplicar
técnicas de minería de datos sobre cualquier dataset CSV o Excel.

---

## 🚀 ¿Qué hace esta app?

Cubre el pipeline completo de Data Science en 4 fases:

| Fase | Contenido |
|------|-----------|
| 🔵 Fase 1 | Partición Train/Val/Test · Data Leakage · Modelo Baseline |
| 🟡 Fase 2 | K-Means · Clustering Jerárquico · Índice Silhouette · Dendrograma |
| 🟠 Fase 3 | Árbol de Decisión · Random Forest · Accuracy · F1 · AUC |
| 🔴 Fase 4 | Curva ROC · Tabla comparativa · Matriz de confusión · Reporte ejecutivo |

---

## 🖥️ Requisitos

- Python 3.11 o superior
- pip

---

## ⚙️ Instalación

### 1. Clona el repositorio
```bash
git clone https://github.com/TU_USUARIO/examen-mineria-datos.git
cd examen-mineria-datos
```

### 2. Instala las dependencias
```bash
pip install -r requirements.txt
```

### 3. Ejecuta la app
```bash
streamlit run app.py
```

### 4. Abre en el navegador

http://localhost:8501

---

## 📁 Estructura del proyecto

examen_mineria/
│
├── app.py                        ← Archivo principal
├── requirements.txt              ← Dependencias
├── README.md                     ← Este archivo
├── .gitignore                    ← Archivos ignorados
│
└── fases/
├── fase1_particion.py        ← Partición + Baseline + Data Leakage
├── fase2_clustering.py       ← K-Means + Clustering Jerárquico
├── fase3_clasificacion.py    ← Árbol de Decisión + Random Forest
└── fase4_evaluacion.py       ← Evaluación comparativa final

---

## 📊 Dataset de ejemplo

Se recomienda usar el dataset **Bank Marketing (UCI)**:
- 🔗 https://archive.ics.uci.edu/static/public/222/bank+marketing.zip
- Archivo a usar: `bank-full.csv`
- Separador: punto y coma (`;`)
- Variable objetivo: columna `y` (yes/no)

---

## 🗂️ Cómo usar la app

1. **Sube tu CSV o Excel** desde el panel lateral izquierdo
2. **Fase 1** → Selecciona la columna objetivo → presiona *Ejecutar*
3. **Fase 2** → Calcula el codo → elige K → ejecuta K-Means y Jerárquico
4. **Fase 3** → Entrena el Árbol de Decisión y el Random Forest
5. **Fase 4** → Revisa la Curva ROC, matrices y el Reporte Ejecutivo

> ⚠️ La Fase 1 siempre debe ejecutarse primero. Las Fases 3 y 4 dependen de ella.

---

## 📋 Requisitos del target (columna objetivo)

La columna objetivo debe ser **categórica**:

| Tipo | Ejemplos | ¿Funciona? |
|------|----------|------------|
| Binaria texto | yes/no · true/false | ✅ |
| Binaria numérica | 0/1 | ✅ |
| Multiclase | alto/medio/bajo | ✅ |
| Numérica continua | edad · saldo | ❌ |

---

## 🛠️ Tecnologías

| Librería | Uso |
|----------|-----|
| `streamlit` | Interfaz web interactiva |
| `pandas` | Manipulación de datos |
| `scikit-learn` | Modelos de ML y métricas |
| `matplotlib` | Gráficas y visualizaciones |
| `seaborn` | Visualizaciones estadísticas |
| `scipy` | Clustering jerárquico |
| `numpy` | Operaciones numéricas |

---

## 👨‍💻 Autor


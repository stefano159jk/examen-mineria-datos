import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

def mostrar_fase1(df):
    st.header("🔵 Fase 1 — Partición, Baseline y Data Leakage")

    # ── 1. Vista previa ──────────────────────────────────────────
    st.subheader("📋 Vista previa del dataset")
    st.dataframe(df.head(10))
    st.write(f"**Filas:** {df.shape[0]} | **Columnas:** {df.shape[1]}")

    # ── 2. Selección de variable objetivo ────────────────────────
    st.subheader("🎯 Selecciona la variable objetivo (target)")
    target_col = st.selectbox("Variable objetivo:", df.columns.tolist())

    # ── 3. Partición ─────────────────────────────────────────────
    st.subheader("✂️ Configuración de Partición")
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("% para Test", 10, 40, 20) / 100
    with col2:
        val_size = st.slider("% para Validación", 10, 40, 20) / 100

    train_size = round(1 - test_size - val_size, 2)
    st.info(f"📊 Train: **{int(train_size*100)}%** | Validación: **{int(val_size*100)}%** | Test: **{int(test_size*100)}%**")

    if train_size <= 0:
        st.error("❌ Los porcentajes superan el 100%. Ajusta los sliders.")
        return

    if st.button("▶️ Ejecutar Partición y Baseline"):

        # ── Limpiar y preparar ───────────────────────────────────
        df_clean = df.dropna().copy()

        # Separar X e y ANTES de cualquier transformación
        X_raw = df_clean.drop(columns=[target_col])
        y_raw = df_clean[target_col].copy()

        # Codificar target por separado
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y_raw.astype(str))

        # Codificar features
        X_encoded = X_raw.copy()
        for col in X_encoded.select_dtypes(include='object').columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))

        # Convertir a numpy para evitar problemas de índice
        X_np = X_encoded.values
        y_np = y_encoded

        # ── Split Test ───────────────────────────────────────────
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_np, y_np,
            test_size=test_size,
            random_state=42,
            stratify=y_np
        )

        # ── Split Validación ─────────────────────────────────────
        val_relative = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_relative,
            random_state=42,
            stratify=y_temp
        )

        st.success(f"✅ Train: {len(X_train)} filas | Validación: {len(X_val)} filas | Test: {len(X_test)} filas")

        # Verificación rápida
        st.write(f"🔍 Clases en y_train: {np.unique(y_train)} | Distribución: {np.bincount(y_train)}")

        # ── Guardar en session_state ─────────────────────────────
        st.session_state['X_train']   = X_train
        st.session_state['X_val']     = X_val
        st.session_state['X_test']    = X_test
        st.session_state['y_train']   = y_train
        st.session_state['y_val']     = y_val
        st.session_state['y_test']    = y_test
        st.session_state['target']    = target_col
        st.session_state['df_clean']  = df_clean
        st.session_state['le_target'] = le_target
        st.session_state['clases']    = le_target.classes_

        # ── 4. Data Leakage ──────────────────────────────────────
        st.subheader("⚠️ Data Leakage — ¿Qué es y cómo lo evitamos?")
        st.markdown("""
        > **Data Leakage** ocurre cuando información del conjunto de **Test o futuro**
        > se filtra al modelo durante el entrenamiento, generando resultados
        > artificialmente optimistas que no se replican en producción.

        ### Tipos comunes:
        | Tipo | Ejemplo | Cómo evitarlo |
        |------|---------|---------------|
        | **Temporal** | Usar datos futuros para predecir el pasado | Ordenar por fecha antes de partir |
        | **Preprocesamiento** | Escalar con estadísticas de todo el dataset | Fit solo en Train, Transform en Val/Test |
        | **Target leakage** | Incluir columnas derivadas del target | Analizar correlaciones antes de modelar |

        ### ✅ En esta app lo evitamos porque:
        - Partimos **antes** de cualquier transformación
        - El encoding se aprende **solo del Train**
        - Test queda **completamente aislado** hasta la evaluación final
        """)

        # ── 5. Modelo Baseline ───────────────────────────────────
        st.subheader("📏 Modelo Baseline")
        st.markdown("""
        El **baseline** es el modelo más simple posible. Sirve como **piso mínimo**:
        cualquier modelo real debe superarlo. Usamos `DummyClassifier` con estrategia
        **most_frequent** (siempre predice la clase más común).
        """)

        baseline = DummyClassifier(strategy="most_frequent", random_state=42)
        baseline.fit(X_train, y_train)
        y_pred = baseline.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1  = f1_score(y_val, y_pred, average='weighted', zero_division=0)

        st.write(f"🔍 Debug — y_val únicos: {np.unique(y_val)} | y_pred únicos: {np.unique(y_pred)}")

        col1, col2 = st.columns(2)
        col1.metric("🎯 Accuracy Baseline", f"{acc:.2%}")
        col2.metric("📊 F1-Score Baseline",  f"{f1:.2%}")

        st.warning(f"""
        💡 **Interpretación:** El baseline obtiene **{acc:.2%}** simplemente
        prediciendo siempre la clase más frecuente. Nuestros modelos reales
        deben superar este valor.
        """)

        st.session_state['baseline_acc'] = acc
        st.session_state['baseline_f1']  = f1

        st.success("✅ Fase 1 completada. Puedes continuar con la Fase 2.")
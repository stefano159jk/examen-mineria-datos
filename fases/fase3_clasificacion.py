import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              confusion_matrix, classification_report,
                              ConfusionMatrixDisplay)
from sklearn.preprocessing import label_binarize

def mostrar_fase3():
    st.header("🟠 Fase 3 — Clasificación")

    # ── Verificar que Fase 1 fue ejecutada ───────────────────────
    if 'X_train' not in st.session_state:
        st.warning("⚠️ Primero ejecuta la Fase 1 para partir los datos.")
        return

    X_train = st.session_state['X_train']
    X_val   = st.session_state['X_val']
    X_test  = st.session_state['X_test']
    y_train = st.session_state['y_train']
    y_val   = st.session_state['y_val']
    y_test  = st.session_state['y_test']

    st.success(f"✅ Datos cargados — Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ════════════════════════════════════════════════════════════
    # SECCIÓN 1 — ÁRBOL DE DECISIÓN
    # ════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("🌳 Árbol de Decisión")
    st.markdown("""
    El **Árbol de Decisión** divide los datos mediante preguntas sucesivas
    sobre las variables (ej: ¿edad > 40? ¿saldo > 1000?) hasta clasificar
    cada cliente. Es muy **interpretable** — se puede visualizar y explicar.

    - **Ventaja:** fácil de entender y explicar
    - **Desventaja:** puede sobreajustarse (overfitting) si es muy profundo
    """)

    col1, col2 = st.columns(2)
    with col1:
        max_depth = st.slider("Profundidad máxima del árbol", 2, 15, 5)
    with col2:
        min_samples = st.slider("Mínimo de muestras por hoja", 1, 50, 10)

    if st.button("🚀 Entrenar Árbol de Decisión"):
        with st.spinner("Entrenando árbol..."):
            dt = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=min_samples,
                random_state=42
            )
            dt.fit(X_train, y_train)

            # Predicciones en validación
            y_pred_val  = dt.predict(X_val)
            y_proba_val = dt.predict_proba(X_val)[:, 1]

            # Predicciones en test
            y_pred_test  = dt.predict(X_test)
            y_proba_test = dt.predict_proba(X_test)[:, 1]

        # Métricas
        acc_val  = accuracy_score(y_val,  y_pred_val)
        f1_val   = f1_score(y_val,  y_pred_val,  average='weighted', zero_division=0)
        auc_val  = roc_auc_score(y_val,  y_proba_val)

        acc_test = accuracy_score(y_test, y_pred_test)
        f1_test  = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
        auc_test = roc_auc_score(y_test, y_proba_test)

        # Guardar para fase 4
        st.session_state['dt_model']      = dt
        st.session_state['dt_proba_test'] = y_proba_test
        st.session_state['dt_pred_test']  = y_pred_test
        st.session_state['dt_metrics']    = {
            'Accuracy': acc_test, 'F1-Score': f1_test, 'AUC': auc_test
        }

        # Mostrar métricas
        st.markdown("#### 📊 Métricas — Árbol de Decisión")
        col1, col2, col3 = st.columns(3)
        col1.metric("🎯 Accuracy (Val)",  f"{acc_val:.2%}")
        col2.metric("📊 F1-Score (Val)",  f"{f1_val:.2%}")
        col3.metric("📈 AUC (Val)",       f"{auc_val:.4f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("🎯 Accuracy (Test)", f"{acc_test:.2%}")
        col5.metric("📊 F1-Score (Test)", f"{f1_test:.2%}")
        col6.metric("📈 AUC (Test)",      f"{auc_test:.4f}")

        # Comparar con baseline
        baseline_acc = st.session_state.get('baseline_acc', 0)
        baseline_f1  = st.session_state.get('baseline_f1',  0)
        mejora_acc = acc_test - baseline_acc
        mejora_f1  = f1_test  - baseline_f1

        st.markdown("#### 📏 Comparación con Baseline")
        col1, col2 = st.columns(2)
        col1.metric("Mejora en Accuracy vs Baseline",
                    f"{acc_test:.2%}", f"{mejora_acc:+.2%}")
        col2.metric("Mejora en F1 vs Baseline",
                    f"{f1_test:.2%}",  f"{mejora_f1:+.2%}")

        # Matriz de confusión
        st.markdown("#### 🔲 Matriz de Confusión — Árbol de Decisión")
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
        cm = confusion_matrix(y_test, y_pred_test)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=['No (0)', 'Yes (1)'])
        disp.plot(ax=ax_cm, colorbar=False, cmap='Blues')
        ax_cm.set_title('Matriz de Confusión — Árbol de Decisión', fontweight='bold')
        st.pyplot(fig_cm)
        plt.close()

        tn, fp, fn, tp = cm.ravel()
        st.markdown(f"""
        | Métrica | Valor | Significado |
        |---------|-------|-------------|
        | **Verdaderos Negativos (TN)** | {tn} | Clientes que NO responden, predichos correctamente como NO |
        | **Falsos Positivos (FP)**     | {fp} | Clientes que NO responden, predichos como SI |
        | **Falsos Negativos (FN)**     | {fn} | Clientes que SI responden, predichos como NO |
        | **Verdaderos Positivos (TP)** | {tp} | Clientes que SI responden, predichos correctamente como SI |
        """)

        # Visualización del árbol
        st.markdown("#### 🌳 Visualización del Árbol")
        fig_tree, ax_tree = plt.subplots(figsize=(20, 8))
        plot_tree(dt, max_depth=3, filled=True, rounded=True,
                  class_names=['No', 'Yes'],
                  ax=ax_tree, fontsize=9)
        ax_tree.set_title('Árbol de Decisión (primeros 3 niveles)', fontweight='bold')
        st.pyplot(fig_tree)
        plt.close()

        # Importancia de variables
        st.markdown("#### 🔑 Importancia de Variables")
        feature_names = st.session_state.get(
            'feature_names',
            [f'Feature_{i}' for i in range(X_train.shape[1])]
        )
        importancias = pd.Series(dt.feature_importances_,
                                  index=feature_names).sort_values(ascending=False).head(10)
        fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
        importancias.plot(kind='bar', ax=ax_imp, color='steelblue')
        ax_imp.set_title('Top 10 Variables más importantes', fontweight='bold')
        ax_imp.set_ylabel('Importancia')
        ax_imp.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig_imp)
        plt.close()

        st.success("✅ Árbol de Decisión completado.")

    # ════════════════════════════════════════════════════════════
    # SECCIÓN 2 — RANDOM FOREST
    # ════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("🌲🌲 Random Forest")
    st.markdown("""
    **Random Forest** entrena **múltiples árboles** sobre subconjuntos
    aleatorios de datos y variables, luego combina sus predicciones por votación.

    - **Ventaja:** mucho más robusto que un solo árbol, menos overfitting
    - **Desventaja:** menos interpretable, más lento de entrenar
    - Generalmente supera al Árbol de Decisión en AUC y F1
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        n_trees    = st.slider("Número de árboles", 50, 300, 100, step=50)
    with col2:
        max_depth_rf = st.slider("Profundidad máxima", 3, 20, 10, key='rf_depth')
    with col3:
        min_samples_rf = st.slider("Mínimo muestras hoja", 1, 50, 5, key='rf_samples')

    if st.button("🚀 Entrenar Random Forest"):
        with st.spinner(f"Entrenando Random Forest con {n_trees} árboles... (puede tardar ~30 seg)"):
            rf = RandomForestClassifier(
                n_estimators=n_trees,
                max_depth=max_depth_rf,
                min_samples_leaf=min_samples_rf,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)

            y_pred_val_rf  = rf.predict(X_val)
            y_proba_val_rf = rf.predict_proba(X_val)[:, 1]

            y_pred_test_rf  = rf.predict(X_test)
            y_proba_test_rf = rf.predict_proba(X_test)[:, 1]

        # Métricas
        acc_val_rf  = accuracy_score(y_val,  y_pred_val_rf)
        f1_val_rf   = f1_score(y_val,  y_pred_val_rf,  average='weighted', zero_division=0)
        auc_val_rf  = roc_auc_score(y_val,  y_proba_val_rf)

        acc_test_rf = accuracy_score(y_test, y_pred_test_rf)
        f1_test_rf  = f1_score(y_test, y_pred_test_rf, average='weighted', zero_division=0)
        auc_test_rf = roc_auc_score(y_test, y_proba_test_rf)

        # Guardar para fase 4
        st.session_state['rf_model']      = rf
        st.session_state['rf_proba_test'] = y_proba_test_rf
        st.session_state['rf_pred_test']  = y_pred_test_rf
        st.session_state['rf_metrics']    = {
            'Accuracy': acc_test_rf, 'F1-Score': f1_test_rf, 'AUC': auc_test_rf
        }

        # Mostrar métricas
        st.markdown("#### 📊 Métricas — Random Forest")
        col1, col2, col3 = st.columns(3)
        col1.metric("🎯 Accuracy (Val)",  f"{acc_val_rf:.2%}")
        col2.metric("📊 F1-Score (Val)",  f"{f1_val_rf:.2%}")
        col3.metric("📈 AUC (Val)",       f"{auc_val_rf:.4f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("🎯 Accuracy (Test)", f"{acc_test_rf:.2%}")
        col5.metric("📊 F1-Score (Test)", f"{f1_test_rf:.2%}")
        col6.metric("📈 AUC (Test)",      f"{auc_test_rf:.4f}")

        # Comparación con baseline
        baseline_acc = st.session_state.get('baseline_acc', 0)
        baseline_f1  = st.session_state.get('baseline_f1',  0)
        st.markdown("#### 📏 Comparación con Baseline")
        col1, col2 = st.columns(2)
        col1.metric("Mejora en Accuracy vs Baseline",
                    f"{acc_test_rf:.2%}", f"{acc_test_rf - baseline_acc:+.2%}")
        col2.metric("Mejora en F1 vs Baseline",
                    f"{f1_test_rf:.2%}",  f"{f1_test_rf - baseline_f1:+.2%}")

        # Matriz de confusión
        st.markdown("#### 🔲 Matriz de Confusión — Random Forest")
        fig_cm2, ax_cm2 = plt.subplots(figsize=(6, 4))
        cm2 = confusion_matrix(y_test, y_pred_test_rf)
        disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2,
                                        display_labels=['No (0)', 'Yes (1)'])
        disp2.plot(ax=ax_cm2, colorbar=False, cmap='Greens')
        ax_cm2.set_title('Matriz de Confusión — Random Forest', fontweight='bold')
        st.pyplot(fig_cm2)
        plt.close()

        tn2, fp2, fn2, tp2 = cm2.ravel()
        st.markdown(f"""
        | Métrica | Valor | Significado |
        |---------|-------|-------------|
        | **Verdaderos Negativos (TN)** | {tn2} | Clientes que NO responden, predichos correctamente |
        | **Falsos Positivos (FP)**     | {fp2} | Clientes que NO responden, predichos como SI |
        | **Falsos Negativos (FN)**     | {fn2} | Clientes que SI responden, predichos como NO |
        | **Verdaderos Positivos (TP)** | {tp2} | Clientes que SI responden, predichos correctamente |
        """)

        # Importancia de variables RF
        st.markdown("#### 🔑 Importancia de Variables — Random Forest")
        feature_names = st.session_state.get(
            'feature_names',
            [f'Feature_{i}' for i in range(X_train.shape[1])]
        )
        imp_rf = pd.Series(rf.feature_importances_,
                            index=feature_names).sort_values(ascending=False).head(10)
        fig_imp2, ax_imp2 = plt.subplots(figsize=(8, 4))
        imp_rf.plot(kind='bar', ax=ax_imp2, color='forestgreen')
        ax_imp2.set_title('Top 10 Variables más importantes — RF', fontweight='bold')
        ax_imp2.set_ylabel('Importancia')
        ax_imp2.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig_imp2)
        plt.close()

        st.success("✅ Random Forest completado. Puedes continuar con la Fase 4.")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, confusion_matrix,
                              ConfusionMatrixDisplay, accuracy_score,
                              f1_score, roc_auc_score, precision_score,
                              recall_score)

def mostrar_fase4():
    st.header("🔴 Fase 4 — Evaluación Comparativa de Modelos")

    # ── Verificar dependencias ───────────────────────────────────
    if 'X_test' not in st.session_state:
        st.warning("⚠️ Primero ejecuta la Fase 1.")
        return
    if 'dt_metrics' not in st.session_state and 'rf_metrics' not in st.session_state:
        st.warning("⚠️ Primero entrena al menos un modelo en la Fase 3.")
        return

    y_test       = st.session_state['y_test']
    baseline_acc = st.session_state.get('baseline_acc', 0)
    baseline_f1  = st.session_state.get('baseline_f1',  0)

    dt_disponible = 'dt_metrics' in st.session_state
    rf_disponible = 'rf_metrics' in st.session_state

    # ════════════════════════════════════════════════════════════
    # SECCIÓN 1 — TABLA COMPARATIVA
    # ════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("📊 Tabla Comparativa de Modelos")

    filas = [{
        'Modelo':    '📏 Baseline (DummyClassifier)',
        'Accuracy':  f"{baseline_acc:.2%}",
        'F1-Score':  f"{baseline_f1:.2%}",
        'AUC':       'N/A',
        'Precisión': 'N/A',
        'Recall':    'N/A'
    }]

    if dt_disponible:
        y_pred_dt  = st.session_state['dt_pred_test']
        y_proba_dt = st.session_state['dt_proba_test']
        filas.append({
            'Modelo':    '🌳 Árbol de Decisión',
            'Accuracy':  f"{accuracy_score(y_test, y_pred_dt):.2%}",
            'F1-Score':  f"{f1_score(y_test, y_pred_dt, average='weighted', zero_division=0):.2%}",
            'AUC':       f"{roc_auc_score(y_test, y_proba_dt):.4f}",
            'Precisión': f"{precision_score(y_test, y_pred_dt, zero_division=0):.2%}",
            'Recall':    f"{recall_score(y_test, y_pred_dt, zero_division=0):.2%}"
        })

    if rf_disponible:
        y_pred_rf  = st.session_state['rf_pred_test']
        y_proba_rf = st.session_state['rf_proba_test']
        filas.append({
            'Modelo':    '🌲 Random Forest',
            'Accuracy':  f"{accuracy_score(y_test, y_pred_rf):.2%}",
            'F1-Score':  f"{f1_score(y_test, y_pred_rf, average='weighted', zero_division=0):.2%}",
            'AUC':       f"{roc_auc_score(y_test, y_proba_rf):.4f}",
            'Precisión': f"{precision_score(y_test, y_pred_rf, zero_division=0):.2%}",
            'Recall':    f"{recall_score(y_test, y_pred_rf, zero_division=0):.2%}"
        })

    df_comp = pd.DataFrame(filas)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    st.markdown("""
    | Métrica | Qué mide |
    |---------|----------|
    | **Accuracy** | % de predicciones correctas en total |
    | **F1-Score** | Balance entre Precisión y Recall (útil con datos desbalanceados) |
    | **AUC** | Capacidad del modelo de distinguir clases (1.0 = perfecto, 0.5 = azar) |
    | **Precisión** | De los que predije como SI, ¿cuántos realmente eran SI? |
    | **Recall** | De todos los SI reales, ¿cuántos detecté correctamente? |
    """)

    # ════════════════════════════════════════════════════════════
    # SECCIÓN 2 — CURVA ROC
    # ════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("📈 Curva ROC Comparativa")
    st.markdown("""
    La **Curva ROC** muestra la relación entre la **Tasa de Verdaderos Positivos**
    y la **Tasa de Falsos Positivos** para distintos umbrales de decisión.
    - Cuanto más arriba y a la izquierda → mejor modelo
    - La línea diagonal punteada = modelo aleatorio (baseline)
    - El **AUC** es el área bajo la curva → más cercano a 1.0 = mejor
    """)

    if dt_disponible or rf_disponible:
        fig_roc, ax_roc = plt.subplots(figsize=(9, 6))

        # Línea baseline
        ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1.5,
                    label='Baseline (AUC = 0.50)', alpha=0.6)

        if dt_disponible:
            fpr_dt, tpr_dt, _ = roc_curve(y_test, y_proba_dt)
            auc_dt = auc(fpr_dt, tpr_dt)
            ax_roc.plot(fpr_dt, tpr_dt, 'b-', linewidth=2,
                        label=f'Árbol de Decisión (AUC = {auc_dt:.4f})')

        if rf_disponible:
            fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
            auc_rf = auc(fpr_rf, tpr_rf)
            ax_roc.plot(fpr_rf, tpr_rf, 'g-', linewidth=2,
                        label=f'Random Forest (AUC = {auc_rf:.4f})')

        ax_roc.set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
        ax_roc.set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12)
        ax_roc.set_title('Curva ROC — Comparativa de Modelos', fontsize=14, fontweight='bold')
        ax_roc.legend(loc='lower right', fontsize=11)
        ax_roc.grid(True, alpha=0.3)
        ax_roc.set_xlim([0, 1])
        ax_roc.set_ylim([0, 1])
        plt.tight_layout()
        st.pyplot(fig_roc)
        plt.close()
    else:
        st.info("Entrena los modelos en Fase 3 para ver la Curva ROC.")

    # ════════════════════════════════════════════════════════════
    # SECCIÓN 3 — MATRIZ DE CONFUSIÓN FINAL
    # ════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("🔲 Matriz de Confusión Final con Métricas Completas")

    def mostrar_matriz_completa(y_true, y_pred, y_proba, nombre, color):
        st.markdown(f"#### {nombre}")
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        col1, col2 = st.columns([1, 1])

        with col1:
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=['No (0)', 'Yes (1)']
            )
            disp.plot(ax=ax_cm, colorbar=False, cmap=color)
            ax_cm.set_title(f'Matriz de Confusión\n{nombre}', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig_cm)
            plt.close()

        with col2:
            acc  = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec  = recall_score(y_true, y_pred, zero_division=0)
            f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            auc_ = roc_auc_score(y_true, y_proba)
            esp  = tn / (tn + fp) if (tn + fp) > 0 else 0

            st.markdown(f"""
            | Métrica | Valor | Fórmula |
            |---------|-------|---------|
            | **Accuracy**      | {acc:.2%}  | (TP+TN) / Total |
            | **Precisión**     | {prec:.2%} | TP / (TP+FP) |
            | **Recall**        | {rec:.2%}  | TP / (TP+FN) |
            | **Especificidad** | {esp:.2%}  | TN / (TN+FP) |
            | **F1-Score**      | {f1:.2%}   | 2*(Prec*Rec)/(Prec+Rec) |
            | **AUC**           | {auc_:.4f} | Área bajo curva ROC |
            | **TP** | {tp} | Clientes SI predichos como SI ✅ |
            | **TN** | {tn} | Clientes NO predichos como NO ✅ |
            | **FP** | {fp} | Clientes NO predichos como SI ❌ |
            | **FN** | {fn} | Clientes SI predichos como NO ❌ |
            """)

    if dt_disponible:
        mostrar_matriz_completa(y_test,
                                st.session_state['dt_pred_test'],
                                st.session_state['dt_proba_test'],
                                "🌳 Árbol de Decisión", "Blues")

    if rf_disponible:
        mostrar_matriz_completa(y_test,
                                st.session_state['rf_pred_test'],
                                st.session_state['rf_proba_test'],
                                "🌲 Random Forest", "Greens")

    # ════════════════════════════════════════════════════════════
    # SECCIÓN 4 — REPORTE EJECUTIVO
    # ════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("📝 Reporte Ejecutivo — Para equipo gerencial")
    st.markdown("""
    > Esta sección traduce los resultados técnicos a **lenguaje de negocio**,
    > sin jerga estadística, para presentar a directivos.
    """)

    if st.button("📋 Generar Reporte Ejecutivo"):
        mejor_modelo = None
        mejor_auc    = 0

        if dt_disponible:
            auc_dt_val = roc_auc_score(y_test, st.session_state['dt_proba_test'])
            if auc_dt_val > mejor_auc:
                mejor_auc    = auc_dt_val
                mejor_modelo = "Árbol de Decisión"

        if rf_disponible:
            auc_rf_val = roc_auc_score(y_test, st.session_state['rf_proba_test'])
            if auc_rf_val > mejor_auc:
                mejor_auc    = auc_rf_val
                mejor_modelo = "Random Forest"

        acc_mejor  = accuracy_score(y_test,
                        st.session_state['rf_pred_test'] if mejor_modelo == "Random Forest"
                        else st.session_state['dt_pred_test'])
        f1_mejor   = f1_score(y_test,
                        st.session_state['rf_pred_test'] if mejor_modelo == "Random Forest"
                        else st.session_state['dt_pred_test'],
                        average='weighted', zero_division=0)

        y_pred_mejor = (st.session_state['rf_pred_test'] if mejor_modelo == "Random Forest"
                        else st.session_state['dt_pred_test'])
        cm_mejor = confusion_matrix(y_test, y_pred_mejor)
        tn_m, fp_m, fn_m, tp_m = cm_mejor.ravel()

        st.markdown(f"""
        ---
        ## 📊 Reporte de Resultados — Campaña de Marketing
        ### Resumen Ejecutivo

        Se analizaron **{len(y_test):,} clientes** del dataset para predecir
        quiénes responderían positivamente a la campaña bancaria.

        ### ✅ Modelo Recomendado: {mejor_modelo}

        Después de comparar múltiples modelos, el **{mejor_modelo}** obtuvo
        los mejores resultados con una capacidad de **{mejor_auc:.1%}** para
        distinguir entre clientes que responden y los que no.

        ### 💼 ¿Qué significa en términos de negocio?

        - De cada 100 clientes contactados, el modelo identifica correctamente
          a **{int(acc_mejor*100)} de ellos**
        - El modelo detectó correctamente **{tp_m:,} clientes** que SÍ
          responderían a la campaña
        - Solo **{fp_m:,} clientes** fueron contactados innecesariamente
          (predijo SI pero era NO)
        - Se perdieron **{fn_m:,} oportunidades** (clientes que sí
          hubieran respondido pero no fueron contactados)

        ### 💰 Impacto estimado

        > Si el costo de contactar un cliente es de $1 y la ganancia
        > por conversión es de $10, usar el modelo permite **enfocar
        > recursos** en los clientes con mayor probabilidad de respuesta,
        > reduciendo costos operativos y aumentando la tasa de conversión.

        ### 📈 Comparación vs Estrategia Actual

        | Estrategia | Clientes bien identificados |
        |---|---|
        | Sin modelo (contactar a todos) | {int(baseline_acc*100)}% |
        | Con modelo {mejor_modelo} | {int(acc_mejor*100)}% |
        | **Mejora** | **+{int((acc_mejor - baseline_acc)*100)} puntos** |

        ---
        *Reporte generado automáticamente por la App de Minería de Datos*
        """)

        st.success("✅ Fase 4 completada — Análisis finalizado.")
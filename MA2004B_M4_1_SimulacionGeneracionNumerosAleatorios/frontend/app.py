import streamlit as st
import subprocess
import pandas as pd
import os

# Compila el backend si no existe el ejecutable
exe_path = os.path.join("backend", "main")
if not os.path.exists(exe_path):
    compile_cmd = [
        "g++",
        "backend/main.cpp",
        "backend/LinearCongruentialGenerator.cpp",
        "-o",
        exe_path
    ]
    try:
        subprocess.run(compile_cmd, check=True, cwd=".")
        st.info("Backend compilado exitosamente.")
    except subprocess.CalledProcessError as e:
        st.error(f"Error al compilar backend: {e}")

st.set_page_config(page_title="Método Congruencial Lineal", layout="centered")
st.title("Método Congruencial Lineal")
st.subheader("Parámetros del generador")

col1, col2 = st.columns(2)
with col1:
    m = st.number_input("m (módulo, > 1)", min_value=2, value=16, step=1)
    a = st.number_input("a (multiplicador)", value=5, step=1)
    c = st.number_input("c (incremento)", value=3, step=1)
with col2:
    seed = st.number_input("Semilla X₀", value=7, step=1)
    n = st.number_input("Cantidad de números a generar (n)", min_value=1, value=10, step=1)

modo = st.selectbox("Modo de generación", ["Pseudoaleatorio (MCL)", "No aleatorio (Xn+1 = Xn + 1)"])
modo_flag = 1 if modo == "Pseudoaleatorio (MCL)" else 0

st.markdown("---")

if st.button("Generar secuencia"):
    if not os.path.exists(exe_path):
        st.error(f"No se encontró el ejecutable C++ en: {exe_path}\n"
                 f"Compila primero el backend (ver README).")
    else:
        # Ejecutar el programa C++ con los argumentos
        cmd = [
            exe_path,
            str(int(m)),
            str(int(a)),
            str(int(c)),
            str(int(seed)),
            str(int(n)),
            str(int(modo_flag))
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd="."  # directorio raíz del proyecto
            )

            if result.returncode != 0:
                st.error("Error al ejecutar el backend C++:")
                st.code(result.stderr)
            else:
                # Leer el CSV generado por el backend (en el directorio raíz)
                csv_path = os.path.join(".", "output.csv")
                if not os.path.exists(csv_path):
                    st.error("No se encontró output.csv después de ejecutar el backend.")
                else:
                    df = pd.read_csv(csv_path)
                    st.success("Secuencia generada correctamente.")
                    st.subheader("Tabla de números generados")
                    st.dataframe(df, use_container_width=True)

                    st.subheader("Gráfica de la secuencia")
                    st.line_chart(df.set_index("index")["value"])

                    # Botón para descargar el CSV
                    csv_bytes = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Descargar CSV",
                        data=csv_bytes,
                        file_name="mcl_resultados.csv",
                        mime="text/csv"
                    )

                    st.markdown("**Mensaje del backend:**")
                    st.code(result.stdout)
        except Exception as e:
            st.error(f"Ocurrió un error al intentar ejecutar el backend: {e}")
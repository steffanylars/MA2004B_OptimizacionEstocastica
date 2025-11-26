import streamlit as st
import subprocess
import pandas as pd
import os

# Obtener el directorio donde está este script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Rutas absolutas
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
EXE_PATH = os.path.join(BACKEND_DIR, "main")
OUTPUT_CSV = os.path.join(BASE_DIR, "output.csv")

# Compila el backend si no existe el ejecutable
if not os.path.exists(EXE_PATH):
    main_cpp = os.path.join(BACKEND_DIR, "main.cpp")
    lcg_cpp = os.path.join(BACKEND_DIR, "LinearCongruentialGenerator.cpp")
    
    # Verificar que existen los archivos fuente
    if not os.path.exists(main_cpp):
        st.error(f"No se encontró: {main_cpp}")
        st.stop()
    if not os.path.exists(lcg_cpp):
        st.error(f"No se encontró: {lcg_cpp}")
        st.stop()
    
    compile_cmd = [
        "g++",
        main_cpp,
        lcg_cpp,
        "-o",
        EXE_PATH
    ]
    
    try:
        compile_result = subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True,
            cwd=BASE_DIR
        )
        if compile_result.returncode != 0:
            st.error(f"Error al compilar backend:\n{compile_result.stderr}")
            st.stop()
        st.info("Backend compilado exitosamente.")
    except Exception as e:
        st.error(f"Error al compilar: {e}")
        st.stop()

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
    if not os.path.exists(EXE_PATH):
        st.error(f"No se encontró el ejecutable C++ en: {EXE_PATH}")
    else:
        cmd = [
            EXE_PATH,
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
                cwd=BASE_DIR  # Ejecutar desde el directorio base
            )

            if result.returncode != 0:
                st.error("Error al ejecutar el backend C++:")
                st.code(result.stderr)
            else:
                csv_path = os.path.join(BASE_DIR, "output.csv")
                if not os.path.exists(csv_path):
                    st.error("No se encontró output.csv después de ejecutar el backend.")
                else:
                    df = pd.read_csv(csv_path)
                    st.success("Secuencia generada correctamente.")
                    st.subheader("Tabla de números generados")
                    st.dataframe(df, use_container_width=True)

                    st.subheader("Gráfica de la secuencia")
                    st.line_chart(df.set_index("index")["value"])

                    csv_bytes = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Descargar CSV",
                        data=csv_bytes,
                        file_name="mcl_resultados.csv",
                        mime="text/csv"
                    )

                    if result.stdout:
                        st.markdown("**Mensaje del backend:**")
                        st.code(result.stdout)
        except Exception as e:
            st.error(f"Ocurrió un error al intentar ejecutar el backend: {e}")

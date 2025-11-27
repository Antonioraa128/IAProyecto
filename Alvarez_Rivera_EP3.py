import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression # Regresión lineal
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # Errores
import seaborn as sns # Heatmap
from sklearn.preprocessing import StandardScaler #scaler

# Configuración de la Página
st.set_page_config(page_title="Proyecto Final IA", layout="wide")

# Leemos el dataset
df = pd.read_csv('computer_prices_all.csv')
df = df.select_dtypes(include=np.number) #Solo numericas

#Limpia de datos y mapeo
#[Pendiente]

# Entrenamiento del modelo: Se entrena en vivo para el demo
X = df[['gpu_tier', 'cpu_tier', 'ram_gb']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Escalado
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)

# ------------------------------------------------------------------
# Interfaz del Dashboard
# ------------------------------------------------------------------
# Barra lateral para navegación
st.sidebar.title("Navegación")
opcion = st.sidebar.radio("Ir a:", ["1. Contexto y Problema", 
                                    "2. Análisis Exploratorio de Datos (EDA)", 
                                    "3. Evaluación del Modelo", 
                                    "4. Predicción en Vivo"])

# ------------------------------------------------------------------
# Sección 1. Contexto y Problema
if opcion == "1. Contexto y Problema":
    st.title("Estimador Inteligente de Precios de computadoras")
    st.markdown("""
    ### El Problema
    El mercado de las computadoras aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    
    ### La Solución
    Un modelo de Machine Learning (Regresion lineal) que sugiere un precio justo basado en características del equipo.
    """)
    
    st.info("Nota: Aquí podrían describir brevemente como limpiaron los datos")
    st.write("Vista previa del Dataset procesado:", df.head())

# ------------------------------------------------------------------
# Sección 2. EDA
elif opcion == "2. Análisis Exploratorio de Datos (EDA)":
    st.title("📊 Análisis Exploratorio de Datos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Relación gpu_tier vs Precio")
        st.scatter_chart(data=df, x='gpu_tier', y='price', color="#3E0080") 
        st.caption("Correlación positiva: AAAAAAAAAAAAAAAAAAAA")
        
    '''with col2:
        st.subheader("Distribución por cpu_tier")
        conteo_zona = df['cpu_tier'].value_counts().rename(index={0: 'Periferia', 1: 'Centro'})
        st.bar_chart(conteo_zona, color="#FF8C0095")
        st.caption("Distribución de propiedades en el dataset.")
    '''

    with col2:
        st.subheader("Relación cpu_tier vs Precio")
        st.scatter_chart(data=df, x='cpu_tier', y='price', color="#3E0080") 
        st.caption("Correlación positiva: AAAAAAAAAAAAAAAAAAAA")

# ------------------------------------------------------------------
# Sección 3. Evaluación del modelo
elif opcion == "3. Evaluación del Modelo":
    st.title("⚙️ Desempeño del Modelo")
    
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("R2 Score", f"{r2:.2%}")
    col2.metric("Error Promedio (MAE)", f"${mae:,.2f}")
    col3.metric("Test Set Size", len(y_test))
    
    st.subheader("Gráfica: Realidad vs. Predicción")
    comparison_df = pd.DataFrame({'Real': y_test, 'Predicho': preds})
    # Colores de alto contraste: Rojo y Azul
    st.line_chart(comparison_df.head(30), color=["#FF0000", "#0000FF"])

# ------------------------------------------------------------------
# Sección 4. Predicción en vivo
elif opcion == "4. Predicción en Vivo":
    st.title("🚀 Prueba del Modelo en Tiempo Real")
    st.markdown("Configure las características de la casa para obtener una valuación instantánea.")
    
    # Inputs interactivos para el profesor/usuario
    col1, col2 = st.columns(2)
    with col1:
        cpu_input = st.slider("Cpu tier", 1, 6, 2)
        ram_input = st.number_input("Gp de ram", 4, 64, 16)

    with col2:
        gpu_input = st.slider("Gpu tier", 1, 6, 2)
    
    '''
    with col2:
        antiguedad_input = st.slider("Antigüedad (Años)", 0, 50, 5)
        zona_input = st.selectbox("Ubicación", ["Periferia", "Centro"])
        zona_binaria = 1 if zona_input == "Centro" else 0
    '''

    # Botón de predicción
    if st.button("Calcular Precio Justo", type="primary"):
        input_data = pd.DataFrame({
            'gpu_tier': [gpu_input],
            'cpu_tier': [cpu_input],
            'ram_gb': [ram_input],
        })
        
        input_data = scaler.transform(input_data) #la escalamos antes de meterla al modelo
        prediccion = model.predict(input_data)[0]
        
        # Detalle visual del resultado
        st.success(f"💰 Precio Estimado: **${prediccion:,.2f} MXN**")
        
        max_price = df['price'].max()
        st.progress(int(min(prediccion / max_price, 1.0) * 100))

    # Para correr: streamlit run demo_clase.py
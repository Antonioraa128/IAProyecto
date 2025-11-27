import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Configuración de la Página
st.set_page_config(page_title="Proyecto Final IA", layout="wide")

# NOTA: Aquí iría su pd.read_csv('datos_reales.csv')

# Generación de datos aleatorios (Precios de venta de casas)
@st.cache_data
def generar_datos():
    np.random.seed(42)
    n = 500
    # Variables independientes
    metros = np.random.randint(80, 400, n) # Casas de 80m2 a 400m2
    habitaciones = np.random.randint(2, 7, n)
    antiguedad = np.random.randint(0, 30, n)
    zona_centro = np.random.randint(0, 2, n) 
    
    # Fórmula de precio de casas (MXN)
    # Base: 1.5 Millones + 25k por m2 + 150k por cuarto - depreciación + plusvalía centro
    precio_base = 1500000 
    precio = precio_base + \
             (metros * 25000) + \
             (habitaciones * 150000) - \
             (antiguedad * 15000) + \
             (zona_centro * 1200000) + \
             np.random.normal(0, 100000, n) # Ruido de +/- 300k
    
    df = pd.DataFrame({
        'Metros_Cuadrados': metros,
        'Habitaciones': habitaciones,
        'Antiguedad_Anios': antiguedad,
        'Zona_Centro': zona_centro,
        'Precio_Venta': precio
    })
    return df

df = generar_datos()

# Entrenamiento del modelo: Se entrena en vivo para el demo
X = df[['Metros_Cuadrados', 'Habitaciones', 'Antiguedad_Anios', 'Zona_Centro']]
y = df['Precio_Venta']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
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
    st.title("🏠 Estimador Inteligente de Precios Inmobiliarios")
    st.markdown("""
    ### El Problema
    El mercado inmobiliario de la ciudad sufre de especulación. Los vendedores suelen fijar precios basándose en emociones / sentimientos y no en datos objetivos.
    
    ### La Solución
    Un modelo de Machine Learning (Random Forest) que sugiere un precio justo basado en características físicas y ubicación, eliminando el sesgo humano.
    """)
    
    st.info("Nota: Aquí podrían describir brevemente como limpiaron los datos")
    st.write("Vista previa del Dataset procesado:", df.head())

# ------------------------------------------------------------------
# Sección 2. EDA
elif opcion == "2. Análisis Exploratorio de Datos (EDA)":
    st.title("📊 Análisis Exploratorio de Datos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Relación Tamaño vs Precio")
        st.scatter_chart(data=df, x='Metros_Cuadrados', y='Precio_Venta', color="#3E0080") 
        st.caption("Correlación positiva: A mayor metraje, mayor costo.")
        
    with col2:
        st.subheader("Distribución por Zona")
        conteo_zona = df['Zona_Centro'].value_counts().rename(index={0: 'Periferia', 1: 'Centro'})
        st.bar_chart(conteo_zona, color="#FF8C0095")
        st.caption("Distribución de propiedades en el dataset.")

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
        metros_input = st.slider("Metros Cuadrados", 50, 300, 120)
        habitaciones_input = st.number_input("Número de Habitaciones", 1, 6, 3)
    
    with col2:
        antiguedad_input = st.slider("Antigüedad (Años)", 0, 50, 5)
        zona_input = st.selectbox("Ubicación", ["Periferia", "Centro"])
        zona_binaria = 1 if zona_input == "Centro" else 0

    # Botón de predicción
    if st.button("Calcular Precio Justo", type="primary"):
        input_data = pd.DataFrame({
            'Metros_Cuadrados': [metros_input],
            'Habitaciones': [habitaciones_input],
            'Antiguedad_Anios': [antiguedad_input],
            'Zona_Centro': [zona_binaria]
        })
        
        prediccion = model.predict(input_data)[0]
        
        # Detalle visual del resultado
        st.success(f"💰 Precio Estimado: **${prediccion:,.2f} MXN**")
        # Barra de progreso ajustada a una escala de 15 Millones
        st.progress(int(min(prediccion / 15000000, 1.0) * 100))

    # Para correr: streamlit run demo_clase.py
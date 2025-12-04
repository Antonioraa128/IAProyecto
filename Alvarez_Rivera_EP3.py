import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression # Regresión lineal
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # Errores
import seaborn as sns # Heatmap
from sklearn.preprocessing import StandardScaler #scaler

# Configuración de la Página
st.set_page_config(page_title="Proyecto Final IA", layout="wide")

# Leemos el dataset
df = pd.read_csv('computer_prices_all.csv')

#Limpia de datos y mapeo
#Mapeo de la variable device type, se vuelve binario si es laptop o no
mapeo_type = {'Desktop': 0, 'Laptop': 1}
df['device_type'] = df['device_type'].map(mapeo_type)
df.head()

#Dummies de las variables categoricas para hacer columnas binarias
df = pd.get_dummies(df, columns=['display_type'], prefix=['DT'], dtype=int)
df.drop(columns=['DT_VA'], inplace=True) #Se quita para evitar redundancia
df.head()

#Dummies de las variables categoricas para hacer columnas binarias
df = pd.get_dummies(df, columns=['gpu_brand'], prefix=['GPU'], dtype=int)
df.drop(columns=['GPU_Intel'], inplace=True) #Se quita para evitar redundancia
df.head()

df = df.select_dtypes(include=np.number) #Dejamos solo las variables numericas

# Entrenamiento del modelo: Se entrena en vivo para el demo
X = df[['cpu_cores','cpu_base_ghz', 'gpu_tier', 'GPU_Apple', 'GPU_NVIDIA', 'GPU_AMD', 'ram_gb', 'device_type', 'display_size_in',  'release_year', 'storage_gb', 'refresh_hz','DT_LED','DT_OLED' ,'DT_IPS', 'DT_QLED', 'DT_Mini-LED']]
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
    st.title("Predicción Inteligente de Precios de Computadoras")
    st.markdown("""
    ### El Problema
    En los últimos años, los precios de las computadoras y sus componentes han subido mucho. Esto ha hecho que sea difícil comprar y actualizar estos sistemas, sobre todo los de gama media y alta, para consumidores y empresas. Este problema ha evolucionado exponencialmente en estos últimos 5 años con la pandemia de COVID-19 y la alta demanda de estas tecnologías para el desarrollo de la inteligencia artificial. Con este rápido aumento de los precios, es difícil para los consumidores y empresas saber si están pagando un precio justo por las computadoras que compran.
    
    ### La Solución
    Decidimos realizar un modelo de Machine Learning (Regresión Lineal Multivariada) que realiza predicciones del precio del sistema, basándose en las características del sistema en general, como el procesador, la tarjeta gráfica, la memoria RAM, el tipo de dispositivo, el tamaño y tipo de pantalla, entre otros.
    """)
    
    st.write("Vista previa del Dataset original:", pd.read_csv('computer_prices_all.csv').head())
    st.write("Vista previa del Dataset procesado:", df.head())
    
    # explicacion de las columnas del dataset procesado
    st.subheader("Descripción de las columnas del dataset procesado:")
    st.markdown("""
    - `device_type`: Tipo de dispositivo (0 para Desktop, 1 para Laptop).
    - `release_year`: Año de lanzamiento del dispositivo.
    - `cpu_cores`: Número de núcleos del procesador.
    - `cpu_base_ghz`: Velocidad base del procesador en GHz.
    - `gpu_tier`: Clasificación de la tarjeta gráfica (1-6).
    - `ram_gb`: Memoria RAM en GB.
    - `storage_gb`: Capacidad de almacenamiento en GB.
    - `display_size_in`: Tamaño de la pantalla en pulgadas.
    - `refresh_hz`: Frecuencia de actualización de la pantalla en Hz.
    - `DT_LED`, `DT_OLED`, `DT_IPS`, `DT_QLED`, `DT_Mini-LED`: Columnas binarias que indican el tipo de pantalla.
    - `GPU_Apple`, `GPU_NVIDIA`, `GPU_AMD`: Columnas binarias que indican la marca de la tarjeta gráfica.
    - `price`: Precio del dispositivo en USD.
    """)
    
    st.info("Nota: Aquí podrían describir brevemente como limpiaron los datos")

# ------------------------------------------------------------------
# Sección 2. EDA
elif opcion == "2. Análisis Exploratorio de Datos (EDA)":
    st.title("📊 Análisis Exploratorio de Datos")
    
    corr = df.corr()
    st.write("Matriz de correlación (Heatmap):")
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    st.pyplot(fig)

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
    comparison_df = pd.DataFrame({'Real': y_test, 'Predicho': preds}).sort_index()
    # Colores de alto contraste: Rojo (Real) y Azul (Predicho)
    st.line_chart(comparison_df.head(30), color=["#FF0000", "#0000FF"], use_container_width=True)

# ------------------------------------------------------------------
# Sección 4. Predicción en vivo
elif opcion == "4. Predicción en Vivo":
    st.title("🚀 Prueba del Modelo en Tiempo Real")
    st.markdown("Configure las características de la computadora para obtener una valuación instantánea.")
    
    # Inputs interactivos para el profesor/usuario
    col1, col2, col3, col4 = st.columns(4)

    #Cosas del CPU
    with col1:
        st.subheader("Caracteristicas del CPU")
        cpu_cores = st.slider("CPU cores", 4, 26, 8)
        cpu_ghz = st.number_input("CPU GHZ base", 2.0, 3.4, 2.6)

    #Cosas del GPU
    with col2:
        st.subheader("Caracteristicas del GPU")
        gpu_input = st.slider("GPU tier", 1, 6, 3)
        gpu_brand_input = st.selectbox("Marca del GPU", ["Apple", "NVIDIA", "AMD", "Intel"])
        apple_binaria = 1 if gpu_brand_input == "Apple" else 0
        nvidia_binaria = 1 if gpu_brand_input == "NVIDIA" else 0
        amd_binaria = 1 if gpu_brand_input == "AMD" else 0

    
    #Otras cosas
    with col3:
        st.subheader("Caracteristicas del Monitor")
        size_input = st.number_input("Tamaño del display", 13.3, 34.0, 16.0)
        hz_input = st.slider("Hercios por segundo", 60, 240, 120)
        display_type_input = st.selectbox("Tipo de display", ["LED", "Mini-LED", "OLED", "QLED", "IPS", "VA"])
        led_binaria = 1 if display_type_input == "LED" else 0
        mled_binaria = 1 if display_type_input == "Mini-LED" else 0
        oled_binaria = 1 if display_type_input == "OLED" else 0
        qled_binaria = 1 if display_type_input == "QLED" else 0
        ips_binaria = 1 if display_type_input == "IPS" else 0

    with col4:
        st.subheader("Otras caracteristicas")
        device_input = st.selectbox("Dispositivo", ["Laptop", "Desktop"])
        year_input = st.slider("Año del dispositivo", 2018, 2025, 2022)
        device_binaria = 1 if device_input == "Laptop" else 0
        ram_input = st.slider("Gb de ram", 8, 144, 32)
        storage_input = st.slider("Almacenamiento", 256, 4096, 512)



    # Botón de predicción
    if st.button("Calcular Precio Justo", type="primary"):
        input_data = pd.DataFrame({
            'cpu_cores': [cpu_cores],
            'cpu_base_ghz': [cpu_ghz],
            'gpu_tier': [gpu_input],
            'GPU_Apple': [apple_binaria],
            'GPU_NVIDIA': [nvidia_binaria],
            'GPU_AMD': [amd_binaria],
            'ram_gb': [ram_input],
            'device_type': [device_binaria],
            'display_size_in': [size_input],
            'release_year': [year_input],
            'storage_gb': [storage_input],
            'refresh_hz': [hz_input],
            'DT_LED': [led_binaria],
            'DT_OLED': [oled_binaria],
            'DT_IPS': [ips_binaria],
            'DT_QLED': [qled_binaria],
            'DT_Mini-LED': [mled_binaria]
        })
        
        input_data = scaler.transform(input_data) #la escalamos antes de meterla al modelo
        prediccion = model.predict(input_data)[0]
        
        # Detalle visual del resultado
        st.success(f"💰 Precio Estimado: **${prediccion:,.2f} USD**")
        
        max_price = df['price'].max()
        st.progress(int(min(prediccion / max_price, 1.0) * 100))

    # Para correr: streamlit run demo_clase.py
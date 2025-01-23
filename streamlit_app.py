import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configuración de la página
st.set_page_config(
    page_title="TI724 FinOps Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personalizado mejorado
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        background-color: #f8f9fa;
    }
    .stPlotlyChart {
        background-color: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stPlotlyChart:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.1);
    }
    .recommendation {
        background-color: #e9ecef;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        border-left: 5px solid #007bff;
        transition: all 0.3s ease;
    }
    .recommendation:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stMetric {
        background-color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTabs {
        background-color: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .stTab {
        border-radius: 10px;
    }
    .stTab[data-baseweb="tab"] {
        height: 60px;
        white-space: pre-wrap;
        background-color: #f1f3f5;
        transition: all 0.3s ease;
    }
    .stTab[aria-selected="true"] {
        background-color: #007bff;
        color: white;
    }
    .stTab:hover {
        transform: translateY(-2px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

# Título y descripción
st.title("📊 TI724 - Panel de Control FinOps Avanzado")
st.markdown("Sistema de Análisis y Optimización de Costos en la Nube")
st.markdown("---")

# Función para cargar datos
@st.cache_data
def cargar_datos():
    try:
        df = pd.read_csv('Datos/part_0_0001.csv', low_memory=False)
        df['Fecha'] = pd.to_datetime(df['BillingPeriodStart'])
        df['BilledCost'] = pd.to_numeric(df['BilledCost'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error al cargar datos: {str(e)}")
        return None

# Cargar datos
with st.spinner('Cargando datos...'):
    df = cargar_datos()
    if df is not None:
        st.success("Datos cargados exitosamente!")
    else:
        st.error("No se pudieron cargar los datos. Por favor, verifica el archivo de datos.")
        st.stop()

# Cálculos FinOps mejorados
def calcular_metricas_finops(df):
    costo_total = df['BilledCost'].sum()
    beneficio_estimado = costo_total * 1.5
    roi_cloud = ((beneficio_estimado - costo_total) / costo_total) * 100
    cur = 0.85
    costo_por_recurso = df.groupby('ResourceId')['BilledCost'].mean()
    unit_economics = costo_por_recurso.mean()
    
    # Nuevas métricas FinOps
    costo_mensual_promedio = df.groupby(df['Fecha'].dt.to_period('M'))['BilledCost'].sum().mean()
    variacion_mensual = df.groupby(df['Fecha'].dt.to_period('M'))['BilledCost'].sum().pct_change().mean() * 100
    eficiencia_recursos = costo_total / len(df['ResourceId'].unique())
    
    # Cálculo del FinOps Score mejorado
    finops_score = (
        (roi_cloud / 150) * 0.25 +
        cur * 0.25 +
        (1 - abs(variacion_mensual) / 100) * 0.2 +
        (1 - eficiencia_recursos / costo_total) * 0.2 +
        (unit_economics / costo_total) * 0.1
    ) * 100
    
    return {
        'costo_total': costo_total,
        'roi_cloud': roi_cloud,
        'cur': cur,
        'unit_economics': unit_economics,
        'costo_mensual_promedio': costo_mensual_promedio,
        'variacion_mensual': variacion_mensual,
        'eficiencia_recursos': eficiencia_recursos,
        'finops_score': finops_score
    }

# Calcular métricas
metricas = calcular_metricas_finops(df)

# Crear pestañas
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Resumen Ejecutivo", 
    "💰 Análisis de Costos", 
    "🔍 Análisis de Recursos",
    "🤖 Predicciones IA",
    "💡 Optimización"
])

with tab1:
    st.header("Resumen Ejecutivo")
    
    # FinOps Score con gráfico de velocímetro
    st.subheader("FinOps Score")
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = metricas['finops_score'],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "FinOps Score", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': 'red'},
                {'range': [50, 80], 'color': 'yellow'},
                {'range': [80, 100], 'color': 'green'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Métricas principales en tarjetas interactivas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Costo Total", f"${metricas['costo_total']:,.2f}", f"{metricas['variacion_mensual']:.1f}% vs mes anterior")
    with col2:
        st.metric("ROI Cloud", f"{metricas['roi_cloud']:.1f}%", "Objetivo: 150%")
    with col3:
        st.metric("Eficiencia de Costos (CUR)", f"{metricas['cur']*100:.1f}%", "Objetivo: 90%")
    with col4:
        st.metric("Costo Unitario Promedio", f"${metricas['unit_economics']:,.2f}", "Por recurso")
    
    # Gráfico de tendencia de costos mejorado
    st.subheader("Tendencia de Costos")
    costos_diarios = df.groupby('Fecha')['BilledCost'].sum().reset_index()
    
    fig = px.area(costos_diarios, x='Fecha', y='BilledCost',
                  labels={'Fecha': 'Fecha', 'BilledCost': 'Costo ($)'},
                  title='Tendencia de Costos Diarios')
    fig.update_traces(fillcolor="rgba(0,100,255,0.2)", line_color="rgba(0,100,255,0.8)")
    fig.update_layout(height=400, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Análisis de Costos")
    
    # Métricas adicionales con visualización mejorada
    col1, col2, col3 = st.columns(3)

    with col1:
        fig = go.Figure(go.Indicator(
            mode="number+delta",
            value=metricas['costo_mensual_promedio'],
            number={
                'prefix': "$",
                'font': {'size': 50, 'color': '#2c3e50'},
                'valueformat': ",.2f"
            },
            title={
                'text': "Costo Mensual Promedio<br><span style='font-size:0.8em;color:gray'>Últimos 30 días</span>",
                'font': {'size': 20}
            },
            delta={
                'reference': metricas['costo_total']/30,
                'relative': True,
                'font': {'size': 20}
            }
        ))
        fig.update_layout(
            height=250,
            margin=dict(t=30, b=0, l=0, r=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure(go.Indicator(
            mode="number+delta",
            value=metricas['variacion_mensual'],
            number={
                'suffix': "%",
                'font': {'size': 50, 'color': '#2c3e50'},
                'valueformat': ".1f"
            },
            title={
                'text': "Variación Mensual<br><span style='font-size:0.8em;color:gray'>vs. Mes Anterior</span>",
                'font': {'size': 20}
            },
            delta={
                'reference': 0,
                'relative': False,
                'font': {'size': 20}
            }
        ))
        fig.update_layout(
            height=250,
            margin=dict(t=30, b=0, l=0, r=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        fig = go.Figure(go.Indicator(
            mode="number+delta",
            value=metricas['eficiencia_recursos'],
            number={
                'prefix': "$",
                'font': {'size': 50, 'color': '#2c3e50'},
                'valueformat': ",.2f"
            },
            title={
                'text': "Eficiencia de Recursos<br><span style='font-size:0.8em;color:gray'>Costo por Recurso</span>",
                'font': {'size': 20}
            },
            delta={
                'reference': metricas['costo_total']/len(df['ResourceId'].unique()),
                'relative': True,
                'font': {'size': 20}
            }
        ))
        fig.update_layout(
            height=250,
            margin=dict(t=30, b=0, l=0, r=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Agregar gráfico de tendencia de costos
    st.subheader("Tendencia de Costos Mensuales")
    costos_mensuales = df.groupby(pd.Grouper(key='Fecha', freq='M'))['BilledCost'].sum().reset_index()
    costos_mensuales['Mes'] = costos_mensuales['Fecha'].dt.strftime('%B %Y')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=costos_mensuales['Mes'],
        y=costos_mensuales['BilledCost'],
        mode='lines+markers',
        name='Costo Mensual',
        line=dict(color='#2c3e50', width=3),
        marker=dict(size=10)
    ))

    # Calcular línea de tendencia
    try:
        z = np.polyfit(range(len(costos_mensuales)), costos_mensuales['BilledCost'], 1)
        p = np.poly1d(z)
        trend_y = p(range(len(costos_mensuales)))
    except np.linalg.LinAlgError:
        # Si polyfit falla, usamos un método más simple
        x = np.arange(len(costos_mensuales))
        y = costos_mensuales['BilledCost'].values
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        trend_y = m * x + c

    fig.add_trace(go.Scatter(
        x=costos_mensuales['Mes'],
        y=trend_y,
        mode='lines',
        name='Tendencia',
        line=dict(color='#e74c3c', width=2, dash='dash')
    ))

    fig.update_layout(
        height=400,
        hovermode='x unified',
        margin=dict(t=30, b=0, l=0, r=0),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        xaxis=dict(
            title="Mes",
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(211,211,211,0.5)'
        ),
        yaxis=dict(
            title="Costo Total ($)",
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(211,211,211,0.5)',
            tickformat="$,.2f"
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    # Agregar análisis de tendencia
    variacion_porcentual = ((costos_mensuales['BilledCost'].iloc[-1] - costos_mensuales['BilledCost'].iloc[0]) / 
                            costos_mensuales['BilledCost'].iloc[0] * 100)
    tendencia = "incremento" if variacion_porcentual > 0 else "reducción"

    st.info(f"""
        **Análisis de Tendencia:**
        - La tendencia general muestra un {tendencia} del {abs(variacion_porcentual):.1f}% en los costos
        - Costo más alto: ${costos_mensuales['BilledCost'].max():,.2f} ({costos_mensuales.loc[costos_mensuales['BilledCost'].idxmax(), 'Mes']})
        - Costo más bajo: ${costos_mensuales['BilledCost'].min():,.2f} ({costos_mensuales.loc[costos_mensuales['BilledCost'].idxmin(), 'Mes']})
        - Costo promedio: ${costos_mensuales['BilledCost'].mean():,.2f}
    """)
    
    # Top servicios por costo con gráfico de barras mejorado
    st.subheader("Top Servicios por Costo")
    costo_por_servicio = df.groupby('ChargeDescription')['BilledCost'].sum().sort_values(ascending=False)
    
    fig = px.bar(costo_por_servicio.head(10), 
                 labels={'index': 'Servicio', 'value': 'Costo ($)'},
                 title='Top 10 Servicios por Costo')
    fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                      marker_line_width=1.5, opacity=0.6)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribución por región con mapa de calor
    st.subheader("Distribución de Costos por Región")
    costo_por_region = df.groupby('x_SkuRegion')['BilledCost'].sum().sort_values(ascending=False)
    
    fig = px.choropleth(
        locations=costo_por_region.index,
        locationmode="country names",
        color=costo_por_region.values,
        hover_name=costo_por_region.index,
        color_continuous_scale=px.colors.sequential.Plasma,
        title="Distribución de Costos por Región"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Análisis de Recursos")
    
    # Top recursos más costosos con gráfico de barras horizontal mejorado
    st.subheader("Top Recursos más Costosos")
    recursos_costosos = df.groupby('ResourceId')['BilledCost'].sum().sort_values(ascending=False).head(10)
    
    fig = px.bar(recursos_costosos, 
                 orientation='h',
                 labels={'index': 'ID del Recurso', 'value': 'Costo ($)'},
                 title='Top 10 Recursos más Costosos')
    fig.update_traces(marker_color='rgb(255,127,14)', marker_line_color='rgb(139,69,19)',
                      marker_line_width=1.5, opacity=0.7)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribución por tipo de cargo con gráfico de dona mejorado
        st.subheader("Distribución por Tipo de Cargo")
        costo_por_tipo = df.groupby('ChargeClass')['BilledCost'].sum().sort_values(ascending=False)
        
        fig = px.pie(values=costo_por_tipo.values, names=costo_por_tipo.index,
                     title='Distribución por Tipo de Cargo', hole=.3)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Análisis de uso por hora del día con gráfico de línea mejorado
        st.subheader("Patrón de Costos por Hora del Día")
        df['Hora'] = df['Fecha'].dt.hour
        uso_por_hora = df.groupby('Hora')['BilledCost'].sum().reset_index()
        
        fig = px.line(uso_por_hora, x='Hora', y='BilledCost',
                      labels={'Hora': 'Hora del Día', 'BilledCost': 'Costo ($)'},
                      title='Patrón de Costos por Hora del Día')
        fig.update_traces(line=dict(color="royalblue", width=4))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Predicciones con Inteligencia Artificial")
    
    # Preparar datos para el modelo
    def preparar_datos_modelo(df):
        df_modelo = df.copy()
        df_modelo['DiaSemana'] = df_modelo['Fecha'].dt.dayofweek
        df_modelo['Mes'] = df_modelo['Fecha'].dt.month
        df_modelo['Dia'] = df_modelo['Fecha'].dt.day
        
        X = df_modelo[['DiaSemana', 'Mes', 'Dia']].values
        y = df_modelo['BilledCost'].values
        
        return X, y

    # Entrenar modelo
    X, y = preparar_datos_modelo(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Escalar datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Entrenar modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Predicciones
    future_dates = pd.date_range(start=df['Fecha'].max(), periods=30, freq='D')
    future_X = np.array([[d.dayofweek, d.month, d.day] for d in future_dates])
    future_X_scaled = scaler.transform(future_X)
    predictions = model.predict(future_X_scaled)

    # Visualizar predicciones con gráfico de línea mejorado
    fig = go.Figure()

    # Datos históricos
    fig.add_trace(go.Scatter(
        x=df['Fecha'],
        y=df['BilledCost'],
        name='Costos Históricos',
        line=dict(color='blue', width=2)
    ))

    # Predicciones
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=predictions,
        name='Predicciones',
        line=dict(color='red', width=2, dash='dash')
    ))

    fig.update_layout(
        title='Predicción de Costos para los Próximos 30 Días',
        xaxis_title='Fecha',
        yaxis_title='Costo Previsto ($)',
        height=500,
        legend=dict(y=0.5, traceorder='reversed', font_size=16)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Métricas de predicción mejoradas
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Indicator(
            mode = "number+delta",
            value = predictions.sum(),
            number = {'prefix': "$"},
            title = {"text": "Costo Total Previsto (30 días)"},
            delta = {'position': "top", 'reference': df['BilledCost'].sum(), 'relative': True}))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = go.Figure(go.Indicator(
            mode = "number+delta",
            value = predictions.mean(),
            number = {'prefix': "$"},
            title = {"text": "Costo Diario Promedio Previsto"},
            delta = {'position': "top", 'reference': df['BilledCost'].mean(), 'relative': True}))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Análisis de factores de influencia con gráfico de barras mejorado
    st.subheader("Factores que Influyen en el Costo")
    importances = model.feature_importances_
    feature_names = ['Día de la Semana', 'Mes', 'Día del Mes']
    
    fig = px.bar(x=feature_names, y=importances,
                 labels={'x': 'Factor', 'y': 'Importancia'},
                 title='Importancia de los Factores en la Predicción de Costos')
    fig.update_traces(marker_color='rgb(55,126,184)', marker_line_color='rgb(8,48,107)',
                      marker_line_width=1.5, opacity=0.7)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.header("Optimización y Ahorro de Costos")

    # Calcular ahorros potenciales
    def calcular_ahorros_potenciales(df):
        costo_por_servicio = df.groupby('ChargeDescription')['BilledCost'].sum().sort_values(ascending=False)
        top_servicios = costo_por_servicio.head(10)
        
        # Estimación de ahorros basada en mejores prácticas
        ahorros_estimados = {
            'Compute Instance': 0.30,
            'Object Storage': 0.25,
            'Block Storage': 0.20,
            'Data Transfer': 0.15,
            'Managed Database': 0.20,
            'Load Balancer': 0.15,
            'Content Delivery': 0.10,
            'Serverless': 0.25,
            'Monitoring': 0.20,
            'Other': 0.10
        }
        
        ahorros = {}
        for servicio, costo in top_servicios.items():
            factor_ahorro = next((v for k, v in ahorros_estimados.items() if k.lower() in servicio.lower()), ahorros_estimados['Other'])
            ahorros[servicio] = costo * factor_ahorro
        
        return ahorros

    ahorros_potenciales = calcular_ahorros_potenciales(df)
    total_ahorros = sum(ahorros_potenciales.values())

    # Mostrar ahorros potenciales con gráfico de indicador
    st.subheader("Ahorros Potenciales Totales")
    fig = go.Figure(go.Indicator(
        mode = "number+gauge+delta",
        value = total_ahorros,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Ahorros Potenciales ($)", 'font': {'size': 24}},
        delta = {'reference': metricas['costo_total'] * 0.2, 'increasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, metricas['costo_total']], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkgreen"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, metricas['costo_total'] * 0.1], 'color': 'lightgreen'},
                {'range': [metricas['costo_total'] * 0.1, metricas['costo_total'] * 0.2], 'color': 'green'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': metricas['costo_total'] * 0.2}}))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    # Visualización de ahorros por servicio con gráfico de barras mejorado
    st.subheader("Ahorros Potenciales por Servicio")
    fig = px.bar(x=list(ahorros_potenciales.keys()), y=list(ahorros_potenciales.values()),
                 labels={'x': 'Servicio', 'y': 'Ahorro Potencial ($)'},
                 title='Ahorros Potenciales por Servicio')
    fig.update_traces(marker_color='rgb(50,205,50)', marker_line_color='rgb(0,100,0)',
                      marker_line_width=1.5, opacity=0.7)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Recomendaciones detalladas
    st.subheader("Recomendaciones de Optimización")

    recomendaciones = {
        'Compute Instance': [
            "Utilizar instancias reservadas para cargas de trabajo constantes",
            "Implementar auto-scaling para ajustar la capacidad según la demanda",
            "Migrar a instancias de nueva generación más eficientes"
        ],
        'Object Storage': [
            "Implementar políticas de ciclo de vida para mover datos menos accedidos",
            "Optimizar la compresión de objetos",
            "Revisar y eliminar datos obsoletos"
        ],
        'Block Storage': [
            "Ajustar el tamaño de los volúmenes según el uso real",
            "Utilizar tipos de almacenamiento más económicos para datos menos críticos",
            "Implementar snapshots incrementales"
        ],
        'Data Transfer': [
            "Optimizar rutas de red",
            "Utilizar CDN para reducir transferencia",
            "Implementar caché en el cliente"
        ],
        'Managed Database': [
            "Optimizar consultas y esquemas",
            "Utilizar réplicas de lectura",
            "Considerar servicios serverless"
        ]
    }

    for servicio, ahorro in sorted(ahorros_potenciales.items(), key=lambda x: x[1], reverse=True):
        with st.expander(f"{servicio} - Ahorro Potencial: ${ahorro:,.2f}"):
            if servicio in recomendaciones:
                for rec in recomendaciones[servicio]:
                    st.markdown(f"• {rec}")
            else:
                st.markdown("• Realizar análisis detallado del uso")

    # Plan de acción personalizado mejorado
    st.subheader("Plan de Acción Personalizado")
    
    acciones = [
        "Optimizar instancias de cómputo",
        "Revisar políticas de almacenamiento",
        "Mejorar estrategias de red",
        "Optimizar bases de datos",
        "Implementar monitoreo avanzado"
    ]
    
    for i, accion in enumerate(acciones, 1):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{i}. {accion}**")
        with col2:
            completado = st.checkbox(f"Completado", key=f"accion_{i}")
        if completado:
            st.success(f"¡Acción '{accion}' completada!")
        else:
            st.info(f"Acción '{accion}' pendiente")
        st.markdown("---")

# Pie de página
st.markdown("---")
st.markdown(f"*Dashboard FinOps TI724 - Actualizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")


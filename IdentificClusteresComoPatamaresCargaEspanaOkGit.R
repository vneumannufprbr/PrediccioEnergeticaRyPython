# ----------------------------------------------------------------
# ANÁLISIS DE CLÚSTERES PARA LA DEMANDA DE ENERGÍA DE ESPAÑA
# ----------------------------------------------------------------

# Paso 0: Instalar paquetes si no los tienes
# Descomenta y ejecuta estas líneas una vez si es necesario.
# install.packages("dplyr")
# install.packages("ggplot2")
# install.packages("cluster")
# install.packages("factoextra")

# Cargar las librerías necesarias para el análisis
library(dplyr)
library(ggplot2)
library(cluster)
library(factoextra)

# ---
# Paso 1: Cargar y preparar los datos
# ---

### Lectura del archivo de datos BancoDatos.csv de Github del Prof. Neumann
url1 <- "https://raw.githubusercontent.com/vneumannufprbr/TrabajosRStudio/main/energy_dataset.csv"
energy_data<- read.csv(url1, stringsAsFactors = FALSE)
View(energy_data)
# Lee el archivo .csv desde el directorio de trabajo.
#tryCatch({
#  energy_data <- read.csv("energy_dataset.csv")
#}, error = function(e) {
#  stop("Error: El archivo 'energy_dataset.csv' no se encuentra. Asegúrate de que esté en tu directorio de trabajo.")
#})

# Seleccionar la columna de interés usando el nombre correcto (con puntos).
demand_data <- energy_data %>%
  select(total_load = total.load.actual) %>%
  na.omit()

# Escalar los datos es un paso crucial para algoritmos basados en distancia.
demand_scaled <- scale(demand_data)

# ---
# Paso 2: Determinar el número óptimo de clústeres (k) con el método del codo
# ---

# Se establece una semilla para que los resultados aleatorios sean reproducibles.
set.seed(123)

# La función fviz_nbclust calcula y grafica el método del codo automáticamente.
elbow_plot <- fviz_nbclust(demand_scaled, kmeans, method = "wss", k.max = 10) +
  geom_vline(xintercept = 3, linetype = 2, color = "steelblue") +
  labs(
    title = "Método del Codo para Determinar k Óptimo",
    subtitle = "El punto de inflexión sugiere k=3",
    x = "Número de Clústeres (k)",
    y = "Suma Total de Cuadrados Internos (WSS)"
  ) +
  theme_minimal()

# Mostrar el gráfico del codo.
print(elbow_plot)

# ---
# Paso 3: Crear un dendrograma con clustering jerárquico
# ---

# Se mantiene la misma lógica de muestreo y cálculo de clustering jerárquico.
set.seed(456)
sample_size <- 500
demand_sample_scaled <- demand_scaled[sample(nrow(demand_scaled), sample_size), ]
dist_matrix <- dist(demand_sample_scaled, method = "euclidean")
hierarchical_cluster <- hclust(dist_matrix, method = "ward.D2")


# En lugar de fviz_dend, usamos las funciones base de R que son más estables.

# 1. Graficar el dendrograma base
plot(hierarchical_cluster,
     main = "Dendrograma de Demanda Energética (Muestra de 500 puntos)",
     xlab = "Muestras de Demanda",
     ylab = "Altura (Distancia de Ward)",
     sub = "",
     labels = FALSE) # Ocultamos las etiquetas individuales para mayor claridad

# 2. Añadir los rectángulos de colores para visualizar los 3 clústeres
rect.hclust(hierarchical_cluster,
            k = 3, # Cortar en 3 clústeres
            border = c("#E7B800", "#2E9FDF", "#00AFBB"))

# ---
# Paso 4: Aplicar el clustering final y visualizar los resultados
# ---

optimal_k <- 3
set.seed(123)
final_clusters <- kmeans(demand_scaled, centers = optimal_k, nstart = 25)

demand_data$cluster <- as.factor(final_clusters$cluster)

cluster_centers <- aggregate(total_load ~ cluster, data = demand_data, FUN = mean)
cluster_centers <- cluster_centers[order(cluster_centers$total_load), ]

level_names <- c("Demanda Baja", "Demanda Media", "Demanda Alta")
cluster_centers$level <- level_names

demand_data <- merge(demand_data, cluster_centers[, c("cluster", "level")], by = "cluster")

# Crear el boxplot final para la visualización de los patamares.
ggplot(demand_data, aes(x = level, y = total_load, fill = level)) +
  geom_boxplot(alpha = 0.8) +
  scale_fill_manual(values = c("Demanda Baja" = "#2E9FDF", "Demanda Media" = "#00AFBB", "Demanda Alta" = "#E7B800")) +
  labs(
    title = "Distribución de la Demanda de Energía por Patamar",
    subtitle = "Clústeres identificados mediante k-Means (k=3)",
    x = "Nivel de Demanda (Patamar)",
    y = "Demanda Total (MW)"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

#  Gráfico de Densidad
ggplot(demand_data, aes(x = total_load, fill = level)) +
  geom_density(alpha = 0.6) + # El 'alpha' crea transparencia para ver las superposiciones
  scale_fill_manual(values = c("Demanda Baja" = "#2E9FDF", "Demanda Media" = "#00AFBB", "Demanda Alta" = "#E7B800")) +
  labs(
    title = "Gráfico de Densidad de los Patamares de Demanda",
    subtitle = "Muestra la distribución y superposición de cada clúster",
    x = "Demanda Total (MW)",
    y = "Densidad",
    fill = "Patamar"
  ) +
  theme_minimal()

# Serie Temporal Coloreada por Clúster 
# Primero, añadimos un índice de tiempo para el eje X
demand_data_con_tiempo <- demand_data %>%
  mutate(time_index = 1:n()) # n() es el número total de filas

# Graficamos la serie temporal COMPLETA, coloreando por nivel de demanda
# Se elimina sample_n() porque el dataset tiene solo 3550 puntos ? Puede cambiar.
ggplot(demand_data_con_tiempo, aes(x = time_index, y = total_load, color = level)) +
  geom_line(alpha = 0.8) +
  scale_color_manual(values = c("Demanda Baja" = "#2E9FDF", "Demanda Media" = "#00AFBB", "Demanda Alta" = "#E7B800")) +
  labs(
    title = "Serie Temporal de Demanda Coloreada por Patamar",
    subtitle = "Muestra cuándo ocurre cada nivel de demanda",
    x = "Tiempo (Índice de Observación)",
    y = "Demanda Total (MW)",
    color = "Patamar"
  ) +
  theme_minimal()
  
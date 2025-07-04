# ----------------------------------------------------------------
# ANÁLISIS DE CLÚSTERES PARA LA DEMANDA DE ENERGÍA DE ESPAÑA
# ----------------------------------------------------------------

# Cargar las librerías
library(dplyr)
library(ggplot2)
library(cluster)
library(factoextra)

# ---
# Paso 1: Cargar y preparar los datos
# ---
url1 <- "https://raw.githubusercontent.com/vneumannufprbr/TrabajosRStudio/main/energy_dataset.csv"
energy_data<- read.csv(url1, stringsAsFactors = FALSE)

demand_data <- energy_data %>%
  select(total_load = total.load.actual) %>%
  na.omit()

demand_scaled <- scale(demand_data)

# ---
# Paso 2: Gráfico del Codo
# ---
set.seed(123)
elbow_plot <- fviz_nbclust(demand_scaled, kmeans, method = "wss", k.max = 10) +
  labs(
    title = "Método del Codo para Optimización de k",
    x = "Número de Clústeres (k)",
    y = "Suma Total de los Cuadrados Internos (WSS)"
  )
print(elbow_plot)

# PAUSA HASTA PRESIONAR ENTER
readline(prompt="GRÁFICO 1/5: Codo. Presiona [Enter] en la consola para continuar...")

# ---
# Paso 3: Dendrograma
# ---
set.seed(456)
# Tomamos una muestra más pequeña solo para el dendrograma, para que sea legible
sample_for_dendro <- demand_scaled[sample(nrow(demand_scaled), 500), ]
dist_matrix <- dist(sample_for_dendro, method = "euclidean")
hierarchical_cluster <- hclust(dist_matrix, method = "ward.D2")

plot(hierarchical_cluster,
     main = "Dendrograma de la Demanda de Carga",
     xlab = "Muestras",
     ylab = "Altura (Distancia de Ward)",
     labels = FALSE,
     sub = NULL)
rect.hclust(hierarchical_cluster, k = 3, border = "red")

# PAUSA
readline(prompt="GRÁFICO 2/5: Dendrograma. Presiona [Enter] en la consola para continuar...")

# ---
# Paso 4: Clustering y Visualizaciones Finales
# ---
optimal_k <- 3
set.seed(123)
final_clusters <- kmeans(demand_scaled, centers = optimal_k, nstart = 25)
demand_data$cluster <- as.factor(final_clusters$cluster)

# Renombrar los clústeres
cluster_centers <- aggregate(total_load ~ cluster, data = demand_data, FUN = mean)
cluster_centers <- cluster_centers[order(cluster_centers$total_load), ]
level_names <- c("Carga Baja", "Carga Media", "Carga Alta") # Traducido
cluster_centers$level <- level_names
demand_data <- merge(demand_data, cluster_centers[, c("cluster", "level")], by = "cluster")

# Gráfico de Cajas (Boxplot)
boxplot_plot <- ggplot(demand_data, aes(x = level, y = total_load, fill = level)) +
  geom_boxplot() +
  labs(
    title = "Distribución por Patamar de Carga",
    x = "Patamar de Carga",
    y = "Carga Total (MW)",
    fill = "Patamar"
  )
print(boxplot_plot)

# PAUSA
readline(prompt="GRÁFICO 3/5: Boxplot. Presiona [Enter] en la consola para continuar...")

# Gráfico de Densidad
density_plot <- ggplot(demand_data, aes(x = total_load, fill = level)) +
  geom_density(alpha = 0.7) +
  labs(
    title = "Densidad de los Patamares de Carga",
    x = "Carga Total (MW)",
    y = "Densidad",
    fill = "Patamar"
  )
print(density_plot)

# PAUSA
readline(prompt="GRÁFICO 4/5: Densidad. Presiona [Enter] en la consola para continuar...")

# Serie Temporal Coloreada
demand_data_con_tiempo <- demand_data %>% mutate(time_index = 1:n())
timeseries_plot <- ggplot(demand_data_con_tiempo, aes(x = time_index, y = total_load, color = level)) +
  geom_line(alpha = 0.8) +
  labs(
    title = "Serie Temporal de la Carga por Patamar",
    x = "Tiempo (Índice de Observación)",
    y = "Carga Total (MW)",
    color = "Patamar"
  )
print(timeseries_plot)

# PAUSA FINAL
readline(prompt="GRÁFICO 5/5: Serie Temporal. Presiona [Enter] en la consola para finalizar.")
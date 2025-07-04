# ----------------------------------------------------------------
# ANÁLISIS CON DETERMINACIÓN AUTOMÁTICA DE K (VERSIÓN CORREGIDA)
# ----------------------------------------------------------------

# Cargar las librerías
library(dplyr)
library(ggplot2)
library(cluster)

# ---
# Paso 1: Cargar y preparar los datos (sin cambios)
# ---
url1 <- "https://raw.githubusercontent.com/vneumannufprbr/TrabajosRStudio/main/energy_dataset.csv"
energy_data<- read.csv(url1, stringsAsFactors = FALSE)

demand_data <- energy_data %>%
  select(total_load = total.load.actual) %>%
  na.omit()

demand_scaled <- scale(demand_data)

# ---
# Paso 2: Determinar k Óptimo (Método Matemático Corregido)
# ---
set.seed(123)
wcss <- vector()

# 1. Calcular WCSS para k de 1 a 10
for (i in 1:10) {
  wcss[i] <- kmeans(demand_scaled, centers = i, nstart = 25)$tot.withinss
}

# 2. Graficar los resultados para visualización
plot(1:10, wcss, type = "b", pch = 19, frame = FALSE,
     xlab = "Número de Clústeres (k)",
     ylab = "Suma Total de Cuadrados Internos (WCSS)",
     main = "Método del Codo (Visualización)")

# 3. Calcular la segunda derivada de WCSS
d2wcss <- diff(diff(wcss))

# 4. Encontrar el 'k' con la fórmula CORRECTA (+2)
optimal_k <- which.max(d2wcss) + 2 # 

# Imprimir el resultado en la consola
cat("El número óptimo de clústeres determinado automáticamente es:", optimal_k, "\n")

# PAUSA HASTA PRESIONAR ENTER
readline(prompt="GRÁFICO 1/5: Codo. Presiona [Enter] en la consola para continuar...")

# ---
# Paso 3: Dendrograma (Ahora usa el k óptimo encontrado)
# ---
set.seed(456)
sample_for_dendro <- demand_scaled[sample(nrow(demand_scaled), 500), ]
dist_matrix <- dist(sample_for_dendro, method = "euclidean")
hierarchical_cluster <- hclust(dist_matrix, method = "ward.D2")

plot(hierarchical_cluster,
     main = "Dendrograma de la Demanda de Carga",
     xlab = "Muestras", ylab = "Altura (Distancia de Ward)",
     labels = FALSE, sub = NULL)
rect.hclust(hierarchical_cluster, k = optimal_k, border = "red")

# PAUSA
readline(prompt="GRÁFICO 2/5: Dendrograma. Presiona [Enter] en la consola para continuar...")

# ---
# Paso 4: Clustering y Visualizaciones Finales (Ahora usa el k óptimo)
# ---
set.seed(123)
final_clusters <- kmeans(demand_scaled, centers = optimal_k, nstart = 25)
demand_data$cluster <- as.factor(final_clusters$cluster)

# Renombrar los clústeres
cluster_centers <- aggregate(total_load ~ cluster, data = demand_data, FUN = mean)
cluster_centers <- cluster_centers[order(cluster_centers$total_load), ]
level_names <- c("Carga Baja", "Carga Media", "Carga Alta")
cluster_centers$level <- level_names
demand_data <- merge(demand_data, cluster_centers[, c("cluster", "level")], by = "cluster")

# Gráfico de Cajas (Boxplot)
boxplot_plot <- ggplot(demand_data, aes(x = level, y = total_load, fill = level)) +
  geom_boxplot() +
  labs(
    title = "Distribución por Patamar de Carga", x = "Patamar de Carga",
    y = "Carga Total (MW)", fill = "Patamar"
  )
print(boxplot_plot)

# PAUSA
readline(prompt="GRÁFICO 3/5: Boxplot. Presiona [Enter] en la consola para continuar...")

# Gráfico de Densidad
density_plot <- ggplot(demand_data, aes(x = total_load, fill = level)) +
  geom_density(alpha = 0.7) +
  labs(
    title = "Densidad de los Patamares de Carga", x = "Carga Total (MW)",
    y = "Densidad", fill = "Patamar"
  )
print(density_plot)

# PAUSA
readline(prompt="GRÁFICO 4/5: Densidad. Presiona [Enter] en la consola para continuar...")

# Serie Temporal Coloreada
demand_data_con_tiempo <- demand_data %>% mutate(time_index = 1:n())
timeseries_plot <- ggplot(demand_data_con_tiempo, aes(x = time_index, y = total_load, color = level)) +
  geom_line(alpha = 0.8) +
  labs(
    title = "Serie Temporal de la Carga por Patamar", x = "Tiempo (Índice de Observación)",
    y = "Carga Total (MW)", color = "Patamar"
  )
print(timeseries_plot)

# PAUSA FINAL
readline(prompt="GRÁFICO 5/5: Serie Temporal. Presiona [Enter] en la consola para finalizar.")
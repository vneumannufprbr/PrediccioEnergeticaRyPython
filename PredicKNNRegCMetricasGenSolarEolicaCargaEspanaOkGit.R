# Instalar paquetes (si no están instalados)
install.packages("tidyverse")   # Para manipulación de datos y gráficos
install.packages("caret")       # Para entrenamiento de modelos
install.packages("lubridate")   # Para manejo de fechas
install.packages("kknn")        # Implementación de K-NN
install.packages("FNN")        # Implementación eficiente de K-NN

# Cargar librerías necesarias
library(dplyr)
library(FNN)
library(lubridate)
library(ggplot2)
library(tidyr)

# Leer y preparar los datos (corrección en el parsing de fechas)
# data <- read.csv("C:/Users/UFPR/Transmissão no Google Drive/Outros computadores/GoogleDrive/Google Drive/Especialização IA UFPR/AAA-Curso IA UFPR/Arquitetura de Dados/Artigos de AI aplicada a Sistemas de Energia/ENCONTRO 02/archive/energy_dataset.csv", stringsAsFactors = FALSE)
### Lectura del archivo de datos BancoDatos.csv de Github del Prof. Neumann
url1 <- "https://raw.githubusercontent.com/vneumannufprbr/TrabajosRStudio/main/energy_dataset.csv"
data<- read.csv(url1, stringsAsFactors = FALSE)
View(data)

# Verificar la columna 'time'
head(data$time)  # Deberías ver algo como "2015-01-01 00:00:00+01:00"

# Parsear la columna 'time' correctamente
data <- data %>%
  mutate(
    time = ymd_hms(time)  # Parsear sin especificar la zona horaria
  ) %>%
  arrange(time)  # Ordenar por tiempo

# Verificar que las fechas se han parseado correctamente
head(data$time)  # Deberías ver fechas en formato POSIXct

# Seleccionar variables objetivo
targets <- c("generation.solar", "generation.wind.onshore", "total.load.actual")
results <- list()
metrics <- list()

# Parámetros del modelo
window_size <- 24 # 24 horas, anterior 168 horas, Ventana de 7 días (24*7)
k_value <- 5
test_size <- 24 # 24 horas, anterior: 30 * 24, 30 días para evaluación
forecast_horizon <- 24 # 24 horas, anterior: 30 * 24, 30 días para pronóstico futuro

# Función para crear características
create_features <- function(serie, window) {
  n <- length(serie)
  features <- matrix(NA, nrow = n - window, ncol = window)
  for(i in 1:window) {
    features[, i] <- serie[i:(n - window + i - 1)]
  }
  target <- serie[(window + 1):n]
  return(data.frame(features, target))
}

# Función para calcular métricas
calculate_metrics <- function(actual, predicted) {
  rmse <- sqrt(mean((actual - predicted)^2))
  mae <- mean(abs(actual - predicted))
  ss_res <- sum((actual - predicted)^2)
  ss_tot <- sum((actual - mean(actual))^2)
  r_squared <- 1 - (ss_res/ss_tot)
  
  return(data.frame(R2 = r_squared, RMSE = rmse, MAE = mae))
}

# Evaluación y pronóstico para cada variable
for(target_var in targets) {
  # Extraer serie completa
  serie_full <- data[[target_var]] %>% na.omit() %>% as.numeric()
  
  # 1. Evaluación en conjunto de prueba -------------------------------------------------
  # Dividir en entrenamiento y prueba
  n_full <- length(serie_full)
  train_series <- serie_full[1:(n_full - test_size)]
  test_series <- serie_full[(n_full - test_size + 1):n_full]
  
  # Crear matriz de entrenamiento
  train_data <- create_features(train_series, window_size)
  
  # Pronóstico recursivo para el conjunto de prueba
  last_window <- tail(train_series, window_size)
  test_predictions <- numeric(test_size)
  
  for(i in 1:test_size) {
    knn_model <- knn.reg(
      train = train_data[, 1:window_size],
      test = matrix(last_window, nrow = 1),
      y = train_data$target,
      k = k_value
    )
    test_predictions[i] <- knn_model$pred
    last_window <- c(last_window[-1], test_predictions[i])
  }
  
  # Calcular métricas de evaluación
  actual_test <- test_series[1:test_size]
  metrics[[target_var]] <- calculate_metrics(actual_test, test_predictions)
  
  # 2. Pronóstico futuro usando toda la data --------------------------------------------
  # Crear matriz con toda la data
  full_data <- create_features(serie_full, window_size)
  
  # Pronóstico recursivo
  last_window_full <- tail(serie_full, window_size)
  future_predictions <- numeric(forecast_horizon)
  
  for(i in 1:forecast_horizon) {
    knn_model <- knn.reg(
      train = full_data[, 1:window_size],
      test = matrix(last_window_full, nrow = 1),
      y = full_data$target,
      k = k_value
    )
    future_predictions[i] <- knn_model$pred
    last_window_full <- c(last_window_full[-1], future_predictions[i])
  }
  
  # Almacenar resultados
  results[[target_var]] <- future_predictions
}

# Mostrar métricas de evaluación
# Puede demorar unos minutos
cat("Métricas de Evaluación en Conjunto de Prueba:\n")
for(target_var in targets) {
  cat("\nVariable:", target_var, "\n")
  print(metrics[[target_var]])
}

# Generar fechas futuras
last_date <- tail(data$time, 1)
future_dates <- seq(last_date + hours(1), by = "hour", length.out = forecast_horizon)

# Crear dataframe con pronósticos futuros
forecast_df <- data.frame(
  time = future_dates,
  solar = results$generation.solar,
  wind_onshore = results$generation.wind.onshore,
  total_load = results$total.load.actual
)

# Visualización de pronósticos
# Va tardar unos minutos!
forecast_df %>%
  gather(key = "variable", value = "value", -time) %>%
  ggplot(aes(x = time, y = value, color = variable)) +
  geom_line() +
  facet_wrap(~variable, scales = "free_y", ncol = 1) +
  labs(title = "Pronóstico a 24 horas usando KNN",
       x = "Fecha", y = "Valor") +
  theme_minimal()


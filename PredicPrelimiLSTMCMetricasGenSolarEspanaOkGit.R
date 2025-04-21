# ============================
# CARGA DE LIBRERÍAS
# Antes debe haberse ejecutado InstalaryConfigurarKerasyTensorflowenR
# ============================
library(tensorflow)
library(keras)
library(reticulate)
library(abind)
library(tidyverse)
library(ggplot2)

# Activar entorno virtual
use_virtualenv("r-tensorflow", required = TRUE)

# ============================
# PARÁMETROS (ENTEROS EXPLÍCITOS)
# ============================
sequence_length <- 24L  # L para enteros
batch_size <- 32L
lstm_units <- 50L
epochs <- 30L

# ============================
# CARGA Y PREPROCESAMIENTO DE DATOS
# ============================
url <- "https://raw.githubusercontent.com/vneumannufprbr/TrabajosRStudio/main/energy_dataset.csv"
df <- read.csv(url, stringsAsFactors = FALSE)

# Limpieza y normalización
df$generation.solar <- as.numeric(df$generation.solar)
df <- df %>% filter(!is.na(generation.solar))

normalize <- function(x) {
  (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
}
serie <- normalize(df$generation.solar)

# ============================
# CREACIÓN DE SECUENCIAS (CONVERSIÓN A ENTEROS)
# ============================
create_sequences <- function(series, seq_length) {
  n_samples <- as.integer(length(series) - seq_length)
  X <- array(0, dim = c(n_samples, as.integer(seq_length), 1L))
  y <- numeric(n_samples)
  
  for (i in 1:n_samples) {
    X[i,,1] <- series[i:(i + seq_length - 1L)]
    y[i] <- series[i + seq_length]
  }
  
  return(list(X = X, y = y))
}

sequences <- create_sequences(serie, sequence_length)

# ============================
# DIVISIÓN ENTRENAMIENTO/PRUEBA
# ============================
total_samples <- dim(sequences$X)[1]
train_size <- as.integer(floor(0.8 * total_samples))

trainX <- sequences$X[1:train_size,, , drop = FALSE]
testX <- sequences$X[(train_size + 1L):total_samples,, , drop = FALSE]

trainY <- matrix(sequences$y[1:train_size], ncol = 1L)
testY <- matrix(sequences$y[(train_size + 1L):total_samples], ncol = 1L)

# ============================
# CONVERSIÓN A FLOAT32
# ============================
np <- import("numpy")
trainX <- np$array(trainX, dtype = np$float32)
testX <- np$array(testX, dtype = np$float32)

# ============================
# MODELO LSTM (UNIDADES EXPLÍCITAS COMO ENTEROS)
# ============================
inputs <- layer_input(shape = c(sequence_length, 1L))  # 1L
x <- layer_lstm(units = lstm_units)(inputs)            # 50L
outputs <- layer_dense(units = 1L)(x)                 # 1L

model <- keras_model(inputs = inputs, outputs = outputs)

model$compile(
  loss = "mse",
  optimizer = "adam",
  metrics = list("mae")
)

# ============================
# ENTRENAMIENTO (PARÁMETROS COMO ENTEROS)
# ============================
history <- model$fit(
  x = trainX,
  y = trainY,
  epochs = epochs,            # 30L
  batch_size = batch_size,    # 32L
  validation_split = 0.2,
  verbose = 1L
)

# ============================
# PREDICCIÓN 
# ============================
pred_test <- model$predict(testX)  # Usar $predict() del modelo

# Verificar dimensiones
print(dim(pred_test))  # Debe ser (708, 1)

# ============================
# VISUALIZACIÓN
# ============================
df_result <- data.frame(
  Observed = as.numeric(testY),
  Predicted = as.numeric(pred_test)
)

ggplot(df_result, aes(x = 1:nrow(df_result))) +
  geom_line(aes(y = Observed), color = "blue", alpha = 0.6) +
  geom_line(aes(y = Predicted), color = "red", alpha = 0.6) +
  labs(title = "Generación Solar: Real vs Predicción LSTM",
       x = "Tiempo",
       y = "Generación Solar (normalizado)") +
  theme_minimal()

# ==========================================
# CÁLCULO DE MÉTRICAS Y DESNORMALIZACIÓN
# ==========================================

# 1. Valores observados y predichos (normalizados)
observed <- df_result$Observed  # Corregir mayúscula
predicted <- df_result$Predicted

# 2. Métricas en escala normalizada
mse <- mean((observed - predicted)^2)
rmse <- sqrt(mse)
mae <- mean(abs(observed - predicted))
ss_total <- sum((observed - mean(observed))^2)
ss_residual <- sum((observed - predicted)^2)
r_squared <- 1 - (ss_residual / ss_total)

# 3. Desnormalizar valores (si es necesario)
if (exists("df") && "generation.solar" %in% colnames(df)) {
  denormalize <- function(x_norm, original_series) {
    min_val <- min(original_series, na.rm = TRUE)
    max_val <- max(original_series, na.rm = TRUE)
    x_norm * (max_val - min_val) + min_val
  }
  
  observed_denorm <- denormalize(observed, df$generation.solar)
  predicted_denorm <- denormalize(predicted, df$generation.solar)
  
  # Métricas en escala original
  mse_denorm <- mean((observed_denorm - predicted_denorm)^2)
  rmse_denorm <- sqrt(mse_denorm)
  mae_denorm <- mean(abs(observed_denorm - predicted_denorm))
  
  cat("\nMétricas en escala ORIGINAL:\n")
  cat("----------------------------\n")
  cat(paste("MSE:", round(mse_denorm, 2), "\n"))
  cat(paste("RMSE:", round(rmse_denorm, 2), "\n"))
  cat(paste("MAE:", round(mae_denorm, 2), "\n"))
}

# Resultados escala normalizada
cat("\nMétricas en escala NORMALIZADA:\n")
cat("------------------------------\n")
cat(paste("MSE:", round(mse, 5), "\n"))
cat(paste("RMSE:", round(rmse, 5), "\n"))
cat(paste("MAE:", round(mae, 5), "\n"))
cat(paste("R²:", round(r_squared, 5), "\n"))


# ============================
# GUARDAR MODELO en formato .keras
# ============================
model$save("modelo_lstm_generation_solar.keras")

# ============================
# Para cargar el modelo después
# ============================
# modelo_cargado <- load_model_tf("modelo_lstm_generation_solar.keras")

# ================================================
# PREDICCIÓN PARA LOS PRÓXIMOS 30 DÍAS (720 HORAS)
# ================================================

# -------------------------------------------------
# FUNCIÓN PARA PREDICCIÓN RECURSIVA
# -------------------------------------------------
predict_future <- function(model, last_sequence, steps = 720L, sequence_length = 24L) {
  np <- import("numpy")
  
  # Convertir la última secuencia a array de NumPy (float32)
  current_window <- np$array(
    array(last_sequence, dim = c(1, sequence_length, 1)),
    dtype = np$float32
  )
  
  predictions <- numeric(steps)
  
  for (i in 1:steps) {
    # Paso 1: Predecir siguiente valor
    next_pred <- model$predict(current_window)
    
    # Paso 2: Almacenar predicción
    predictions[i] <- as.numeric(next_pred)
    
    # Paso 3: Actualizar ventana (eliminar primer elemento + añadir predicción)
    current_window <- np$array(
      array(
        data = c(current_window[1, -1, 1], next_pred),
        dim = c(1, sequence_length, 1)
      ),
      dtype = np$float32
    )
  }
  
  return(predictions)
}

# -------------------------------------------------
# EJECUCIÓN PRINCIPAL
# -------------------------------------------------

# 1. Obtener última secuencia de 24 horas (normalizada)
last_24h_normalized <- tail(serie, sequence_length)

# 2. Generar predicciones (normalizadas)
predicciones_normalizadas <- predict_future(
  model = model,
  last_sequence = last_24h_normalized,
  steps = 720L
)

# 3. Desnormalizar predicciones
denormalize <- function(x_norm) {
  min_val <- min(df$generation.solar, na.rm = TRUE)
  max_val <- max(df$generation.solar, na.rm = TRUE)
  x_norm * (max_val - min_val) + min_val
}

predicciones_MW <- denormalize(predicciones_normalizadas)

# 4. Crear dataframe con fechas futuras
ultima_fecha <- as.POSIXct(tail(df$time, 1))
fechas_prediccion <- ultima_fecha + hours(1:720)

df_predicciones <- data.frame(
  Fecha = fechas_prediccion,
  Generacion_MW = round(predicciones_MW, 2)
)

# 5. Visualizar primeros 7 días
ggplot(df_predicciones[1:168, ], aes(x = Fecha, y = Generacion_MW)) +
  geom_line(color = "#FF6B6B", linewidth = 0.8) +
  labs(
    title = "Predicción de Generación Solar - Próximos 7 Días",
    x = "Fecha",
    y = "Generación (MW)"
  ) +
  theme_minimal() +
  scale_x_datetime(date_labels = "%d %b %H:%M")

# 6. Guardar resultados
write_csv(df_predicciones, "prediccion_30dias_generacion_solar.csv")

# -------------------------------------------------
# VERIFICACIONES FINALES
# -------------------------------------------------

# Dimensiones correctas
print(dim(df_predicciones))  # Debe ser: 720 filas x 2 columnas

# Ejemplo de salida
head(df_predicciones)





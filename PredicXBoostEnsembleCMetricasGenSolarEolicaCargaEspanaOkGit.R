# --------------------------------------
# INSTALACIÓN Y CARGA DE PAQUETES
# --------------------------------------
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  tidyverse,    # Manipulación de datos y gráficos
  lubridate,    # Manejo de fechas
  xgboost,      # Algoritmo XGBoost
  ggplot2       # Visualización
)

# --------------------------------------
# LECTURA Y PREPARACIÓN DE DATOS
# --------------------------------------
url <- "https://raw.githubusercontent.com/vneumannufprbr/TrabajosRStudio/main/energy_dataset.csv"

data <- read.csv(url, stringsAsFactors = FALSE) %>% 
  mutate(time = ymd_hms(time)) %>% 
  arrange(time) %>% 
  select(time, 
         generation.solar, 
         generation.wind.onshore, 
         total.load.actual) %>% 
  na.omit()

# --------------------------------------
# CONFIGURACIÓN DE PARÁMETROS
# --------------------------------------
targets <- c("generation.solar", "generation.wind.onshore", "total.load.actual")
window_size <- 24 # 24 horas, anterior 168 horas, Ventana de 7 días (24*7)
test_size <- 48 # 48 horas porqué con 24 horas no es suficiente para el RF - anterior: 30 * 24, 30 días para evaluación
forecast_horizon <- 24 # 24 horas, anterior: 30 * 24, 30 días para pronóstico futuro

# --------------------------------------
# FUNCIONES AUXILIARES
# --------------------------------------
# Función para crear ventanas temporales
create_features <- function(serie, window) {
  n <- length(serie)
  features <- matrix(NA, nrow = n - window, ncol = window)
  for (i in 1:window) {
    features[, i] <- serie[i:(n - window + i - 1)]
  }
  target <- serie[(window + 1):n]
  return(data.frame(features, target))
}

# Función para calcular métricas (robusta a errores)
safe_calculate_metrics <- function(actual, predicted) {
  if (length(actual) != length(predicted) || length(actual) == 0) {
    return(data.frame(R2 = NA, RMSE = NA, MAE = NA))
  }
  rmse <- sqrt(mean((actual - predicted)^2))
  mae <- mean(abs(actual - predicted))
  ss_res <- sum((actual - predicted)^2)
  ss_tot <- sum((actual - mean(actual))^2)
  r_squared <- ifelse(ss_tot == 0, NA, 1 - (ss_res/ss_tot))
  
  return(data.frame(R2 = r_squared, RMSE = rmse, MAE = mae))
}

# --------------------------------------
# MODELADO Y PRONÓSTICO
# --------------------------------------
results <- list()
metrics <- list()

for (target_var in targets) {
  # Extraer serie y dividir datos
  serie <- data[[target_var]] %>% as.numeric()
  n <- length(serie)
  train_series <- serie[1:(n - test_size)]
  test_series <- serie[(n - test_size + 1):n]
  
  # Crear conjuntos de entrenamiento y prueba
  train_data <- create_features(train_series, window_size)
  test_data <- create_features(test_series, window_size)
  
  # Convertir a formato XGBoost
  dtrain <- xgb.DMatrix(
    data = as.matrix(train_data[, 1:window_size]),
    label = train_data$target
  )
  
  dtest <- xgb.DMatrix(
    data = as.matrix(test_data[, 1:window_size]),
    label = test_data$target
  )
  
  # Parámetros del modelo
  params <- list(
    objective = "reg:squarederror",
    max_depth = 6,
    eta = 0.1,
    subsample = 0.8,
    colsample_bytree = 0.8,
    eval_metric = "rmse"
  )
  
  # Entrenamiento con validación
  xgb_model <- xgb.train(
    params,
    dtrain,
    nrounds = 500,
    watchlist = list(train = dtrain, test = dtest),
    early_stopping_rounds = 20,
    verbose = 1
  )
  
  # Pronóstico en conjunto de prueba
  test_preds <- predict(xgb_model, as.matrix(test_data[, 1:window_size]))
  metrics[[target_var]] <- safe_calculate_metrics(test_data$target, test_preds)
  
  # Pronóstico futuro (usando todos los datos)
  full_data <- create_features(serie, window_size)
  dfull <- xgb.DMatrix(
    data = as.matrix(full_data[, 1:window_size]),
    label = full_data$target
  )
  
  xgb_full <- xgb.train(
    params,
    dfull,
    nrounds = xgb_model$best_iteration,
    verbose = 0
  )
  
  # Pronóstico recursivo
  last_window <- tail(serie, window_size)
  future_preds <- numeric(forecast_horizon)
  
  for (i in 1:forecast_horizon) {
    current_input <- matrix(last_window, nrow = 1)
    future_preds[i] <- predict(xgb_full, current_input)
    last_window <- c(last_window[-1], future_preds[i])
  }
  
  results[[target_var]] <- future_preds
}

# --------------------------------------
# RESULTADOS Y VISUALIZACIÓN
# --------------------------------------
# Mostrar métricas
cat("\nMétricas de Evaluación:\n")
for (target_var in targets) {
  cat("\nVariable:", target_var, "\n")
  print(metrics[[target_var]])
}

# Generar fechas futuras
last_date <- tail(data$time, 1)
future_dates <- seq(last_date + hours(1), by = "hour", length.out = forecast_horizon)

# Crear dataframe para gráficos
forecast_df <- data.frame(
  time = future_dates,
  solar = results$generation.solar,
  wind = results$generation.wind.onshore,
  load = results$total.load.actual
) %>% 
  pivot_longer(-time, names_to = "variable", values_to = "value")

# Visualización
ggplot(forecast_df, aes(x = time, y = value, color = variable)) +
  geom_line(linewidth = 1) +
  facet_wrap(~variable, scales = "free_y", ncol = 1) +
  labs(
    title = paste("Pronóstico a", forecast_horizon/24, "día(s) usando XGBoost"),
    x = "Fecha",
    y = "Valor",
    color = "Variable"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

metrics_df <- data.frame(
  Algoritmo = rep(c("KNN", "SVM", "XGBoost"), each = 3),
  Variable = rep(c("Solar", "Eólica", "Carga"), 3),
  R2 = c(0.292, -0.906, -0.228, 0.503, -0.698, -0.334, 0.988, 0.938, 0.977) # para horizonte de 30 dias - Debes cambiar con tus resultados para 24 horas
)

# PAUSA HASTA PRESIONAR ENTER
readline(prompt="GRÁFICO 2/2. Presiona [Enter] en la consola para continuar...")

ggplot(metrics_df, aes(x = Variable, y = R2, fill = Algoritmo)) +
  geom_col(position = "dodge") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Comparación de R² entre Algoritmos",
       subtitle = "XGBoost supera significativamente a KNN y SVM",
       y = "Coeficiente R²") +
  theme_minimal()

# --------------------------------------
# INSTALACIÓN Y CARGA DE PAQUETES
# --------------------------------------
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  tidyverse,    # Manipulación de datos y gráficos
  lubridate,    # Manejo de fechas
  caret,        # Framework para entrenamiento y evaluación de modelos (incluye RF)
  randomForest, # Necesario para method = "rf" en caret
  ggplot2       # Visualización
)

# --------------------------------------
# LECTURA Y PREPARACIÓN DE DATOS
# --------------------------------------
# (Igual que antes)
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
# Parámetros del modelo
window_size <- 24 # 24 horas, anterior 168 horas, Ventana de 7 días (24*7)
test_size <- 48 # 48 horas porqué con 24 horas no es suficiente para el RF - anterior: 30 * 24, 30 días para evaluación
forecast_horizon <- 24 # 24 horas, anterior: 30 * 24, 30 días para pronóstico futuro

# --------------------------------------
# FUNCIONES AUXILIARES
# --------------------------------------
# Función para crear ventanas temporales
create_features <- function(serie, window) {
  n <- length(serie)
  if (n <= window) {
    stop("La longitud de la serie debe ser mayor que el tamaño de la ventana.")
  }
  features <- matrix(NA, nrow = n - window, ncol = window)
  for (i in 1:window) {
    features[, i] <- serie[i:(n - window + i - 1)]
  }
  target <- serie[(window + 1):n]
  
  # Crear nombres de columna estándar para las características
  colnames(features) <- paste0("X", 1:window)
  
  return(data.frame(features, target))
}

# Función para calcular métricas (robusta a errores)
# (Igual que antes)
safe_calculate_metrics <- function(actual, predicted) {
  if (length(actual) != length(predicted) || length(actual) == 0) {
    return(data.frame(R2 = NA, RMSE = NA, MAE = NA))
  }
  # Asegurarse que no haya NAs infinitos o NaN que rompan los cálculos
  valid_indices <- is.finite(actual) & is.finite(predicted)
  if(sum(valid_indices) < 2) { # Necesitamos al menos 2 puntos válidos para R2
    return(data.frame(R2 = NA, RMSE = NA, MAE = NA))
  }
  actual <- actual[valid_indices]
  predicted <- predicted[valid_indices]
  
  rmse <- sqrt(mean((actual - predicted)^2))
  mae <- mean(abs(actual - predicted))
  ss_res <- sum((actual - predicted)^2)
  ss_tot <- sum((actual - mean(actual))^2)
  # Evitar división por cero si todos los valores reales son iguales
  r_squared <- ifelse(ss_tot < .Machine$double.eps, NA, 1 - (ss_res/ss_tot)) 
  
  return(data.frame(R2 = r_squared, RMSE = rmse, MAE = mae))
}


# --------------------------------------
# MODELADO Y PRONÓSTICO CON RANDOM FOREST
# --------------------------------------
results <- list()
metrics <- list()

# Configuración de control para caret (Validación Cruzada)
# Usaremos CV para ajustar mtry de forma más robusta
# Nota: RF no usa 'early stopping' como XGBoost. Se entrena un número fijo de árboles.
train_control <- trainControl(
  method = "cv",      # Validación cruzada
  number = 5,         # 5 folds (reducido de 10 para rapidez, ajustar si es necesario)
  # verboseIter = TRUE, # Descomentar para ver progreso de caret
  allowParallel = TRUE # Usar procesamiento en paralelo si está configurado
)


for (target_var in targets) {
  cat("\nProcesando variable:", target_var, "\n")
  
  # Extraer serie y dividir datos
  serie <- data[[target_var]] %>% as.numeric()
  n <- length(serie)
  train_series <- serie[1:(n - test_size)]
  test_series <- serie[(n - test_size + 1):n]
  
  # Crear conjuntos de entrenamiento y prueba
  # Añadir manejo de error si la serie es demasiado corta
  tryCatch({
    train_data <- create_features(train_series, window_size)
    # Asegurarse que test_series tenga suficientes datos para crear al menos una ventana + target
    if(length(test_series) > window_size) {
      test_data <- create_features(test_series, window_size)
    } else {
      cat("  Skipping test evaluation for", target_var, "- not enough data.\n")
      test_data <- NULL # Marcar que no hay datos de test
    }
  }, error = function(e) {
    cat("  Error creando features para", target_var, ":", e$message, "\n")
    train_data <- NULL
    test_data <- NULL
  })
  
  # Continuar solo si tenemos datos de entrenamiento
  if (!is.null(train_data)) {
    
    # --- Entrenamiento con caret y Random Forest ---
    cat("  Entrenando modelo Random Forest...puede demorar unos minutos \n")
    
    # Definir grid para mtry (opcional, caret puede hacerlo por defecto)
    # Por defecto caret prueba floor(sqrt(num_predictores)) y algunos más
    # num_predictores <- ncol(train_data) - 1 # -1 por la columna target
    # tune_grid_rf <- expand.grid(.mtry = floor(num_predictores * c(0.25, 0.33, 0.5))) # Ejemplo
    
    set.seed(1912) # Para reproducibilidad del entrenamiento
    rf_model <- train(
      target ~ .,                 # Fórmula: predecir target usando el resto
      data = train_data,          # Datos de entrenamiento
      method = "rf",              # Especifica Random Forest
      trControl = train_control,  # Usa la configuración de CV
      # tuneGrid = tune_grid_rf,  # Descomentar para probar mtry específicos
      # importance = TRUE,        # Descomentar si quieres calcular importancia de variables
      ntree = 100                 # Reducir ntree para rapidez (default es 500)
      # Aumentar para mejor rendimiento si el tiempo lo permite
    )
    
    cat("  Mejor mtry encontrado:", rf_model$bestTune$mtry, "\n")
    # print(rf_model) # Descomentar para ver detalles del modelo caret
    
    # --- Evaluación en Prueba (si hay datos de prueba) ---
    if (!is.null(test_data)) {
      cat("  Evaluando en conjunto de prueba...\n")
      # Predecir en el conjunto de prueba (excluyendo la columna target)
      test_preds <- predict(rf_model, newdata = test_data %>% select(-target))
      metrics[[target_var]] <- safe_calculate_metrics(test_data$target, test_preds)
    } else {
      metrics[[target_var]] <- data.frame(R2 = NA, RMSE = NA, MAE = NA) # No hay métricas si no hay test set
    }
    
    
    # --- Reentrenamiento y Pronóstico Futuro ---
    cat("  Reentrenando modelo final con todos los datos...\n")
    # Reentrenar usando la configuración óptima en todos los datos disponibles
    # (O simplemente usar rf_model si CV fue robusta y el dataset es grande)
    # Aquí reentrenamos para consistencia con el enfoque original
    
    # Crear features con toda la serie
    tryCatch({
      full_data <- create_features(serie, window_size)
    }, error = function(e) {
      cat("  Error creando features para datos completos en", target_var, ":", e$message, "\n")
      full_data <- NULL
    })
    
    if(!is.null(full_data)) {
      set.seed(1912)
      rf_full <- train(
        target ~ .,
        data = full_data,
        method = "rf",
        # Usar el mejor mtry encontrado, sin remuestreo adicional
        tuneGrid = expand.grid(.mtry = rf_model$bestTune$mtry),
        trControl = trainControl(method = "none"), # Sin remuestreo
        ntree = 100 # Usar el mismo ntree
      )
      
      # --- Pronóstico Recursivo ---
      cat("  Generando pronóstico futuro...\n")
      last_window <- tail(serie, window_size)
      future_preds <- numeric(forecast_horizon)
      feature_names <- paste0("X", 1:window_size) # Nombres esperados por el modelo
      
      for (i in 1:forecast_horizon) {
        # Crear data frame para la predicción con nombres correctos
        current_input_df <- as.data.frame(matrix(last_window, nrow = 1))
        colnames(current_input_df) <- feature_names
        
        # Predecir
        pred_value <- predict(rf_full, newdata = current_input_df)
        future_preds[i] <- pred_value
        
        # Actualizar ventana para la siguiente predicción
        last_window <- c(last_window[-1], pred_value)
      }
      results[[target_var]] <- future_preds # Guarda los pronósticos futuros
    } else {
      results[[target_var]] <- rep(NA, forecast_horizon) # No hay pronóstico si falló full_data
    }
  } else {
    # Si falló la creación inicial de train_data
    metrics[[target_var]] <- data.frame(R2 = NA, RMSE = NA, MAE = NA)
    results[[target_var]] <- rep(NA, forecast_horizon)
  }
}

# --------------------------------------
# RESULTADOS Y VISUALIZACIÓN
# --------------------------------------
# Mostrar métricas
cat("\nMétricas de Evaluación (Random Forest):\n")
for (target_var in targets) {
  cat("\nVariable:", target_var, "\n")
  # Imprimir métricas redondeadas
  if(!is.null(metrics[[target_var]])) {
    print(metrics[[target_var]] %>% mutate(across(everything(), ~round(.x, 3))))
  } else {
    print("Métricas no disponibles.")
  }
}

# Generar fechas futuras
last_date <- tail(data$time, 1)
future_dates <- seq(last_date + hours(1), by = "hour", length.out = forecast_horizon)

# Crear dataframe para gráficos
# Asegurarse que los resultados tengan la longitud correcta
valid_results <- results[sapply(results, length) == forecast_horizon]
if(length(valid_results) > 0) {
  forecast_df_list <- lapply(names(valid_results), function(name) {
    data.frame(time = future_dates, variable = name, value = valid_results[[name]])
  })
  forecast_df <- bind_rows(forecast_df_list)
  
  # Cambiar nombres para el gráfico si es necesario (ej. quitar generation.)
  forecast_df <- forecast_df %>%
    mutate(variable = case_when(
      variable == "generation.solar" ~ "Solar",
      variable == "generation.wind.onshore" ~ "Eólica",
      variable == "total.load.actual" ~ "Carga",
      TRUE ~ variable
    ))
  
  readline(prompt="GRÁFICO 1/3. Presiona [Enter] en la consola para continuar...")
  
  # Visualización del Pronóstico
  print( # Asegurarse que el gráfico se imprima
    ggplot(forecast_df, aes(x = time, y = value, color = variable)) +
      geom_line(linewidth = 1) +
      facet_wrap(~variable, scales = "free_y", ncol = 1) +
      labs(
        title = paste("Pronóstico a", forecast_horizon/24, "día(s) usando Random Forest"),
        x = "Fecha",
        y = "Valor",
        color = "Variable"
      ) +
      theme_minimal() +
      theme(legend.position = "none")
  )
} else {
  cat("\nNo se generaron pronósticos válidos para graficar.\n")
}


# Visualización Comparativa (Actualizar con métricas RF si se desea)
# Necesitarías guardar las métricas de RF y combinarlas con las de KNN/SVM
 metrics_rf_solar <- metrics$generation.solar$R2
 metrics_rf_wind <- metrics$generation.wind.onshore$R2
 metrics_rf_load <- metrics$total.load.actual$R2

# # Crear un dataframe similar a metrics_df pero con los valores de RF
 # Los resultados de metricas de KNN, SVM y XGBoost se obtuvieron anteriormente
 metrics_comp_df <- data.frame(
   Algoritmo = rep(c("KNN", "SVM","XGBoost","Random Forest"), each = 3), # Asumiendo que tienes los de KNN/SVM/"XGBoost"
   Variable = rep(c("Solar", "Eólica", "Carga"), 4),
   R2 = c( 0.292213, -0.9057994, -0.228126,  # Valores KNN para horizonte de 30 dias - Debes cambiar con tus resultados para 24 horas
          0.503, -0.698, -0.334,  # Valores SVM para horizonte de 30 dias - Debes cambiar con tus resultados para 24 horas
          0.988, 0.938134, 0.9765706,  # Valores XGBoost para horizonte de 30 dias - Debes cambiar con tus resultados para 24 horas
          metrics_rf_solar, metrics_rf_wind, metrics_rf_load) # Valores RF calculados
 )

 readline(prompt="GRÁFICO 2/3. Presiona [Enter] en la consola para continuar...")
 
# Graficar comparación (si tienes los datos de KNN/SVM)
 print(
     ggplot(metrics_comp_df, aes(x = Variable, y = R2, fill = Algoritmo)) +
       geom_col(position = "dodge") +
       geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
       labs(title = "Comparación de R² entre Algoritmos",
            y = "Coeficiente R²") +
       theme_minimal()
 )
 
 # Supuestos: teniendo los valores de R² para RF en estas variables
  metrics_rf_solar <- metrics$generation.solar
  metrics_rf_wind <- metrics$generation.wind.onshore
  metrics_rf_load <- metrics$total.load.actual
 
 # Crear un dataframe con los 3 tipos de métricas
 metrics_comp_df <- data.frame(
   Algoritmo = rep(c("KNN", "SVM", "XGBoost", "Random Forest"), each = 3 * 3),  # 3 variables x 3 métricas
   Variable = rep(rep(c("Solar", "Eólica", "Carga"), each = 3), 4),  # Repetir por métrica
   Metrica = rep(c("R2", "RMSE", "MAE"), times = 3 * 4),  # Cada métrica se repite
   Valor = c(
     # --- KNN --- para horizonte de 30 dias - Debes cambiar con tus resultados para 24 horas
     0.292213, 1556.813, 1101.061,
     -0.9057994, 3953.569, 3243.733,
     -0.228126, 4468.826, 3584.862,
     
     # --- SVM --- para horizonte de 30 dias - Debes cambiar con tus resultados para 24 horas
     0.503, 1304.703, 966.0549,
     -0.698, 3731.805, 2952.332,
     -0.334, 4657.628, 3913.831,
     
     # --- XGBoost --- para horizonte de 30 dias - Debes cambiar con tus resultados para 24 horas
     0.988, 201.488, 137.3568,
     0.938134, 640.2372, 380.0813,
     0.9765706, 563.9765, 408.3985,
     
     # --- Random Forest (métricas reales) ---
     metrics_rf_solar$R2,  metrics_rf_solar$RMSE,  metrics_rf_solar$MAE, # Valores RF calculados
     metrics_rf_wind$R2,   metrics_rf_wind$RMSE,   metrics_rf_wind$MAE,  # Valores RF calculados
     metrics_rf_load$R2,   metrics_rf_load$RMSE,   metrics_rf_load$MAE   # Valores RF calculados
   )
 )
 
 # Visualización con ggplot2 agrupada por métrica
 library(ggplot2)
 
 # PAUSA HASTA PRESIONAR ENTER
 readline(prompt="GRÁFICO 3/3. Presiona [Enter] en la consola para continuar...")
 
 ggplot(metrics_comp_df, aes(x = Variable, y = Valor, fill = Algoritmo)) +
   geom_col(position = "dodge") +
   facet_wrap(~Metrica, scales = "free_y") +
   geom_hline(data = subset(metrics_comp_df, Metrica == "R2"), aes(yintercept = 0),
              linetype = "dashed", color = "red") +
   labs(
     title = "Comparación de R², RMSE y MAE entre Algoritmos",
     y = "Valor de la Métrica",
     x = "Variable",
     fill = "Algoritmo"
   ) +
   theme_minimal(base_size = 13)
 
 
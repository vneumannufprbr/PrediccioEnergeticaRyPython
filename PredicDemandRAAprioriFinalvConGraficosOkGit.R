# 1. Instalación y carga de librerías
if(!require(arules)){install.packages("arules")}
if(!require(arulesViz)){install.packages("arulesViz")}
library(arules)
library(arulesViz)

# 2. Creación del conjunto de datos
datos <- data.frame(
  Hora = c(15, 15, 9, 7, 22, 15, 15, 20, 21, 14),
  DiaSemana = factor(c(2, 6, 2, 2, 6, 2, 7, 2, 6, 3),
                     levels = 1:7, labels = c("Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo")),
  Mes = c(7, 7, 1, 1, 1, 12, 5, 12, 8, 6),
  Temperatura = factor(c("Alta", "Alta", "Media", "Baja", "Baja", "Alta", "Media", "Baja", "Alta", "Media"),
                       levels = c("Baja", "Media", "Alta")),
  Humedad = factor(c("Media", "Media", "Baja", "Alta", "Baja", "Media", "Alta", "Baja", "Alta", "Media"),
                   levels = c("Baja", "Media", "Alta")),
  Festivo = factor(c("No", "No", "No", "No", "No", "Si", "No", "No", "Si", "No"),
                   levels = c("No", "Si")),
  Demanda = factor(c("Alta", "Alta", "Media", "Baja", "Baja", "Alta", "Media", "Baja", "Media", "Alta"),
                   levels = c("Baja", "Media", "Alta"))
)

# 3. Discretización de variables numéricas
datos$Hora <- cut(datos$Hora, breaks = c(0, 6, 12, 17, 23),
                  labels = c("0-6", "7-12", "13-17", "18-23"),
                  include.lowest = TRUE, right = FALSE)
datos$Mes <- factor(datos$Mes, levels = as.character(sort(unique(datos$Mes))))

# 4. Conversión a formato transaccional
transacciones <- as(datos, "transactions") 

# 5. Aplicación del Algoritmo Apriori
reglas <- apriori(transacciones, 
                  parameter = list(supp = 0.2, conf = 0.6, minlen = 2),
                  appearance = list(rhs = c("Demanda=Alta", "Demanda=Media", "Demanda=Baja"), default = "lhs"))

# 6. Nueva instancia de entrada
nueva_instancia <- data.frame(
  Hora = 16,
  DiaSemana = "Sabado",
  Mes = 7,
  Temperatura = "Baja",
  Humedad = "Media",
  Festivo = "No"
)

# 6.1 Discretización manual de la nueva instancia
nueva_instancia_discretizada <- data.frame(
  Hora = factor("13-17", levels = c("0-6", "7-12", "13-17", "18-23")),
  DiaSemana = factor("Sabado", levels = levels(datos$DiaSemana)),
  Mes = factor(7, levels = sort(unique(datos$Mes))),
  Temperatura = factor("Baja", levels = levels(datos$Temperatura)),
  Humedad = factor("Media", levels = levels(datos$Humedad)),
  Festivo = factor("No", levels = levels(datos$Festivo))
)

# 6.2 Conversión a transacción
nueva_instancia_transaccion <- as(nueva_instancia_discretizada, "transactions")

# 7. Filtrado de reglas aplicables
reglas_aplicables <- subset(reglas, subset = is.subset(lhs(reglas), nueva_instancia_transaccion, sparse = FALSE))

# 8. Selección de la mejor regla y predicción
if (length(reglas_aplicables) > 0) {
  mejor_regla <- head(sort(reglas_aplicables, by = "lift", decreasing = TRUE), n = 1)
  
  if (!is.null(mejor_regla)) {
    regla_string <- capture.output(inspect(mejor_regla))
    prediccion_demanda <- gsub(".*=> ", "", regla_string[1])
    print(paste("Demanda predicha:", prediccion_demanda))
    inspect(mejor_regla)
  } else {
    print("No se encontraron reglas aplicables que cumplan los criterios de soporte y confianza.")
  }
} else {
  print("No se encontraron reglas aplicables.")
}

# 9. VISUALIZACIONES (solo con las 20 mejores reglas por lift)
mejores_reglas <- head(sort(reglas, by = "lift", decreasing = TRUE), 20)

# PAUSA HASTA PRESIONAR ENTER
readline(prompt="GRÁFICO 1/3: Presiona [Enter] en la consola para continuar...")

# 9.1 de Predicción de Demanda
if(length(mejores_reglas) >= 1){
  plot(mejores_reglas, method = "scatterplot", 
       measure = c("support", "confidence"), 
       shading = "lift", jitter = 0)
} else {
  print("No hay suficientes reglas para el scatterplot.")
}

# PAUSA HASTA PRESIONAR ENTER
readline(prompt="GRÁFICO 2/3: Presiona [Enter] en la consola para continuar...")

# 9.2 Diagrama agrupado (grouped matrix)
if(length(mejores_reglas) >= 2){
  plot(mejores_reglas, method = "grouped")
} else {
  print("No hay suficientes reglas para el gráfico agrupado.")
}

# PAUSA HASTA PRESIONAR ENTER
readline(prompt="GRÁFICO 3/3: Presiona [Enter] en la consola para continuar...")

# 9.3 Diagrama de red (grafo)
if(length(mejores_reglas) >= 1){
  plot(mejores_reglas, method = "graph", engine = "htmlwidget")
} else {
  print("No hay reglas suficientes para el diagrama de red.")
}

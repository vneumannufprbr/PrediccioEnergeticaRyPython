# 1. Instalación y carga de librerías
if(!require(arules)){install.packages("arules")}
library(arules)
#	Verifica si la librería arules está instalada; si no, la instala.
#	Carga la librería arules, que se usa para descubrir patrones en datos transaccionales mediante Apriori.

# 2. Creación del conjunto de datos
datos <- data.frame(
  Hora = c(15, 15, 9, 7, 22, 15, 15, 20, 21, 14),
  DiaSemana = factor(c(2, 6, 2, 2, 6, 2, 7, 2, 6, 3), levels = 1:7, labels = c("Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo")),
  Mes = c(7, 7, 1, 1, 1, 12, 5, 12, 8, 6),
  Temperatura = factor(c("Alta", "Alta", "Media", "Baja", "Baja", "Alta", "Media", "Baja", "Alta", "Media"), levels = c("Baja", "Media", "Alta")),
  Humedad = factor(c("Media", "Media", "Baja", "Alta", "Baja", "Media", "Alta", "Baja", "Alta", "Media"), levels = c("Baja", "Media", "Alta")),
  Festivo = factor(c("No", "No", "No", "No", "No", "Si", "No", "No", "Si", "No"), levels = c("No", "Si")),
  Demanda = factor(c("Alta", "Alta", "Media", "Baja", "Baja", "Alta", "Media", "Baja", "Media", "Alta"), levels = c("Baja", "Media", "Alta"))
)
#	Se define un dataframe con variables categóricas como Día de la semana, Hora, Temperatura, Humedad, Festivo y Demanda.
#	DiaSemana se transforma en factor con etiquetas (Lunes, Martes, etc.).
#	Temperatura, Humedad, Festivo y Demanda también se convierten en factores.

# Se visualiza la "lista de compras" de los datos
View(datos)

# 3. Discretización de variables numéricas
datos$Hora <- cut(datos$Hora, breaks = c(0, 6, 12, 17, 23), labels = c("0-6", "7-12", "13-17", "18-23"), include.lowest = TRUE, right=FALSE)
datos$Mes <- factor(datos$Mes, levels = as.character(sort(unique(datos$Mes))))
#	Hora se agrupa en rangos: 
#	0-6 (madrugada)
#	7-12 (mañana)
#	13-17 (tarde)
#	18-23 (noche)
#	Mes se convierte en factor con los valores únicos ordenados.

# 4. Conversión a formato transaccional
transacciones <- as(datos, "transactions") 
#	Convierte los datos a formato transaccional, que es requerido por el algoritmo Apriori.

# 5. Aplicación del Algoritmo Apriori
reglas <- apriori(transacciones, 
                  parameter = list(supp = 0.2, conf = 0.6, minlen=2),
                  appearance = list(rhs = c("Demanda=Alta", "Demanda=Media", "Demanda=Baja"), default="lhs"))
#	Se ejecuta Apriori con: 
# 	Soporte mínimo = 0.2 (20%)
# 	Confianza mínima = 0.6 (60%)
# 	minlen=2 (reglas con al menos dos elementos)
#	Se especifica que Demanda (Alta, Media o Baja) debe estar en el lado derecho (rhs) de las reglas.

# 6. Creación de una nueva instancia para predicción 
# Estos datos pueden ser cambiados, y tambien cambiar en Discretización manual
nueva_instancia <- data.frame(
  Hora = 16,
  DiaSemana = "Sabado",
  Mes = 7,
  Temperatura = "Baja",
  Humedad = "Media",
  Festivo = "No"
) #	Se define un nuevo registro con valores específicos.

# Discretización manual de la nueva instancia: 
  nueva_instancia_discretizada <- data.frame(
    Hora = factor("13-17", levels = c("0-6", "7-12", "13-17", "18-23")),
    DiaSemana = factor("Sabado", levels = levels(datos$DiaSemana)),
    Mes = factor(7, levels = sort(unique(datos$Mes))),
    Temperatura = factor("Baja", levels = levels(datos$Temperatura)),
    Humedad = factor("Media", levels = levels(datos$Humedad)),
    Festivo = factor("No", levels = levels(datos$Festivo))
  ) #	Se asignan los valores discretizados, asegurando que las variables tengan los mismos niveles que en datos.

# Conversión a transacciones
  nueva_instancia_transaccion <- as(nueva_instancia_discretizada, "transactions") 
  #	Se convierte la nueva instancia a transacciones, para poder compararla con las reglas.
  
#  7. Filtrado de reglas aplicables
  reglas_aplicables <- subset(reglas, subset = is.subset(lhs(reglas), nueva_instancia_transaccion, sparse = FALSE)) 
  #	Se seleccionan solo las reglas cuyo lado izquierdo (lhs) es un subconjunto de la nueva instancia
  
#  8. Selección de la mejor regla y predicción
  if (length(reglas_aplicables) > 0) {
    mejor_regla <- head(sort(reglas_aplicables, by = "lift", decreasing = TRUE), n = 1)
    
    if (!is.null(mejor_regla)) {  # Verificar si mejor_regla NO es NULL
      # Extraer la demanda predicha usando inspect() y gsub()
      regla_string <- capture.output(inspect(mejor_regla)) # Capturar la salida de inspect()
      prediccion_demanda <- gsub(".*=> ", "", regla_string[1]) # Extraer solo la demanda de la primera línea
      print(paste("Demanda predicha:", prediccion_demanda))
      inspect(mejor_regla) # Mostrar la regla utilizada para la predicción (opcional)
    } else {
      print("No se encontraron reglas aplicables que cumplan los criterios de soporte y confianza.")
    }
  } else {
    print("No se encontraron reglas aplicables.")
  }
  #	Si hay reglas aplicables, selecciona la mejor según el lift.
  #	Extrae la predicción de demanda desde la mejor regla.
  #	Imprime la predicción y la regla utilizada.
  #	Si no hay reglas aplicables, muestra un mensaje indicando que no hay reglas relevantes
 
  # Resumen: ¿Qué hace el código?
  #  1.	Prepara los datos (discretiza variables, transforma en transacciones).
  #  2.	Genera reglas de asociación con Apriori.
  #  3.	Define una nueva instancia (un nuevo caso para predecir demanda).
  #  4.	Filtra reglas relevantes para la nueva instancia.
  #  5.	Elige la mejor regla y predice la demanda basada en patrones descubiertos.
  
    
  
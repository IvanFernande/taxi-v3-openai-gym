# Q-Learning en el Entorno "Taxi-v3" de Gym

Este repositorio contiene un script que implementa un agente de aprendizaje por refuerzo utilizando el algoritmo de Q-learning en el entorno "Taxi-v3" de Gym. El entorno "Taxi-v3" es un escenario de aprendizaje por refuerzo donde un taxi debe recoger y dejar pasajeros en ubicaciones específicas dentro de una cuadrícula. El objetivo del agente es aprender una política óptima para completar esta tarea de manera eficiente.

## Descripción del Algoritmo

Q-learning es un método de aprendizaje por refuerzo basado en una tabla de valores (Q-table), donde cada fila representa un estado del entorno y cada columna representa una acción posible. El valor en cada celda indica el valor esperado de tomar una acción específica en un estado determinado.

En este script, el agente comienza con una tabla Q inicializada con ceros y aprende a través de la interacción con el entorno. Durante el entrenamiento, el agente recibe recompensas y ajusta los valores en la tabla Q para reflejar las mejores acciones para cada estado, utilizando la siguiente fórmula de actualización:

\[Q(s, a) = Q(s, a) + \alpha \times (R + \gamma \times \max_{a'} Q(s', a') - Q(s, a))\]

donde:
- \(s\) es el estado actual,
- \(a\) es la acción tomada,
- \(R\) es la recompensa obtenida,
- \(s'\) es el siguiente estado,
- \(a'\) son las posibles acciones del siguiente estado,
- \(\alpha\) es la tasa de aprendizaje, y
- \(\gamma\) es el factor de descuento.

## Estructura del Código

1. **Parámetros y Configuración**:
   - Se configura el entorno "Taxi-v3" utilizando `gym.make`.
   - Se definen los parámetros del modelo: `alpha` (tasa de aprendizaje), `gamma` (factor de descuento), y `epsilon` (probabilidad de exploración).

2. **Entrenamiento**:
   - El agente se entrena durante 1000 episodios, tomando acciones, obteniendo recompensas y actualizando la tabla Q.
   - El agente explora tomando acciones aleatorias con una probabilidad `epsilon` y explota el conocimiento tomando la acción con el mayor valor en la tabla Q.
   - Se recopilan las recompensas obtenidas para visualizar el progreso del entrenamiento.

3. **Evaluación**:
   - Después del entrenamiento, el agente se evalúa durante 100 episodios de prueba.
   - Durante la evaluación, el agente toma la acción con el mayor valor en la tabla Q, explotando la política aprendida.
   - Se calcula el número de episodios completados con éxito.

4. **Visualización**:
   - Se visualiza el entorno con `env.render()` para observar el comportamiento del taxi después del entrenamiento.
   - Se utiliza un gráfico para mostrar la evolución de las recompensas a lo largo del entrenamiento.

5. **Cierre del Entorno**:
   - Se cierra el entorno con `env.close()` al final del script para liberar recursos.

## Requisitos

- Python 3.x
- Gym
- Numpy
- Matplotlib

## Cómo Usar el Código

1. Clona el repositorio en tu máquina local.
2. Asegúrate de tener los requisitos necesarios instalados.
3. Ejecuta el script para entrenar al agente y visualizar su comportamiento en el entorno "Taxi-v3".
4. Observa el gráfico de recompensas y la simulación del entorno para evaluar el rendimiento del agente.


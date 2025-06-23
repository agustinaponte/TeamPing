# TeamPing

**TeamPing** es una herramienta de seguimiento del estado de hosts, diseñada para ofrecer monitoreo en tiempo real y notificaciones personalizadas cuando un host se active o se caiga.  

Ideal para equipos de infraestructura IT y profesionales del área, **TeamPing** envía pings continuos a los hosts especificados, rastrea su estado y notifica cualquier cambio de forma inmediata.  

---

## ¿Por qué usar TeamPing?

TeamPing está pensado como una herramienta de monitoreo que puede implementarse rapidamente para trabajar de forma colaborativa en equipos de infraestructura. Puede ser útil en cambios de configuración, apagado o encendido de muchos equipos, situaciones donde tener una vista en tiempo real y de fácil configuración es útil para trabajar de forma más eficiente.

Hay una gran cantidad de herramientas de monitoreo disponibles con una amplia variedad de configuraciones y funciones. Sin embargo, para algunas actividades estas herramientas pueden ser demasiado complejas o lentas de configurar.

Si tiene una lista de servidores preparada, es tan fácil como ejecutar TeamPing y pegar la lista de servidores en la interfaz web.

---

## 🚀 Características

- ✅ **Monitoreo en Tiempo Real:** Supervisa de forma continua el estado de los hosts.  
- 🔔 **Notificaciones Personalizadas:** Configura alertas cuando los hosts suban o bajen.  
- 🔄 **Actualizaciones vía WebSocket:** Recibe cambios de estado en vivo.  
- 🌐 **API REST:** Permite agregar, eliminar y configurar hosts mediante solicitudes HTTP.  

---
## 📥 Instrucciones de uso

Descargar la última versión y ejecutarla como administrador. La interfaz web estará disponible en http://localhost:9123

La aplicación busca un archivo hosts.csv en el directorio de ejecución y lo crea si no existe. Este archivo mantiene una lista de los hosts que deben ser monitoreados.

Tambien genera un archivo host_monitor.log para registrar la actividad de la aplicación.

---

## 🖥️ Uso y Endpoints de la API (avanzado/desarrolladores)

La aplicación ofrece tanto una interfaz web como endpoints RESTful para la gestión de hosts.

📡 WebSocket para Actualizaciones en Vivo

Endpoint: /ws

Uso: Conéctate para recibir actualizaciones en tiempo real sobre el estado de los hosts.

📌 Endpoints de la API

🔹 1. Agregar un Host

Endpoint: POST /hosts

Parámetro: address (form data): Dirección IP o nombre del host a monitorear.

Descripción: Agrega un nuevo host y comienza a monitorearlo de inmediato.

🔹 2. Obtener Todos los Hosts

Endpoint: GET /hosts

Descripción: Recupera el estado actual y las estadísticas de todos los hosts monitoreados.

🔹 3. Eliminar un Host

Endpoint: DELETE /hosts/{host_id}

Descripción: Elimina un host de la lista de monitoreo.

🔹 4. Alternar Monitoreo del Host

Endpoint: PUT /hosts/{host_id}/toggle-monitoring

Descripción: Activa o desactiva el monitoreo para un host específico.

🔹 5. Configurar Modo de Notificación

Endpoint: PUT /hosts/{host_id}/notification-mode

Parámetro (Query): mode: Acepta notify_up, notify_down o disabled.

Descripción: Configura cuándo se deben enviar notificaciones para el host.

🔹 6. Obtener Detalles del Host

Endpoint: GET /hosts/{host_id}/details

Descripción: Proporciona registros detallados y estadísticas de un host en particular.

🔹 7. Servir la Interfaz Web

Endpoint: GET /

Descripción: Sirve la página principal (index.html) para la interacción básica.

---

⚖️ Licencia
Este proyecto está bajo la Licencia MIT.

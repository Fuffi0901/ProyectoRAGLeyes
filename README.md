# ⚖️ Asistente Legal AI: Multimodal RAG con LangGraph

Sistema avanzado de asistencia legal capaz de procesar documentos PDF y evidencia visual (imágenes) utilizando una arquitectura de **Grafo de Estados** y modelos de lenguaje de última generación.

---

## 🚀 Características Principales

* **🖼️ Procesamiento Multimodal:** Análisis de imágenes de evidencia y documentos PDF mediante **Gemini 2.0 Flash**.
* **🧠 Arquitectura de Grafo (LangGraph):** Implementa un flujo de trabajo inteligente que incluye:
    * **Router:** Clasifica la consulta en categorías legales (Penal, Civil, etc.).
    * **Rewriter:** Optimiza la pregunta del usuario para mejorar la búsqueda vectorial.
    * **Ranker:** Re-rankeo de resultados con Cross-Encoder (**BGE-Reranker**).
* **📑 Ingesta Jerárquica:** Sistema de *Chunks Padre-Hijo* para mantener el contexto global sin perder precisión semántica.
* **💻 Interfaz Completa:** UI moderna en **Streamlit** con soporte para Texto-a-Voz (TTS) y visualización de evidencia integrada.

---

## 🛠️ Stack Tecnológico

| Componente | Tecnología |
| :--- | :--- |
| **LLMs** | Google Gemini (Visión), Groq/Llama 3 |
| **Embeddings** | `intfloat/multilingual-e5-small` |
| **Vector DB** | ChromaDB (Persistente) |
| **Orquestación** | LangGraph & LangChain |
| **Frontend** | Streamlit |

---

## 📁 Estructura del Proyecto

```text
ProyectoRag
├── chroma_db/          # Base de datos vectorial persistente
├── data/
│   ├── Pdfs/           # Documentos legales para indexar
│   └── Img/            # Evidencia visual
├── api/
│   └── backend.py      # API FastAPI y lógica del Grafo de Estados
├── app/
│   ├── .streamlit/
│   └── RagStreamlit.py # Interfaz de usuario Streamlit
├── src/
│   ├── cargar_pdfs.py  # Pipeline de ingesta y análisis visual
│   └── utils.py        # Funciones auxiliares
├── requirements.txt
└── .env                # Variables de entorno (API Keys)

## ⚙️ Configuración e Instalación
Requisitos Previos
- Python 3.10 + Claves de API para Google Gemini y Groq.

Instalación
# Clonar el repositorio
git clone https://github.com/Fuffi0901/ProyectoRAGLeyes.git
cd ProyectoRAGLeyes

# Instalar dependencias
pip install -r requirements.txt

📖 Guía de Uso
Paso 1: Ingesta de Datos
    Ejecuta el script para procesar leyes y evidencias. Gemini analizará las imágenes (OCR) y las categorizará.
    Bash
    python src/cargar_pdfs.py
Paso 2: Iniciar el Backend
    La API gestiona el flujo de razonamiento y la búsqueda.
    Bash
    uvicorn api.backend:app --reload --port 8000
Paso 3: Lanzar la Interfaz
    Bash
    streamlit run app/RagStreamlit.py

🔄 Lógica de la RAG Multimodal
El sistema sigue un flujo de razonamiento cíclico para asegurar la precisión:
Pregunta: Recepción vía texto o archivos.
Categorización: Clasificación automática del dominio legal.
Query Rewriting: Transformación a consulta técnica optimizada.
Búsqueda (Top 10): Recuperación semántica en ChromaDB.
Re-ranker: Evaluación con Cross-Encoder para eliminar ruido.
Análisis Gemini: Integración de pruebas visuales/escaneadas.
Respuesta: Generación fundamentada con soporte de audio.

📊 Reporte de Calidad (RAGAS)
Con Ground Truth ManualFidelidad (Anti-Alucinación): 57.9%
Relevancia de Respuesta: 3.26 / 5.0
Exactitud (vs Ground Truth): 3.66 / 5.0
Context Recall (Chunks): 23.7%

🏆 Comparativa de Estrategias de Chunking
Modelo: intfloat/multilingual-e5-small (Recomendado)
Configuración,Hit Rate @5,MRR,Latencia Media
"v1 (400 size, 50 overlap)",94.7%,0.742,24.38 ms
"v2 (800 size, 100 overlap)",13.2%,0.32,22.90 ms
"v3 (950 size, 150 overlap)",6.2%,0.05,21.95 ms

🛡️ Notas de Implementación
Seguridad: El sistema incluye reglas críticas para prevenir comandos de jailbreak y asegurar que el modelo no revele sus instrucciones internas (System Prompt).
Reranking: Se utiliza BAAI/bge-reranker-v2-m3 para garantizar que el contexto inyectado sea el más pertinente.

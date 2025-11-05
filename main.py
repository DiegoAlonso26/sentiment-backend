import os
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from googleapiclient.discovery import build
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv

# --- 1. Configuración Inicial ---
load_dotenv()

app = FastAPI(
    title="API de Análisis de Sentimientos",
    description="Una API para analizar comentarios de YouTube usando VADER."
)

# --- 2. Configuración de CORS ---
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    # --- ¡CORRECCIÓN 1! ---
    # Quité la barra "/" al final. Debe ser la URL exacta.
    "https://sentiment-frontend-gamma.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. Carga de API Key y Modelos ---
API_KEY = os.environ.get("YOUTUBE_API_KEY")

if not API_KEY:
    print("ERROR: YOUTUBE_API_KEY no encontrada. Asegúrate de crear el archivo .env")

analyzer = SentimentIntensityAnalyzer()


# --- 4. NUEVA FUNCIÓN: Obtener información del video ---
async def obtener_info_video(video_id: str):
    """
    Obtiene información básica del video: título, canal, miniatura y fecha de publicación.
    """
    if not API_KEY:
        return None

    try:
        youtube = build("youtube", "v3", developerKey=API_KEY)

        request = youtube.videos().list(
            part="snippet",
            id=video_id
        )
        response = request.execute()

        if not response.get('items'):
            return None

        video_data = response['items'][0]['snippet']

        return {
            "title": video_data.get('title', 'Título no disponible'),
            "channel": video_data.get('channelTitle', 'Canal no disponible'),
            "thumbnail": video_data.get('thumbnails', {}).get('medium', {}).get('url', ''),
            "published_at": video_data.get('publishedAt', '')
        }

    except Exception as e:
        print(f"Error al obtener info del video: {e}")
        return None


# --- 5. Función para obtener comentarios (MEJORADA) ---
async def obtener_comentarios_youtube(video_id: str):
    """
    Obtiene HASTA 100 comentarios. Esto previene timeouts en Render.
    """
    if not API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Servicio no disponible: Falta configuración de API Key en el servidor."
        )

    try:
        youtube = build("youtube", "v3", developerKey=API_KEY)
        comentarios = []
        next_page_token = None


        # Cambiado de 500 a 100 para evitar el error 502 Bad Gateway
        while len(comentarios) < 100:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                textFormat="plainText",
                pageToken=next_page_token
            )
            response = request.execute()

            for item in response.get('items', []):
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comentarios.append(comment)

            next_page_token = response.get('nextPageToken')

            if not next_page_token:
                break


        return comentarios[:100]

    except Exception as e:
        print(f"Error al llamar a la API de YouTube: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al contactar la API de YouTube: {str(e)}"
        )


# --- 6. Función para analizar sentimientos ---
async def analizar_sentimientos_vader(lista_comentarios: list):
    """
    Analiza una lista de comentarios y devuelve:
    1. Un conteo de sentimientos (dict)
    2. Una lista de comentarios clasificados (list)
    """
    counts = {"Positivo": 0, "Neutral": 0, "Negativo": 0}
    classified_list = []

    for comentario in lista_comentarios:
        scores = analyzer.polarity_scores(comentario)
        compound_score = scores['compound']

        sentimiento = "Neutral"
        if compound_score >= 0.05:
            sentimiento = "Positivo"
            counts["Positivo"] += 1
        elif compound_score <= -0.05:
            sentimiento = "Negativo"
            counts["Negativo"] += 1
        else:
            counts["Neutral"] += 1

        classified_list.append({
            "comentario": comentario,
            "sentimiento": sentimiento
        })

    return counts, classified_list


# --- 7. Endpoint Principal (ACTUALIZADO) ---
@app.get("/analizar/")
async def analizar_video(
        video_id: str = Query(
            ...,
            min_length=11,
            max_length=11,
            description="El ID de 11 caracteres del video de YouTube"
        )
):
    """
    Endpoint principal: Recibe un video_id, obtiene información del video,
    comentarios y devuelve el análisis completo.
    """
    # Paso A: Obtener información del video
    video_info = await obtener_info_video(video_id)

    # Paso B: Obtener comentarios
    comentarios = await obtener_comentarios_youtube(video_id)

    if not comentarios:
        raise HTTPException(
            status_code=404,
            detail="No se encontraron comentarios para este video."
        )

    # Paso C: Analizar sentimientos
    counts, classified_list = await analizar_sentimientos_vader(comentarios)

    # Paso D: Devolver respuesta completa con información del video
    return {
        "video_id": video_id,
        "video_info": video_info,  # ← NUEVO: Información del video
        "total_comentarios": len(comentarios),
        "sentimientos": counts,
        "lista_comentarios": classified_list
    }


@app.get("/")
async def root():
    return {
        "message": "Bienvenido a la API de Análisis de Sentimientos. Usa /docs para ver la documentación."
    }
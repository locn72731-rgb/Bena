import os
import json
import asyncio
import audioop

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types

app = FastAPI()

API_KEY = os.environ.get("GEMINI_API_KEY")

MODEL = "gemini-3.8-live"

SYSTEM_PROMPT = """
Bạn là Bé Na, một robot trợ lý AI thân thiện.
Bạn đang trò chuyện với Lộc bằng tiếng Việt.
Luôn trả lời tự nhiên, ngắn gọn, dễ hiểu.
Gọi người dùng là Lộc.
Không tự ý điều khiển robot khi chưa có yêu cầu rõ ràng.
"""

@app.get("/")
async def home():
    return {
        "robot": "Be Na",
        "status": "online",
        "ai": "Gemini Live"
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    if not API_KEY:
        await ws.send_text(json.dumps({
            "action": "none",
            "text": "Chưa cấu hình Gemini API Key"
        }))
        await ws.close()
        return

    client = genai.Client(api_key=API_KEY)

    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        system_instruction=types.Content(
            parts=[types.Part(text=SYSTEM_PROMPT)]
        ),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig()
    )

    try:
        async with client.aio.live.connect(
            model=MODEL,
            config=config
        ) as session:

            await ws.send_text(json.dumps({
                "action": "none",
                "text": "Bé Na đã kết nối bộ não AI!"
            }, ensure_ascii=False))

            listening = False

            async def receive_ai():
                async for response in session.receive():
                    content = response.server_content

                    if not content:
                        continue

                    if content.output_transcription:
                        text = content.output_transcription.text
                        if text:
                            await ws.send_text(json.dumps({
                                "action": "none",
                                "text": text
                            }, ensure_ascii=False))

                    if content.model_turn:
                        for part in content.model_turn.parts:
                            if part.inline_data:
                                audio = part.inline_data.data

                                # Gemini: 24kHz
                                # ESP32: 16kHz
                                audio_16k, _ = audioop.ratecv(
                                    audio, 2, 1, 24000, 16000, None
                                )

                                await ws.send_bytes(audio_16k)

            ai_task = asyncio.create_task(receive_ai())

            try:
                while True:
                    message = await ws.receive()

                    if message["type"] == "websocket.disconnect":
                        break

                    if message.get("text"):
                        data = json.loads(message["text"])
                        msg_type = data.get("type", "")

                        if msg_type == "listen":
                            state = data.get("state", "")

                            if state == "start":
                                listening = True

                            elif state == "stop":
                                listening = False
                                await session.send_realtime_input(
                                    audio_stream_end=True
                                )

                    elif message.get("bytes") is not None:
                        if listening:
                            await session.send_realtime_input(
                                audio=types.Blob(
                                    data=message["bytes"],
                                    mime_type="audio/pcm;rate=16000"
                                )
                            )

            finally:
                ai_task.cancel()
                await asyncio.gather(
                    ai_task, return_exceptions=True
                )

    except WebSocketDisconnect:
        pass

    except Exception as e:
        print("Lỗi Bé Na:", repr(e))

    finally:
        print("Đã đóng kết nối Bé Na")

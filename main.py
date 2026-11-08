
import os
import json
import asyncio
import audioop
import traceback

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types

app = FastAPI()

# =====================================
# CONFIGURATION
# =====================================

API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = "gemini-3.8-live"

SYSTEM_PROMPT = """
Bạn là Bé Na, một robot trợ lý AI thân thiện.
Bạn trò chuyện với Lộc bằng tiếng Việt.
Trả lời tự nhiên, ngắn gọn, dễ hiểu.
Không tự ý điều khiển robot.
Không tự ý ra lệnh di chuyển robot.
"""

# =====================================
# HEALTH CHECK
# =====================================

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


# =====================================
# SEND JSON
# =====================================

async def send_message(ws, action, message):
    await ws.send_text(json.dumps({
        "action": action,
        "text": message
    }, ensure_ascii=False))


# =====================================
# WEBSOCKET
# =====================================

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):

    await ws.accept()

    if not API_KEY:
        print("ERROR: GEMINI_API_KEY is missing")
        await send_message(
            ws, "none", "Chưa cấu hình Gemini API Key."
        )
        await ws.close()
        return

    client = genai.Client(
        api_key=API_KEY,
        http_options=types.HttpOptions(
            api_version="v1beta"
        )
    )

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=SYSTEM_PROMPT,
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig()
    )

    ai_task = None
    listening = False

    try:
        print("Connecting to Gemini Live...")
        print("Model:", MODEL)

        async with client.aio.live.connect(
            model=MODEL,
            config=config
        ) as session:

            print("GEMINI CONNECTED SUCCESSFULLY")

            await send_message(
                ws,
                "none",
                "Bé Na đã kết nối bộ não AI!"
            )

            # =================================
            # RECEIVE GEMINI RESPONSE
            # =================================

            async def receive_ai():

                audio_state = None

                try:
                    async for response in session.receive():

                        content = response.server_content

                        if not content:
                            continue

                        # Text transcription
                        if content.output_transcription:

                            text = content.output_transcription.text

                            if text:
                                print("GEMINI:", text)

                                await send_message(
                                    ws, "none", text
                                )

                        # Audio response
                        if content.model_turn:

                            for part in content.model_turn.parts:

                                if part.inline_data:

                                    audio = part.inline_data.data

                                    if not audio:
                                        continue

                                    audio_16k, audio_state = audioop.ratecv(
                                        audio,
                                        2,
                                        1,
                                        24000,
                                        16000,
                                        audio_state
                                    )

                                    if audio_16k:
                                        await ws.send_bytes(audio_16k)

                        if content.turn_complete:
                            audio_state = None
                            print("Gemini turn complete")

                except asyncio.CancelledError:
                    raise

                except Exception:
                    print("ERROR RECEIVING GEMINI RESPONSE")
                    traceback.print_exc()

            ai_task = asyncio.create_task(receive_ai())

            # =================================
            # RECEIVE ESP32 DATA
            # =================================

            while True:

                message = await ws.receive()

                if message["type"] == "websocket.disconnect":
                    print("ESP32 disconnected")
                    break

                if message.get("text"):

                    try:
                        data = json.loads(message["text"])
                    except json.JSONDecodeError:
                        continue

                    if data.get("type") == "listen":

                        state = data.get("state")

                        if state == "start":
                            listening = True
                            print("LISTENING START")

                        elif state == "stop":
                            listening = False
                            print("LISTENING STOP")

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

    except WebSocketDisconnect:
        print("ESP32 disconnected")

    except Exception as e:

        print("========== GEMINI CONNECTION ERROR ==========")
        print("ERROR TYPE:", type(e).__name__)
        print("ERROR MESSAGE:", str(e))

        response = getattr(e, "response", None)

        if response is not None:
            print("HTTP STATUS:", getattr(response, "status_code", None))
            print("HTTP BODY:", getattr(response, "body", None))

        traceback.print_exc()
        print("==============================================")

        try:
            await send_message(
                ws,
                "none",
                "Lỗi kết nối AI: " + type(e).__name__
            )
        except Exception:
            pass

    finally:

        if ai_task:
            ai_task.cancel()
            await asyncio.gather(
                ai_task,
                return_exceptions=True
            )

        print("BENA CONNECTION CLOSED")

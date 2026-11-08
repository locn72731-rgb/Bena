import os
import json
import asyncio
import audioop

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types

app = FastAPI()

# ==============================
# CẤU HÌNH GEMINI
# ==============================

API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = "gemini-3.8-live"

SYSTEM_PROMPT = """
Bạn là Bé Na, một robot trợ lý AI thân thiện.
Bạn đang trò chuyện với Lộc bằng tiếng Việt.
Luôn trả lời tự nhiên, ngắn gọn, dễ hiểu.
Gọi người dùng là Lộc.
Không tự ý điều khiển robot khi chưa có yêu cầu rõ ràng.
Không tự ý ra lệnh di chuyển robot.
"""


# ==============================
# TRANG KIỂM TRA SERVER
# ==============================

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


# ==============================
# HÀM GỬI THÔNG BÁO CHO ESP32
# ==============================

async def send_message(ws, action, text):
    await ws.send_text(json.dumps({
        "action": action,
        "text": text
    }, ensure_ascii=False))


# ==============================
# WEBSOCKET BÉ NA
# ==============================

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):

    await ws.accept()

    if not API_KEY:
        await send_message(
            ws,
            "none",
            "Chưa cấu hình Gemini API Key."
        )
        await ws.close()
        return

    client = genai.Client(api_key=API_KEY)

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=SYSTEM_PROMPT,
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig()
    )

    ai_task = None
    listening = False
    audio_state = None

    try:

        async with client.aio.live.connect(
            model=MODEL,
            config=config
        ) as session:

            await send_message(
                ws,
                "none",
                "Bé Na đã kết nối bộ não AI!"
            )

            # ==============================
            # NHẬN CÂU TRẢ LỜI TỪ GEMINI
            # ==============================

            async def receive_ai():

                nonlocal audio_state

                try:
                    async for response in session.receive():

                        content = response.server_content

                        if not content:
                            continue

                        # Nhận nội dung văn bản
                        if content.output_transcription:

                            text = content.output_transcription.text

                            if text:
                                await send_message(
                                    ws,
                                    "none",
                                    text
                                )

                        # Nhận âm thanh trả lời
                        if content.model_turn:

                            for part in content.model_turn.parts:

                                if part.inline_data:

                                    audio = part.inline_data.data

                                    if not audio:
                                        continue

                                    # Đổi âm thanh 24kHz
                                    # thành 16kHz cho ESP32
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

                        # Kết thúc một lượt trả lời
                        if content.turn_complete:
                            audio_state = None

                except asyncio.CancelledError:
                    raise

                except Exception as e:
                    print("Lỗi nhận AI:", repr(e))

            ai_task = asyncio.create_task(receive_ai())

            # ==============================
            # NHẬN DỮ LIỆU TỪ ESP32
            # ==============================

            while True:

                message = await ws.receive()

                if message["type"] == "websocket.disconnect":
                    break

                # Nhận JSON
                if message.get("text"):

                    try:
                        data = json.loads(message["text"])
                    except json.JSONDecodeError:
                        continue

                    msg_type = data.get("type", "")

                    # Bắt đầu nghe
                    if msg_type == "listen":

                        state = data.get("state", "")

                        if state == "start":

                            listening = True
                            audio_state = None

                            print("Bé Na đang nghe...")

                        # Kết thúc nghe
                        elif state == "stop":

                            listening = False

                            print("Đã kết thúc thu âm.")

                            await session.send_realtime_input(
                                audio_stream_end=True
                            )

                # Nhận âm thanh từ ESP32
                elif message.get("bytes") is not None:

                    if listening:

                        audio_data = message["bytes"]

                        if audio_data:

                            await session.send_realtime_input(
                                audio=types.Blob(
                                    data=audio_data,
                                    mime_type="audio/pcm;rate=16000"
                                )
                            )

    except WebSocketDisconnect:
        print("ESP32 đã ngắt kết nối.")

    except Exception as e:

        print("Lỗi Bé Na:", repr(e))

        try:
            await send_message(
                ws,
                "none",
                "Bé Na đang gặp lỗi kết nối AI."
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

        print("Đã đóng kết nối Bé Na.")

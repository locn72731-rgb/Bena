import os
import json
import asyncio
import audioop
import traceback

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types

app = FastAPI()

# =================================
# CẤU HÌNH
# =================================

API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = "gemini-3.8-live"

SYSTEM_PROMPT = """
Bạn là Bé Na, một robot trợ lý AI thân thiện.
Bạn đang trò chuyện với Lộc bằng tiếng Việt.
Luôn trả lời tự nhiên, ngắn gọn, dễ hiểu.
Gọi người dùng là Lộc.
Không tự ý điều khiển robot.
Không tự ý ra lệnh di chuyển robot.
"""

# =================================
# TRANG CHỦ
# =================================

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


# =================================
# GỬI THÔNG BÁO
# =================================

async def send_message(ws, action, text):
    await ws.send_text(json.dumps({
        "action": action,
        "text": text
    }, ensure_ascii=False))


# =================================
# WEBSOCKET BÉ NA
# =================================

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):

    await ws.accept()

    if not API_KEY:
        print("LỖI: Chưa tìm thấy GEMINI_API_KEY")
        await send_message(
            ws, "none", "Chưa cấu hình Gemini API Key."
        )
        await ws.close()
        return

    client = genai.Client(api_key=API_KEY)

    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        system_instruction=SYSTEM_PROMPT,
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig()
    )

    ai_task = None
    listening = False
    audio_state = None

    try:

        print("Đang kết nối Gemini Live...")

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
            # NHẬN PHẢN HỒI TỪ GEMINI
            # =================================

            async def receive_ai():

                nonlocal audio_state

                try:

                    async for response in session.receive():

                        content = response.server_content

                        if not content:
                            continue

                        # Văn bản AI trả lời
                        if content.output_transcription:

                            ai_text = content.output_transcription.text

                            if ai_text:
                                print("BÉ NA:", ai_text)

                                await send_message(
                                    ws,
                                    "none",
                                    ai_text
                                )

                        # Âm thanh AI trả lời
                        if content.model_turn:

                            for part in content.model_turn.parts:

                                if part.inline_data:

                                    audio = part.inline_data.data

                                    if not audio:
                                        continue

                                    # Chuyển 24kHz sang 16kHz
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
                            print("Gemini hoàn thành câu trả lời.")

                except asyncio.CancelledError:
                    raise

                except Exception as e:

                    print("LỖI NHẬN PHẢN HỒI GEMINI:")
                    traceback.print_exc()

                    try:
                        await send_message(
                            ws,
                            "none",
                            "Lỗi nhận phản hồi Gemini: " + type(e).__name__
                        )
                    except Exception:
                        pass

            ai_task = asyncio.create_task(receive_ai())

            # =================================
            # NHẬN DỮ LIỆU TỪ ESP32
            # =================================

            while True:

                message = await ws.receive()

                if message["type"] == "websocket.disconnect":
                    print("ESP32 đã ngắt kết nối.")
                    break

                # Nhận JSON
                if message.get("text"):

                    try:
                        data = json.loads(message["text"])
                    except json.JSONDecodeError:
                        print("JSON không hợp lệ.")
                        continue

                    msg_type = data.get("type", "")

                    if msg_type == "listen":

                        state = data.get("state", "")

                        # Bắt đầu thu âm
                        if state == "start":

                            listening = True
                            audio_state = None

                            print("BÉ NA ĐANG NGHE...")

                        # Kết thúc thu âm
                        elif state == "stop":

                            listening = False

                            print("ĐÃ KẾT THÚC THU ÂM.")

                            await session.send_realtime_input(
                                audio_stream_end=True
                            )

                # Nhận âm thanh PCM từ ESP32
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

        print("WebSocket đã ngắt kết nối.")

    except Exception as e:

        print("========== GEMINI ERROR ==========")
        traceback.print_exc()
        print("==================================")

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

        print("ĐÃ ĐÓNG KẾT NỐI BÉ NA.")

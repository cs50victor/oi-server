# import sys
# if sys.version_info < (3, 12):
# else:
#     from typing import TypedDict

# https://docs.pydantic.dev/2.8/errors/usage_errors/#typed-dict-version
from typing_extensions import TypedDict
from typing import Any, Callable, Optional, Union, Dict, List, Literal
from fastapi import FastAPI, APIRouter, Depends, Request, HTTPException, UploadFile, File, Form, WebSocket, status
from pydantic import BaseModel, field_validator
from fastapi.responses import FileResponse
from interpreter import OpenInterpreter  # type: ignore
from datetime import datetime
import shutil
import socket
import logging
import json
import os
import uvicorn
import asyncio
import uvloop  # uvloop makes asyncio 2-4x faster.

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# FOR READER: file class structure for easy skimming
# also cool read / side not: - python threading side note: https://stackoverflow.com/questions/11431637/how-can-i-kill-a-thread-in-python
# - imports
# - Pydanctic Structs for LMC, Ping, etc
# - AsnycInterpreter
# - OISever
# - ServerRoutes
# TODO: maybe add better typing to interpreter core message array


class RunCode(TypedDict):
    language: str
    code: str


class Ping(BaseModel):
    type: Literal["ping"]


class LMC(BaseModel):
    id: Optional[str] = str(datetime.now().timestamp())
    role: Literal["user", "assistant", "computer", "server"]
    type: Optional[Literal["message", "console", "command", "status", "image", "code", "audio", "error"]]
    format: Optional[
        Literal[
            "active_line",
            "output",
            "base64",
            "base64.png",
            "base64.jpeg",
            "path",
            "html",
            "javascript",
            "python",
            "r",
            "applescript",
            "shell",
            "wav",
        ]
    ] = None
    content: Optional[str]
    run_code: Optional[RunCode] = None
    start: Optional[bool] = None
    end: Optional[bool] = None

    @field_validator("start")
    def start_must_be_true(cls, start: bool):
        if not start:
            raise ValueError("start must be true in LCM message")
        return start

    @field_validator("end")
    def end_must_be_true(cls, end: bool):
        if not end:
            raise ValueError("end must be true in LCM message")
        return end

    def is_server_complete_msg(self) -> bool:
        return self.role == "server" and self.type == "status" and self.content == "complete"


### FOR OPENAI COMPATIBLE ENDPOINT
class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]


class ChatCompletionRequest(BaseModel):
    model: str = "default-model"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: Optional[bool] = False


class ClientMsg(BaseModel):
    inner: Union[LMC, Ping, bytes]

    @field_validator("inner")
    def validate_server_msg(cls, inner: Any) -> Union[LMC, Ping, bytes]:
        if isinstance(inner, bytes):
            return inner

        if isinstance(inner, str):
            try:
                inner_dict = json.loads(inner)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON message: {inner} | {e}")
        elif isinstance(inner, dict):
            inner_dict: Dict[str, Any] = inner
        else:
            raise ValueError(f"Invalid websocket message type: {type(inner)}")

        if "type" not in inner_dict:
            raise ValueError("Message missing 'type' field")

        try:
            if inner_dict["type"] == "ping":
                return Ping(**inner_dict)
            elif inner_dict["type"] in ["message", "console", "image", "code", "audio"]:
                return LMC(**inner_dict)
            else:
                raise ValueError(f"Unknown message type: {inner_dict['type']}")
        except Exception as e:
            raise ValueError(f"Error creating object: {e}")

    def is_lcm(self) -> bool:
        return isinstance(self.inner, LMC)

    def is_ping(self) -> bool:
        return isinstance(self.inner, Ping)

    def is_bytes(self) -> bool:
        return isinstance(self.inner, bytes)


class ServerInterpreter(OpenInterpreter):
    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)  # type: ignore
        self.messages: List[LMC]
        self.output_queue: asyncio.Queue[Union[LMC, bytes]] = asyncio.Queue()
        self.id = str(os.getenv("INTERPRETER_ID", datetime.now().timestamp()))
        self.print = False  # Will print output

    async def output(self):
        # If queue is empty, wait until an item is available.
        return await self.output_queue.get()

    # TODO: protential bug here, test later
    def not_similar_to_last(self, lmc_message: LMC) -> bool:
        if len(self.messages) == 0:  # type: ignore
            return False
        last_msg: Any = self.messages[-1]  # type: ignore
        return lmc_message.type != last_msg.get("type") or lmc_message.format != last_msg.get("format")

    def response_complete_lmc(self) -> LMC:
        return LMC(role="server", type="status", content="complete")

    def accumulate(self, message: ClientMsg):
        if not isinstance(message.inner, (bytes, LMC)):
            raise ValueError(f"Invalid message type: {type(message.inner)}")
        if message.is_bytes():
            if not self.messages:
                raise ValueError("Cannot accumulate bytes message when messages list is empty")
            # We initialize as an empty string ^
            # But it actually should be bytes
            if self.messages[-1]["content"] == "":  # type: ignore
                self.messages[-1]["content"] = b""  # type: ignore
            self.messages[-1]["content"] += message.inner  # type: ignore
            return

        assert isinstance(message.inner, LMC)  # This just helps type checking since lsp doesn't pick up is_bytes()

        if message.inner.content == "active_line":
            return

        # possible logical bug in this code block below
        curr_msg_is_similar_to_prev = self.not_similar_to_last(message.inner)
        if message.inner.start or curr_msg_is_similar_to_prev:
            curr_msg = message.inner
            # TODO: is this needed?
            if curr_msg.start:
                del curr_msg.start
            if curr_msg.content is None:
                curr_msg.content = ""
            self.messages.append(curr_msg)
        elif message.inner.content and not curr_msg_is_similar_to_prev:
            if not self.messages:
                raise ValueError("You must send a 'start: True' chunk first to create this message.")
            # Append to an existing message
            last_msg = self.messages[-1]
            if not last_msg.type:  # It was created with a type-less start message
                last_msg.type = message.inner.type
            if message.inner.format and not last_msg.format:  # It was created with a type-less start message
                last_msg.format = message.inner.format
                if last_msg.content is None:
                    last_msg.content = message.inner.content
            else:
                if last_msg.content is None:
                    last_msg.content = ""
                last_msg.content += message.inner.content


class OIServer:
    app: FastAPI
    config: uvicorn.Config
    uvicorn_server: uvicorn.Server
    authenticate: Callable[[str], bool]

    DEFAULT_HOST = "127.0.0.1"
    DEFAULT_PORT = 8000

    def __init__(self, server_interpreter: ServerInterpreter) -> None:
        # logging setup
        self.log_level = logging.INFO
        logging.basicConfig(
            level=self.log_level, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.logger = logging.getLogger(__name__)

        # fast api setup
        # "dependencies are the preferred way to create a middleware for authentication"
        # see https://stackoverflow.com/questions/66632841/fastapi-dependency-vs-middleware
        self.app = FastAPI(dependencies=[Depends(self.http_auth_middleware())])
        self.app.include_router(
            ServerRoutes(server_interpreter=server_interpreter, logger=self.logger).router,
            # for fun
            responses={418: {"description": "I'm a teapot"}},
        )

        # server config setup
        host = os.getenv("HOST", OIServer.DEFAULT_HOST)
        port = int(os.getenv("PORT", OIServer.DEFAULT_PORT))

        # TODO: optimize later
        # 1080 minutes / 18 hours
        ws_ping_timeout_in_minutes: float = 1080.0 * 60.0
        _ws_ping_interval_in_secs: float = 30.0
        _ws_max_msg_queue: int = 100
        ws_max_msg_size_in_bytes: int = 500 * (1024 * 1024)  # ( n ) Megabytes
        self.config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            ws_max_size=ws_max_msg_size_in_bytes,
            ws_ping_timeout=ws_ping_timeout_in_minutes,
            ws_max_queue=_ws_max_msg_queue,
            ws_ping_interval=_ws_ping_interval_in_secs,
            log_level=self.log_level,
        )
        self.connection_auth = self.default_connection_auth

    @property
    def host(self):
        return self.config.host

    @host.setter
    def host(self, value: str):
        self.config.host = value

    @property
    def port(self):
        return self.config.port

    @port.setter
    def port(self, value: int):
        self.config.port = value

    #  ------- ws config start -------
    @property
    def ws_timeout_in_mins(self) -> float:
        return self.ws_ping_timeout_in_minutes

    @ws_timeout_in_mins.setter
    def ws_timeout_in_minutes(self, value: float):
        minutes = value * 60.0
        self.ws_ping_timeout_in_minutes = minutes
        self.config.ws_ping_timeout = minutes

    @property
    def ws_max_msg_size_in_bytes(self) -> int:
        return self._ws_max_msg_size_in_bytes

    @ws_max_msg_size_in_bytes.setter
    def ws_max_msg_size_in_mb(self, value: int):
        mb = value * 1024 * 1024
        self._ws_max_msg_size_in_bytes = mb
        self.config.ws_max_size = mb

    #  ------- ws config end -------

    # TODO: clean up or think about it harder later
    def default_connection_auth(self, key: str | None) -> bool:
        ws_api_key = os.getenv("INTERPRETER_API_KEY")
        #  ws_api_key is None ??
        return ws_api_key is None or (key is not None and ws_api_key == key)

    def http_auth_middleware(self):
        async def _http_auth_middleware(request: Request):
            api_key = request.headers.get("X-API-KEY")
            if request.url.path == "/heartbeat" or not self.connection_auth(api_key):
                return

            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Http Middleware Authentication failed. X-API-KEY is invalid.",
            )

        return _http_auth_middleware

    def run(self):
        uvicorn_server = uvicorn.Server(self.config)

        # for more info : https://stackoverflow.com/questions/20778771/what-is-the-difference-between-0-0-0-0-127-0-0-1-and-localhost

        # Print server ip disclaimer
        msg = f"Server will run at http://{self.config.host}:{self.config.port}"
        if self.config.host == "0.0.0.0":
            self.logger.warning(
                "Warning: Using host `0.0.0.0` will expose Open Interpreter over your local network. `127.0.0.1` is recommended in most cases. For more info, see https://stackoverflow.com/questions/20778771/what-is-the-difference-between-0-0-0-0-127-0-0-1-and-localhost"
            )
            # TODO: readup on this later
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))  # Google's public DNS server
            msg = f"Server will run at http://{s.getsockname()[0]}:{self.config.port}"
            s.close()
        self.logger.info(msg)
        uvicorn_server.run()


class ServerRoutes:
    def __init__(self, logger: logging.Logger, server_interpreter: ServerInterpreter):
        self.logger = logger
        self.router = APIRouter()
        self.interpreter_task: asyncio.Task[Any] | None = None
        self.server_interpreter = server_interpreter
        # routes setup
        self.router.add_api_route("/", self.home, methods=["GET"])
        self.router.add_api_websocket_route("/ws", self.ws_endpoint)
        self.router.add_api_route("/heartbeat", self.heartbeat, methods=["GET"])
        self.router.add_api_route("/settings", self.set_settings, methods=["POST"])
        self.router.add_api_route("/settings/{setting}", self.get_setting, methods=["GET"])
        # TODO
        # self.router.add_api_route("/openai/chat/completions", self.chat_completion, methods=["POST"])
        if os.getenv("INTERPRETER_INSECURE_ROUTES", "").lower() == "true":
            self.router.add_api_route("/run", self.run_code, methods=["POST"])
            self.router.add_api_route("/upload", self.upload_file, methods=["POST"])
            self.router.add_api_route("/download/{filename}", self.download_file, methods=["GET"])

    async def home(self):
        return {"message": "Open Interpreter Local Server"}

    async def heartbeat(self):
        return {"status": "alive"}

    async def run_code(self, payload: Dict[str, Any]):
        try:
            code_meta = RunCode(**payload)
            language, code = code_meta["language"], code_meta["code"]
        except Exception as e:
            return {"error": str(e)}, 400

        try:
            if self.interpreter_task and not self.interpreter_task.done():
                self.interpreter_task.cancel()
            self.logger.info(f"Running {language}:", code)
            self.interpreter_task = asyncio.create_task(
                asyncio.to_thread(self.server_interpreter.computer.run, language, code)  # type: ignore
            )
            code_output = await self.interpreter_task
            self.logger.info("Code output:", code_output)
            return {"output": code_output}
        except Exception as e:
            return {"error": str(e)}, 500

    async def get_setting(self, setting: str):
        if hasattr(self.server_interpreter, setting):
            setting_value = getattr(self.server_interpreter, setting)
            try:
                return json.dumps({setting: setting_value})
            except TypeError:
                return {"error": "Failed to serialize the setting value"}, 500
        else:
            return json.dumps({"error": "Setting not found"}), 404

    async def set_settings(self, payload: Dict[str, Any]):
        for key, value in payload.items():
            if key == "auto_run":
                return {
                    "error": f"The setting {key} is not modifiable through the server due to security constraints."
                }, 403

            self.logger.info(f"Updating settings: {key} = {value}")

            if key in ["llm", "computer"] and isinstance(value, dict):
                if not hasattr(self.server_interpreter, key):
                    return ({"error": f"Setting {key} not found"}, 404)
                for sub_key, sub_value in value.items():  # type: ignore
                    if not hasattr(getattr(self.server_interpreter, key), sub_key):  # type: ignore
                        return ({"error": f"Sub-setting {sub_key} not found in {key}"}, 404)
                    setattr(getattr(self.server_interpreter, key), sub_key, sub_value)  # type: ignore
            elif hasattr(self.server_interpreter, key):
                setattr(self.server_interpreter, key, value)
            else:
                return ({"error": f"Setting {key} not found"}, 404)
        return {"status": "success"}

    async def ws_endpoint(self, websocket: WebSocket):
        await websocket.accept()
        msg_queue: asyncio.Queue[ClientMsg] = asyncio.Queue()

        async def receive_msg_from_client():
            try:
                msg = await websocket.receive()
                self.logger.info(f"received message from client: {msg}")
                # validate msg
                # if invalid, log and send error msg to client, else put
                try:
                    client_msg = ClientMsg(**msg)
                    if client_msg.is_ping():
                        return await websocket.send_json({"type": "Pong"})
                    msg_queue.put_nowait(client_msg)
                except Exception as e:
                    err_msg = f"Error validating websocket message: {msg} | {e}"
                    self.logger.error(err_msg)
            except Exception as e:
                self.logger.warning(f"Client disconnected from websocket or other error occured | {e}")

        async def send_response_to_client():
            while True:
                try:
                    output = await self.server_interpreter.output()
                    if isinstance(output, LMC):
                        await websocket.send_text(json.dumps(output))
                    else:
                        await websocket.send_bytes(output)
                except Exception as e:
                    self.logger.error(f"Error sending output to client: {e}")

        def run_code(language: str, code: str):
            if os.getenv("INTERPRETER_INSECURE_ROUTES", "").lower() != "true":
                self.logger.warning("Running code is disabled due to security constraints.")
                return
            try:
                self.logger.info(f"Running {language}:", code)
                output: Any = self.server_interpreter.computer.run(language, code)  # type: ignore
                self.logger.info("Code Output:", output)
                self.server_interpreter.output_queue.put_nowait(LMC(role="server", type="error", content=output))
            except Exception as e:
                self.logger.error(f"Running {language}:", code)
                self.server_interpreter.output_queue.put_nowait(LMC(role="server", type="error", content=str(e)))

        def client_msg_to_task(client_msg: ClientMsg) -> asyncio.Task[None] | None:
            # for each message we either accumulate or starting responding
            if isinstance(client_msg.inner, bytes) or (
                isinstance(client_msg.inner, LMC) and client_msg.inner.end is not True
            ):
                self.server_interpreter.accumulate(client_msg)
            elif isinstance(client_msg.inner, LMC):
                lmc_msg = client_msg.inner
                if lmc_msg.run_code:
                    # non-async long running code
                    blocking_coroutine = asyncio.to_thread(
                        run_code, lmc_msg.run_code["language"], lmc_msg.run_code["code"]
                    )
                else:
                    blocking_coroutine = asyncio.to_thread(
                        start_blocking_response_logic, self.server_interpreter, self.logger
                    )

                return asyncio.create_task(blocking_coroutine)

        async def process_client_msg():
            client_msg = await msg_queue.get()
            self.interpreter_task = client_msg_to_task(client_msg)
            while True:
                client_msg = await msg_queue.get()
                if self.interpreter_task and not self.interpreter_task.done():
                    self.interpreter_task.cancel()
                self.interpreter_task = client_msg_to_task(client_msg)

        await asyncio.gather(receive_msg_from_client(), process_client_msg(), send_response_to_client())

    # https://github.com/fastapi/fastapi/discussions/9062
    # https://python.plainenglish.io/file-uploads-and-downloads-in-fastapi-a-comprehensive-guide-06e0b18bb245
    # aiofile is really slow
    async def upload_file(self, uploaded_file: UploadFile = File(...), path: str = Form(...)):
        try:
            with open(path, "w+b") as file:
                shutil.copyfileobj(uploaded_file.file, file)
            return {
                "status": "success",
                "file": uploaded_file.filename,
                "content": uploaded_file.content_type,
                "path": path,
            }
        except Exception as e:
            return {"error": str(e)}, 500

    async def download_file(self, file_path: str):
        if os.path.isfile(file_path):
            return FileResponse(path=file_path, media_type="application/octet-stream")
        return {"error": f"file {file_path} not found or not a file"}, 404

    # TODO: deeply look into the chat completions later functions, not my top priority atm
    # async def openai_compatible_generator(self):


def start_blocking_response_logic(interpreter: ServerInterpreter, logger: logging.Logger):
    # we got an end chunk
    if not interpreter.messages:
        return

    last_msg = interpreter.messages[-1]
    # TODO: run_code logic might have a bug, double check
    run_code = interpreter.auto_run
    if last_msg.type == "command":
        command = last_msg.content
        interpreter.messages.pop()
        if command == "go":
            run_code = True

    # TODO: _ means private, maybe change in core?
    # this should return a generator of LMC chunks
    for chunk_og in interpreter._respond_and_store():  # type: ignore
        chunk: Dict[str, Any] = chunk_og.copy()
        try:
            lmc_chunk = LMC(**chunk)
        except Exception as e:
            logger.error(f"Error parsing LMC chunk produced by interpreter-core: {chunk} | {e}")
            continue
        if chunk.get("type") == "confirmation":
            if not run_code:
                break
            run_code = False
            continue

        # maybe use logger.info instead of print ???
        if interpreter.print:
            if lmc_chunk.start:
                print("\n")
            if lmc_chunk.type in ["code", "console"] and lmc_chunk.format:
                if lmc_chunk.start:
                    print(
                        "\n------------\n\n```" + chunk["format"],
                        flush=True,
                    )
                if lmc_chunk.end:
                    print("\n```\n\n------------\n\n", flush=True)
            if lmc_chunk.format != "active_line":
                if lmc_chunk.format == "base64":
                    print("\n[An image was produced]")
                else:
                    content = str(lmc_chunk.content or "").encode("ascii", "ignore").decode("ascii")
                    print(content, end="")

        if interpreter.debug:
            logger.debug("Interpreter produced this chunk:", chunk)

        interpreter.output_queue.put_nowait(lmc_chunk)
        resp_complete_lmc = interpreter.response_complete_lmc()
        interpreter.output_queue.put_nowait(resp_complete_lmc)
        if interpreter.debug:
            logger.debug("Server response complete")

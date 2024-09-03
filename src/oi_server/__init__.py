# https://docs.pydantic.dev/2.8/errors/usage_errors/#typed-dict-version
from typing_extensions import TypedDict
from typing import Any, Callable, Optional, Union, Dict, List, Literal, Coroutine, Annotated
from fastapi import FastAPI, APIRouter, Depends, HTTPException, UploadFile, File, Form, WebSocket, Header, status
from pydantic import BaseModel, field_validator, ValidationInfo
from fastapi.responses import FileResponse
from interpreter import OpenInterpreter  # type: ignore
from starlette.websockets import WebSocketState
from datetime import datetime
import shutil
import socket
import logging
import json
import os
import uvicorn
import asyncio

print('----------- ❄️" -------------')

# FOR READER: file class structure for easy skimming
# also cool read / side not: - python threading side note: https://stackoverflow.com/questions/11431637/how-can-i-kill-a-thread-in-python
# - imports
# - Pydanctic Structs for LMC, ClientMsg, and other json to model stuff
# - AsnycInterpreter
# - OISever
# - ServerRoutes
# TODO: maybe add better typing to interpreter core message array


class RunCode(TypedDict):
    language: str
    code: str


# should maybe be an enum
class LMC(BaseModel):
    role: Literal["user", "assistant", "computer", "server"]
    type: Optional[
        Literal["message", "console", "command", "status", "image", "code", "audio", "error", "confirmation"]
    ] = None
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
    content: Optional[str] = None
    run_code: Optional[RunCode] = None
    start: Optional[bool] = None
    end: Optional[bool] = None

    @field_validator("content")
    @classmethod
    def ensure_content_when_not_status(cls, v: Optional[str], info: ValidationInfo):
        if info.data.get("type") != "status" and v is None:
            raise ValueError("Content is required for non 'status' LMC messages")
        return v

    @field_validator("start")
    @classmethod
    def start_must_be_true(cls, start: bool, info: ValidationInfo):
        if not start:
            raise ValueError("start must be true in LCM message")
        if info.data.get("content") is not None:
            raise ValueError("content must be None in LCM message if start chunk")
        return start

    @field_validator("end")
    @classmethod
    def end_must_be_true(cls, end: bool, info: ValidationInfo):
        if not end:
            raise ValueError("end must be true in LCM message")
        if info.data.get("content") is not None:
            raise ValueError("content must be None in LCM message if end chunk")
        return end


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
    inner: Union[LMC, bytes]

    def is_lcm(self) -> bool:
        return isinstance(self.inner, LMC)

    def is_bytes(self) -> bool:
        return isinstance(self.inner, bytes)


class ServerInterpreter(OpenInterpreter):
    # TODO: connected to the core, need to figure out how to type this
    llm: Any  # Replace 'Any' with the actual type of llm
    computer: Any  # Replace 'Any' with the actual type of computer

    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)  # type: ignore
        self.messages: List[LMC]
        self.output_queue: asyncio.Queue[Union[LMC, bytes]] = asyncio.Queue()
        self.id = str(os.getenv("INTERPRETER_ID", datetime.now().timestamp()))
        self.print = False  # Will print output

    async def output(self):
        try:
            return await self.output_queue.get()
        except asyncio.CancelledError as e:
            print("async queue cancel error exception ", e)
        except Exception as e:
            print("async queue exception ", e)

    # TODO: protential bug here, test later
    def not_similar_to_last(self, lmc_message: LMC) -> bool:
        if len(self.messages) == 0:  # type: ignore
            return False
        last_msg: Any = self.messages[-1]  # type: ignore
        return lmc_message.type != last_msg.type or lmc_message.format != last_msg.format

    def response_complete_lmc(self) -> LMC:
        return LMC(role="server", type="status", content="complete")

    def accumulate(self, message: ClientMsg):
        if isinstance(message.inner, bytes):
            if not self.messages:
                raise ValueError("Cannot accumulate bytes message when messages list is empty")
            # We initialize as an empty string ^
            # But it actually should be bytes
            if self.messages[-1]["content"] == "":  # type: ignore
                self.messages[-1]["content"] = b""  # type: ignore
            self.messages[-1]["content"] += message.inner  # type: ignore
            return

        if message.inner.content == "active_line":
            return

        # possible logical bug in this code block below
        #
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
    authenticate: Callable[[str | None], bool]
    interpreter: ServerInterpreter

    DEFAULT_HOST = "127.0.0.1"
    DEFAULT_PORT = 8000

    def __init__(self, server_interpreter: ServerInterpreter) -> None:
        # logging setup
        self.log_level = logging.INFO
        logging.basicConfig(
            level=self.log_level,
            format="%(levelname)s: %(message)s",
        )
        self.logger = logging.getLogger(__name__)

        # fast api setup
        self.app = FastAPI()
        self.interpreter = server_interpreter
        self.connection_auth = self.default_connection_auth
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
            timeout_graceful_shutdown=2,
        )

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
    async def default_connection_auth(self, jwt: str | None) -> bool:
        ws_api_key = os.getenv("INTERPRETER_API_KEY")
        #  ws_api_key is None ??
        return ws_api_key is None or (jwt is not None and ws_api_key == jwt)

    def http_auth_middleware(self):
        async def _http_auth_middleware(x_api_key: Annotated[str | None, Header()]):
            if not await self.connection_auth(x_api_key):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Http Middleware Authentication failed. X-API-KEY is invalid.",
                )

        return _http_auth_middleware

    def run(self):
        # "dependencies are the preferred way to create a middleware for authentication"
        # see https://stackoverflow.com/questions/66632841/fastapi-dependency-vs-middleware
        routers = ServerRoutes(
            server_interpreter=self.interpreter,
            logger=self.logger,
            http_auth_middleware=self.http_auth_middleware(),
            core_auth=self.connection_auth,
        ).routers
        for router in routers:
            self.app.include_router(router)

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
    def __init__(
        self,
        logger: logging.Logger,
        server_interpreter: ServerInterpreter,
        http_auth_middleware: Callable[[str], Coroutine[Any, Any, None]],
        core_auth: Callable[[str | None], Coroutine[Any, Any, bool]],
    ):
        self.logger = logger
        self.core_connection_auth = core_auth
        self.auth_routes = APIRouter(dependencies=[Depends(http_auth_middleware)], tags=["auth routes"])
        self.no_auth_routes = APIRouter()
        self.interpreter_task: asyncio.Task[Any] | None = None
        self.server_interpreter = server_interpreter
        # routes setup
        self.no_auth_routes.add_api_route("/", self.home, methods=["GET"])
        self.auth_routes.add_api_websocket_route("/ws", self.ws_endpoint)
        self.no_auth_routes.add_api_websocket_route("/ws/web", self.ws_endpoint)
        self.no_auth_routes.add_api_route("/heartbeat", self.heartbeat, methods=["GET"])
        self.auth_routes.add_api_route("/settings", self.set_settings, methods=["POST"])
        self.auth_routes.add_api_route("/settings/{setting}", self.get_setting, methods=["GET"])
        # TODO
        # self.router.add_api_route("/openai/chat/completions", self.chat_completion, methods=["POST"])
        if os.getenv("INTERPRETER_INSECURE_ROUTES", "").lower() == "true":
            self.auth_routes.add_api_route("/run", self.run_code, methods=["POST"])
            self.auth_routes.add_api_route("/upload", self.upload_file, methods=["POST"])
            self.auth_routes.add_api_route("/download/{filename}", self.download_file, methods=["GET"])

        self.routers = [self.auth_routes, self.no_auth_routes]

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
        is_browser_connection = websocket.url.path.endswith("/web")
        await websocket.accept()
        self.logger.info("WebSocket connection opened / accepted")
        msg_queue: asyncio.Queue[ClientMsg] = asyncio.Queue()

        async def receive_msg_from_client():
            browser_connection_authenticated = False
            while websocket.client_state == WebSocketState.CONNECTED:
                try:
                    async for ws_msg in websocket.iter_json():
                        self.logger.info(f"received message from client: {ws_msg}")
                        try:
                            if ws_msg.get("type") == "ping":
                                await websocket.send_json({"type": "pong"})
                                continue

                            # first message after connection for browser clients should be "auth" with a jwt
                            if is_browser_connection and not browser_connection_authenticated:
                                if not (ws_msg.get("auth") and await self.core_connection_auth(ws_msg.get("auth"))):
                                    return await websocket.close(
                                        code=1003,
                                        reason="Authentication failed. Invalid JWT on first connection message.",
                                    )
                                browser_connection_authenticated = True
                                await websocket.send_json({"type": "browser_connection_authenticated"})
                                continue

                            # validate msg
                            # if invalid, log and send error msg to client, else put
                            inner_msg = LMC(**ws_msg)
                            client_msg = ClientMsg(inner=inner_msg)
                            msg_queue.put_nowait(client_msg)
                        except Exception as e:
                            err_msg = f"Invalid websocket message: {ws_msg} | {e}"
                            self.logger.error(err_msg)
                            await websocket.send_json(
                                LMC(role="server", type="error", content=err_msg).model_dump(exclude_none=True)
                            )
                except Exception as e:
                    self.logger.warning(f"Client disconnected from websocket or other error occured | {e}")

        async def send_response_to_client():
            while websocket.client_state == WebSocketState.CONNECTED:
                try:
                    output = await self.server_interpreter.output()
                    if output is None:
                        await websocket.close(code=1001, reason="Interpreter closed the connection")
                        break
                    if isinstance(output, LMC):
                        await websocket.send_json(output.model_dump(exclude_none=True))
                    else:
                        await websocket.send_bytes(output)
                    self.server_interpreter.output_queue.task_done()
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
            if isinstance(client_msg.inner, bytes) or (client_msg.inner.end is not True):
                self.server_interpreter.accumulate(client_msg)
            else:
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
            while websocket.client_state == WebSocketState.CONNECTED:
                client_msg = await msg_queue.get()
                if self.interpreter_task and not self.interpreter_task.done():
                    self.interpreter_task.cancel()
                self.interpreter_task = client_msg_to_task(client_msg)

        try:
            await asyncio.gather(receive_msg_from_client(), process_client_msg(), send_response_to_client())
        except Exception as e:
            self.logger.error(f"Error in websocket connection: {e}")
        finally:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close()
            self.logger.info("WebSocket connection closed")

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
    if not interpreter.messages:
        return

    # we got an end chunk

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
    # change the messages to dicts
    # TODO: optimize later
    interpreter.messages = [msg.model_dump(exclude_none=True, exclude_defaults=True) for msg in interpreter.messages]  # type: ignore
    # this is a hack to remove the first message, remove later. not having this cause an uncaught exception
    interpreter.messages.pop(0)
    # print("msgs refined", interpreter.messages)
    for chunk_og in interpreter._respond_and_store():  # type: ignore
        print("chunk", chunk_og)
        chunk: Dict[str, Any] = chunk_og.copy()
        try:
            lmc_chunk = LMC(**chunk)
        except Exception as e:
            logger.error(f"Error parsing LMC chunk produced by interpreter-core: {chunk} | {e}")
            continue
        if lmc_chunk.type == "confirmation":
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
                        "\n------------\n\n```" + lmc_chunk.format,
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

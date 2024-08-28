from oi_server import OIServer, ServerInterpreter
import os
from fastapi.middleware.cors import CORSMiddleware


def authenticate_function(_jwt: str):
    return True


if __name__ == "__main__":
    port = 63863
    os.environ["INTERPRETER_REQUIRE_ACKNOWLEDGE"] = "True"
    async_interpreter = ServerInterpreter()
    async_interpreter.llm.api_base = "https://api.openinterpreter.com/v0/"
    async_interpreter.computer.save_skills = False
    async_interpreter.computer.import_computer_api = True

    server = OIServer(async_interpreter)
    server.port = port
    server.connection_auth = authenticate_function
    # TODO: be more specific about the origins
    origins = ["*"]
    server.app.add_middleware(
        CORSMiddleware,
        allow_origins=[origins],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    server.run()

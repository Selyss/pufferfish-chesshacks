
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import time
import chess
from chess.pgn import read_game
import io
import os

from src.utils import chess_manager
from src import main

app = FastAPI()


@app.post("/")
async def root():
    return JSONResponse(content={"running": True})


@app.post("/move")
async def get_move(request: Request):
    try:
        data = await request.json()
    except Exception as e:
        return JSONResponse(content={"error": "Missing pgn or timeleft"}, status_code=400)

    if ("pgn" not in data or "timeleft" not in data):
        return JSONResponse(content={"error": "Missing pgn or timeleft"}, status_code=400)

    pgn = data["pgn"]
    timeleft = data["timeleft"]  # in milliseconds

    chess_manager.set_context(pgn, timeleft)
    print("pgn", pgn)

    # Try model move; on failure return a safe fallback instead of 500
    try:
        start_time = time.perf_counter()
        move, move_probs, logs = chess_manager.get_model_move()
        end_time = time.perf_counter()
        time_taken = (end_time - start_time) * 1000
    except Exception as e:
        time_taken = (time.perf_counter() - start_time) * 1000
        # Fallback: reconstruct board from PGN and pick the first legal move
        try:
            game = read_game(io.StringIO(pgn))
            board = game.board()
            for m in game.mainline_moves():
                board.push(m)
            fallback_move = next(iter(board.legal_moves), None)
            fallback_move_uci = fallback_move.uci() if fallback_move else None
        except Exception:
            fallback_move_uci = None

        return JSONResponse(
            content={
                "move": fallback_move_uci,
                "move_probs": {},
                "time_taken": time_taken,
                "error": "Bot raised an exception",
                "logs": None,
                "exception": str(e),
            },
            status_code=200,
        )

    # Normalize move and move_probs; avoid failing the endpoint over types
    if isinstance(move, chess.Move):
        move_uci = move.uci()
    elif isinstance(move, str):
        move_uci = move
    else:
        move_uci = None

    # Translate move_probs to Dict[str, float] safely
    move_probs_dict = {}
    if isinstance(move_probs, dict):
        for m, prob in move_probs.items():
            if isinstance(m, chess.Move) and isinstance(prob, (int, float)):
                move_probs_dict[m.uci()] = float(prob)

    return JSONResponse(
        content={
            "move": move_uci,
            "error": None,
            "time_taken": time_taken,
            "move_probs": move_probs_dict,
            "logs": logs,
        },
        status_code=200,
    )

if __name__ == "__main__":
    port = int(os.getenv("SERVE_PORT", "5058"))
    uvicorn.run(app, host="0.0.0.0", port=port)

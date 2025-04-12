import cv2
import depthai as dai
import numpy as np
import chess
import chess.engine
from time import sleep

# Initialize Oak-D Lite pipeline
def init_oakd():
    pipeline = dai.Pipeline()
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.setPreviewSize(640, 480)  # Adjust for your setup
    cam_rgb.setInterleaved(False)
    cam_rgb.preview.link(xout_rgb.input)
    return pipeline

# Detect chessboard and map squares to coordinates
def detect_chessboard(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Find chessboard corners (8x8 internal corners = 7x7 grid)
    ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)
    if not ret:
        print("Chessboard not detected!")
        return None, None

    # Refine corners
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                              criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

    # Get 8x8 square coordinates
    squares = {}
    board_size = 8
    for i in range(board_size):
        for j in range(board_size):
            # Map to chess notation (a1 = bottom-left, h8 = top-right)
            file = chr(ord('a') + j)  # a-h
            rank = str(board_size - i)  # 8-1
            square_name = f"{file}{rank}"
            # Approximate square center (interpolate from corners)
            idx = i * 7 + j if i < 7 and j < 7 else (i-1) * 7 + (j-1)
            if idx < len(corners):
                x, y = corners[idx][0]
                squares[square_name] = (int(x), int(y))
            else:
                # Extrapolate for outer edges
                x = corners[-1][0][0] + (j-6) * (corners[-1][0][0] - corners[-8][0][0])
                y = corners[-1][0][1] + (i-6) * (corners[-1][0][1] - corners[-8][0][1])
                squares[square_name] = (int(x), int(y))

    return squares, corners

# Detect pieces by checking square occupancy
def detect_pieces(frame, squares):
    if squares is None:
        return None

    board_state = {}
    for square, (x, y) in squares.items():
        # Sample a small region at square center
        region = frame[y-10:y+10, x-10:x+10]
        if region.size == 0:
            continue
        # Simple color-based detection (tune thresholds for your pieces)
        mean_color = np.mean(region, axis=(0, 1))
        # Assume pieces are darker/brighter than empty squares
        if np.mean(mean_color) < 100 or np.mean(mean_color) > 200:  # Adjust thresholds
            board_state[square] = 'piece'  # Placeholder (can refine to piece type)
        else:
            board_state[square] = 'empty'

    return board_state

# Convert board state to FEN (simplified)
def board_state_to_fen(board_state, turn='w'):
    if board_state is None:
        return None

    board = chess.Board()
    # Clear board
    board.clear()
    # Map detected pieces to approximate FEN
    for square_name, state in board_state.items():
        if state == 'piece':
            # Placeholder: assume pawns for simplicity (refine with piece recognition)
            square_idx = chess.parse_square(square_name)
            board.set_piece_at(square_idx, chess.Piece(chess.PAWN, chess.WHITE if square_name[1] in '12' else chess.BLACK))

    fen = board.fen()
    return fen

# Get Stockfish counter-move
def get_stockfish_move(fen, stockfish_path="./stockfish"):
    if fen is None:
        return None

    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        board = chess.Board(fen)
        result = engine.play(board, chess.engine.Limit(time=0.1))  # Fast thinking
        move = result.move
        engine.quit()
        return move.uci()  # e.g., "e7e5"
    except Exception as e:
        print(f"Stockfish error: {e}")
        return None

# Main pipeline
def main():
    # Initialize Oak-D Lite
    pipeline = init_oakd()
    with dai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        prev_board_state = None
        stockfish_path = "./stockfish"  # Adjust to your Stockfish binary path

        while True:
            # Capture frame
            in_rgb = q_rgb.get()
            frame = in_rgb.getCvFrame()

            # Detect chessboard
            squares, corners = detect_chessboard(frame)
            if squares is None:
                continue

            # Visualize board (optional, for debugging)
            if corners is not None:
                cv2.drawChessboardCorners(frame, (7, 7), corners, True)
                for square, (x, y) in squares.items():
                    cv2.putText(frame, square, (x-10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Detect pieces
            board_state = detect_pieces(frame, squares)
            if board_state is None:
                continue

            # Check for board change
            if prev_board_state is not None and board_state != prev_board_state:
                print("Board changed!")
                # Convert to FEN
                fen = board_state_to_fen(board_state)
                if fen:
                    print(f"Current FEN: {fen}")
                    # Get Stockfish move
                    move = get_stockfish_move(fen, stockfish_path)
                    if move:
                        print(f"Stockfish counter-move: {move}")
                        # Output move for arm (to be implemented later)
                        # e.g., move = "e7e5" -> send to arm control

            prev_board_state = board_state

            # Show frame
            cv2.imshow("Chessboard", frame)
            if cv2.waitKey(1) == ord('q'):
                break

            # Throttle loop
            sleep(0.1)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
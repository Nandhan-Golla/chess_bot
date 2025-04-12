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
    cam_rgb.setPreviewSize(640, 640)  # Square resolution for 28.5 cm board
    cam_rgb.setInterleaved(False)
    cam_rgb.preview.link(xout_rgb.input)
    return pipeline

# Detect chessboard and map squares to coordinates
def detect_chessboard(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Find chessboard corners (8x8 board = 7x7 internal corners)
    ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)
    if not ret:
        print("Chessboard not detected!")
        return None, None

    # Refine corners
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                              criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

    # Map 8x8 squares (28.5 cm board, ~3.56 cm per square)
    squares = {}
    board_size = 8
    for i in range(board_size):
        for j in range(board_size):
            file = chr(ord('a') + j)  # a-h
            rank = str(board_size - i)  # 8-1
            square_name = f"{file}{rank}"
            # Interpolate square centers
            if i < 7 and j < 7:
                idx = i * 7 + j
                x, y = corners[idx][0]
            else:
                # Extrapolate outer edges
                if i == 7:
                    x = corners[(i-1) * 7 + j][0][0] + (corners[(i-1) * 7 + j][0][0] - corners[(i-2) * 7 + j][0][0])
                    y = corners[(i-1) * 7 + j][0][1] + (corners[(i-1) * 7 + j][0][1] - corners[(i-2) * 7 + j][0][1])
                elif j == 7:
                    x = corners[i * 7 + (j-1)][0][0] + (corners[i * 7 + (j-1)][0][0] - corners[i * 7 + (j-2)][0][0])
                    y = corners[i * 7 + (j-1)][0][1] + (corners[i * 7 + (j-1)][0][1] - corners[i * 7 + (j-2)][0][1])
                else:
                    x = corners[-1][0][0] + (j-6) * (corners[-1][0][0] - corners[-8][0][0])
                    y = corners[-1][0][1] + (i-6) * (corners[-1][0][1] - corners[-8][0][1])
            squares[square_name] = (int(x), int(y))

    return squares, corners

# Detect pieces on squares
def detect_pieces(frame, squares):
    if squares is None:
        return None

    board_state = {}
    square_size_cm = 3.56  # 28.5 cm / 8
    pixels_per_cm = 640 / 28.5  # Approx, adjust after calibration
    region_size = int(square_size_cm * pixels_per_cm / 4)  # Sample ~1/4 square

    for square, (x, y) in squares.items():
        # Sample center region
        region = frame[max(0, y-region_size):y+region_size, max(0, x-region_size):x+region_size]
        if region.size == 0:
            continue
        # Color-based detection (tune for your pieces)
        mean_color = np.mean(region, axis=(0, 1))
        # Threshold for piece vs. empty (adjust after testing)
        if np.mean(mean_color) < 120 or np.mean(mean_color) > 180:
            board_state[square] = 'piece'
        else:
            board_state[square] = 'empty'

    return board_state

# Convert board state to FEN (simplified)
def board_state_to_fen(board_state, turn='w'):
    if board_state is None:
        return None

    board = chess.Board()
    board.clear()
    for square_name, state in board_state.items():
        if state == 'piece':
            # Placeholder: use pawns (refine later)
            square_idx = chess.parse_square(square_name)
            color = chess.WHITE if square_name[1] in '12' else chess.BLACK
            board.set_piece_at(square_idx, chess.Piece(chess.PAWN, color))

    fen = board.fen()
    return fen

# Get Stockfish counter-move
def get_stockfish_move(fen, stockfish_path="./stockfish"):
    if fen is None:
        return None

    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        board = chess.Board(fen)
        result = engine.play(board, chess.engine.Limit(time=0.1))
        move = result.move
        engine.quit()
        return move.uci()
    except Exception as e:
        print(f"Stockfish error: {e}")
        return None

# Main pipeline
def main():
    pipeline = init_oakd()
    with dai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        prev_board_state = None
        stockfish_path = "./stockfish"  # Update path

        while True:
            in_rgb = q_rgb.get()
            frame = in_rgb.getCvFrame()

            # Detect chessboard
            squares, corners = detect_chessboard(frame)
            if squares is None:
                continue

            # Visualize (for debugging)
            if corners is not None:
                cv2.drawChessboardCorners(frame, (7, 7), corners, True)
                for square, (x, y) in squares.items():
                    cv2.putText(frame, square, (x-10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Detect pieces
            board_state = detect_pieces(frame, squares)
            if board_state is None:
                continue

            # Check for change
            if prev_board_state is not None and board_state != prev_board_state:
                print("Board changed!")
                fen = board_state_to_fen(board_state)
                if fen:
                    print(f"Current FEN: {fen}")
                    move = get_stockfish_move(fen, stockfish_path)
                    if move:
                        print(f"Stockfish counter-move: {move}")

            prev_board_state = board_state

            # Display
            cv2.imshow("Chessboard", frame)
            if cv2.waitKey(1) == ord('q'):
                break

            sleep(0.1)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
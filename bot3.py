import cv2
import numpy as np
import depthai as dai
from chessboard import detect_chessboard
from stockfish import Stockfish
import chess
from time import sleep

class ChessVisionSystem:
    def __init__(self):
        # Initialize camera pipeline
        self.pipeline = dai.Pipeline()
        
        # Configure camera
        self.cam_rgb = self.pipeline.create(dai.node.ColorCamera)
        self.cam_rgb.setPreviewSize(800, 800)
        self.cam_rgb.setInterleaved(False)
        self.cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        
        # Create output
        self.xout_rgb = self.pipeline.create(dai.node.XLinkOut)
        self.xout_rgb.setStreamName("rgb")
        self.cam_rgb.preview.link(self.xout_rgb.input)
        
        # Initialize Stockfish
        self.stockfish = Stockfish(path="/usr/games/stockfish")  # Update path as needed
        self.board = chess.Board()
        
        # Chessboard state tracking
        self.previous_board = None
        self.current_board = None
        self.move_count = 0
        
    def detect_chess_pieces(self, frame):
        """Detect chess pieces on the board using computer vision"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (7,7), None)
        
        if not ret:
            return None
            
        # Refine corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        
        # Warp perspective to get top-down view
        pts1 = np.float32([corners[0][0], corners[6][0], corners[42][0], corners[48][0]])
        pts2 = np.float32([[0,0], [700,0], [0,700], [700,700]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(frame, matrix, (700,700))
        
        # Split into squares
        square_size = 700 // 8
        squares = []
        for row in range(8):
            for col in range(8):
                x1 = col * square_size
                y1 = row * square_size
                x2 = x1 + square_size
                y2 = y1 + square_size
                square = warped[y1:y2, x1:x2]
                squares.append((row, col, square))
        
        # Analyze each square for piece presence
        board_state = [[None for _ in range(8)] for _ in range(8)]
        
        for row, col, square in squares:
            # Simple threshold-based piece detection (replace with ML model for better accuracy)
            gray_square = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_square, 120, 255, cv2.THRESH_BINARY)
            
            # Calculate percentage of non-white pixels
            non_white = np.sum(thresh < 200)
            total_pixels = square_size * square_size
            ratio = non_white / total_pixels
            
            if ratio > 0.1:  # Threshold indicating piece presence
                # Determine piece color (simple method - replace with better classifier)
                mean_color = np.mean(square, axis=(0,1))
                if mean_color[0] > mean_color[1] and mean_color[0] > mean_color[2]:
                    board_state[row][col] = 'w'  # White piece (more red)
                else:
                    board_state[row][col] = 'b'  # Black piece
        
        return board_state
    
    def board_to_fen(self, board_state):
        """Convert board state array to FEN string"""
        fen_parts = []
        empty_count = 0
        
        for row in range(8):
            row_str = ""
            empty_count = 0
            
            for col in range(8):
                piece = board_state[row][col]
                
                if piece is None:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        row_str += str(empty_count)
                        empty_count = 0
                    row_str += piece
                    
            if empty_count > 0:
                row_str += str(empty_count)
                
            fen_parts.append(row_str)
        
        # Join with slashes and add other FEN parts (simplified)
        fen = "/".join(fen_parts) + " w KQkq - 0 1"
        return fen
    
    def get_computer_move(self):
        """Get best move from Stockfish"""
        self.stockfish.set_fen_position(self.board.fen())
        best_move = self.stockfish.get_best_move()
        return best_move
    
    def update_board(self, move):
        """Update the chess board with the new move"""
        try:
            chess_move = chess.Move.from_uci(move)
            if chess_move in self.board.legal_moves:
                self.board.push(chess_move)
                return True
        except:
            pass
        return False
    
    def find_move_difference(self, prev_board, current_board):
        """Find the move made between two board states"""
        from_square = None
        to_square = None
        
        for row in range(8):
            for col in range(8):
                if prev_board[row][col] != current_board[row][col]:
                    if prev_board[row][col] is not None and current_board[row][col] is None:
                        from_square = chr(ord('a') + col) + str(8 - row)
                    elif prev_board[row][col] is None and current_board[row][col] is not None:
                        to_square = chr(ord('a') + col) + str(8 - row)
                    elif prev_board[row][col] != current_board[row][col]:
                        # Handle captures
                        from_square = chr(ord('a') + col) + str(8 - row)
                        to_square = chr(ord('a') + col) + str(8 - row)
        
        if from_square and to_square:
            return from_square + to_square
        return None
    
    def run(self):
        with dai.Device(self.pipeline) as device:
            rgb_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            
            print("Chess Vision System Ready. Waiting for moves...")
            
            while True:
                in_rgb = rgb_queue.get()
                frame = in_rgb.getCvFrame()
                
                # Detect current board state
                self.current_board = self.detect_chess_pieces(frame)
                
                if self.current_board is None:
                    cv2.imshow("Chessboard", frame)
                    if cv2.waitKey(1) == ord('q'):
                        break
                    continue
                
                # If we have a previous state, compare to detect moves
                if self.previous_board is not None:
                    move = self.find_move_difference(self.previous_board, self.current_board)
                    
                    if move and self.update_board(move):
                        print(f"Detected move: {move}")
                        
                        # Get computer's response
                        computer_move = self.get_computer_move()
                        print(f"Computer responds: {computer_move}")
                        
                        # Here you would convert computer_move to robot arm commands
                        # For now just print it
                        print(f"Robot arm should move: {computer_move}")
                        
                        # Update the board with computer's move (for visualization)
                        self.update_board(computer_move)
                
                # Update previous board state
                self.previous_board = self.current_board
                
                # Display
                cv2.imshow("Chessboard", frame)
                if cv2.waitKey(1) == ord('q'):
                    break

if __name__ == "__main__":
    chess_system = ChessVisionSystem()
    chess_system.run()
    cv2.destroyAllWindows()
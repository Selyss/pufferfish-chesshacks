# memory_test.py
"""
Memory profiling script that plays a full game against itself.
Tracks memory usage throughout the game.
"""
import chess
import psutil
import os
import sys

# Add mini folder to path
sys.path.insert(0, os.path.dirname(__file__))

from inference_model import load_model
from search import find_best_move_tactical


def print_memory_usage(label):
    """Print current memory usage."""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"{label}: {mem_mb:.2f} MB")
    return mem_mb


def play_game(model, max_moves=200):
    """
    Play a full game against itself.
    
    Args:
        model: Loaded TinyModel instance
        max_moves: Maximum number of moves before declaring draw
    
    Returns:
        Tuple of (result, move_count, memory_samples)
    """
    board = chess.Board()
    move_count = 0
    memory_samples = []
    
    print("\n" + "="*60)
    print("Starting self-play game...")
    print("="*60)
    print(board)
    print()
    
    while not board.is_game_over() and move_count < max_moves:
        move_count += 1
        
        # Find best move
        move = find_best_move_tactical(board, model)
        
        if move is None:
            print(f"\nNo legal moves available at move {move_count}")
            break
        
        # Make the move
        board.push(move)
        
        # Track memory every 10 moves
        if move_count % 10 == 0:
            mem = print_memory_usage(f"After move {move_count}")
            memory_samples.append((move_count, mem))
        
        # Print position every 20 moves
        if move_count % 20 == 0:
            print(f"\nPosition after move {move_count}:")
            print(board)
            print()
    
    print("\n" + "="*60)
    print("Game Over!")
    print("="*60)
    print(f"Final position after {move_count} moves:")
    print(board)
    print()
    
    # Determine result
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        result = f"Checkmate - {winner} wins!"
    elif board.is_stalemate():
        result = "Stalemate"
    elif board.is_insufficient_material():
        result = "Draw (insufficient material)"
    elif board.is_fifty_moves():
        result = "Draw (50-move rule)"
    elif board.is_repetition():
        result = "Draw (repetition)"
    elif move_count >= max_moves:
        result = f"Draw (max moves {max_moves} reached)"
    else:
        result = "Game ended"
    
    print(f"Result: {result}")
    print(f"Total moves: {move_count}")
    
    return result, move_count, memory_samples


def main():
    print("="*60)
    print("Chess Bot Memory Profiling Test")
    print("="*60)
    print()
    
    # Baseline memory
    baseline = print_memory_usage("Baseline (imports only)")
    
    # Load model
    print("\nLoading model...")
    model = load_model("tiny_model.pt")
    after_model = print_memory_usage("After loading model")
    model_overhead = after_model - baseline
    print(f"Model overhead: {model_overhead:.2f} MB")
    
    # Play a game
    result, move_count, memory_samples = play_game(model, max_moves=200)
    
    # Final memory report
    print("\n" + "="*60)
    print("Memory Usage Summary")
    print("="*60)
    print(f"Baseline:            {baseline:.2f} MB")
    print(f"After model load:    {after_model:.2f} MB")
    print(f"Model overhead:      {model_overhead:.2f} MB")
    
    if memory_samples:
        print("\nMemory during game:")
        for move_num, mem in memory_samples:
            print(f"  Move {move_num:3d}: {mem:.2f} MB (Δ {mem - after_model:+.2f} MB)")
        
        min_mem = min(mem for _, mem in memory_samples)
        max_mem = max(mem for _, mem in memory_samples)
        avg_mem = sum(mem for _, mem in memory_samples) / len(memory_samples)
        
        print(f"\nStatistics:")
        print(f"  Minimum: {min_mem:.2f} MB")
        print(f"  Maximum: {max_mem:.2f} MB")
        print(f"  Average: {avg_mem:.2f} MB")
        print(f"  Range:   {max_mem - min_mem:.2f} MB")
    
    final = print_memory_usage("\nFinal memory")
    game_overhead = final - after_model
    print(f"Game overhead: {game_overhead:.2f} MB")
    
    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()

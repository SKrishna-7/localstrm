"""
Milestone 5: PagedAttention Block Manager & Virtual Memory
----------------------------------------------------------
Implements the Operating System-style virtual memory paging system
for GPU VRAM:
  - PhysicalBlock: Fixed 16-token chunk with reference counts.
  - BlockTable: Maps a sequence's logical blocks to non-contiguous physical blocks.
  - BlockManager: Allocates on prompt arrival, dynamically allocates on block
    overflow, and recycles blocks on sequence eviction.
"""

import os
import sys
import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config import add_common_engine_args, get_logger

logger = get_logger("mini_vllm.block_manager")


@dataclass
class PhysicalBlock:
    """Represents a physical chunk of memory in the KV cache pool."""
    block_id: int
    block_size: int = 16  # Capacity: holds 16 tokens
    num_tokens: int = 0   # Number of tokens currently stored
    ref_count: int = 0    # Reference count for sharing (Prefix Caching / COW)

    @property
    def is_full(self) -> bool:
        return self.num_tokens >= self.block_size

    @property
    def free_slots(self) -> int:
        return self.block_size - self.num_tokens


class BlockTable:
    """Maps a sequence's logical blocks to physical block IDs in VRAM."""
    def __init__(self, seq_id: int):
        self.seq_id = seq_id
        self.physical_block_ids: List[int] = []

    def append_block(self, block_id: int):
        self.physical_block_ids.append(block_id)

    def __len__(self):
        return len(self.physical_block_ids)

    def __repr__(self):
        return f"BlockTable(Seq #{self.seq_id} -> PhysBlocks: {self.physical_block_ids})"


class BlockManager:
    """
    Manages the physical KV cache memory pool.
    Acts like the OS Virtual Memory Manager for GPU VRAM.
    """
    def __init__(self, total_blocks: int, block_size: int = 16):
        self.block_size = block_size
        self.total_blocks = total_blocks

        # Pre-allocate physical blocks pool
        self.blocks: List[PhysicalBlock] = [
            PhysicalBlock(block_id=i, block_size=block_size)
            for i in range(total_blocks)
        ]
        self.free_block_ids: List[int] = list(range(total_blocks))
        self.block_tables: Dict[int, BlockTable] = {}

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    @property
    def total_memory_utilization(self) -> float:
        used = self.total_blocks - self.num_free_blocks
        return (used / self.total_blocks) * 100.0

    def can_allocate(self, num_tokens: int) -> bool:
        blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        return self.num_free_blocks >= blocks_needed

    def allocate_for_prompt(self, seq_id: int, prompt_tokens: int) -> BlockTable:
        blocks_needed = (prompt_tokens + self.block_size - 1) // self.block_size

        if self.num_free_blocks < blocks_needed:
            raise MemoryError(
                f"Out of GPU KV Cache Memory! Needed {blocks_needed} blocks, "
                f"but only {self.num_free_blocks} available."
            )

        table = BlockTable(seq_id=seq_id)
        remaining_tokens = prompt_tokens

        for _ in range(blocks_needed):
            block_id = self.free_block_ids.pop(0)
            block = self.blocks[block_id]
            block.ref_count = 1

            tokens_in_this_block = min(remaining_tokens, self.block_size)
            block.num_tokens = tokens_in_this_block
            remaining_tokens -= tokens_in_this_block

            table.append_block(block_id)

        self.block_tables[seq_id] = table
        return table

    def append_token(self, seq_id: int) -> Optional[int]:
        """
        Appends 1 new generated token to the sequence's KV cache.
        If the current last block is full, allocates a new physical block!
        """
        table = self.block_tables[seq_id]
        last_block_id = table.physical_block_ids[-1]
        last_block = self.blocks[last_block_id]

        if not last_block.is_full:
            last_block.num_tokens += 1
            return None
        else:
            if self.num_free_blocks == 0:
                raise MemoryError(f"OOM: Cannot allocate new block for Seq #{seq_id}!")

            new_block_id = self.free_block_ids.pop(0)
            new_block = self.blocks[new_block_id]
            new_block.ref_count = 1
            new_block.num_tokens = 1

            table.append_block(new_block_id)
            return new_block_id

    def free_sequence(self, seq_id: int):
        """Frees all physical blocks assigned to a sequence back to the free pool."""
        if seq_id not in self.block_tables:
            return

        table = self.block_tables.pop(seq_id)
        for block_id in table.physical_block_ids:
            block = self.blocks[block_id]
            block.ref_count -= 1

            if block.ref_count == 0:
                block.num_tokens = 0
                self.free_block_ids.append(block_id)


def print_pool_state(manager: BlockManager, label: str):
    """Visualizes the VRAM block pool status."""
    print(f"\n--- {label} ---")
    print(f"Free Blocks: {manager.num_free_blocks} / {manager.total_blocks} "
          f"({manager.total_memory_utilization:.1f}% VRAM used)")

    grid = []
    for b in manager.blocks:
        if b.ref_count == 0:
            grid.append("[ . ]")
        else:
            grid.append(f"[{b.num_tokens:2d}]")

    for i in range(0, len(grid), 8):
        print("  " + " ".join(grid[i:i+8]))
    print("-" * 45)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PagedAttention Block Manager Simulation")
    add_common_engine_args(parser)
    parser.add_argument("--total-blocks", type=int, default=12, help="Total physical blocks (default: 12)")
    parser.add_argument("--block-size", type=int, default=16, help="Tokens per block (default: 16)")
    args = parser.parse_args()

    print("=" * 65)
    print("       PAGED ATTENTION BLOCK MANAGER VERIFICATION")
    print("=" * 65)

    manager = BlockManager(total_blocks=args.total_blocks, block_size=args.block_size)
    print_pool_state(manager, "Initial State (All Free)")

    # 1. User #1 Prompt = 35 tokens -> 3 blocks
    t1 = manager.allocate_for_prompt(seq_id=1, prompt_tokens=35)
    if args.verbose:
        print("\n>>> Event 1: Seq #1 arrives (Prompt = 35 tokens)")
        print(f"  Allocated: {t1}")
        print_pool_state(manager, "After Seq #1 Prompt Allocation")

    # 2. User #2 Prompt = 20 tokens -> 2 blocks
    t2 = manager.allocate_for_prompt(seq_id=2, prompt_tokens=20)
    if args.verbose:
        print("\n>>> Event 2: Seq #2 arrives (Prompt = 20 tokens)")
        print(f"  Allocated: {t2}")
        print_pool_state(manager, "After Seq #2 Prompt Allocation")

    # 3. User #1 generates 14 tokens in decode
    for step in range(1, 15):
        new_blk = manager.append_token(seq_id=1)
        if new_blk is not None and args.verbose:
            print(f"  [Step {step}] Block filled up! Dynamically allocated new Physical Block #{new_blk}")

    if args.verbose:
        print(f"  Updated: {manager.block_tables[1]}")
        print_pool_state(manager, "After Seq #1 Generation & Dynamic Allocation")

    # 4. User #1 finishes -> Free blocks
    manager.free_sequence(seq_id=1)
    if args.verbose:
        print("\n>>> Event 4: Seq #1 finishes and is evicted!")
        print_pool_state(manager, "After Seq #1 Evicted & Blocks Recycled")

    # 5. User #3 arrives (Prompt = 60 tokens -> 4 blocks)
    t3 = manager.allocate_for_prompt(seq_id=3, prompt_tokens=60)
    print_pool_state(manager, "Final State: Recycled Block Reuse")
    print(f"  • Seq #3 Allocated: {t3}")
    print(f"  • Remaining Free Blocks: {manager.num_free_blocks}/{manager.total_blocks}")
    print("=" * 65)

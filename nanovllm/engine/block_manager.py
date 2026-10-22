from collections import deque
from itertools import islice

from nanovllm.engine.sequence import Sequence
from nanovllm.engine.radix_cache import RadixCache, RadixEvictionQueue


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0

    def reset(self):
        self.ref_count = 1


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.radix = RadixCache()
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self._radix_evictions: RadixEvictionQueue | None = None
        self._pending_match = None

    def _allocate_block(self) -> int:
        if self._radix_evictions is not None:
            block_id = self._radix_evictions.pop()
        else:
            # Prefer empty pages; otherwise discard the least recently used
            # *leaf*. An ancestor remains addressable until its children go.
            block_id = next((i for i in self.free_block_ids
                             if i not in self.radix.by_block), None)
            if block_id is None:
                block_id = self.radix.eviction_candidate(set(self.free_block_ids))
                if block_id is None:
                    raise RuntimeError("No evictable radix KV page")
            self.free_block_ids.remove(block_id)
            self.radix.remove(block_id)
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        cached = self.cached_blocks(seq.prompt_token_ids, seq.num_blocks - 1)
        self._pending_match = (seq, cached, self.radix.clock)
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for block_id in cached:
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def cached_blocks(self, tokens: list[int], limit: int | None = None) -> list[int]:
        limit = min(len(tokens) // self.block_size, limit if limit is not None else len(tokens) // self.block_size)
        return self.radix.match(tokens, self.block_size, limit)

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        pending = self._pending_match
        self._pending_match = None
        if (pending is not None and pending[0] is seq and
                pending[2] == self.radix.clock and
                len(pending[1]) == num_cached_blocks and
                all(i in self.radix.by_block for i in pending[1])):
            matched = pending[1]
        else:
            matched = self.cached_blocks(seq.token_ids, num_cached_blocks)
        if len(matched) != num_cached_blocks:
            raise RuntimeError("Prefix was evicted before allocation")
        last_node = None
        for block_id in matched:
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
            node = self.radix.by_block[block_id][0]
            if node is not last_node:
                self.radix.touch(node)
                last_node = node
        num_new_blocks = seq.num_blocks - num_cached_blocks
        # The next pages are empty, so no eviction ordering is needed.
        if (len(self.free_block_ids) >= num_new_blocks and
                all(block_id not in self.radix.by_block for block_id in
                    islice(self.free_block_ids, num_new_blocks))):
            for _ in range(num_new_blocks):
                seq.block_table.append(self._allocate_block())
        else:
            self._radix_evictions = RadixEvictionQueue(self.radix, self.free_block_ids)
            try:
                for _ in range(num_new_blocks):
                    seq.block_table.append(self._allocate_block())
            finally:
                if self._radix_evictions is not None:
                    remaining = self._radix_evictions.free
                    self.free_block_ids = deque(i for i in self.free_block_ids if i in remaining)
                self._radix_evictions = None
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def cache_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        self.radix.insert(seq.token_ids, self.block_size, seq.block_table, end)

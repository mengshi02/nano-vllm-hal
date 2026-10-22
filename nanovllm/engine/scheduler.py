from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.decode_only = False

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def _prefix_being_built(self, seq: Sequence, active: list[Sequence]) -> bool:
        """Wait only if another prefill is producing our next missing full block."""
        if seq.block_table or not any(other is not seq and other.is_prefill for other in active):
            return False
        cached = len(self.block_manager.cached_blocks(seq.prompt_token_ids,
                                                        seq.num_blocks - 1))
        end = (cached + 1) * self.block_size
        if seq.num_prompt_tokens < end:
            return False
        active_ids = {other.seq_id for other in active if other is not seq and
                      other.is_prefill and other.num_cached_tokens < end and
                      other.num_prompt_tokens >= end}
        return self.block_manager.radix.being_built(seq.prompt_token_ids,
                                                      self.block_size, active_ids)

    def _order_waiting_by_prefix(self, scheduled: list[Sequence]):
        if len(self.waiting) < 2:
            return
        active = [s for s in (*self.running, *scheduled)
                  if s.is_prefill and s.num_cached_tokens < s.num_prompt_tokens]
        def priority(item):
            position, seq = item
            if seq.block_table:  # Finish an already admitted prompt chunk first.
                return (-1, 0, position)
            cached = len(self.block_manager.cached_blocks(seq.prompt_token_ids))
            awaiting = self._prefix_being_built(seq, active)
            return (int(awaiting), -cached, position)
        self.waiting = deque(seq for _, seq in sorted(enumerate(self.waiting), key=priority))

    def schedule(self, allow_empty: bool = False) -> list[Sequence]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # Running requests get their next token before new prompts use the budget.
        while self.running and len(scheduled_seqs) < self.max_num_seqs and num_batched_tokens < self.max_num_batched_tokens:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.decode_only:
                    self.running.appendleft(seq)
                    raise RuntimeError("Decode engine is out of KV blocks; refusing to recompute the prompt")
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
                num_batched_tokens += 1
        self.running.extendleft(reversed(scheduled_seqs))

        # Fill the remaining token and sequence budget with prompt chunks.
        self._order_waiting_by_prefix(scheduled_seqs)
        skipped = 0
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if self._prefix_being_built(seq, [*self.running, *scheduled_seqs]):
                # Let the first request materialize only the missing shared
                # block; independent suffixes can still batch together.
                self.waiting.rotate(-1)
                skipped += 1
                if skipped == len(self.waiting):
                    break
                continue
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    self.waiting.rotate(-1)
                    skipped += 1
                    if skipped == len(self.waiting):
                        break
                    continue
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            skipped = 0
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            # A sole prompt that completes in this step cannot be observed as
            # in-flight by another schedule(); skip a redundant tree claim.
            if (len(self.waiting) > 1 or
                    seq.num_cached_tokens + seq.num_scheduled_tokens < seq.num_prompt_tokens):
                self.block_manager.radix.claim(seq.seq_id, seq.prompt_token_ids,
                                               self.block_size)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if not scheduled_seqs:
            if allow_empty:
                return []
            assert False, (f"No runnable requests: waiting={[(s.seq_id, s.num_tokens, s.num_cached_tokens, s.block_table) for s in self.waiting]}, "
                           f"running={[(s.seq_id, s.num_tokens, s.block_table) for s in self.running]}, "
                           f"free_blocks={len(self.block_manager.free_block_ids)}")
        return scheduled_seqs

    def preempt(self, seq: Sequence):
        self.block_manager.radix.release(seq.seq_id)
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.cache_blocks(seq)
            self.block_manager.radix.release(seq.seq_id)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if seq.is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            seq.is_prefill = False
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)


class DataParallelScheduler:
    """Route each request to one attention rank and schedule the ranks together."""

    def __init__(self, config: Config):
        self.size = config.data_parallel_size
        self.schedulers = [Scheduler(config) for _ in range(self.size)]

    def add(self, seq: Sequence):
        # Global sequence IDs are stable and make request placement deterministic.
        self.schedulers[seq.seq_id % self.size].add(seq)

    def is_finished(self):
        return all(scheduler.is_finished() for scheduler in self.schedulers)

    def schedule_ranks(self) -> list[list[Sequence]]:
        rank_seqs = [scheduler.schedule(allow_empty=True)
                     for scheduler in self.schedulers]
        if not any(rank_seqs):
            raise RuntimeError("No runnable requests on any data-parallel rank")
        return rank_seqs

    def postprocess_ranks(self, rank_seqs: list[list[Sequence]],
                          rank_token_ids: list[list[int]]):
        if len(rank_seqs) != self.size or len(rank_token_ids) != self.size:
            raise ValueError("DP scheduler received incomplete rank results")
        for scheduler, seqs, token_ids in zip(self.schedulers, rank_seqs,
                                               rank_token_ids):
            scheduler.postprocess(seqs, token_ids)

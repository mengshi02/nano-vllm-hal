from dataclasses import dataclass, field
from collections import deque
from heapq import heapify, heappop, heappush


Page = tuple[int, ...]


class RadixEvictionQueue:
    """Reuse the leaf/LRU ordering while one request allocates many pages."""

    def __init__(self, radix: "RadixCache", free_ids):
        self.radix = radix
        self.free = set(free_ids)
        self.empty = deque(i for i in free_ids if i not in radix.by_block)
        self.order = {block_id: i for i, block_id in enumerate(radix.by_block)}
        self.leaves = []
        self.any = None
        for block_id, (node, offset) in radix.by_block.items():
            if block_id not in self.free:
                continue
            entry = (node.last_used, self.order[block_id], block_id, node, offset)
            if not node.children and offset == len(node.block_ids) - 1:
                self.leaves.append(entry)
        heapify(self.leaves)

    def _valid(self, entry, leaf=False):
        age, _, block_id, node, offset = entry
        return (block_id in self.free and self.radix.by_block.get(block_id) == (node, offset)
                and node.last_used == age and
                (not leaf or not node.children and offset == len(node.block_ids) - 1))

    def _take(self, heap, leaf=False):
        while heap:
            entry = heappop(heap)
            if self._valid(entry, leaf):
                return entry[2], entry[3]
        return None

    def pop(self):
        while self.empty:
            block_id = self.empty.popleft()
            if block_id in self.free and block_id not in self.radix.by_block:
                self.free.remove(block_id)
                return block_id
        picked = self._take(self.leaves, leaf=True)
        if picked is None:
            if self.any is None:
                self.any = [(node.last_used, self.order[block_id], block_id, node, offset)
                            for block_id, (node, offset) in self.radix.by_block.items()
                            if block_id in self.free]
                heapify(self.any)
            picked = self._take(self.any)
        if picked is None:
            raise RuntimeError("No evictable radix KV page")
        block_id, node = picked
        had_descendants = bool(node.children)
        self.free.remove(block_id)
        self.radix.remove(block_id)
        # Only the nearest surviving ancestor can become a new leaf.
        while node is not None and (not node.block_ids or node.children):
            node = node.parent
        if node is not None:
            terminal = node.block_ids[-1]
            if terminal in self.free:
                entry = self.radix.by_block.get(terminal)
                if entry == (node, len(node.block_ids) - 1):
                    heappush(self.leaves, (node.last_used, self.order[terminal],
                                           terminal, node, entry[1]))
        # Removing an indexed ancestor may invalidate its entire subtree.
        # Those still-free physical pages are now available as empty pages.
        if had_descendants:
            self.empty.extend(i for i in self.free if i not in self.radix.by_block)
        return block_id


@dataclass(eq=False)
class RadixNode:
    parent: "RadixNode | None" = None
    key: tuple[Page, ...] = ()
    block_ids: tuple[int, ...] = ()
    children: dict[Page, "RadixNode"] = field(default_factory=dict)
    building: set[int] = field(default_factory=set)
    last_used: int = 0


class RadixCache:
    """Page-aligned compressed radix tree over resident KV and in-flight pages."""

    def __init__(self):
        self.root = RadixNode()
        self.by_block: dict[int, tuple[RadixNode, int]] = {}
        self.claims: dict[int, RadixNode] = {}
        self.clock = 0

    @staticmethod
    def _pages(tokens: list[int], block_size: int, limit: int) -> tuple[Page, ...]:
        return tuple(tuple(tokens[i * block_size:(i + 1) * block_size])
                     for i in range(limit))

    def _walk(self, pages: tuple[Page, ...]) -> tuple[RadixNode, int, list[int]]:
        node = self.root
        matched: list[int] = []
        pos = 0
        while pos < len(pages):
            child = node.children.get(pages[pos])
            if child is None:
                break
            common = 0
            while (common < len(child.key) and pos + common < len(pages)
                   and child.key[common] == pages[pos + common]):
                common += 1
            resident = min(common, len(child.block_ids))
            matched.extend(child.block_ids[:resident])
            pos += resident
            if common != len(child.key) or resident != len(child.key):
                break
            node = child
        return node, pos, matched

    def match(self, tokens: list[int], block_size: int, limit: int) -> list[int]:
        """Return the longest resident full-page prefix."""
        node = self.root
        matched = []
        pos = 0
        while pos < limit:
            start = pos * block_size
            child = node.children.get(tuple(tokens[start:start + block_size]))
            if child is None:
                break
            common = 1
            while common < len(child.key) and pos + common < limit:
                start = (pos + common) * block_size
                if child.key[common] != tuple(tokens[start:start + block_size]):
                    break
                common += 1
            resident = min(common, len(child.block_ids))
            matched.extend(child.block_ids[:resident])
            pos += resident
            if common != len(child.key) or resident != len(child.key):
                break
            node = child
        return matched

    def _split(self, node: RadixNode, length: int) -> RadixNode:
        """Split a compressed edge at a page boundary, preserving page owners."""
        assert 0 < length < len(node.key)
        parent = node.parent
        prefix = RadixNode(parent=parent, key=node.key[:length],
                           block_ids=node.block_ids[:length], last_used=node.last_used)
        parent.children[prefix.key[0]] = prefix
        node.key = node.key[length:]
        node.block_ids = node.block_ids[length:]
        node.parent = prefix
        prefix.children[node.key[0]] = node
        for i, block_id in enumerate(prefix.block_ids):
            self.by_block[block_id] = (prefix, i)
        for i, block_id in enumerate(node.block_ids):
            self.by_block[block_id] = (node, i)
        return prefix

    def insert(self, tokens: list[int], block_size: int, block_ids: list[int], limit: int):
        pages = self._pages(tokens, block_size, limit)
        node = self.root
        pos = 0
        while pos < limit:
            child = node.children.get(pages[pos])
            if child is None:
                child = RadixNode(parent=node, key=pages[pos:],
                                  block_ids=tuple(block_ids[pos:limit]))
                node.children[pages[pos]] = child
                for offset, block_id in enumerate(child.block_ids):
                    self.by_block[block_id] = (child, offset)
                self.touch(child)
                return
            common = 0
            while (common < len(child.key) and pos + common < limit
                   and child.key[common] == pages[pos + common]):
                common += 1
            if common < len(child.key):
                child = self._split(child, common)
            if not child.block_ids:
                # A scheduled prefill claimed the next page before writing it.
                child.block_ids = (block_ids[pos],)
                self.by_block[block_ids[pos]] = (child, 0)
            pos += common
            node = child
            self.touch(node)

    def touch(self, node: RadixNode):
        self.clock += 1
        node.last_used = self.clock

    def claim(self, seq_id: int, tokens: list[int], block_size: int):
        """Mark the next missing complete page on this request's tree path."""
        self.release(seq_id)
        limit = len(tokens) // block_size
        pages = self._pages(tokens, block_size, limit)
        node = self.root
        pos = 0
        while pos < limit:
            child = node.children.get(pages[pos])
            if child is None:
                child = RadixNode(parent=node, key=(pages[pos],))
                node.children[pages[pos]] = child
            else:
                common = 0
                while (common < len(child.key) and pos + common < limit
                       and child.key[common] == pages[pos + common]):
                    common += 1
                if common < len(child.key):
                    child = self._split(child, common)
                if len(child.block_ids) == len(child.key):
                    pos += common
                    node = child
                    continue
            child.building.add(seq_id)
            self.claims[seq_id] = child
            return

    def being_built(self, tokens: list[int], block_size: int,
                    active_ids: set[int]) -> bool:
        pages = self._pages(tokens, block_size, len(tokens) // block_size)
        node, pos, _ = self._walk(pages)
        if pos >= len(pages):
            return False
        child = node.children.get(pages[pos])
        return bool(child and child.building & active_ids)

    def release(self, seq_id: int):
        node = self.claims.pop(seq_id, None)
        if node is None:
            return
        node.building.discard(seq_id)
        self._prune(node)

    def _prune(self, node: RadixNode):
        while (node.parent is not None and not node.block_ids
               and not node.children and not node.building):
            parent = node.parent
            del parent.children[node.key[0]]
            node = parent

    def eviction_candidate(self, free_ids: set[int]) -> int | None:
        """Find the oldest free leaf, or the oldest free page if no leaf exists.

        The caller needs only one victim. Sorting every resident page on every
        allocation makes a prefix miss quadratic in the number of free pages.
        Strict comparisons preserve the previous insertion-order tie break.
        """
        oldest_leaf = oldest_page = None
        leaf_age = page_age = None
        for block_id, (node, offset) in self.by_block.items():
            if block_id not in free_ids:
                continue
            if page_age is None or node.last_used < page_age:
                oldest_page, page_age = block_id, node.last_used
            if not node.children and offset == len(node.block_ids) - 1:
                if leaf_age is None or node.last_used < leaf_age:
                    oldest_leaf, leaf_age = block_id, node.last_used
        return oldest_leaf if oldest_leaf is not None else oldest_page

    def remove(self, block_id: int):
        entry = self.by_block.pop(block_id, None)
        if entry is None:
            return
        node, offset = entry
        if offset == len(node.block_ids) - 1 and not node.children and not node.building:
            # The normal leaf eviction has no subtree to invalidate. Keep
            # the remaining edge and its indexed pages in place.
            parent = node.parent
            first_page = node.key[0]
            node.block_ids = node.block_ids[:offset]
            node.key = node.key[:offset]
            if not node.key:
                del parent.children[first_page]
                self._prune(parent)
            return
        # Dropping an ancestor invalidates the subtree's *index*, not any
        # physical descendant page still owned by a running request.
        descendants = [node]
        while descendants:
            child = descendants.pop()
            descendants.extend(child.children.values())
            for seq_id in child.building:
                self.claims.pop(seq_id, None)
            child.building.clear()
            if child is node:
                for old_id in child.block_ids[offset:]:
                    self.by_block.pop(old_id, None)
                child.block_ids = child.block_ids[:offset]
                child.key = child.key[:offset]
            else:
                for old_id in child.block_ids:
                    self.by_block.pop(old_id, None)
            child.children.clear()
        if not node.key:
            parent = node.parent
            if parent is not None:
                for key, child in list(parent.children.items()):
                    if child is node:
                        del parent.children[key]
                        break
                self._prune(parent)
        else:
            self._prune(node)

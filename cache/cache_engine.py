"""
Core Cache Engine module for RAG pipelines.

Provides foundational caching and rate limiting primitives including consistent hashing
ring routing, LRU eviction with TTL support, and condition-based leaky bucket rate limiting.
"""

import hashlib
import threading
import time
from typing import Any, Dict, List, Optional


class ConsistentHashRing:
    """
    Consistent Hash Ring implementation using MD5 projection onto a 64-bit integer space.

    Provides deterministic request routing across physical nodes with uniform key distribution
    via virtual node replication.
    """

    def __init__(self, virtual_nodes: int = 150) -> None:
        """
        Initializes the consistent hash ring.

        Args:
            virtual_nodes: Number of virtual node replicas to place per physical node.
        """
        if virtual_nodes < 1:
            raise ValueError("virtual_nodes must be greater than or equal to 1.")

        self.virtual_nodes: int = virtual_nodes
        self._ring: Dict[int, str] = {}
        self._sorted_keys: List[int] = []
        self._lock: threading.Lock = threading.Lock()

    def _hash(self, key: str) -> int:
        """Computes a signed 64-bit integer hash from MD5 digest."""
        digest = hashlib.md5(key.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=True)

    def _insert_sorted_key(self, key_hash: int) -> None:
        """Inserts key_hash into _sorted_keys preserving sorted order."""
        low = 0
        high = len(self._sorted_keys)
        while low < high:
            mid = (low + high) // 2
            if self._sorted_keys[mid] < key_hash:
                low = mid + 1
            else:
                high = mid
        self._sorted_keys.insert(low, key_hash)

    def _remove_sorted_key(self, key_hash: int) -> None:
        """Removes key_hash from _sorted_keys."""
        low = 0
        high = len(self._sorted_keys) - 1
        while low <= high:
            mid = (low + high) // 2
            if self._sorted_keys[mid] == key_hash:
                del self._sorted_keys[mid]
                return
            elif self._sorted_keys[mid] < key_hash:
                low = mid + 1
            else:
                high = mid - 1

    def add_node(self, node: str) -> None:
        """
        Adds a physical node and its virtual replicas to the hash ring.

        Args:
            node: Physical node identifier string.
        """
        with self._lock:
            for i in range(self.virtual_nodes):
                vnode_key = f"{node}#vnode-{i}"
                vnode_hash = self._hash(vnode_key)
                if vnode_hash not in self._ring:
                    self._ring[vnode_hash] = node
                    self._insert_sorted_key(vnode_hash)

    def remove_node(self, node: str) -> None:
        """
        Removes a physical node and all corresponding virtual replicas from the ring.

        Args:
            node: Physical node identifier string to remove.
        """
        with self._lock:
            for i in range(self.virtual_nodes):
                vnode_key = f"{node}#vnode-{i}"
                vnode_hash = self._hash(vnode_key)
                if vnode_hash in self._ring:
                    del self._ring[vnode_hash]
                    self._remove_sorted_key(vnode_hash)

    def get_node(self, key: str) -> Optional[str]:
        """
        Finds the responsible physical node for a given cache key.

        Args:
            key: Cache lookup key string.

        Returns:
            The mapped physical node identifier, or None if the ring is empty.
        """
        with self._lock:
            if not self._sorted_keys:
                return None

            key_hash = self._hash(key)
            low = 0
            high = len(self._sorted_keys) - 1

            while low <= high:
                mid = (low + high) // 2
                if self._sorted_keys[mid] < key_hash:
                    low = mid + 1
                else:
                    high = mid - 1

            idx = low if low < len(self._sorted_keys) else 0
            vnode_hash = self._sorted_keys[idx]
            return self._ring.get(vnode_hash)


class _LRUNode:
    """Internal doubly linked list node for LRU cache tracking."""

    def __init__(
        self,
        key: Any = None,
        value: Any = None,
        expiry_time: Optional[float] = None,
    ) -> None:
        self.key: Any = key
        self.value: Any = value
        self.expiry_time: Optional[float] = expiry_time
        self.prev: Optional["_LRUNode"] = None
        self.next: Optional["_LRUNode"] = None


class LRUCache:
    """
    Least Recently Used (LRU) cache with O(1) operational complexity and TTL support.

    Combines a hash map for constant-time key lookups with a doubly linked list
    for constant-time eviction ordering.
    """

    def __init__(self, capacity: int) -> None:
        """
        Initializes LRUCache with fixed capacity.

        Args:
            capacity: Maximum number of entries stored before eviction occurs.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be greater than 0.")

        self.capacity: int = capacity
        self._cache: Dict[Any, _LRUNode] = {}
        self._head: _LRUNode = _LRUNode()
        self._tail: _LRUNode = _LRUNode()
        self._head.next = self._tail
        self._tail.prev = self._head
        self._lock: threading.Lock = threading.Lock()

    def _add_node(self, node: _LRUNode) -> None:
        """Inserts node at MRU position directly after head sentinel."""
        node.prev = self._head
        node.next = self._head.next
        if self._head.next:
            self._head.next.prev = node
        self._head.next = node

    def _remove_node(self, node: _LRUNode) -> None:
        """Detaches node from the doubly linked list."""
        prev_node = node.prev
        next_node = node.next
        if prev_node:
            prev_node.next = next_node
        if next_node:
            next_node.prev = prev_node
        node.prev = None
        node.next = None

    def _move_to_head(self, node: _LRUNode) -> None:
        """Moves existing node to MRU position."""
        self._remove_node(node)
        self._add_node(node)

    def _pop_tail(self) -> Optional[_LRUNode]:
        """Removes and returns LRU node directly before tail sentinel."""
        lru_node = self._tail.prev
        if lru_node is self._head or lru_node is None:
            return None
        self._remove_node(lru_node)
        return lru_node

    def get(self, key: Any) -> Any:
        """
        Retrieves cached value by key if present and unexpired.

        Args:
            key: Lookup key.

        Returns:
            Cached value if key exists and is valid, else None.
        """
        with self._lock:
            if key not in self._cache:
                return None

            node = self._cache[key]
            if node.expiry_time is not None and time.time() > node.expiry_time:
                self._remove_node(node)
                del self._cache[key]
                return None

            self._move_to_head(node)
            return node.value

    def put(self, key: Any, value: Any, ttl: Optional[float] = None) -> None:
        """
        Inserts or updates key-value pair with optional time-to-live.

        Args:
            key: Target cache key.
            value: Data payload.
            ttl: Optional duration in seconds before entry expiration.
        """
        expiry_time = (time.time() + ttl) if (ttl is not None and ttl > 0) else None

        with self._lock:
            if key in self._cache:
                node = self._cache[key]
                node.value = value
                node.expiry_time = expiry_time
                self._move_to_head(node)
            else:
                if len(self._cache) >= self.capacity:
                    lru_node = self._pop_tail()
                    if lru_node and lru_node.key in self._cache:
                        del self._cache[lru_node.key]

                new_node = _LRUNode(key=key, value=value, expiry_time=expiry_time)
                self._cache[key] = new_node
                self._add_node(new_node)


class LeakyBucketRateLimiter:
    """
    Rate limiter implementing the Leaky Bucket algorithm using threading.Condition.

    Regulates request rates by accumulating volume up to a max capacity and continuously
    leaking at a fixed rate per second.
    """

    def __init__(self, capacity: float, leak_rate: float) -> None:
        """
        Initializes LeakyBucketRateLimiter.

        Args:
            capacity: Maximum bucket storage volume.
            leak_rate: Outflow rate in units per second.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be greater than 0.")
        if leak_rate <= 0:
            raise ValueError("Leak rate must be greater than 0.")

        self.capacity: float = float(capacity)
        self.leak_rate: float = float(leak_rate)
        self._water_level: float = 0.0
        self._last_leak_time: float = time.monotonic()
        self._lock: threading.Lock = threading.Lock()
        self._condition: threading.Condition = threading.Condition(self._lock)

    def _leak(self) -> None:
        """Calculates and deducts volume leaked since last check."""
        now = time.monotonic()
        elapsed = now - self._last_leak_time
        if elapsed > 0:
            leaked = elapsed * self.leak_rate
            self._water_level = max(0.0, self._water_level - leaked)
            self._last_leak_time = now

    def acquire(self, amount: float = 1.0, timeout: Optional[float] = None) -> bool:
        """
        Acquires bucket capacity, blocking until space is available or timeout occurs.

        Args:
            amount: Request volume to add to the bucket.
            timeout: Optional maximum duration in seconds to wait.

        Returns:
            True if capacity was successfully acquired, False on timeout.
        """
        if amount <= 0:
            raise ValueError("Amount must be greater than 0.")
        if amount > self.capacity:
            raise ValueError(
                f"Requested amount {amount} exceeds maximum bucket capacity {self.capacity}."
            )

        start_time = time.monotonic()

        with self._condition:
            while True:
                self._leak()

                if self._water_level + amount <= self.capacity:
                    self._water_level += amount
                    self._condition.notify_all()
                    return True

                needed_leak = (self._water_level + amount) - self.capacity
                wait_seconds = needed_leak / self.leak_rate

                if timeout is not None:
                    elapsed_total = time.monotonic() - start_time
                    remaining_timeout = timeout - elapsed_total
                    if remaining_timeout <= 0:
                        return False
                    wait_seconds = min(wait_seconds, remaining_timeout)

                self._condition.wait(timeout=wait_seconds)

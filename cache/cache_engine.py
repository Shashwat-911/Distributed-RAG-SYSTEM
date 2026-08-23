"""
Core Cache Engine module for RAG pipelines.

Implements the AeroCache distributed caching architecture:
1. ConsistentHashRing: MD5 projection with 150 virtual nodes per partition for uniform distribution.
2. LRUCache: O(1) hash map + doubly linked list with millisecond TTL support.
3. PredictiveEvictionPolicy: Frequency-recency decay model for predictive cold-key eviction.
4. ShardedAeroCache: Multi-partition sharded caching engine eliminating lock contention.
5. LeakyBucketRateLimiter: Condition-based smooth request rate limiter.
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
from typing import Any, Dict, List, Optional, Tuple


class ConsistentHashRing:
    """
    Consistent Hash Ring implementation using MD5 projection onto a 64-bit integer space.

    Provides deterministic request routing across physical nodes/partitions with uniform
    key distribution via virtual node replication (default: 150 virtual nodes).
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
        """Inserts key_hash into _sorted_keys preserving sorted order via binary search."""
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
        Finds the responsible physical node/partition for a given cache key.

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

    def get_all_nodes(self) -> List[str]:
        """Returns list of unique physical nodes in the ring."""
        with self._lock:
            return sorted(list(set(self._ring.values())))


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
        self.access_count: int = 1
        self.last_accessed: float = time.time()
        self.prev: Optional["_LRUNode"] = None
        self.next: Optional["_LRUNode"] = None


class PredictiveEvictionPolicy:
    """
    AeroCache Predictive AI Eviction Policy.

    Evaluates cache entry utility based on access frequency, time-decayed recency,
    and access velocity. Proactively identifies 'cold' keys for early eviction
    before memory saturation, with O(1) LRU fallback.
    """

    def __init__(self, half_life_seconds: float = 300.0) -> None:
        self.half_life: float = half_life_seconds

    def score(self, node: _LRUNode, now: Optional[float] = None) -> float:
        """
        Computes the utility score of a cache node. Higher score = higher utility (retain).
        Lower score = cold key (candidate for eviction).

        Formula:
            Utility = log(1 + access_count) * exp(-ln(2) * (now - last_accessed) / half_life)
        """
        current_time = now if now is not None else time.time()
        elapsed = max(0.0, current_time - node.last_accessed)
        decay = math.exp(-0.693147 * elapsed / self.half_life)
        frequency_weight = math.log1p(node.access_count)
        return frequency_weight * decay


class LRUCache:
    """
    Least Recently Used (LRU) cache with O(1) operational complexity, TTL support,
    and AeroCache Predictive Eviction integration.
    """

    def __init__(
        self,
        capacity: int,
        predictive_eviction: bool = True,
    ) -> None:
        """
        Initializes LRUCache with fixed capacity.

        Args:
            capacity: Maximum number of entries stored before eviction occurs.
            predictive_eviction: Whether to apply frequency-recency predictive eviction.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be greater than 0.")

        self.capacity: int = capacity
        self.predictive_eviction: bool = predictive_eviction
        self._policy: PredictiveEvictionPolicy = PredictiveEvictionPolicy()
        self._cache: Dict[Any, _LRUNode] = {}
        self._head: _LRUNode = _LRUNode()
        self._tail: _LRUNode = _LRUNode()
        self._head.next = self._tail
        self._tail.prev = self._head
        self._lock: threading.Lock = threading.Lock()
        self.evictions: int = 0
        self.predictive_evictions: int = 0

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
        """Moves existing node to MRU position and updates access telemetry."""
        node.access_count += 1
        node.last_accessed = time.time()
        self._remove_node(node)
        self._add_node(node)

    def _pop_tail(self) -> Optional[_LRUNode]:
        """Removes and returns LRU node directly before tail sentinel."""
        lru_node = self._tail.prev
        if lru_node is self._head or lru_node is None:
            return None
        self._remove_node(lru_node)
        return lru_node

    def _find_coldest_node(self, sample_size: int = 6) -> Optional[_LRUNode]:
        """
        AeroCache AI Predictive scan: samples candidate tail nodes and evicts
        the node with the lowest utility score.
        """
        candidates: List[_LRUNode] = []
        curr = self._tail.prev
        while curr and curr is not self._head and len(candidates) < sample_size:
            candidates.append(curr)
            curr = curr.prev

        if not candidates:
            return None

        now = time.time()
        coldest = min(candidates, key=lambda n: self._policy.score(n, now))
        self._remove_node(coldest)
        return coldest

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
                    evicted: Optional[_LRUNode] = None
                    if self.predictive_eviction and len(self._cache) > 4:
                        evicted = self._find_coldest_node()
                        if evicted:
                            self.predictive_evictions += 1
                    if evicted is None:
                        evicted = self._pop_tail()

                    if evicted and evicted.key in self._cache:
                        del self._cache[evicted.key]
                        self.evictions += 1

                new_node = _LRUNode(key=key, value=value, expiry_time=expiry_time)
                self._cache[key] = new_node
                self._add_node(new_node)

    def delete(self, key: Any) -> bool:
        """Removes a key from the cache. Returns True if removed."""
        with self._lock:
            if key in self._cache:
                node = self._cache[key]
                self._remove_node(node)
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Evicts all keys from this cache partition."""
        with self._lock:
            self._cache.clear()
            self._head.next = self._tail
            self._tail.prev = self._head

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)


class ShardedAeroCache:
    """
    Distributed Sharded AeroCache Engine.

    Distributes cache entries across N independent LRUCache partitions using
    a ConsistentHashRing with 150 virtual nodes per partition. Eliminates single-lock
    contention for high-concurrency read/write operations and maintains global hit/miss stats.
    """

    def __init__(
        self,
        num_partitions: int = 4,
        partition_capacity: int = 250,
        virtual_nodes: int = 150,
        default_ttl: Optional[float] = 3600.0,
    ) -> None:
        """
        Args:
            num_partitions: Number of distinct cache partitions/shards.
            partition_capacity: Max keys per partition (Total capacity = num_partitions * partition_capacity).
            virtual_nodes: Number of virtual node tokens placed per partition on hash ring.
            default_ttl: Default entry expiration duration in seconds.
        """
        self.num_partitions: int = max(1, num_partitions)
        self.partition_capacity: int = max(10, partition_capacity)
        self.default_ttl: Optional[float] = default_ttl
        self.ring: ConsistentHashRing = ConsistentHashRing(virtual_nodes=virtual_nodes)

        self._partitions: Dict[str, LRUCache] = {}
        for i in range(self.num_partitions):
            p_id = f"partition-{i}"
            self._partitions[p_id] = LRUCache(
                capacity=self.partition_capacity,
                predictive_eviction=True,
            )
            self.ring.add_node(p_id)

        self._hits: int = 0
        self._misses: int = 0
        self._semantic_hits: int = 0
        self._epoch: int = 1
        self._stats_lock: threading.Lock = threading.Lock()

        # Semantic Vector Cache Store
        self._semantic_entries: List[Dict[str, Any]] = []
        self._semantic_lock: threading.Lock = threading.Lock()
        self._max_semantic_capacity: int = 500

    def _get_partition(self, key: str) -> LRUCache:
        """Determines responsible partition via consistent hashing."""
        partition_id = self.ring.get_node(str(key))
        if partition_id is None or partition_id not in self._partitions:
            return self._partitions["partition-0"]
        return self._partitions[partition_id]

    def get(self, key: str) -> Optional[Any]:
        """Retrieves entry across sharded partitions in O(1) time."""
        partition = self._get_partition(key)
        val = partition.get(key)
        with self._stats_lock:
            if val is not None:
                self._hits += 1
            else:
                self._misses += 1
        return val

    def get_semantic(
        self,
        exact_key: str,
        query_vector: Optional[Any] = None,
        similarity_threshold: float = 0.88,
    ) -> Tuple[Optional[Any], float, str]:
        """
        Retrieves cached response via exact key match or semantic vector similarity.

        Args:
            exact_key: Deterministic cache key string.
            query_vector: Dense query embedding vector.
            similarity_threshold: Minimum cosine similarity required for a semantic cache hit (default 0.88).

        Returns:
            Tuple of (response_value, similarity_score, hit_type)
            where hit_type is 'exact', 'semantic', or 'miss'.
        """
        # 1. Check exact O(1) partition match
        partition = self._get_partition(exact_key)
        exact_val = partition.get(exact_key)
        if exact_val is not None:
            with self._stats_lock:
                self._hits += 1
            return exact_val, 1.0, "exact"

        # 2. Check semantic vector similarity
        if query_vector is not None and len(self._semantic_entries) > 0:
            import numpy as np
            q_vec = np.asarray(query_vector, dtype=np.float32)
            q_norm = np.linalg.norm(q_vec)
            if q_norm > 0:
                q_unit = q_vec / q_norm
                now = time.time()

                with self._semantic_lock:
                    # Clean expired entries
                    self._semantic_entries = [
                        e for e in self._semantic_entries
                        if e.get("expiry_time") is None or now <= e["expiry_time"]
                    ]

                    if self._semantic_entries:
                        vectors = np.stack([e["vector"] for e in self._semantic_entries])
                        # Vectors are stored unit-normalized, so dot product = cosine similarity
                        sims = np.dot(vectors, q_unit)
                        best_idx = int(np.argmax(sims))
                        best_sim = float(sims[best_idx])

                        if best_sim >= similarity_threshold:
                            matched_entry = self._semantic_entries[best_idx]
                            matched_entry["last_accessed"] = now
                            matched_entry["access_count"] = matched_entry.get("access_count", 1) + 1
                            with self._stats_lock:
                                self._hits += 1
                                self._semantic_hits += 1
                            return matched_entry["response"], best_sim, "semantic"

        with self._stats_lock:
            self._misses += 1
        return None, 0.0, "miss"

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Inserts or updates entry in responsible partition."""
        partition = self._get_partition(key)
        target_ttl = ttl if ttl is not None else self.default_ttl
        partition.put(key, value, ttl=target_ttl)

    def put_semantic(
        self,
        key: str,
        query_text: str,
        query_vector: Optional[Any],
        response: Any,
        ttl: Optional[float] = None,
    ) -> None:
        """
        Stores response in both exact partition cache and semantic vector index.

        Args:
            key: Deterministic cache key string.
            query_text: Raw user question.
            query_vector: Query embedding vector.
            response: Response payload.
            ttl: Time to live in seconds.
        """
        self.put(key, response, ttl=ttl)

        if query_vector is not None:
            import numpy as np
            q_vec = np.asarray(query_vector, dtype=np.float32)
            q_norm = np.linalg.norm(q_vec)
            if q_norm > 0:
                unit_vec = q_vec / q_norm
                now = time.time()
                expiry = (now + ttl) if (ttl is not None and ttl > 0) else None

                with self._semantic_lock:
                    # Enforce capacity
                    if len(self._semantic_entries) >= self._max_semantic_capacity:
                        self._semantic_entries.pop(0)

                    self._semantic_entries.append({
                        "key": key,
                        "query_text": query_text,
                        "vector": unit_vec,
                        "response": response,
                        "created_at": now,
                        "last_accessed": now,
                        "access_count": 1,
                        "expiry_time": expiry,
                    })

    def delete(self, key: str) -> bool:
        """Deletes a key from its responsible partition."""
        partition = self._get_partition(key)
        with self._semantic_lock:
            self._semantic_entries = [e for e in self._semantic_entries if e.get("key") != key]
        return partition.delete(key)

    def clear(self) -> None:
        """Flushes all partitions and resets hit/miss counters."""
        with self._stats_lock:
            for p in self._partitions.values():
                p.clear()
            with self._semantic_lock:
                self._semantic_entries.clear()
            self._hits = 0
            self._misses = 0
            self._semantic_hits = 0
            self._epoch += 1

    def stats(self) -> Dict[str, Any]:
        """Aggregates telemetry across all AeroCache partitions."""
        with self._stats_lock:
            total_size = sum(p.size for p in self._partitions.values())
            total_evictions = sum(p.evictions for p in self._partitions.values())
            total_predictive = sum(p.predictive_evictions for p in self._partitions.values())
            total_reqs = self._hits + self._misses
            hit_rate = (self._hits / total_reqs) if total_reqs > 0 else 0.0

            return {
                "size": total_size,
                "capacity": self.num_partitions * self.partition_capacity,
                "num_partitions": self.num_partitions,
                "virtual_nodes_per_partition": self.ring.virtual_nodes,
                "hits": self._hits,
                "misses": self._misses,
                "semantic_hits": self._semantic_hits,
                "semantic_entries": len(self._semantic_entries),
                "hit_rate": hit_rate,
                "evictions": total_evictions,
                "predictive_evictions": total_predictive,
                "epoch": self._epoch,
            }


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

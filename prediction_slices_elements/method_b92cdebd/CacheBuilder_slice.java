// Source-based slice around line 766
// Method: <com.google.common.cache.CacheBuilder: long getExpireAfterWriteNanos()>

        expireAfterWriteNanos == UNSET_INT,
        "expireAfterWrite was already set to %s ns",
        expireAfterWriteNanos);
    checkArgument(duration >= 0, "duration cannot be negative: %s %s", duration, unit);
    this.expireAfterWriteNanos = unit.toNanos(duration);
    return this;
  }

  @SuppressWarnings("GoodTime") // nanos internally, should be Duration
  long getExpireAfterWriteNanos() {
    return (expireAfterWriteNanos == UNSET_INT) ? DEFAULT_EXPIRATION_NANOS : expireAfterWriteNanos;
  }

  /**
   * Specifies that each entry should be automatically removed from the cache once a fixed duration
   * has elapsed after the entry's creation, the most recent replacement of its value, or its last
   * access. Access time is reset by all cache read and write operations (including {@code
   * Cache.asMap().get(Object)} and {@code Cache.asMap().put(K, V)}), but not by {@code
   * containsKey(Object)}, nor by operations on the collection-views of {@link Cache#asMap}}. So,
   * for example, iterating through {@code Cache.asMap().entrySet()} does not reset access time for

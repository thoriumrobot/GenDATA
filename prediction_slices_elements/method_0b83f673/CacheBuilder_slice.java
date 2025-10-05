// Source-based slice around line 844
// Method: <com.google.common.cache.CacheBuilder: long getExpireAfterAccessNanos()>

        expireAfterAccessNanos == UNSET_INT,
        "expireAfterAccess was already set to %s ns",
        expireAfterAccessNanos);
    checkArgument(duration >= 0, "duration cannot be negative: %s %s", duration, unit);
    this.expireAfterAccessNanos = unit.toNanos(duration);
    return this;
  }

  @SuppressWarnings("GoodTime") // nanos internally, should be Duration
  long getExpireAfterAccessNanos() {
    return (expireAfterAccessNanos == UNSET_INT)
        ? DEFAULT_EXPIRATION_NANOS
        : expireAfterAccessNanos;
  }

  /**
   * Specifies that active entries are eligible for automatic refresh once a fixed duration has
   * elapsed after the entry's creation, or the most recent replacement of its value. The semantics
   * of refreshes are specified in {@link LoadingCache#refresh}, and are performed by calling {@link
   * CacheLoader#reload}.

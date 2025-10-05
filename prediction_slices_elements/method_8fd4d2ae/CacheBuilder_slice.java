// Source-based slice around line 468
// Method: <com.google.common.cache.CacheBuilder: int getConcurrencyLevel()>

    checkState(
        this.concurrencyLevel == UNSET_INT,
        "concurrency level was already set to %s",
        this.concurrencyLevel);
    checkArgument(concurrencyLevel > 0);
    this.concurrencyLevel = concurrencyLevel;
    return this;
  }

  int getConcurrencyLevel() {
    return (concurrencyLevel == UNSET_INT) ? DEFAULT_CONCURRENCY_LEVEL : concurrencyLevel;
  }

  /**
   * Specifies the maximum number of entries the cache may contain.
   *
   * <p>Note that the cache <b>may evict an entry before this limit is exceeded</b>. For example, in
   * the current implementation, when {@code concurrencyLevel} is greater than {@code 1}, each
   * resulting segment inside the cache <i>independently</i> limits its own size to approximately
   * {@code maximumSize / concurrencyLevel}.

// Source-based slice around line 155
// Method: <com.google.common.collect.MapMaker: int getInitialCapacity()>

    checkState(
        this.initialCapacity == UNSET_INT,
        "initial capacity was already set to %s",
        this.initialCapacity);
    checkArgument(initialCapacity >= 0);
    this.initialCapacity = initialCapacity;
    return this;
  }

  int getInitialCapacity() {
    return (initialCapacity == UNSET_INT) ? DEFAULT_INITIAL_CAPACITY : initialCapacity;
  }

  /**
   * Guides the allowed concurrency among update operations. Used as a hint for internal sizing. The
   * table is internally partitioned to try to permit the indicated number of concurrent updates
   * without contention. Because assignment of entries to these partitions is not necessarily
   * uniform, the actual concurrency observed may vary. Ideally, you should choose a value to
   * accommodate as many threads as will ever concurrently modify the table. Using a significantly
   * higher value than you need can waste space and time, and a significantly lower value can lead

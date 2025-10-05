// Source-based slice around line 189
// Method: <com.google.common.collect.MapMaker: int getConcurrencyLevel()>

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
   * Specifies that each key (not value) stored in the map should be wrapped in a {@link
   * WeakReference} (by default, strong references are used).
   *
   * <p><b>Warning:</b> when this method is used, the resulting map will use identity ({@code ==})
   * comparison to determine equality of keys, which is a technical violation of the {@link Map}
   * specification, and may not be what you expect.

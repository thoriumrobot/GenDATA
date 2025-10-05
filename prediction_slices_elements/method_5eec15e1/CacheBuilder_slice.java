// Source-based slice around line 698
// Method: <com.google.common.cache.CacheBuilder: Strength getValueStrength()>

  }

  @CanIgnoreReturnValue
  CacheBuilder<K, V> setValueStrength(Strength strength) {
    checkState(valueStrength == null, "Value strength was already set to %s", valueStrength);
    valueStrength = checkNotNull(strength);
    return this;
  }

  Strength getValueStrength() {
    return MoreObjects.firstNonNull(valueStrength, Strength.STRONG);
  }

  /**
   * Specifies that each entry should be automatically removed from the cache once a fixed duration
   * has elapsed after the entry's creation, or the most recent replacement of its value.
   *
   * <p>When {@code duration} is zero, this method hands off to {@link #maximumSize(long)
   * maximumSize}{@code (0)}, ignoring any otherwise-specified maximum size or weight. This can be
   * useful in testing, or to disable caching temporarily without a code change.

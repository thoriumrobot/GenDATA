// Source-based slice around line 2356
// Method: <com.google.common.collect.MapMakerInternalMap: V get(Object)>

    Segment<K, V, E, S>[] segments = this.segments;
    long sum = 0;
    for (int i = 0; i < segments.length; ++i) {
      sum += segments[i].count;
    }
    return Ints.saturatedCast(sum);
  }

  @Override
  public @Nullable V get(@Nullable Object key) {
    if (key == null) {
      return null;
    }
    int hash = hash(key);
    return segmentFor(hash).get(key, hash);
  }

  /**
   * Returns the internal entry for the specified key. The entry may be computing or partially
   * collected. Does not impact recency ordering.

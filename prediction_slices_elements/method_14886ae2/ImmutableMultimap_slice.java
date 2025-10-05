// Source-based slice around line 576
// Method: <com.google.common.collect.ImmutableMultimap: int size()>

    return map.containsKey(key);
  }

  @Override
  public boolean containsValue(@Nullable Object value) {
    return value != null && super.containsValue(value);
  }

  @Override
  public int size() {
    return size;
  }

  // views

  /**
   * Returns an immutable set of the distinct keys in this multimap, in the same order as they
   * appear in this multimap.
   */
  @Override

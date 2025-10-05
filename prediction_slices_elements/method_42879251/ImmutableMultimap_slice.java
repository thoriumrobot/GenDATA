// Source-based slice around line 559
// Method: <com.google.common.collect.ImmutableMultimap: boolean isPartialView()>

    throw new UnsupportedOperationException();
  }

  /**
   * Returns {@code true} if this immutable multimap's implementation contains references to
   * user-created objects that aren't accessible via this multimap's methods. This is generally used
   * to determine whether {@code copyOf} implementations should make an explicit copy to avoid
   * memory leaks.
   */
  boolean isPartialView() {
    return map.isPartialView();
  }

  // accessors

  @Override
  public boolean containsKey(@Nullable Object key) {
    return map.containsKey(key);
  }


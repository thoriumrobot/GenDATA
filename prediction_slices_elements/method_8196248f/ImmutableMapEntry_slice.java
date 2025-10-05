// Source-based slice around line 81
// Method: <com.google.common.collect.ImmutableMapEntry: ImmutableMapEntry getNextInValueBucket()>

  @ParametricNullness
  public final V setValue(@ParametricNullness V value) {
    return super.setValue(value);
  }

  @Nullable ImmutableMapEntry<K, V> getNextInKeyBucket() {
    return null;
  }

  @Nullable ImmutableMapEntry<K, V> getNextInValueBucket() {
    return null;
  }

  /**
   * Returns true if this entry has no bucket links and can safely be reused as a terminal entry in
   * a bucket in another map.
   */
  boolean isReusable() {
    return true;
  }

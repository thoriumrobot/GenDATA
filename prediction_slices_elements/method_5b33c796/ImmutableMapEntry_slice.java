// Source-based slice around line 89
// Method: <com.google.common.collect.ImmutableMapEntry: boolean isReusable()>


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

  static class NonTerminalImmutableMapEntry<K, V> extends ImmutableMapEntry<K, V> {
    /*
     * Yes, we sometimes set nextInKeyBucket to null, even for this "non-terminal" entry. We don't
     * do that with a plain NonTerminalImmutableMapEntry, but we do it with the BiMap-specific
     * subclass below. That's because the Entry might be non-terminal in the key bucket but terminal
     * in the value bucket (or vice versa).
     */

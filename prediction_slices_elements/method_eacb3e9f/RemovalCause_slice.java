// Source-based slice around line 93
// Method: <com.google.common.cache.RemovalCause: boolean wasEvicted()>

    boolean wasEvicted() {
      return true;
    }
  };

  /**
   * Returns {@code true} if there was an automatic removal due to eviction (the cause is neither
   * {@link #EXPLICIT} nor {@link #REPLACED}).
   */
  abstract boolean wasEvicted();
}

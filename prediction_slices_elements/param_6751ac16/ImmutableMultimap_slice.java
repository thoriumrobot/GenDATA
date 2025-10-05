// Source-based slice around line 386
// Method: <com.google.common.collect.ImmutableMultimap: ImmutableMultimap copyOf(Multimap)>

   * Returns an immutable multimap containing the same mappings as {@code multimap}, in the
   * "key-grouped" iteration order described in the class documentation.
   *
   * <p>Despite the method name, this method attempts to avoid actually copying the data when it is
   * safe to do so. The exact circumstances under which a copy will or will not be performed are
   * undocumented and subject to change.
   *
   * @throws NullPointerException if any key or value in {@code multimap} is null
   */
  public static <K, V> ImmutableMultimap<K, V> copyOf(Multimap<? extends K, ? extends V> multimap) {
    if (multimap instanceof ImmutableMultimap) {
      @SuppressWarnings("unchecked") // safe since multimap is not writable
      ImmutableMultimap<K, V> kvMultimap = (ImmutableMultimap<K, V>) multimap;
      if (!kvMultimap.isPartialView()) {
        return kvMultimap;
      }
    }
    return ImmutableListMultimap.copyOf(multimap);
  }


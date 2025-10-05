// Source-based slice around line 468
// Method: <com.google.common.collect.testing.TestsForMapsInJavaUtil: M populate(M,Entry[])>

  // TODO: IdentityHashMap, AbstractMap

  private static Map<String, String> toHashMap(Entry<String, String>[] entries) {
    return populate(new HashMap<String, String>(), entries);
  }

  // TODO: call conversion constructors or factory methods instead of using
  // populate() on an empty map
  @CanIgnoreReturnValue
  private static <T, M extends Map<T, String>> M populate(M map, Entry<T, String>[] entries) {
    for (Entry<T, String> entry : entries) {
      map.put(entry.getKey(), entry.getValue());
    }
    return map;
  }

  static <T> Comparator<T> arbitraryNullFriendlyComparator() {
    return new NullFriendlyComparator<>();
  }


// Source-based slice around line 475
// Method: <com.google.common.collect.testing.TestsForMapsInJavaUtil: Comparator arbitraryNullFriendlyComparator()>

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

  private static final class NullFriendlyComparator<T> implements Comparator<T>, Serializable {
    @Override
    public int compare(T left, T right) {
      return String.valueOf(left).compareTo(String.valueOf(right));
    }
  }
}

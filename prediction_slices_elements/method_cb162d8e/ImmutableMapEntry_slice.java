// Source-based slice around line 48
// Method: <com.google.common.collect.ImmutableMapEntry: ImmutableMapEntry[] createEntryArray(int)>

   * Creates an {@code ImmutableMapEntry} array to hold parameterized entries. The result must never
   * be upcast back to ImmutableMapEntry[] (or Object[], etc.), or allowed to escape the class.
   *
   * <p>The returned array has all its elements set to their initial null values. However, we don't
   * declare it as {@code @Nullable ImmutableMapEntry[]} because our checker doesn't require newly
   * created arrays to have a {@code @Nullable} element type even when they're created directly with
   * {@code new ImmutableMapEntry[...]}, so it seems silly to insist on that only here.
   */
  @SuppressWarnings("unchecked") // Safe as long as the javadocs are followed
  static <K, V> ImmutableMapEntry<K, V>[] createEntryArray(int size) {
    return (ImmutableMapEntry<K, V>[]) new ImmutableMapEntry<?, ?>[size];
  }

  ImmutableMapEntry(K key, V value) {
    super(key, value);
    checkEntryNotNull(key, value);
  }

  // Redeclare methods to make them `final`, just to be extra-safe.


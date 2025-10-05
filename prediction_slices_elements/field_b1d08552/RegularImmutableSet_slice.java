// Source-based slice around line 46
// Method: com.google.common.collect.RegularImmutableSet.mask

  private static final Object[] EMPTY_ARRAY = new Object[0];
  static final RegularImmutableSet<Object> EMPTY =
      new RegularImmutableSet<>(EMPTY_ARRAY, 0, EMPTY_ARRAY, 0);

  private final transient Object[] elements;
  private final transient int hashCode;
  // the same values as `elements` in hashed positions (plus nulls)
  @VisibleForTesting final transient @Nullable Object[] table;
  // 'and' with an int to get a valid table index.
  private final transient int mask;

  RegularImmutableSet(Object[] elements, int hashCode, @Nullable Object[] table, int mask) {
    this.elements = elements;
    this.hashCode = hashCode;
    this.table = table;
    this.mask = mask;
  }

  @Override
  public boolean contains(@Nullable Object target) {

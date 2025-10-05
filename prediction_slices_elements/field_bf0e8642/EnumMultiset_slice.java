// Source-based slice around line 88
// Method: com.google.common.collect.EnumMultiset.counts

   */
  public static <E extends Enum<E>> EnumMultiset<E> create(Iterable<E> elements, Class<E> type) {
    EnumMultiset<E> result = create(type);
    Iterables.addAll(result, elements);
    return result;
  }

  private transient Class<E> type;
  private transient E[] enumConstants;
  private transient int[] counts;
  private transient int distinctElements;
  private transient long size;

  /** Creates an empty {@code EnumMultiset}. */
  private EnumMultiset(Class<E> type) {
    this.type = type;
    checkArgument(type.isEnum());
    this.enumConstants = type.getEnumConstants();
    this.counts = new int[enumConstants.length];
  }

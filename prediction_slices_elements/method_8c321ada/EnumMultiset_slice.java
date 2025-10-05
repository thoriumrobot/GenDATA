// Source-based slice around line 80
// Method: <com.google.common.collect.EnumMultiset: EnumMultiset create(Iterable,Class)>

    return multiset;
  }

  /**
   * Returns a new {@code EnumMultiset} instance containing the given elements. Unlike {@link
   * EnumMultiset#create(Iterable)}, this method does not produce an exception on an empty iterable.
   *
   * @since 14.0
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

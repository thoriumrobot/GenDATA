// Source-based slice around line 66
// Method: <com.google.common.collect.EnumMultiset: EnumMultiset create(Iterable)>


  /**
   * Creates a new {@code EnumMultiset} containing the specified elements.
   *
   * <p>This implementation is highly efficient when {@code elements} is itself a {@link Multiset}.
   *
   * @param elements the elements that the multiset should contain
   * @throws IllegalArgumentException if {@code elements} is empty
   */
  public static <E extends Enum<E>> EnumMultiset<E> create(Iterable<E> elements) {
    Iterator<E> iterator = elements.iterator();
    checkArgument(iterator.hasNext(), "EnumMultiset constructor passed empty Iterable");
    EnumMultiset<E> multiset = new EnumMultiset<>(iterator.next().getDeclaringClass());
    Iterables.addAll(multiset, elements);
    return multiset;
  }

  /**
   * Returns a new {@code EnumMultiset} instance containing the given elements. Unlike {@link
   * EnumMultiset#create(Iterable)}, this method does not produce an exception on an empty iterable.

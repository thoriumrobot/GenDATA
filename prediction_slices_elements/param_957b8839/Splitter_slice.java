// Source-based slice around line 399
// Method: <com.google.common.base.Splitter: List splitToList(CharSequence)>


  /**
   * Splits {@code sequence} into string components and returns them as an immutable list. If you
   * want an {@link Iterable} which may be lazily evaluated, use {@link #split(CharSequence)}.
   *
   * @param sequence the sequence of characters to split
   * @return an immutable list of the segments split from the parameter
   * @since 15.0
   */
  public List<String> splitToList(CharSequence sequence) {
    checkNotNull(sequence);

    Iterator<String> iterator = splittingIterator(sequence);
    List<String> result = new ArrayList<>();

    while (iterator.hasNext()) {
      result.add(iterator.next());
    }

    return Collections.unmodifiableList(result);

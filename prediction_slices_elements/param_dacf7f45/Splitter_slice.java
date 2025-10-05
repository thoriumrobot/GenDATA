// Source-based slice around line 368
// Method: <com.google.common.base.Splitter: Iterable split(CharSequence)>


  /**
   * Splits {@code sequence} into string components and makes them available through an {@link
   * Iterator}, which may be lazily evaluated. If you want an eagerly computed {@link List}, use
   * {@link #splitToList(CharSequence)}. Java 8+ users may prefer {@link #splitToStream} instead.
   *
   * @param sequence the sequence of characters to split
   * @return an iteration over the segments split from the parameter
   */
  public Iterable<String> split(CharSequence sequence) {
    checkNotNull(sequence);

    return new Iterable<String>() {
      @Override
      public Iterator<String> iterator() {
        return splittingIterator(sequence);
      }

      @Override
      public String toString() {

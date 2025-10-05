// Source-based slice around line 52
// Method: <com.google.common.collect.testing.TestStringSortedSetGenerator: List order(List)>


  /** Sorts the elements by their natural ordering. */
  /*
   * While the current implementation returns `this`, that's not something we mean to guarantee.
   * Callers of TestContainerGenerator.order need to be prepared for implementations to return a new
   * collection.
   */
  @SuppressWarnings("CanIgnoreReturnValueSuggester")
  @Override
  public List<String> order(List<String> insertionOrder) {
    sort(insertionOrder);
    return insertionOrder;
  }

  @Override
  public String belowSamplesLesser() {
    return "!! a";
  }

  @Override

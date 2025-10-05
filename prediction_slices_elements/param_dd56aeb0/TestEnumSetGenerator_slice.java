// Source-based slice around line 65
// Method: <com.google.common.collect.testing.TestEnumSetGenerator: List order(List)>


  /** Sorts the enums according to their natural ordering. */
  /*
   * While the current implementation returns `this`, that's not something we mean to guarantee.
   * Callers of TestContainerGenerator.order need to be prepared for implementations to return a new
   * collection.
   */
  @SuppressWarnings("CanIgnoreReturnValueSuggester")
  @Override
  public List<AnEnum> order(List<AnEnum> insertionOrder) {
    sort(insertionOrder);
    return insertionOrder;
  }
}

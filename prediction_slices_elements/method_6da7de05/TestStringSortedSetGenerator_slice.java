// Source-based slice around line 58
// Method: <com.google.common.collect.testing.TestStringSortedSetGenerator: String belowSamplesLesser()>

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
  public String belowSamplesGreater() {
    return "!! b";
  }

  @Override
  public String aboveSamplesLesser() {

// Source-based slice around line 42
// Method: <com.google.common.collect.testing.TestStringSortedSetGenerator: SortedSet create(String[])>

public abstract class TestStringSortedSetGenerator extends TestStringSetGenerator
    implements TestSortedSetGenerator<String> {

  @Override
  public SortedSet<String> create(Object... elements) {
    return (SortedSet<String>) super.create(elements);
  }

  @Override
  protected abstract SortedSet<String> create(String[] elements);

  /** Sorts the elements by their natural ordering. */
  /*
   * While the current implementation returns `this`, that's not something we mean to guarantee.
   * Callers of TestContainerGenerator.order need to be prepared for implementations to return a new
   * collection.
   */
  @SuppressWarnings("CanIgnoreReturnValueSuggester")
  @Override
  public List<String> order(List<String> insertionOrder) {

// Source-based slice around line 37
// Method: <com.google.common.collect.testing.TestStringSortedSetGenerator: SortedSet create(Object)>

 *
 * @author Jared Levy
 */
@GwtCompatible
@NullMarked
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

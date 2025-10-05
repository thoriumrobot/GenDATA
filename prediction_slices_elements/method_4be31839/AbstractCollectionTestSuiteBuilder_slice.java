// Source-based slice around line 51
// Method: <com.google.common.collect.testing.AbstractCollectionTestSuiteBuilder: List getTesters()>

 *
 * @author George van den Driessche
 */
@GwtIncompatible
public abstract class AbstractCollectionTestSuiteBuilder<
        B extends AbstractCollectionTestSuiteBuilder<B, E>, E>
    extends PerCollectionSizeTestSuiteBuilder<B, TestCollectionGenerator<E>, Collection<E>, E> {
  @SuppressWarnings("rawtypes") // class literals
  @Override
  protected List<Class<? extends AbstractTester>> getTesters() {
    return Arrays.asList(
        CollectionAddAllTester.class,
        CollectionAddTester.class,
        CollectionClearTester.class,
        CollectionContainsAllTester.class,
        CollectionContainsTester.class,
        CollectionCreationTester.class,
        CollectionEqualsTester.class,
        CollectionIsEmptyTester.class,
        CollectionIteratorTester.class,

// Source-based slice around line 44
// Method: <com.google.common.collect.testing.NavigableMapTestSuiteBuilder: NavigableMapTestSuiteBuilder using(TestSortedMapGenerator)>

import junit.framework.TestSuite;

/**
 * Creates, based on your criteria, a JUnit test suite that exhaustively tests a NavigableMap
 * implementation.
 */
@GwtIncompatible
public class NavigableMapTestSuiteBuilder<K, V> extends SortedMapTestSuiteBuilder<K, V> {
  public static <K, V> NavigableMapTestSuiteBuilder<K, V> using(
      TestSortedMapGenerator<K, V> generator) {
    NavigableMapTestSuiteBuilder<K, V> result = new NavigableMapTestSuiteBuilder<>();
    result.usingGenerator(generator);
    return result;
  }

  @SuppressWarnings("rawtypes") // class literals
  @Override
  protected List<Class<? extends AbstractTester>> getTesters() {
    List<Class<? extends AbstractTester>> testers = copyToList(super.getTesters());
    testers.add(NavigableMapNavigationTester.class);

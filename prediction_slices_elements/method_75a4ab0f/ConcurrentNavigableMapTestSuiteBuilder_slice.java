// Source-based slice around line 34
// Method: <com.google.common.collect.testing.ConcurrentNavigableMapTestSuiteBuilder: ConcurrentNavigableMapTestSuiteBuilder using(TestSortedMapGenerator)>

 * Creates, based on your criteria, a JUnit test suite that exhaustively tests a
 * ConcurrentNavigableMap implementation.
 *
 * @author Louis Wasserman
 */
@GwtIncompatible
public class ConcurrentNavigableMapTestSuiteBuilder<K, V>
    extends NavigableMapTestSuiteBuilder<K, V> {

  public static <K, V> ConcurrentNavigableMapTestSuiteBuilder<K, V> using(
      TestSortedMapGenerator<K, V> generator) {
    ConcurrentNavigableMapTestSuiteBuilder<K, V> result =
        new ConcurrentNavigableMapTestSuiteBuilder<>();
    result.usingGenerator(generator);
    return result;
  }

  @SuppressWarnings("rawtypes") // class literals
  @Override
  protected List<Class<? extends AbstractTester>> getTesters() {

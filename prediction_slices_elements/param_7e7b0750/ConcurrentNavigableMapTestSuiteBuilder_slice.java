// Source-based slice around line 35
// Method: <com.google.common.collect.testing.ConcurrentNavigableMapTestSuiteBuilder: ConcurrentNavigableMapTestSuiteBuilder using(TestSortedMapGenerator)>

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
    List<Class<? extends AbstractTester>> testers = copyToList(super.getTesters());

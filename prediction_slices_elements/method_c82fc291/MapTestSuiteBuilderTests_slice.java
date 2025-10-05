// Source-based slice around line 58
// Method: <com.google.common.collect.testing.MapTestSuiteBuilderTests: Test suite()>

/**
 * Tests {@link MapTestSuiteBuilder} by using it against maps that have various negative behaviors.
 *
 * @author George van den Driessche
 */
@AndroidIncompatible // test-suite builders
public final class MapTestSuiteBuilderTests extends TestCase {
  private MapTestSuiteBuilderTests() {}

  public static Test suite() {
    TestSuite suite = new TestSuite(MapTestSuiteBuilderTests.class.getSimpleName());
    suite.addTest(testsForHashMapNullKeysForbidden());
    suite.addTest(testsForHashMapNullValuesForbidden());
    suite.addTest(testsForSetUpTearDown());
    return suite;
  }

  private abstract static class WrappedHashMapGenerator extends TestStringMapGenerator {
    @Override
    protected final Map<String, String> create(Entry<String, String>[] entries) {

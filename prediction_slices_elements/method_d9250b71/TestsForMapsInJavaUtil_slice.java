// Source-based slice around line 58
// Method: <com.google.common.collect.testing.TestsForMapsInJavaUtil: Test suite()>

/**
 * Generates a test suite covering the {@link Map} implementations in the {@link java.util} package.
 * Can be subclassed to specify tests that should be suppressed.
 *
 * @author Kevin Bourrillion
 */
@GwtIncompatible
public class TestsForMapsInJavaUtil {

  public static Test suite() {
    return new TestsForMapsInJavaUtil().allTests();
  }

  public Test allTests() {
    TestSuite suite = new TestSuite("java.util Maps");
    suite.addTest(testsForCheckedMap());
    suite.addTest(testsForCheckedSortedMap());
    suite.addTest(testsForEmptyMap());
    suite.addTest(testsForSingletonMap());
    suite.addTest(testsForHashMap());

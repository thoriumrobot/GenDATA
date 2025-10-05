// Source-based slice around line 62
// Method: <com.google.common.collect.testing.TestsForMapsInJavaUtil: Test allTests()>

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
    suite.addTest(testsForHashtable());
    suite.addTest(testsForLinkedHashMap());
    suite.addTest(testsForTreeMapNatural());
    suite.addTest(testsForTreeMapWithComparator());

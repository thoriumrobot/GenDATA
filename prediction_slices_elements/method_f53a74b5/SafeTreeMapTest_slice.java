// Source-based slice around line 46
// Method: <com.google.common.collect.testing.SafeTreeMapTest: Test suite()>

import junit.framework.TestSuite;

/**
 * Tests for SafeTreeMap.
 *
 * @author Louis Wasserman
 */
public class SafeTreeMapTest extends TestCase {
  @AndroidIncompatible // test-suite builders
  public static Test suite() {
    TestSuite suite = new TestSuite();
    suite.addTestSuite(SafeTreeMapTest.class);
    suite.addTest(
        NavigableMapTestSuiteBuilder.using(
                new TestStringSortedMapGenerator() {
                  @Override
                  protected SortedMap<String, String> create(Entry<String, String>[] entries) {
                    NavigableMap<String, String> map = new SafeTreeMap<>(Ordering.natural());
                    for (Entry<String, String> entry : entries) {
                      map.put(entry.getKey(), entry.getValue());

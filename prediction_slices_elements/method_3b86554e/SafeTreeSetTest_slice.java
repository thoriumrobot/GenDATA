// Source-based slice around line 41
// Method: <com.google.common.collect.testing.SafeTreeSetTest: Test suite()>

import java.util.NavigableSet;
import java.util.Set;
import java.util.SortedSet;
import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

public class SafeTreeSetTest extends TestCase {
  @AndroidIncompatible // test-suite builders
  public static Test suite() {
    TestSuite suite = new TestSuite();
    suite.addTestSuite(SafeTreeSetTest.class);
    suite.addTest(
        NavigableSetTestSuiteBuilder.using(
                new TestStringSetGenerator() {
                  @Override
                  protected Set<String> create(String[] elements) {
                    return new SafeTreeSet<>(asList(elements));
                  }


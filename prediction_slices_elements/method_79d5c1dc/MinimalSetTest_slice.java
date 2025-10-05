// Source-based slice around line 32
// Method: <com.google.common.collect.testing.MinimalSetTest: Test suite()>

import junit.framework.TestCase;

/**
 * Unit test for {@link MinimalSet}.
 *
 * @author Regina O'Dell
 */
@AndroidIncompatible // test-suite builders
public class MinimalSetTest extends TestCase {
  public static Test suite() {
    return SetTestSuiteBuilder.using(
            new TestStringSetGenerator() {
              @Override
              protected Set<String> create(String[] elements) {
                return MinimalSet.of(elements);
              }
            })
        .named("MinimalSet")
        .withFeatures(
            CollectionFeature.ALLOWS_NULL_VALUES, CollectionFeature.NONE, CollectionSize.ANY)

// Source-based slice around line 32
// Method: <com.google.common.collect.testing.MinimalCollectionTest: Test suite()>

import junit.framework.TestCase;

/**
 * Unit test for {@link MinimalCollection}.
 *
 * @author Kevin Bourrillion
 */
@AndroidIncompatible // test-suite builders
public class MinimalCollectionTest extends TestCase {
  public static Test suite() {
    return CollectionTestSuiteBuilder.using(
            new TestStringCollectionGenerator() {
              @Override
              public Collection<String> create(String[] elements) {
                // TODO: MinimalCollection should perhaps throw
                for (Object element : elements) {
                  if (element == null) {
                    throw new NullPointerException();
                  }
                }

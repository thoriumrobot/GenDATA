// Source-based slice around line 40
// Method: <com.google.common.collect.testing.IteratorTesterTest: void testCanCatchDifferentLengthOfIteration()>

/**
 * Unit test for IteratorTester.
 *
 * @author Mick Killianey
 */
@GwtCompatible
@SuppressWarnings("serial") // No serialization is used in this test
public class IteratorTesterTest extends TestCase {

  public void testCanCatchDifferentLengthOfIteration() {
    IteratorTester<Integer> tester =
        new IteratorTester<Integer>(
            4, MODIFIABLE, newArrayList(1, 2, 3), IteratorTester.KnownOrder.KNOWN_ORDER) {
          @Override
          protected Iterator<Integer> newTargetIterator() {
            return Lists.newArrayList(1, 2, 3, 4).iterator();
          }
        };
    assertFailure(tester);
  }

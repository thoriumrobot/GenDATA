// Source-based slice around line 46
// Method: <com.google.common.collect.testing.HelpersTest: void testNullsBeforeB()>

import junit.framework.TestCase;

/**
 * Unit test for {@link Helpers}.
 *
 * @author Chris Povirk
 */
@GwtCompatible
public class HelpersTest extends TestCase {
  public void testNullsBeforeB() {
    testComparator(NullsBeforeB.INSTANCE, "a", "azzzzzz", null, "b", "c");
  }

  public void testIsEmpty_iterable() {
    List<Object> list = new ArrayList<>();
    assertEmpty(list);
    assertEmpty(() -> emptyIterator());

    list.add("a");
    try {

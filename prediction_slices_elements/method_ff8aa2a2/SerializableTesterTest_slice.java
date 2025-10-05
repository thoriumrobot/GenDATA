// Source-based slice around line 32
// Method: <com.google.common.testing.SerializableTesterTest: void testStringAssertions()>

import org.jspecify.annotations.Nullable;

/**
 * Tests for {@link SerializableTester}.
 *
 * @author Nick Kralevich
 */
@NullUnmarked
public class SerializableTesterTest extends TestCase {
  public void testStringAssertions() {
    String original = "hello world";
    String copy = SerializableTester.reserializeAndAssert(original);
    assertEquals(original, copy);
    assertNotSame(original, copy);
  }

  public void testClassWhichDoesNotImplementEquals() {
    ClassWhichDoesNotImplementEquals orig = new ClassWhichDoesNotImplementEquals();
    boolean errorNotThrown = false;
    try {

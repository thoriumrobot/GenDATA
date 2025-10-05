// Source-based slice around line 39
// Method: <com.google.common.testing.SerializableTesterTest: void testClassWhichDoesNotImplementEquals()>

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
      SerializableTester.reserializeAndAssert(orig);
      errorNotThrown = true;
    } catch (AssertionFailedError error) {
      // expected
      assertContains("must be Object#equals to", error.getMessage());
    }
    assertFalse(errorNotThrown);

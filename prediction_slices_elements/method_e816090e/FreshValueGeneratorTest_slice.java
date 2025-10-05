// Source-based slice around line 120
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testFreshInstance()>

/**
 * Tests for {@link FreshValueGenerator}.
 *
 * @author Ben Yu
 */
@NullUnmarked
public class FreshValueGeneratorTest extends TestCase {

  @AndroidIncompatible // problem with equality of Type objects?
  public void testFreshInstance() {
    assertFreshInstances(
        String.class,
        CharSequence.class,
        Appendable.class,
        StringBuffer.class,
        StringBuilder.class,
        Pattern.class,
        MatchResult.class,
        Number.class,
        int.class,

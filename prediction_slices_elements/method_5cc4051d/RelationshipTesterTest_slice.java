// Source-based slice around line 31
// Method: <com.google.common.testing.RelationshipTesterTest: void testNulls()>

import org.jspecify.annotations.NullUnmarked;

/**
 * Tests for {@link RelationshipTester}.
 *
 * @author Ben Yu
 */
@NullUnmarked
public class RelationshipTesterTest extends TestCase {
  public void testNulls() {
    new ClassSanityTester()
        .setDefault(ItemReporter.class, /* itemReporter */ Item::toString)
        .testNulls(RelationshipTester.class);
  }
}

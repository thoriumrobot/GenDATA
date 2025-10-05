// Source-based slice around line 44
// Method: <com.google.common.testing.EquivalenceTesterTest: void setUp()>

 * @author Gregory Kick
 */
@GwtCompatible
@NullUnmarked
public class EquivalenceTesterTest extends TestCase {
  private EquivalenceTester<Object> tester;
  private MockEquivalence equivalenceMock;

  @Override
  public void setUp() throws Exception {
    super.setUp();
    this.equivalenceMock = new MockEquivalence();
    this.tester = EquivalenceTester.of(equivalenceMock);
  }

  /** Test null reference yields error */
  public void testOf_nullPointerException() {
    assertThrows(NullPointerException.class, () -> EquivalenceTester.of(null));
  }


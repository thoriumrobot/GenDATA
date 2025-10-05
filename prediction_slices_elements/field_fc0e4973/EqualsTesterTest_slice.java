// Source-based slice around line 43
// Method: com.google.common.testing.EqualsTesterTest.equalsTester

 * Unit tests for {@link EqualsTester}.
 *
 * @author Jim McMaster
 */
@GwtCompatible
@SuppressWarnings("MissingTestCall")
@NullUnmarked
public class EqualsTesterTest extends TestCase {
  private ValidTestObject reference;
  private EqualsTester equalsTester;
  private ValidTestObject equalObject1;
  private ValidTestObject equalObject2;
  private ValidTestObject notEqualObject1;

  @Override
  public void setUp() throws Exception {
    super.setUp();
    reference = new ValidTestObject(1, 2);
    equalsTester = new EqualsTester();
    equalObject1 = new ValidTestObject(1, 2);

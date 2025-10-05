// Source-based slice around line 37
// Method: <com.google.common.testing.TestLogHandlerTest: void setUp()>

 * @author kevinb
 */
@NullUnmarked
public class TestLogHandlerTest extends TestCase {

  private TestLogHandler handler;
  private final TearDownStack stack = new TearDownStack();

  @Override
  protected void setUp() throws Exception {
    super.setUp();

    handler = new TestLogHandler();

    // You could also apply it higher up the Logger hierarchy than this
    ExampleClassUnderTest.logger.addHandler(handler);

    ExampleClassUnderTest.logger.setUseParentHandlers(false); // optional

    stack.addTearDown(

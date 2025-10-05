// Source-based slice around line 35
// Method: <com.google.common.testing.TearDownStackTest: void testSingleTearDown()>

/**
 * @author Luiz-Otavio "Z" Zorzella
 */
@GwtCompatible
@NullUnmarked
public class TearDownStackTest extends TestCase {

  private final TearDownStack tearDownStack = new TearDownStack();

  public void testSingleTearDown() throws Exception {
    TearDownStack stack = buildTearDownStack();

    SimpleTearDown tearDown = new SimpleTearDown();
    stack.addTearDown(tearDown);

    assertEquals(false, tearDown.ran);

    stack.runTearDown();

    assertEquals("tearDown should have run", true, tearDown.ran);

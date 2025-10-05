// Source-based slice around line 49
// Method: com.google.common.testing.anotherpackage.ForwardingWrapperTesterTest.tester


/**
 * Tests for {@link ForwardingWrapperTester}. Live in a different package to detect reflection
 * access issues, if any.
 *
 * @author Ben Yu
 */
public class ForwardingWrapperTesterTest extends TestCase {

  private final ForwardingWrapperTester tester = new ForwardingWrapperTester();

  public void testGoodForwarder() {
    tester.testForwarding(
        Arithmetic.class,
        new Function<Arithmetic, Arithmetic>() {
          @Override
          public Arithmetic apply(Arithmetic arithmetic) {
            return new ForwardingArithmetic(arithmetic);
          }
        });

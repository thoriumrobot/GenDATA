// Source-based slice around line 36
// Method: <com.google.common.collect.testing.OpenJdk6QueueTests: Test suite()>


/**
 * Tests the {@link Queue} implementations of {@link java.util}, suppressing tests that trip known
 * OpenJDK 6 bugs.
 *
 * @author Kevin Bourrillion
 */
@AndroidIncompatible // test-suite builders
public class OpenJdk6QueueTests extends TestsForQueuesInJavaUtil {
  public static Test suite() {
    return new OpenJdk6QueueTests().allTests();
  }

  private static final List<Method> PQ_SUPPRESS = asList(getCreateWithNullUnsupportedMethod());

  @Override
  protected Collection<Method> suppressForPriorityBlockingQueue() {
    return PQ_SUPPRESS;
  }


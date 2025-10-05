// Source-based slice around line 40
// Method: com.google.common.collect.testing.OpenJdk6QueueTests.PQ_SUPPRESS

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

  @Override
  protected Collection<Method> suppressForPriorityQueue() {
    return PQ_SUPPRESS;
  }

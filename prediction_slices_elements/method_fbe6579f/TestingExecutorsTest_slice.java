// Source-based slice around line 40
// Method: <com.google.common.util.concurrent.testing.TestingExecutorsTest: void testNoOpScheduledExecutor()>


/**
 * Tests for TestingExecutors.
 *
 * @author Eric Chang
 */
public class TestingExecutorsTest extends TestCase {
  private volatile boolean taskDone;

  public void testNoOpScheduledExecutor() throws InterruptedException {
    taskDone = false;
    Runnable task =
        new Runnable() {
          @Override
          public void run() {
            taskDone = true;
          }
        };
    ScheduledFuture<?> future =
        TestingExecutors.noOpScheduledExecutor().schedule(task, 10, MILLISECONDS);

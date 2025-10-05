// Source-based slice around line 38
// Method: com.google.common.util.concurrent.testing.TestingExecutorsTest.taskDone

import java.util.concurrent.ScheduledFuture;
import junit.framework.TestCase;

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

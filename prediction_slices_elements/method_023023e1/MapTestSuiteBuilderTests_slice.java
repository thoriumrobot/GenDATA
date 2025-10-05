// Source-based slice around line 313
// Method: <com.google.common.collect.testing.MapTestSuiteBuilderTests: Test testsForSetUpTearDown()>

      } catch (InvocationTargetException e) {
        throw e.getCause();
      } catch (IllegalAccessException e) {
        throw newLinkageError(e);
      }
    }
  }

  /** Verifies that {@code setUp} and {@code tearDown} are called in all map test cases. */
  private static Test testsForSetUpTearDown() {
    AtomicBoolean setUpRan = new AtomicBoolean();
    Runnable setUp =
        new Runnable() {
          @Override
          public void run() {
            assertFalse("previous tearDown should have run before setUp", setUpRan.getAndSet(true));
          }
        };
    Runnable tearDown =
        new Runnable() {

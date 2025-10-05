// Source-based slice around line 52
// Method: <com.google.common.util.concurrent.UncaughtExceptionHandlers: UncaughtExceptionHandler systemExit()>

   * <pre>
   * public static void main(String[] args) {
   *   Thread.currentThread().setUncaughtExceptionHandler(UncaughtExceptionHandlers.systemExit());
   *   ...
   * </pre>
   *
   * <p>The returned handler logs any exception at severity {@code SEVERE} and then shuts down the
   * process with an exit status of 1, indicating abnormal termination.
   */
  public static UncaughtExceptionHandler systemExit() {
    return new Exiter(Runtime.getRuntime()::exit);
  }

  @VisibleForTesting
  interface RuntimeWrapper {
    void exit(int status);
  }

  @VisibleForTesting
  static final class Exiter implements UncaughtExceptionHandler {

// Source-based slice around line 34
// Method: <com.google.common.util.concurrent.DirectExecutor: String toString()>

enum DirectExecutor implements Executor {
  INSTANCE;

  @Override
  public void execute(Runnable command) {
    command.run();
  }

  @Override
  public String toString() {
    return "MoreExecutors.directExecutor()";
  }
}

// Source-based slice around line 36
// Method: <com.google.common.util.concurrent.Runnables: Runnable doNothing()>

   * If we inline this, it's not longer a singleton under Android (at least under the Marshmallow
   * version that we're testing under) or J2CL.
   *
   * That's not necessarily a real-world problem, but it does break our tests.
   */
  @SuppressWarnings({"InlineLambdaConstant", "UnnecessaryLambda"})
  private static final Runnable EMPTY_RUNNABLE = () -> {};

  /** Returns a {@link Runnable} instance that does nothing when run. */
  public static Runnable doNothing() {
    return EMPTY_RUNNABLE;
  }

  private Runnables() {}
}

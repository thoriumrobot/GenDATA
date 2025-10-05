// Source-based slice around line 43
// Method: <com.google.common.util.concurrent.internal.InternalFutures: Throwable tryInternalFastPathGetFailure(InternalFutureFailureAccess)>

   * <ul>
   *   <li>{@code isDone()} must return {@code true}
   *   <li>{@code isCancelled()} must return {@code false}
   *   <li>{@code get()} must not block, and it must throw an {@code ExecutionException} with the
   *       return value of this method as its cause
   * </ul>
   */
  public static
      Throwable
      tryInternalFastPathGetFailure(InternalFutureFailureAccess future) {
    return future.tryInternalFastPathGetFailure();
  }

  private InternalFutures() {}
}

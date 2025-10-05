// Source-based slice around line 53
// Method: <com.google.common.util.concurrent.internal.InternalFutureFailureAccess: Throwable tryInternalFastPathGetFailure()>

   *       return value of this method as its cause
   * </ul>
   *
   * <p>This method is {@code protected} so that classes like {@code
   * com.google.common.util.concurrent.SettableFuture} do not expose it to their users as an
   * instance method. In the unlikely event that you need to call this method, call {@link
   * InternalFutures#tryInternalFastPathGetFailure(InternalFutureFailureAccess)}.
   */
  protected abstract
      Throwable
      tryInternalFastPathGetFailure();
}

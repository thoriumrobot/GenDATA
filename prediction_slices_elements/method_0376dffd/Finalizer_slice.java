// Source-based slice around line 214
// Method: <com.google.common.base.internal.Finalizer: Method getFinalizeReferentMethod()>

    try {
      finalizeReferentMethod.invoke(reference);
    } catch (Throwable t) {
      logger.log(Level.SEVERE, "Error cleaning up after reference.", t);
    }
    return true;
  }

  /** Looks up FinalizableReference.finalizeReferent() method. */
  private @Nullable Method getFinalizeReferentMethod() {
    Class<?> finalizableReferenceClass = finalizableReferenceClassReference.get();
    if (finalizableReferenceClass == null) {
      /*
       * FinalizableReference's class loader was reclaimed. While there's a chance that other
       * finalizable references could be enqueued subsequently (at which point the class loader
       * would be resurrected by virtue of us having a strong reference to it), we should pretty
       * much just shut down and make sure we don't keep it alive any longer than necessary.
       */
      return null;
    }

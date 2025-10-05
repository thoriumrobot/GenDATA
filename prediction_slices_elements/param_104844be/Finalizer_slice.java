// Source-based slice around line 191
// Method: <com.google.common.base.internal.Finalizer: boolean finalizeReference(Reference,Method)>

    }
  }

  /**
   * Cleans up the given reference. Catches and logs all throwables.
   *
   * @return true if the caller should continue to clean up references from the queue, false if the
   *     associated FinalizableReferenceQueue is no longer referenced.
   */
  private boolean finalizeReference(Reference<?> reference, Method finalizeReferentMethod) {
    /*
     * This is for the benefit of phantom references. Weak and soft references will have already
     * been cleared by this point.
     */
    reference.clear();

    if (reference == frqReference) {
      /*
       * The client no longer has a reference to the FinalizableReferenceQueue. We can stop.
       */

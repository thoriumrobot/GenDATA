// Source-based slice around line 160
// Method: <com.google.common.base.internal.Finalizer: boolean cleanUp(Reference)>

  }

  /**
   * Cleans up the given reference and any other references already in the queue. Catches and logs
   * all throwables.
   *
   * @return true if the caller should continue to wait for more references to be added to the
   *     queue, false if the associated FinalizableReferenceQueue is no longer referenced.
   */
  private boolean cleanUp(Reference<?> firstReference) {
    Method finalizeReferentMethod = getFinalizeReferentMethod();
    if (finalizeReferentMethod == null) {
      return false;
    }

    if (!finalizeReference(firstReference, finalizeReferentMethod)) {
      return false;
    }

    /*

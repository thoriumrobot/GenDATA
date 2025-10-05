// Source-based slice around line 236
// Method: <com.google.common.base.FinalizableReferenceQueue: void cleanUp()>

    frqRef.enqueue();
    cleanUp();
  }

  /**
   * Repeatedly dequeues references from the queue and invokes {@link
   * FinalizableReference#finalizeReferent()} on them until the queue is empty. This method is a
   * no-op if the background thread was created successfully.
   */
  void cleanUp() {
    if (threadStarted) {
      return;
    }

    Reference<?> reference;
    while ((reference = queue.poll()) != null) {
      /*
       * This is for the benefit of phantom references. Weak and soft references will have already
       * been cleared by this point.
       */

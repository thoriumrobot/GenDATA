// Source-based slice around line 226
// Method: <com.google.common.base.FinalizableReferenceQueue: void close()>

          "Failed to start reference finalizer thread."
              + " Reference cleanup will only occur when new references are created.",
          t);
    }

    this.threadStarted = threadStarted;
  }

  @Override
  public void close() {
    frqRef.enqueue();
    cleanUp();
  }

  /**
   * Repeatedly dequeues references from the queue and invokes {@link
   * FinalizableReference#finalizeReferent()} on them until the queue is empty. This method is a
   * no-op if the background thread was created successfully.
   */
  void cleanUp() {

// Source-based slice around line 141
// Method: <com.google.common.base.internal.Finalizer: void run()>

    this.finalizableReferenceClassReference = new WeakReference<>(finalizableReferenceClass);

    // Keep track of the FRQ that started us so we know when to stop.
    this.frqReference = frqReference;
  }

  /** Loops continuously, pulling references off the queue and cleaning them up. */
  @SuppressWarnings("InfiniteLoopStatement")
  @Override
  public void run() {
    while (true) {
      try {
        if (!cleanUp(queue.remove())) {
          break;
        }
      } catch (InterruptedException e) {
        // ignore
      }
    }
  }

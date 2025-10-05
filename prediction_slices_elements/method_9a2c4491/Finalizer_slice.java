// Source-based slice around line 62
// Method: <com.google.common.base.internal.Finalizer: void startFinalizer(Class,ReferenceQueue,PhantomReference)>

  /**
   * Starts the Finalizer thread. FinalizableReferenceQueue calls this method reflectively.
   *
   * @param finalizableReferenceClass FinalizableReference.class.
   * @param queue a reference queue that the thread will poll.
   * @param frqReference a phantom reference to the FinalizableReferenceQueue, which will be queued
   *     either when the FinalizableReferenceQueue is no longer referenced anywhere, or when its
   *     close() method is called.
   */
  public static void startFinalizer(
      Class<?> finalizableReferenceClass,
      ReferenceQueue<Object> queue,
      PhantomReference<Object> frqReference) {
    /*
     * We use FinalizableReference.class for two things:
     *
     * 1) To invoke FinalizableReference.finalizeReferent()
     *
     * 2) To detect when FinalizableReference's class loader has to be garbage collected, at which
     * point, Finalizer can stop running

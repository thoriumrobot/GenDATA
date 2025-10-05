// Source-based slice around line 196
// Method: com.google.common.base.FinalizableReferenceQueue.queue

  private static final Method startFinalizer;

  static {
    Class<?> finalizer =
        loadFinalizer(new SystemLoader(), new DecoupledLoader(), new DirectLoader());
    startFinalizer = getStartFinalizer(finalizer);
  }

  /** The actual reference queue that our background thread will poll. */
  final ReferenceQueue<Object> queue;

  final PhantomReference<Object> frqRef;

  /** Whether or not the background thread started successfully. */
  final boolean threadStarted;

  /** Constructs a new queue. */
  public FinalizableReferenceQueue() {
    // We could start the finalizer lazily, but I'd rather it blow up early.
    queue = new ReferenceQueue<>();

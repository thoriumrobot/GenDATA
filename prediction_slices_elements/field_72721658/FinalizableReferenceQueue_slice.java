// Source-based slice around line 198
// Method: com.google.common.base.FinalizableReferenceQueue.frqRef

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
    frqRef = new PhantomReference<>(this, queue);
    boolean threadStarted = false;

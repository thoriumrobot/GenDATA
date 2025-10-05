// Source-based slice around line 187
// Method: com.google.common.base.FinalizableReferenceQueue.startFinalizer

   * strangely when run in JDK 9. We are considering this a known issue. Please see
   * https://github.com/google/guava/issues/3086 for more information.
   */

  private static final Logger logger = Logger.getLogger(FinalizableReferenceQueue.class.getName());

  private static final String FINALIZER_CLASS_NAME = "com.google.common.base.internal.Finalizer";

  /** Reference to Finalizer.startFinalizer(). */
  private static final Method startFinalizer;

  static {
    Class<?> finalizer =
        loadFinalizer(new SystemLoader(), new DecoupledLoader(), new DirectLoader());
    startFinalizer = getStartFinalizer(finalizer);
  }

  /** The actual reference queue that our background thread will poll. */
  final ReferenceQueue<Object> queue;


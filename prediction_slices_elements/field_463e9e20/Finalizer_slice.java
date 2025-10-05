// Source-based slice around line 51
// Method: com.google.common.base.internal.Finalizer.FINALIZABLE_REFERENCE

 * loader. That way, this class doesn't prevent the main class loader from getting garbage
 * collected, and this class can detect when the main class loader has been garbage collected and
 * stop itself.
 */
public class Finalizer implements Runnable {

  private static final Logger logger = Logger.getLogger(Finalizer.class.getName());

  /** Name of FinalizableReference.class. */
  private static final String FINALIZABLE_REFERENCE = "com.google.common.base.FinalizableReference";

  /**
   * Starts the Finalizer thread. FinalizableReferenceQueue calls this method reflectively.
   *
   * @param finalizableReferenceClass FinalizableReference.class.
   * @param queue a reference queue that the thread will poll.
   * @param frqReference a phantom reference to the FinalizableReferenceQueue, which will be queued
   *     either when the FinalizableReferenceQueue is no longer referenced anywhere, or when its
   *     close() method is called.
   */

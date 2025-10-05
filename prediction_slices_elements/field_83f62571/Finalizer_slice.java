// Source-based slice around line 122
// Method: com.google.common.base.internal.Finalizer.inheritableThreadLocals

  private final PhantomReference<Object> frqReference;
  private final ReferenceQueue<Object> queue;

  // By preference, we will use the Thread constructor that has an `inheritThreadLocals` parameter.
  // But before Java 9, our only way not to inherit ThreadLocals is to zap them after the thread
  // is created, by accessing a private field.
  private static final @Nullable Constructor<Thread> bigThreadConstructor =
      getBigThreadConstructor();

  private static final @Nullable Field inheritableThreadLocals =
      (bigThreadConstructor == null) ? getInheritableThreadLocalsField() : null;

  /** Constructs a new finalizer thread. */
  private Finalizer(
      Class<?> finalizableReferenceClass,
      ReferenceQueue<Object> queue,
      PhantomReference<Object> frqReference) {
    this.queue = queue;

    this.finalizableReferenceClassReference = new WeakReference<>(finalizableReferenceClass);

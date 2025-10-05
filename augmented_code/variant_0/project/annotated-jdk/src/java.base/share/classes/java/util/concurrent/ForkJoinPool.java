/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
    @Positive
 */
    @Positive
package java.util.concurrent;

    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.lang.Thread.UncaughtExceptionHandler;
    @Positive
import java.lang.invoke.MethodHandles;
    @Positive
import java.lang.invoke.VarHandle;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.AccessControlContext;
    @Positive
import java.security.Permission;
    @Positive
import java.security.Permissions;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.security.ProtectionDomain;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Collection;
    @Positive
import java.util.Collections;
    @Positive
import java.util.List;
    @Positive
import java.util.function.Predicate;
    @Positive
import java.util.concurrent.atomic.AtomicInteger;
    @Positive
import java.util.concurrent.locks.LockSupport;
    @Positive
import java.util.concurrent.locks.ReentrantLock;
    @Positive
import java.util.concurrent.locks.Condition;

    @Positive
public class ForkJoinPool extends AbstractExecutorService {

    @Positive
    @SuppressWarnings("removal")
    @Positive
    static AccessControlContext contextWithPermissions(Permission... perms);

    @Positive
    public static interface ForkJoinWorkerThreadFactory {

    @Positive
        public ForkJoinWorkerThread newThread(ForkJoinPool pool);
    @Positive
    }

    @Positive
    static final class DefaultForkJoinWorkerThreadFactory implements ForkJoinWorkerThreadFactory {

    @Positive
        @SuppressWarnings("removal")
    @Positive
        public final ForkJoinWorkerThread newThread(ForkJoinPool pool);
    @Positive
    }

    @Positive
    static final class DefaultCommonPoolForkJoinWorkerThreadFactory implements ForkJoinWorkerThreadFactory {

    @Positive
        @SuppressWarnings("removal")
    @Positive
        public final ForkJoinWorkerThread newThread(ForkJoinPool pool);
    @Positive
    }

    @Positive
    static final class WorkQueue {

    @Positive
        static final ForkJoinTask<?> getSlot(ForkJoinTask<?>[] a, int i);

    @Positive
        static final ForkJoinTask<?> getAndClearSlot(ForkJoinTask<?>[] a, int i);

    @Positive
        static final void setSlotVolatile(ForkJoinTask<?>[] a, int i, ForkJoinTask<?> v);

    @Positive
        static final boolean casSlotToNull(ForkJoinTask<?>[] a, int i, ForkJoinTask<?> c);

    @Positive
        final boolean tryLock();

    @Positive
        final void setBaseOpaque(int b);

    @Positive
        final int getPoolIndex();

    @Positive
        final int queueSize();

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
        final boolean isEmpty();

    @Positive
        final void push(ForkJoinTask<?> task, ForkJoinPool pool);

    @Positive
        final boolean lockedPush(ForkJoinTask<?> task);

    @Positive
        final void growArray();

    @Positive
        final boolean tryUnpush(ForkJoinTask<?> task);

    @Positive
        final boolean externalTryUnpush(ForkJoinTask<?> task);

    @Positive
        final boolean tryRemove(ForkJoinTask<?> task, boolean owned);

    @Positive
        final ForkJoinTask<?> tryPoll();

    @Positive
        final ForkJoinTask<?> nextLocalTask(int cfg);

    @Positive
        final ForkJoinTask<?> nextLocalTask();

    @Positive
        @Pure
    @Positive
        final ForkJoinTask<?> peek();

    @Positive
        final void topLevelExec(ForkJoinTask<?> task, WorkQueue q);

    @Positive
        final int helpComplete(ForkJoinTask<?> task, boolean owned, int limit);

    @Positive
        final void helpAsyncBlocker(ManagedBlocker blocker);

    @Positive
        @SuppressWarnings("removal")
    @Positive
        final void initializeInnocuousWorker();

    @Positive
        final boolean isApparentlyUnblocked();
    @Positive
    }

    @Positive
    public static final ForkJoinWorkerThreadFactory defaultForkJoinWorkerThreadFactory;

    @Positive
    final String nextWorkerThreadName();

    @Positive
    final void registerWorker(WorkQueue w);

    @Positive
    final void deregisterWorker(ForkJoinWorkerThread wt, Throwable ex);

    @Positive
    final void signalWork();

    @Positive
    final void runWorker(WorkQueue w);

    @Positive
    final boolean canStop();

    @Positive
    final void uncompensate();

    @Positive
    final int helpJoin(ForkJoinTask<?> task, WorkQueue w, boolean canHelp);

    @Positive
    final int helpComplete(ForkJoinTask<?> task, WorkQueue w, boolean owned);

    @Positive
    final int helpQuiescePool(WorkQueue w, long nanos, boolean interruptible);

    @Positive
    final int externalHelpQuiescePool(long nanos, boolean interruptible);

    @Positive
    final ForkJoinTask<?> nextTaskFor(WorkQueue w);

    @Positive
    final WorkQueue submissionQueue();

    @Positive
    final void externalPush(ForkJoinTask<?> task);

    @Positive
    static WorkQueue commonQueue();

    @Positive
    final WorkQueue externalQueue();

    @Positive
    static void helpAsyncBlocker(Executor e, ManagedBlocker blocker);

    @Positive
    static int getSurplusQueuedTaskCount();

    @Positive
    public ForkJoinPool() {
    @Positive
    }

    @Positive
    public ForkJoinPool(int parallelism) {
    @Positive
    }

    @Positive
    public ForkJoinPool(int parallelism, ForkJoinWorkerThreadFactory factory, UncaughtExceptionHandler handler, boolean asyncMode) {
    @Positive
    }

    @Positive
    public ForkJoinPool(int parallelism, ForkJoinWorkerThreadFactory factory, UncaughtExceptionHandler handler, boolean asyncMode, int corePoolSize, int maximumPoolSize, int minimumRunnable, Predicate<? super ForkJoinPool> saturate, long keepAliveTime, TimeUnit unit) {
    @Positive
    }

    @Positive
    public static ForkJoinPool commonPool();

    @Positive
    public <T> T invoke(ForkJoinTask<T> task);

    @Positive
    public void execute(ForkJoinTask<?> task);

    @Positive
    @Override
    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public void execute(Runnable task);

    @Positive
    public <T> ForkJoinTask<T> submit(ForkJoinTask<T> task);

    @Positive
    @Override
    @Positive
    public <T> ForkJoinTask<T> submit(Callable<T> task);

    @Positive
    @Override
    @Positive
    public <T> ForkJoinTask<T> submit(Runnable task, T result);

    @Positive
    @Override
    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public ForkJoinTask<?> submit(Runnable task);

    @Positive
    @Override
    @Positive
    public <T> List<Future<T>> invokeAll(Collection<? extends Callable<T>> tasks);

    @Positive
    @Override
    @Positive
    public <T> List<Future<T>> invokeAll(Collection<? extends Callable<T>> tasks, long timeout, TimeUnit unit) throws InterruptedException;

    @Positive
    static final class InvokeAnyRoot<E> extends ForkJoinTask<E> {

    @Positive
        final void tryComplete(Callable<E> c);

    @Positive
        public final boolean exec();

    @Positive
        public final E getRawResult();

    @Positive
        public final void setRawResult(E v);
    @Positive
    }

    @Positive
    static final class InvokeAnyTask<E> extends ForkJoinTask<E> {

    @Positive
        public final boolean exec();

    @Positive
        public final boolean cancel(boolean mayInterruptIfRunning);

    @Positive
        public final void setRawResult(E v);

    @Positive
        public final E getRawResult();
    @Positive
    }

    @Positive
    @Override
    @Positive
    public <T> T invokeAny(Collection<? extends Callable<T>> tasks) throws InterruptedException, ExecutionException;

    @Positive
    @Override
    @Positive
    public <T> T invokeAny(Collection<? extends Callable<T>> tasks, long timeout, TimeUnit unit) throws InterruptedException, ExecutionException, TimeoutException;

    @Positive
    public ForkJoinWorkerThreadFactory getFactory();

    @Positive
    public UncaughtExceptionHandler getUncaughtExceptionHandler();

    @Positive
    public int getParallelism();

    @Positive
    public static int getCommonPoolParallelism();

    @Positive
    public int getPoolSize();

    @Positive
    public boolean getAsyncMode();

    @Positive
    public int getRunningThreadCount();

    @Positive
    public int getActiveThreadCount();

    @Positive
    public boolean isQuiescent();

    @Positive
    public long getStealCount();

    @Positive
    public long getQueuedTaskCount();

    @Positive
    public int getQueuedSubmissionCount();

    @Positive
    public boolean hasQueuedSubmissions();

    @Positive
    protected ForkJoinTask<?> pollSubmission();

    @Positive
    protected int drainTasksTo(Collection<? super ForkJoinTask<?>> c);

    @Positive
    public String toString();

    @Positive
    public void shutdown();

    @Positive
    public List<Runnable> shutdownNow();

    @Positive
    public boolean isTerminated();

    @Positive
    public boolean isTerminating();

    @Positive
    public boolean isShutdown();

    @Positive
    public boolean awaitTermination(long timeout, TimeUnit unit) throws InterruptedException;

    @Positive
    public boolean awaitQuiescence(long timeout, TimeUnit unit);

    @Positive
    public static interface ManagedBlocker {

    @Positive
        boolean block() throws InterruptedException;

    @Positive
        boolean isReleasable();
    @Positive
    }

    @Positive
    public static void managedBlock(ManagedBlocker blocker) throws InterruptedException;

    @Positive
    @Override
    @Positive
    protected <T> RunnableFuture<T> newTaskFor(Runnable runnable, T value);

    @Positive
    @Override
    @Positive
    protected <T> RunnableFuture<T> newTaskFor(Callable<T> callable);
    @Positive
}

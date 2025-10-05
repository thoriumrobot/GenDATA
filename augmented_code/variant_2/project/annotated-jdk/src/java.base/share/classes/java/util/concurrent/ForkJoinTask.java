/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive << 1 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
    @Positive
 */
    @Positive
package java.util.concurrent;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.Serializable;
    @Positive
import java.lang.invoke.MethodHandles;
    @Positive
import java.lang.invoke.VarHandle;
    @Positive
import java.lang.reflect.Constructor;
    @Positive
import java.util.Collection;
    @Positive
import java.util.List;
    @Positive
import java.util.RandomAccess;
    @Positive
import java.util.concurrent.locks.LockSupport;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public abstract class ForkJoinTask<V> implements Future<V>, Serializable {

    @Positive
    static final class Aux {

    @Positive
        final boolean casNext(Aux c, Aux v);
    @Positive
    }

    @Positive
    final int trySetThrown(Throwable ex);

    @Positive
    int trySetException(Throwable ex);

    @Positive
    public ForkJoinTask() {
    @Positive
    }

    @Positive
    static boolean isExceptionalStatus(int s);

    @Positive
    final int doExec();

    @Positive
    static final void cancelIgnoringExceptions(Future<?> t);

    @Positive
    static void rethrow(Throwable ex);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    static <T extends Throwable> void uncheckedThrow(Throwable t) throws T;

    @Positive
    public final ForkJoinTask<V> fork();

    @Positive
    public final V join();

    @Positive
    public final V invoke();

    @Positive
    public static void invokeAll(ForkJoinTask<?> t1, ForkJoinTask<?> t2);

    @Positive
    public static void invokeAll(ForkJoinTask<?>... tasks);

    @Positive
    public static <T extends ForkJoinTask<?>> Collection<T> invokeAll(Collection<T> tasks);

    @Positive
    public boolean cancel(boolean mayInterruptIfRunning);

    @Positive
    public final boolean isDone();

    @Positive
    public final boolean isCancelled();

    @Positive
    public final boolean isCompletedAbnormally();

    @Positive
    public final boolean isCompletedNormally();

    @Positive
    public final Throwable getException();

    @Positive
    public void completeExceptionally(Throwable ex);

    @Positive
    public void complete(V value);

    @Positive
    public final void quietlyComplete();

    @Positive
    public final V get() throws InterruptedException, ExecutionException;

    @Positive
    public final V get(long timeout, TimeUnit unit) throws InterruptedException, ExecutionException, TimeoutException;

    @Positive
    public final void quietlyJoin();

    @Positive
    public final void quietlyInvoke();

    @Positive
    final void awaitPoolInvoke(ForkJoinPool pool);

    @Positive
    final void awaitPoolInvoke(ForkJoinPool pool, long nanos);

    @Positive
    final V joinForPoolInvoke(ForkJoinPool pool);

    @Positive
    final V getForPoolInvoke(ForkJoinPool pool) throws InterruptedException, ExecutionException;

    @Positive
    final V getForPoolInvoke(ForkJoinPool pool, long nanos) throws InterruptedException, ExecutionException, TimeoutException;

    @Positive
    public static void helpQuiesce();

    @Positive
    public void reinitialize();

    @Positive
    public static ForkJoinPool getPool();

    @Positive
    public static boolean inForkJoinPool();

    @Positive
    public boolean tryUnfork();

    @Positive
    public static int getQueuedTaskCount();

    @Positive
    public static int getSurplusQueuedTaskCount();

    @Positive
    public abstract V getRawResult();

    @Positive
    protected abstract void setRawResult(V value);

    @Positive
    protected abstract boolean exec();

    @Positive
    protected static ForkJoinTask<?> peekNextLocalTask();

    @Positive
    protected static ForkJoinTask<?> pollNextLocalTask();

    @Positive
    protected static ForkJoinTask<?> pollTask();

    @Positive
    protected static ForkJoinTask<?> pollSubmission();

    @Positive
    public final short getForkJoinTaskTag();

    @Positive
    public final short setForkJoinTaskTag(short newValue);

    @Positive
    public final boolean compareAndSetForkJoinTaskTag(short expect, short update);

    @Positive
    static final class AdaptedRunnable<T> extends ForkJoinTask<T> implements RunnableFuture<T> {

    @Positive
        public final T getRawResult();

    @Positive
        public final void setRawResult(T v);

    @Positive
        public final boolean exec();

    @Positive
        public final void run();

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    static final class AdaptedRunnableAction extends ForkJoinTask<Void> implements RunnableFuture<Void> {

    @Positive
        public final Void getRawResult();

    @Positive
        public final void setRawResult(Void v);

    @Positive
        public final boolean exec();

    @Positive
        public final void run();

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    static final class RunnableExecuteAction extends ForkJoinTask<Void> {

    @Positive
        public final Void getRawResult();

    @Positive
        public final void setRawResult(Void v);

    @Positive
        public final boolean exec();

    @Positive
        int trySetException(Throwable ex);
    @Positive
    }

    @Positive
    static final class AdaptedCallable<T> extends ForkJoinTask<T> implements RunnableFuture<T> {

    @Positive
        public final T getRawResult();

    @Positive
        public final void setRawResult(T v);

    @Positive
        public final boolean exec();

    @Positive
        public final void run();

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    static final class AdaptedInterruptibleCallable<T> extends ForkJoinTask<T> implements RunnableFuture<T> {

    @Positive
        public final T getRawResult();

    @Positive
        public final void setRawResult(T v);

    @Positive
        public final boolean exec();

    @Positive
        public final void run();

    @Positive
        public final boolean cancel(boolean mayInterruptIfRunning);

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    public static ForkJoinTask<?> adapt(Runnable runnable);

    @Positive
    public static <T> ForkJoinTask<T> adapt(Runnable runnable, T result);

    @Positive
    public static <T> ForkJoinTask<T> adapt(Callable<? extends T> callable);
    @Positive
}

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
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
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
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.nullness.qual.PolyNull;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import static java.lang.ref.Reference.reachabilityFence;
    @Positive
import java.security.AccessControlContext;
    @Positive
import java.security.AccessControlException;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.security.PrivilegedActionException;
    @Positive
import java.security.PrivilegedExceptionAction;
    @Positive
import java.util.Collection;
    @Positive
import java.util.List;
    @Positive
import java.util.concurrent.atomic.AtomicInteger;
    @Positive
import sun.security.util.SecurityConstants;

    @Positive
@AnnotatedFor({ "interning", "nullness" })
    @Positive
@UsesObjectEquals
    @Positive
public class Executors {

    @Positive
    public static ExecutorService newFixedThreadPool(int nThreads);

    @Positive
    public static ExecutorService newWorkStealingPool(int parallelism);

    @Positive
    public static ExecutorService newWorkStealingPool();

    @Positive
    public static ExecutorService newFixedThreadPool(int nThreads, ThreadFactory threadFactory);

    @Positive
    public static ExecutorService newSingleThreadExecutor();

    @Positive
    public static ExecutorService newSingleThreadExecutor(ThreadFactory threadFactory);

    @Positive
    public static ExecutorService newCachedThreadPool();

    @Positive
    public static ExecutorService newCachedThreadPool(ThreadFactory threadFactory);

    @Positive
    public static ScheduledExecutorService newSingleThreadScheduledExecutor();

    @Positive
    public static ScheduledExecutorService newSingleThreadScheduledExecutor(ThreadFactory threadFactory);

    @Positive
    public static ScheduledExecutorService newScheduledThreadPool(int corePoolSize);

    @Positive
    public static ScheduledExecutorService newScheduledThreadPool(int corePoolSize, ThreadFactory threadFactory);

    @Positive
    public static ExecutorService unconfigurableExecutorService(ExecutorService executor);

    @Positive
    public static ScheduledExecutorService unconfigurableScheduledExecutorService(ScheduledExecutorService executor);

    @Positive
    public static ThreadFactory defaultThreadFactory();

    @Positive
    @Deprecated()
    @Positive
    public static ThreadFactory privilegedThreadFactory();

    @Positive
    public static <T> Callable<T> callable(Runnable task, T result);

    @Positive
    public static Callable<@Nullable Object> callable(Runnable task);

    @Positive
    public static Callable<@PolyNull Object> callable(final PrivilegedAction<@PolyNull ?> action);

    @Positive
    public static Callable<@PolyNull Object> callable(final PrivilegedExceptionAction<@PolyNull ?> action);

    @Positive
    @Deprecated()
    @Positive
    public static <T> Callable<T> privilegedCallable(Callable<T> callable);

    @Positive
    @Deprecated()
    @Positive
    public static <T> Callable<T> privilegedCallableUsingCurrentClassLoader(Callable<T> callable);

    @Positive
    private static final class RunnableAdapter<T> implements Callable<T> {

    @Positive
        public T call();

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    private static final class PrivilegedCallable<T> implements Callable<T> {

    @Positive
        @SuppressWarnings("removal")
    @Positive
        public T call() throws Exception;

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    private static final class PrivilegedCallableUsingCurrentClassLoader<T> implements Callable<T> {

    @Positive
        @SuppressWarnings("removal")
    @Positive
        public T call() throws Exception;

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    private static class DefaultThreadFactory implements ThreadFactory {

    @Positive
        public Thread newThread(Runnable r);
    @Positive
    }

    @Positive
    private static class PrivilegedThreadFactory extends DefaultThreadFactory {

    @Positive
        public Thread newThread(final Runnable r);
    @Positive
    }

    @Positive
    private static class DelegatedExecutorService implements ExecutorService {

    @Positive
        public void execute(Runnable command);

    @Positive
        public void shutdown();

    @Positive
        public List<Runnable> shutdownNow();

    @Positive
        public boolean isShutdown();

    @Positive
        public boolean isTerminated();

    @Positive
        public boolean awaitTermination(long timeout, TimeUnit unit) throws InterruptedException;

    @Positive
        public Future<?> submit(Runnable task);

    @Positive
        public <T> Future<T> submit(Callable<T> task);

    @Positive
        public <T> Future<T> submit(Runnable task, T result);

    @Positive
        public <T> List<Future<T>> invokeAll(Collection<? extends Callable<T>> tasks) throws InterruptedException;

    @Positive
        public <T> List<Future<T>> invokeAll(Collection<? extends Callable<T>> tasks, long timeout, TimeUnit unit) throws InterruptedException;

    @Positive
        public <T> T invokeAny(Collection<? extends Callable<T>> tasks) throws InterruptedException, ExecutionException;

    @Positive
        public <T> T invokeAny(Collection<? extends Callable<T>> tasks, long timeout, TimeUnit unit) throws InterruptedException, ExecutionException, TimeoutException;
    @Positive
    }

    @Positive
    private static class FinalizableDelegatedExecutorService extends DelegatedExecutorService {

    @Positive
        @SuppressWarnings("deprecation")
    @Positive
        protected void finalize();
    @Positive
    }

    @Positive
    private static class DelegatedScheduledExecutorService extends DelegatedExecutorService implements ScheduledExecutorService {

    @Positive
        public ScheduledFuture<?> schedule(Runnable command, long delay, TimeUnit unit);

    @Positive
        public <V> ScheduledFuture<V> schedule(Callable<V> callable, long delay, TimeUnit unit);

    @Positive
        public ScheduledFuture<?> scheduleAtFixedRate(Runnable command, long initialDelay, long period, TimeUnit unit);

    @Positive
        public ScheduledFuture<?> scheduleWithFixedDelay(Runnable command, long initialDelay, long delay, TimeUnit unit);
    @Positive
    }
    @Positive
}

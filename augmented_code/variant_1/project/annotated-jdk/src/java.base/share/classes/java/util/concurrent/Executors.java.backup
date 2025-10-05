/*
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package java.util.concurrent;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.checker.nullness.qual.PolyNull;
import org.checkerframework.framework.qual.AnnotatedFor;
import static java.lang.ref.Reference.reachabilityFence;
import java.security.AccessControlContext;
import java.security.AccessControlException;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.security.PrivilegedActionException;
import java.security.PrivilegedExceptionAction;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import sun.security.util.SecurityConstants;

@AnnotatedFor({ "interning", "nullness" })
@UsesObjectEquals
public class Executors {

    public static ExecutorService newFixedThreadPool(int nThreads);

    public static ExecutorService newWorkStealingPool(int parallelism);

    public static ExecutorService newWorkStealingPool();

    public static ExecutorService newFixedThreadPool(int nThreads, ThreadFactory threadFactory);

    public static ExecutorService newSingleThreadExecutor();

    public static ExecutorService newSingleThreadExecutor(ThreadFactory threadFactory);

    public static ExecutorService newCachedThreadPool();

    public static ExecutorService newCachedThreadPool(ThreadFactory threadFactory);

    public static ScheduledExecutorService newSingleThreadScheduledExecutor();

    public static ScheduledExecutorService newSingleThreadScheduledExecutor(ThreadFactory threadFactory);

    public static ScheduledExecutorService newScheduledThreadPool(int corePoolSize);

    public static ScheduledExecutorService newScheduledThreadPool(int corePoolSize, ThreadFactory threadFactory);

    public static ExecutorService unconfigurableExecutorService(ExecutorService executor);

    public static ScheduledExecutorService unconfigurableScheduledExecutorService(ScheduledExecutorService executor);

    public static ThreadFactory defaultThreadFactory();

    @Deprecated()
    public static ThreadFactory privilegedThreadFactory();

    public static <T> Callable<T> callable(Runnable task, T result);

    public static Callable<@Nullable Object> callable(Runnable task);

    public static Callable<@PolyNull Object> callable(final PrivilegedAction<@PolyNull ?> action);

    public static Callable<@PolyNull Object> callable(final PrivilegedExceptionAction<@PolyNull ?> action);

    @Deprecated()
    public static <T> Callable<T> privilegedCallable(Callable<T> callable);

    @Deprecated()
    public static <T> Callable<T> privilegedCallableUsingCurrentClassLoader(Callable<T> callable);

    private static final class RunnableAdapter<T> implements Callable<T> {

        public T call();

        public String toString();
    }

    private static final class PrivilegedCallable<T> implements Callable<T> {

        @SuppressWarnings("removal")
        public T call() throws Exception;

        public String toString();
    }

    private static final class PrivilegedCallableUsingCurrentClassLoader<T> implements Callable<T> {

        @SuppressWarnings("removal")
        public T call() throws Exception;

        public String toString();
    }

    private static class DefaultThreadFactory implements ThreadFactory {

        public Thread newThread(Runnable r);
    }

    private static class PrivilegedThreadFactory extends DefaultThreadFactory {

        public Thread newThread(final Runnable r);
    }

    private static class DelegatedExecutorService implements ExecutorService {

        public void execute(Runnable command);

        public void shutdown();

        public List<Runnable> shutdownNow();

        public boolean isShutdown();

        public boolean isTerminated();

        public boolean awaitTermination(long timeout, TimeUnit unit) throws InterruptedException;

        public Future<?> submit(Runnable task);

        public <T> Future<T> submit(Callable<T> task);

        public <T> Future<T> submit(Runnable task, T result);

        public <T> List<Future<T>> invokeAll(Collection<? extends Callable<T>> tasks) throws InterruptedException;

        public <T> List<Future<T>> invokeAll(Collection<? extends Callable<T>> tasks, long timeout, TimeUnit unit) throws InterruptedException;

        public <T> T invokeAny(Collection<? extends Callable<T>> tasks) throws InterruptedException, ExecutionException;

        public <T> T invokeAny(Collection<? extends Callable<T>> tasks, long timeout, TimeUnit unit) throws InterruptedException, ExecutionException, TimeoutException;
    }

    private static class FinalizableDelegatedExecutorService extends DelegatedExecutorService {

        @SuppressWarnings("deprecation")
        protected void finalize();
    }

    private static class DelegatedScheduledExecutorService extends DelegatedExecutorService implements ScheduledExecutorService {

        public ScheduledFuture<?> schedule(Runnable command, long delay, TimeUnit unit);

        public <V> ScheduledFuture<V> schedule(Callable<V> callable, long delay, TimeUnit unit);

        public ScheduledFuture<?> scheduleAtFixedRate(Runnable command, long initialDelay, long period, TimeUnit unit);

        public ScheduledFuture<?> scheduleWithFixedDelay(Runnable command, long initialDelay, long delay, TimeUnit unit);
    }
}

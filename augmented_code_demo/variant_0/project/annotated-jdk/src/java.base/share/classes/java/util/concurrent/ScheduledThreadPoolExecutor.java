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
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmpty;
    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.checker.nonempty.qual.NonEmpty;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.nullness.qual.PolyNull;
    @Positive
import org.checkerframework.checker.signedness.qual.UnknownSignedness;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import static java.util.concurrent.TimeUnit.MILLISECONDS;
    @Positive
import static java.util.concurrent.TimeUnit.NANOSECONDS;
    @Positive
import java.util.AbstractQueue;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Collection;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.List;
    @Positive
import java.util.NoSuchElementException;
    @Positive
import java.util.Objects;
    @Positive
import java.util.concurrent.atomic.AtomicLong;
    @Positive
import java.util.concurrent.locks.Condition;
    @Positive
import java.util.concurrent.locks.ReentrantLock;

    @Positive
public class ScheduledThreadPoolExecutor extends ThreadPoolExecutor implements ScheduledExecutorService {

    @Positive
    private class ScheduledFutureTask<V> extends FutureTask<V> implements RunnableScheduledFuture<V> {

    @Positive
        public long getDelay(TimeUnit unit);

    @Positive
        public int compareTo(Delayed other);

    @Positive
        public boolean isPeriodic();

    @Positive
        public boolean cancel(boolean mayInterruptIfRunning);

    @Positive
        public void run();
    @Positive
    }

    @Positive
    boolean canRunInCurrentRunState(RunnableScheduledFuture<?> task);

    @Positive
    void reExecutePeriodic(RunnableScheduledFuture<?> task);

    @Positive
    @Override
    @Positive
    void onShutdown();

    @Positive
    protected <V> RunnableScheduledFuture<V> decorateTask(Runnable runnable, RunnableScheduledFuture<V> task);

    @Positive
    protected <V> RunnableScheduledFuture<V> decorateTask(Callable<V> callable, RunnableScheduledFuture<V> task);

    @Positive
    public ScheduledThreadPoolExecutor(int corePoolSize) {
    @Positive
    }

    @Positive
    public ScheduledThreadPoolExecutor(int corePoolSize, ThreadFactory threadFactory) {
    @Positive
    }

    @Positive
    public ScheduledThreadPoolExecutor(int corePoolSize, RejectedExecutionHandler handler) {
    @Positive
    }

    @Positive
    public ScheduledThreadPoolExecutor(int corePoolSize, ThreadFactory threadFactory, RejectedExecutionHandler handler) {
    @Positive
    }

    @Positive
    long triggerTime(long delay);

    @Positive
    public ScheduledFuture<?> schedule(Runnable command, long delay, TimeUnit unit);

    @Positive
    public <V> ScheduledFuture<V> schedule(Callable<V> callable, long delay, TimeUnit unit);

    @Positive
    public ScheduledFuture<?> scheduleAtFixedRate(Runnable command, long initialDelay, long period, TimeUnit unit);

    @Positive
    public ScheduledFuture<?> scheduleWithFixedDelay(Runnable command, long initialDelay, long delay, TimeUnit unit);

    @Positive
    public void execute(Runnable command);

    @Positive
    public Future<?> submit(Runnable task);

    @Positive
    public <T> Future<T> submit(Runnable task, T result);

    @Positive
    public <T> Future<T> submit(Callable<T> task);

    @Positive
    public void setContinueExistingPeriodicTasksAfterShutdownPolicy(boolean value);

    @Positive
    public boolean getContinueExistingPeriodicTasksAfterShutdownPolicy();

    @Positive
    public void setExecuteExistingDelayedTasksAfterShutdownPolicy(boolean value);

    @Positive
    public boolean getExecuteExistingDelayedTasksAfterShutdownPolicy();

    @Positive
    public void setRemoveOnCancelPolicy(boolean value);

    @Positive
    public boolean getRemoveOnCancelPolicy();

    @Positive
    public void shutdown();

    @Positive
    public List<Runnable> shutdownNow();

    @Positive
    public BlockingQueue<Runnable> getQueue();

    @Positive
    static class DelayedWorkQueue extends AbstractQueue<Runnable> implements BlockingQueue<Runnable> {

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean contains(@UnknownSignedness Object x);

    @Positive
        public boolean remove(@UnknownSignedness Object x);

    @Positive
        @Pure
    @Positive
        public int size();

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
        public boolean isEmpty();

    @Positive
        public int remainingCapacity();

    @Positive
        @Pure
    @Positive
        public RunnableScheduledFuture<?> peek();

    @Positive
        public boolean offer(Runnable x);

    @Positive
        public void put(Runnable e);

    @Positive
        @EnsuresNonEmpty("this")
    @Positive
        public boolean add(Runnable e);

    @Positive
        public boolean offer(Runnable e, long timeout, TimeUnit unit);

    @Positive
        public RunnableScheduledFuture<?> poll();

    @Positive
        public RunnableScheduledFuture<?> take() throws InterruptedException;

    @Positive
        public RunnableScheduledFuture<?> poll(long timeout, TimeUnit unit) throws InterruptedException;

    @Positive
        public void clear();

    @Positive
        public int drainTo(Collection<? super Runnable> c);

    @Positive
        public int drainTo(Collection<? super Runnable> c, int maxElements);

    @Positive
        public Object[] toArray();

    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        @Nullable
    @Positive
        public <T> T[] toArray(@PolyNull T[] a);

    @Positive
        public Iterator<Runnable> iterator();

    @Positive
        private class Itr implements Iterator<Runnable> {

    @Positive
            @Pure
    @Positive
            @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
            public boolean hasNext();

    @Positive
            @SideEffectsOnly("this")
    @Positive
            public Runnable next(@NonEmpty Itr this);

    @Positive
            public void remove();
    @Positive
        }
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0

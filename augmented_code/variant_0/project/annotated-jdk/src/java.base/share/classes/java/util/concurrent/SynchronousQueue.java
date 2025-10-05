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
import org.checkerframework.checker.index.qual.PolyGrowShrink;
    @Positive
import org.checkerframework.checker.index.qual.Shrinkable;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.checker.nonempty.qual.PolyNonEmpty;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.nullness.qual.PolyNull;
    @Positive
import org.checkerframework.checker.signedness.qual.PolySigned;
    @Positive
import org.checkerframework.checker.signedness.qual.UnknownSignedness;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.lang.invoke.MethodHandles;
    @Positive
import java.lang.invoke.VarHandle;
    @Positive
import java.util.AbstractQueue;
    @Positive
import java.util.Collection;
    @Positive
import java.util.Collections;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Spliterator;
    @Positive
import java.util.Spliterators;
    @Positive
import java.util.concurrent.locks.LockSupport;
    @Positive
import java.util.concurrent.locks.ReentrantLock;

    @Positive
@AnnotatedFor({ "nullness" })
    @Positive
public class SynchronousQueue<E extends @NonNull Object> extends AbstractQueue<E> implements BlockingQueue<E>, java.io.Serializable {

    @Positive
    abstract static class Transferer<E> {

    @Positive
        abstract E transfer(E e, boolean timed, long nanos);
    @Positive
    }

    @Positive
    static final class TransferStack<E> extends Transferer<E> {

    @Positive
        static boolean isFulfilling(int m);

    @Positive
        static final class SNode implements ForkJoinPool.ManagedBlocker {

    @Positive
            boolean casNext(SNode cmp, SNode val);

    @Positive
            boolean tryMatch(SNode s);

    @Positive
            boolean tryCancel();

    @Positive
            boolean isCancelled();

    @Positive
            public final boolean isReleasable();

    @Positive
            public final boolean block();

    @Positive
            void forgetWaiter();
    @Positive
        }

    @Positive
        boolean casHead(SNode h, SNode nh);

    @Positive
        static SNode snode(@Nullable SNode s, Object e, SNode next, int mode);

    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        E transfer(E e, boolean timed, long nanos);

    @Positive
        void clean(SNode s);
    @Positive
    }

    @Positive
    static final class TransferQueue<E> extends Transferer<E> {

    @Positive
        static final class QNode implements ForkJoinPool.ManagedBlocker {

    @Positive
            boolean casNext(QNode cmp, QNode val);

    @Positive
            boolean casItem(Object cmp, Object val);

    @Positive
            boolean tryCancel(Object cmp);

    @Positive
            boolean isCancelled();

    @Positive
            boolean isOffList();

    @Positive
            void forgetWaiter();

    @Positive
            boolean isFulfilled();

    @Positive
            public final boolean isReleasable();

    @Positive
            public final boolean block();
    @Positive
        }

    @Positive
        void advanceHead(QNode h, QNode nh);

    @Positive
        void advanceTail(QNode t, QNode nt);

    @Positive
        boolean casCleanMe(QNode cmp, QNode val);

    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        E transfer(E e, boolean timed, long nanos);

    @Positive
        void clean(QNode pred, QNode s);
    @Positive
    }

    @Positive
    public SynchronousQueue() {
    @Positive
    }

    @Positive
    public SynchronousQueue(boolean fair) {
    @Positive
    }

    @Positive
    public void put(E e) throws InterruptedException;

    @Positive
    public boolean offer(E e, long timeout, TimeUnit unit) throws InterruptedException;

    @Positive
    public boolean offer(E e);

    @Positive
    public E take(@GuardSatisfied @Shrinkable SynchronousQueue<E> this) throws InterruptedException;

    @Positive
    public E poll(@GuardSatisfied @Shrinkable SynchronousQueue<E> this, long timeout, TimeUnit unit) throws InterruptedException;

    @Positive
    public E poll(@GuardSatisfied @Shrinkable SynchronousQueue<E> this);

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
    public boolean isEmpty();

    @Positive
    @Pure
    @Positive
    public int size();

    @Positive
    public int remainingCapacity();

    @Positive
    public void clear(@GuardSatisfied @Shrinkable SynchronousQueue<E> this);

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
    public boolean contains(@GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Positive
    public boolean remove(@Shrinkable SynchronousQueue<E> this, @GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Positive
    @Pure
    @Positive
    public boolean containsAll(Collection<? extends @UnknownSignedness Object> c);

    @Positive
    public boolean removeAll(@Shrinkable SynchronousQueue<E> this, Collection<? extends @UnknownSignedness Object> c);

    @Positive
    public boolean retainAll(@GuardSatisfied @Shrinkable SynchronousQueue<E> this, Collection<? extends @UnknownSignedness Object> c);

    @Positive
    @Pure
    @Positive
    public E peek();

    @Positive
    @SideEffectFree
    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    public Iterator<E> iterator(@PolyGrowShrink @PolyNonEmpty SynchronousQueue<E> this);

    @Positive
    @SideEffectFree
    @Positive
    public Spliterator<E> spliterator();

    @Positive
    @SideEffectFree
    @Positive
    @PolyNull
    @Positive
    @PolySigned
    @Positive
    public Object[] toArray(SynchronousQueue<@PolyNull @PolySigned E> this);

    @Positive
    @SideEffectFree
    @Positive
    @Nullable
    @Positive
    public <T> T[] toArray(@PolyNull T[] a);

    @Positive
    public String toString();

    @Positive
    public int drainTo(@GuardSatisfied @Shrinkable SynchronousQueue<E> this, Collection<? super E> c);

    @Positive
    public int drainTo(@GuardSatisfied @Shrinkable SynchronousQueue<E> this, Collection<? super E> c, int maxElements);

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static class WaitQueue implements java.io.Serializable {
    @Positive
    }

    @Positive
    static class LifoWaitQueue extends WaitQueue {
    @Positive
    }

    @Positive
    static class FifoWaitQueue extends WaitQueue {
    @Positive
    }
    @Positive
}

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
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmpty;
    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.checker.nonempty.qual.NonEmpty;
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
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import java.lang.invoke.MethodHandles;
    @Positive
import java.lang.invoke.VarHandle;
    @Positive
import java.util.AbstractQueue;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Collection;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.NoSuchElementException;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Queue;
    @Positive
import java.util.Spliterator;
    @Positive
import java.util.Spliterators;
    @Positive
import java.util.concurrent.locks.LockSupport;
    @Positive
import java.util.function.Consumer;
    @Positive
import java.util.function.Predicate;

    @Positive
public class LinkedTransferQueue<E> extends AbstractQueue<E> implements TransferQueue<E>, java.io.Serializable {

    @Positive
    static final class Node implements ForkJoinPool.ManagedBlocker {

    @Positive
        final boolean casNext(Node cmp, Node val);

    @Positive
        final boolean casItem(Object cmp, Object val);

    @Positive
        final void selfLink();

    @Positive
        final void appendRelaxed(Node next);

    @Positive
        final boolean isMatched();

    @Positive
        final boolean tryMatch(Object cmp, Object val);

    @Positive
        final boolean cannotPrecede(boolean haveData);

    @Positive
        public final boolean isReleasable();

    @Positive
        public final boolean block();
    @Positive
    }

    @Positive
    final Node firstDataNode();

    @Positive
    public String toString();

    @Positive
    @PolyNull
    @Positive
    @PolySigned
    @Positive
    public Object[] toArray(LinkedTransferQueue<@PolyNull @PolySigned E> this);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @Nullable
    @Positive
    public <T> T[] toArray(@PolyNull T[] a);

    @Positive
    final class Itr implements Iterator<E> {

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public final boolean hasNext();

    @Positive
        @SideEffectsOnly("this")
    @Positive
        public final E next(@NonEmpty Itr this);

    @Positive
        public void forEachRemaining(Consumer<? super E> action);

    @Positive
        public final void remove();
    @Positive
    }

    @Positive
    final class LTQSpliterator implements Spliterator<E> {

    @Positive
        public Spliterator<E> trySplit();

    @Positive
        public void forEachRemaining(Consumer<? super E> action);

    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        public boolean tryAdvance(Consumer<? super E> action);

    @Positive
        public long estimateSize();

    @Positive
        public int characteristics();
    @Positive
    }

    @Positive
    public Spliterator<E> spliterator();

    @Positive
    final void unsplice(Node pred, Node s);

    @Positive
    public LinkedTransferQueue() {
    @Positive
    }

    @Positive
    public LinkedTransferQueue(Collection<? extends E> c) {
    @Positive
    }

    @Positive
    public void put(E e);

    @Positive
    public boolean offer(E e, long timeout, TimeUnit unit);

    @Positive
    public boolean offer(E e);

    @Positive
    @EnsuresNonEmpty("this")
    @Positive
    public boolean add(E e);

    @Positive
    public boolean tryTransfer(@GuardSatisfied @Shrinkable LinkedTransferQueue<E> this, E e);

    @Positive
    public void transfer(@GuardSatisfied @Shrinkable LinkedTransferQueue<E> this, E e) throws InterruptedException;

    @Positive
    public boolean tryTransfer(@GuardSatisfied @Shrinkable LinkedTransferQueue<E> this, E e, long timeout, TimeUnit unit) throws InterruptedException;

    @Positive
    public E take(@GuardSatisfied @Shrinkable LinkedTransferQueue<E> this) throws InterruptedException;

    @Positive
    public E poll(@GuardSatisfied @Shrinkable LinkedTransferQueue<E> this, long timeout, TimeUnit unit) throws InterruptedException;

    @Positive
    public E poll(@GuardSatisfied @Shrinkable LinkedTransferQueue<E> this);

    @Positive
    public int drainTo(@GuardSatisfied @Shrinkable LinkedTransferQueue<E> this, Collection<? super E> c);

    @Positive
    public int drainTo(@GuardSatisfied @Shrinkable LinkedTransferQueue<E> this, Collection<? super E> c, int maxElements);

    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    public Iterator<E> iterator(@PolyGrowShrink @PolyNonEmpty LinkedTransferQueue<E> this);

    @Positive
    @Pure
    @Positive
    public E peek();

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
    public boolean isEmpty();

    @Positive
    public boolean hasWaitingConsumer();

    @Positive
    @Pure
    @Positive
    public int size();

    @Positive
    public int getWaitingConsumerCount();

    @Positive
    public boolean remove(@Shrinkable LinkedTransferQueue<E> this, @GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
    public boolean contains(@GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Positive
    public int remainingCapacity();

    @Positive
    public boolean removeIf(@Shrinkable LinkedTransferQueue<E> this, Predicate<? super E> filter);

    @Positive
    public boolean removeAll(@Shrinkable LinkedTransferQueue<E> this, Collection<? extends @NonNull @UnknownSignedness Object> c);

    @Positive
    public boolean retainAll(@GuardSatisfied @Shrinkable LinkedTransferQueue<E> this, Collection<? extends @NonNull @UnknownSignedness Object> c);

    @Positive
    public void clear(@GuardSatisfied @Shrinkable LinkedTransferQueue<E> this);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    void forEachFrom(Consumer<? super E> action, Node p);

    @Positive
    public void forEach(Consumer<? super E> action);
    @Positive
}

// CFWR semantic augmentation - variant 0

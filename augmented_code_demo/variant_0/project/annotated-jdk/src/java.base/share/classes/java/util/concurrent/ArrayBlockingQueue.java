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
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.lang.ref.WeakReference;
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
import java.util.Spliterator;
    @Positive
import java.util.Spliterators;
    @Positive
import java.util.concurrent.locks.Condition;
    @Positive
import java.util.concurrent.locks.ReentrantLock;
    @Positive
import java.util.function.Consumer;
    @Positive
import java.util.function.Predicate;

    @Positive
@AnnotatedFor({ "nullness" })
    @Positive
public class ArrayBlockingQueue<E extends Object> extends AbstractQueue<E> implements BlockingQueue<E>, java.io.Serializable {

    @Positive
    static final int inc(int i, int modulus);

    @Positive
    static final int dec(int i, int modulus);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    final E itemAt(int i);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    static <E> E itemAt(Object[] items, int i);

    @Positive
    void removeAt(final int removeIndex);

    @Positive
    public ArrayBlockingQueue(int capacity) {
    @Positive
    }

    @Positive
    public ArrayBlockingQueue(int capacity, boolean fair) {
    @Positive
    }

    @Positive
    public ArrayBlockingQueue(int capacity, boolean fair, Collection<? extends E> c) {
    @Positive
    }

    @Positive
    @EnsuresNonEmpty("this")
    @Positive
    public boolean add(E e);

    @Positive
    public boolean offer(E e);

    @Positive
    public void put(E e) throws InterruptedException;

    @Positive
    public boolean offer(E e, long timeout, TimeUnit unit) throws InterruptedException;

    @Positive
    public E poll(@GuardSatisfied @Shrinkable ArrayBlockingQueue<E> this);

    @Positive
    public E take(@GuardSatisfied @Shrinkable ArrayBlockingQueue<E> this) throws InterruptedException;

    @Positive
    public E poll(@GuardSatisfied @Shrinkable ArrayBlockingQueue<E> this, long timeout, TimeUnit unit) throws InterruptedException;

    @Positive
    @Pure
    @Positive
    public E peek();

    @Positive
    @Pure
    @Positive
    public int size();

    @Positive
    public int remainingCapacity();

    @Positive
    public boolean remove(@Shrinkable ArrayBlockingQueue<E> this, @GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
    public boolean contains(@GuardSatisfied @Nullable @UnknownSignedness Object o);

    @Positive
    @PolyNull
    @Positive
    @PolySigned
    @Positive
    public Object[] toArray(ArrayBlockingQueue<@PolyNull @PolySigned E> this);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @Nullable
    @Positive
    public <T> T[] toArray(@PolyNull T[] a);

    @Positive
    public String toString();

    @Positive
    public void clear(@GuardSatisfied @Shrinkable ArrayBlockingQueue<E> this);

    @Positive
    public int drainTo(@GuardSatisfied @Shrinkable ArrayBlockingQueue<E> this, Collection<? super E> c);

    @Positive
    public int drainTo(@GuardSatisfied @Shrinkable ArrayBlockingQueue<E> this, Collection<? super E> c, int maxElements);

    @Positive
    @PolyGrowShrink
    @Positive
    @PolyNonEmpty
    @Positive
    public Iterator<E> iterator(@PolyGrowShrink @PolyNonEmpty ArrayBlockingQueue<E> this);

    @Positive
    class Itrs {

    @Positive
        private class Node extends WeakReference<Itr> {
    @Positive
        }

    @Positive
        void doSomeSweeping(boolean tryHarder);

    @Positive
        void register(Itr itr);

    @Positive
        void takeIndexWrapped();

    @Positive
        void removedAt(int removedIndex);

    @Positive
        void queueIsEmpty();

    @Positive
        void elementDequeued();
    @Positive
    }

    @Positive
    private class Itr implements Iterator<E> {

    @Positive
        boolean isDetached();

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean hasNext();

    @Positive
        @SideEffectsOnly("this")
    @Positive
        public E next(@NonEmpty Itr this);

    @Positive
        public void forEachRemaining(Consumer<? super E> action);

    @Positive
        public void remove();

    @Positive
        void shutdown();

    @Positive
        boolean removedAt(int removedIndex);

    @Positive
        boolean takeIndexWrapped();
    @Positive
    }

    @Positive
    public Spliterator<E> spliterator();

    @Positive
    public void forEach(Consumer<? super E> action);

    @Positive
    public boolean removeIf(@Shrinkable ArrayBlockingQueue<E> this, Predicate<? super E> filter);

    @Positive
    public boolean removeAll(@Shrinkable ArrayBlockingQueue<E> this, Collection<? extends @NonNull @UnknownSignedness Object> c);

    @Positive
    public boolean retainAll(@GuardSatisfied @Shrinkable ArrayBlockingQueue<E> this, Collection<? extends @NonNull @UnknownSignedness Object> c);

    @Positive
    void checkInvariants();
    @Positive
}

// CFWR semantic augmentation - variant 0
